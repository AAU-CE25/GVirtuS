# Gusto — resultados de solidificacion (2026-08-02)

Todo lo de aqui se ejecuto. Cada afirmacion lleva su evidencia. Lo que no se ejecuto se dice.

---

## 1. Causa raiz verificada: el experimento que motivaba el trabajo era inerte

**Lo que se creia**: un pool de 8 slots colapsaba con llama a CONC=8 y 32 lo arreglaba, luego
umbral y capacidad del pool son un par acoplado.

**Lo medido**: el pool que consume H2D lo anuncia el **backend** (`send_rma_setup` empaqueta sus
`rx_slots`; el cliente hace `ucp_put` sobre ellos). El arnes ponia `GVIRTUS_RMA_SLOTS` en el
contenedor del **frontend**, y `reset_backend_fault.sh` no propaga esa variable. Resultado en
los DOS brazos, identico:

```
[GVS] RMA fast path declined: message 3170196 B exceeds the peer's largest slot 131072 B
[GVS] rma_setup: epoch 5 arrived, 0/2 slots in flight
```

El backend se quedo en su default de **2 slots de 128 KiB** y las transferencias de 3,17 MB de
llama **nunca tomaron el camino RMA**. Por eso 8, 32 y 64 daban 591,6 identico: la perilla era
inerte y los tres brazos corrian por AM eager.

**Arreglo**: `~/reset_backend_pool.sh` (dpu-01) propaga `GVIRTUS_RMA_SLOTS`,
`GVIRTUS_RMA_SLOT_CAP_MB`, `GVIRTUS_RMA_SLOT_MIN_MB` y `GVIRTUS_RMA_PREALLOC`, y **verifica el
env efectivo dentro del contenedor** antes de devolver el control. Verificacion de que ahora
si se ejercita: `rma_setup: epoch 2 arrived, 0/8 slots in flight` y cero lineas de decline.

## 2. Relacion umbral → demanda de RMA (medida, no inferida)

llama-7B, CONC=8, ventana 45 s, backend con 8 slots de 4 MiB, 1 conexion GVirtuS:

| politica | reservas RMA | ops totales | % a RMA | **peak_inflight** | waited | declines |
|---|---:|---:|---:|---:|---:|---:|
| scalar 4 MiB | 65 | 1 531 381 | 0,004 % | **2** de 8 | 0 | 0 |
| quadrant | 17 097 | 1 300 310 | 1,31 % | **2** de 8 | 0 | 0 |

- Bajar el umbral multiplica las **admisiones por ×263**.
- El **pico de slots vivos no se mueve**: 2 de 8 en las dos politicas.
- **Refuta** «bajar el suelo ×512 multiplica la demanda de slots ×512». La carga es
  peticion/respuesta sincrona: el ack de consumo libera el slot antes de que haga falta el
  siguiente.

La afirmacion defendible es la que el encargo autoriza: *lowering the threshold increases the
fraction of transfers admitted to RMA and therefore increases the number of RMA operations*,
**pero no el numero de slots simultaneamente vivos en esta carga**.

## 3. El colapso no es agotamiento del pool

En la corrida que abortó: `peak_inflight=2` de 8, `peak_waiters=0`, `waited=0`, y
`decline_capacity = decline_timeout = decline_swap = decline_epfail = 0`.

**El backend nunca muere.** Contenedor `Up` y listener en `25.25.25.2:32222` intactos. Quien
aborta es el **frontend**:

```
ggml_backend_cuda_buffer_get_tensor  →  cudaStreamSynchronize((cudaStream_t)0x2)
CUDA error: invalid argument  →  ggml_abort()
```

`0x2` es `cudaStreamPerThread`. Pila: `server_slot::prompt_save` → `state_seq_get_data` →
`llama_io_write_host` (D2H de estado de KV cache).

Los **248 021 «fallos»** del registro original son **consecuencia**: con el proceso muerto el
generador recibe conexión rechazada al instante, lo que da ~5 500 fallos/s en 45 s. Un
agotamiento de pool habría dado esperas de 30 s y decenas de fallos.

### 3.1 Frecuencia: 1 de 9, y el intervalo es enorme

Corridas de llama con la MISMA configuración (quadrant, 8 slots de 4 MiB en el backend,
umbrales D2H por defecto):

| tanda | corridas | aborts |
|---|---:|---:|
| verificación inicial | 1 | **1** |
| aislamiento, brazo B | 3 | 0 |
| B4, quadrant | 5 | 0 |
| **total** | **9** | **1** |

Control con `scalar` en las mismas condiciones: **0 de 5**.

**Lo que se puede afirmar**: el abort existe, está capturado con traza completa, y ocurrió una
vez en nueve corridas. **Lo que NO se puede afirmar**: una tasa. Con un solo evento el
intervalo de confianza va de «muy raro» a «uno de cada dos», y separar quadrant de scalar
(1/9 frente a 0/5) no alcanza significación con estos tamaños. Hace falta orden de 50-100
corridas por brazo para acotarlo, y eso son ~10 h de banco.

**Descartado por medida**, no por argumento: agotamiento de pool (contadores arriba), muerte
del backend (sigue escuchando), y admisión de D2H — subir `GVIRTUS_RMA_MIN_D2H_{PINNED,
PAGEABLE}` a 8 MiB deja las reservas prácticamente iguales (17 073 frente a 17 097), luego lo
que llama admite al camino RMA es **H2D**, y mover el umbral D2H no cambia el régimen.

**Causa raíz: abierta.** La hipótesis viva es la interacción entre `cudaStreamPerThread` y el
reenvío de streams del frontend, que es donde `0x2` tendría que traducirse; no está
comprobada y no se presenta como conclusión.

## 4. Coste de memoria del provisioning (medido)

Medido en el backend con `GVS_SLOT_CAP_MB=256`, `GVS_SLOT_MIN_MB=4`, `GVS_PREALLOC=1`, trafico
real de 4 MiB. **El pool es memoria host pinned (`cudaHostAlloc`): no aparece en `nvidia-smi`.**
La primera medida miraba solo la GPU y por eso daba cero; se rehizo mirando el RSS del
contenedor del backend.

| slots | RSS en reposo (MiB) | RSS con cliente | delta GPU (MiB) | slots anunciados |
|---:|---:|---:|---:|---:|
| 8 | 2 183 | 2 190 | 0 | 8 |
| 16 | 4 240 | 4 247 | 0 | 16 |
| 24 | 6 297 | 6 305 | 0 | 24 |
| 32 | 8 355 | 8 363 | 0 | 32 |
| 64 | 16 588 | 36 874 | 11 924 | 64 |

**Lectura**: el coste es **lineal y de ~257 MiB por slot** (2 183 → 4 240 → 6 297 → 8 355 →
16 588; diferencias de 2 057 MiB por cada 8 slots). 257 MiB es el **techo**
`GVIRTUS_RMA_SLOT_CAP_MB`, **no** el tamano del trafico: con `GVIRTUS_RMA_PREALLOC=1`,
`rma_slot_cap_for(0)` devuelve el techo porque todavia no hay evidencia de trafico. El pool se
reserva entero al arrancar, antes de que ninguna transferencia lo justifique.

Consecuencias de diseno, las dos accionables:

1. **«Sube el numero de slots» no es gratis.** Pasar de 8 a 32 slots cuesta **6,2 GB de RAM
   fijada** en el backend; a 64, **16,6 GB**. En un backend multi-inquilino eso es memoria que
   se le quita a los demas.
2. **El techo importa mas que el numero.** Con el techo ajustado al trafico (4 MiB en vez de
   256 MB) los mismos 64 slots costarian ~260 MiB en total. La perilla cara es
   `SLOT_CAP_MB`, no `SLOTS`, y hoy el arnes las movia sin distinguirlas.

A 64 slots aparece ademas un salto de **11,9 GB de memoria de GPU** y +20 GB de RSS al conectar
el cliente, que no aparece a 32 ni por debajo. El shadow GPU es **condicional** (solo si la
conexion prueba trafico device-destined), asi que el salto es consistente con que el shadow se
materialice en esa configuracion y no en las otras. **No esta explicado por que solo a 64**, y
se deja dicho como observacion, no como mecanismo.

## 5. Matriz de inyeccion de fallos

### 5.1 Un defecto del arnes, encontrado y corregido

La primera version puso las tres inyecciones en el backend. **Solo dos viven ahi**:

| inyeccion | funcion | lado |
|---|---|---|
| `DupAck`, `StaleAck` | `send_slot_consumed` (`:2351,2361`) | **servidor** |
| `DelayAck` | `release_remote_slot` (`:2270`) | **cliente** |

Con `delay_ack` en el backend las celdas corrian **inertes**. Mismo patron que la perilla del
pool: variable puesta en el lado que no la lee.

### 5.2 Un hueco de instrumentacion, encontrado y tapado

`release_remote_slot` solo contaba `(state==InFlight && generacion!=actual)`. Un ack duplicado
que llega con el slot ya `Free` **no incrementaba nada** ⇒ «la guarda lo rechazo» y «el fallo
no se ejercito» daban los dos cero. Anadidos `ack_on_free` y `ack_applied`.

### 5.3 Resultados con el arnes y la instrumentacion corregidos

`rma_checksum 4 MiB × 96`, backend 8 slots, politica quadrant:

| celda | fallo | lado | ablacion | reservas | applied | on_free | gen_mm | FAIL | backend |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 control | — | — | completo | 96 | 96 | 0 | 0 | 0 | VIVO |
| 2 | dup_ack | backend | completo | 96 | 96 | **96** | 0 | 0 | VIVO |
| 3 | dup_ack | backend | sin generacion | 96 | 96 | **96** | 0 | 0 | VIVO |
| 4 | delay_ack | frontend | completo | 96 | 96 | **94** | 0 | 0 | VIVO |
| 5 | delay_ack | frontend | sin generacion | 96 | 96 | **94** | 0 | 0 | VIVO |
| **6** | **stale_ack** | **backend** | **completo** | **7** | **0** | 0 | **1** | 0 | VIVO |
| **7** | **stale_ack** | **backend** | **sin generacion** | **96** | **96** | 0 | **96** | 0 | VIVO |
| 8 | delay_ack | frontend | sin epoch | 96 | 96 | 94 | 0 | 0 | VIVO |

**Las celdas 6 y 7 son la demostracion.** Es el unico par donde retirar la guarda cambia el
comportamiento, y lo cambia por completo:

- **Con la guarda (6)**: el ack de generacion obsoleta se **rechaza** (`gen_mismatch=1`), el
  slot **no** vuelve a `Free`, el pool se queda sin slots reutilizables tras **7** reservas y
  el sistema **degrada al camino AM**. Cero corrupcion, backend vivo. Es exactamente el
  comportamiento que se quiere de una proteccion: prefiere perder rendimiento antes que
  liberar un buffer que el par sigue usando.
- **Sin la guarda (7)**: las **96** transferencias siguen por RMA y se cuentan **96
  violaciones** del protocolo, cada una liberando un slot cuya generacion ya no casa. El
  checksum no detecta corrupcion **en esta carga**, porque es sincrona y el backend ya habia
  terminado de consumir; pero el invariante esta roto 96 veces y solo el contador lo ve.

Ese ultimo matiz importa y no se debe suavizar: **ausencia de corrupcion no es prueba de
correctitud**. La celda 7 pasa el checksum y viola el protocolo 96 veces.

**Las celdas 2-5 y 8 no demuestran nada sobre la guarda de generacion**, y decirlo es parte del
resultado. `on_free` alto con `gen_mismatch=0` significa que el ack duplicado o retrasado cayo
sobre un slot **ya libre**: lo rechaza la comprobacion de **estado**, no la de generacion. Por
eso quitar la guarda (celdas 3, 5) no cambia una sola cifra. La condicion ABA -- ack viejo que
llega cuando el slot ya se reasigno -- **no se construye** con un retardo fijo sobre una carga
sincrona, y que se construya o no depende del reloj.

### 5.4 `hold_ack`: la condicion ABA por construccion, y las tres iteraciones que costo

**El mecanismo**. `GVS_FAULT=hold_ack` (lado cliente): el ack de (slot S, generacion G) se
aplica con normalidad **y se guarda una copia**; cuando `WriteIovRma` vuelve a reservar ese
mismo slot -- ya con generacion G+k -- la copia se **re-entrega**. Es un duplicado tardio que
llega despues de la reasignacion: la condicion ABA por construccion, no por temporizacion.
Contadores nuevos: `ack_held`, `ack_released`.

**Las tres versiones, porque los dos fallos son instructivos**:

1. *Retener el ack* (v1): se auto-bloquea. Si el ack no se aplica, el slot se queda `InFlight`
   para siempre, luego nunca se reasigna, luego la copia nunca se entrega. Medido:
   `released=0`, reservas paradas en 7. La inyeccion impedia la condicion que queria provocar.
2. *Replay armado en el PRIMER ack* (v2): arma pero no entrega. La traza lo dijo:
   `replay? armado slot=0 vs reasignado slot=9`, 190 veces. El primer ack viene de una
   transferencia de arranque contra la layout **inicial**; luego el pool crece, cambia el
   epoch, y el regimen estable usa otro slot. El slot 0 pertenece a una layout superada y no
   vuelve.
3. *Replay armado en el ack numero 20* (v3): la layout ya es la definitiva. Funciona.

**Resultado (v3)** — `rma_checksum 4 MiB x 96`, backend 8 slots, quadrant:

| celda | ablacion | reservas | applied | held | released | gen_mm | **on_free** | FAIL | backend |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| control | completo | 96 | 96 | 0 | 0 | 0 | 0 | 0 | VIVO |
| `hold_ack` | **completo** | 96 | 96 | 1 | 1 | **1** | **0** | 0 | VIVO |
| `hold_ack` | **sin generacion** | 96 | 96 | 1 | 1 | 1 | **1** | 0 | VIVO |

**La firma que separa las dos variantes es `on_free`**, y el mecanismo es exactamente el que
la guarda existe para atajar:

- **Con la guarda**: el ack rancio llega, el slot esta `InFlight` con otra generacion, se
  **rechaza** (`gen_mismatch=1`) y el slot **sigue en vuelo**. El ack legitimo llega despues y
  lo libera con normalidad. `on_free=0`.
- **Sin la guarda**: el ack rancio **libera un slot vivo** -- esa es la violacion. Cuando llega
  el ack legitimo de la transferencia que seguia en curso, encuentra el slot ya libre:
  `on_free=1`. Liberacion prematura, observable y reproducible.

**Sin corrupcion en ninguna de las dos, y hay que decirlo asi**: con transferencias sincronas
de 4 MiB el backend ya habia terminado de consumir cuando el slot se libero de mas, asi que el
checksum no ve nada. **La violacion del invariante es real y esta medida; que se materialice en
corrupcion depende de la carga.** Ausencia de corrupcion no es prueba de correctitud.

## 5.5 Validacion del provisioning (F4)

`gusto_validate_pool_cfg()` se llama antes de reservar nada e imprime la configuracion
efectiva. Seis casos, todos verificados contra el backend real:

| caso | `GVIRTUS_RMA_SLOTS` | cap | presupuesto | resultado | listener |
|---|---|---|---|---|---|
| valido | 8 | 32 MB | — | acepta: `slots=8 cap=32.1 MiB total=513.0 MiB` | SI |
| cero | 0 | 32 MB | — | **RECHAZADO** (no es entero positivo) | SI |
| basura | `abc` | 32 MB | — | **RECHAZADO** | SI |
| enorme | 99999 | 32 MB | — | **RECHAZADO** (fuera de [1,1024]) | SI |
| sobre presupuesto | 32 | 256 MB | 100 MiB | **RECHAZADO** (pide 16 388 MiB) | SI |
| bajo presupuesto | 8 | 32 MB | 8 GiB | acepta | SI |

**En los cuatro rechazos el listener sigue en pie**: una configuracion invalida no construye el
pool y el transporte se queda en el camino AM. Es degradacion, no fallo, que es el requisito.

Dos defectos propios encontrados al validar, ambos de la misma familia:

1. `GVIRTUS_RMA_SLOTS=0` **no llegaba** a la validacion: el lambda `env_size` convierte 0 en el
   default 2 en silencio. Se comprueba ahora el valor **crudo** antes de normalizar.
2. `GUSTO_RMA_HOST_POOL_BUDGET_BYTES` **no se propagaba** al contenedor del backend, asi que la
   comprobacion existia en el codigo y no se ejercitaba nunca. Tercera aparicion esta noche de
   «perilla puesta en el lado que no la lee»; ahora `reset_backend_pool.sh` la reenvia.

## 5.6 Contencion del pool: no existe en ninguna configuracion medida

La prueba de pool insuficiente con varios clientes **no ejercita nada**, y el motivo es
estructural: el pool es **por conexion** (`UcxCommunicator.cpp:1406`). Cuatro clientes son
cuatro pools independientes. Para tener mas de una transferencia viva sobre UN pool hace falta
concurrencia **dentro** de una conexion (`GVIRTUS_ASYNC_DISPATCH=1`):

| slots | async | reservas | **peak_inflight** | waited | peak_waiters | declines | fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | si | 6 562 | **2** | 0 | 0 | 0 | 0 |
| 4 | si | 6 455 | **2** | 0 | 0 | 0 | 0 |
| 8 | si | 6 278 | **2** | 0 | 0 | 0 | 0 |
| 8 | no | 6 287 | **2** | 0 | 0 | 0 | 0 |

Y con clientes separados (1, 2 y 4 slots x 4 clientes): `DEGRADA_SIN_ROMPER`, cero corrupcion,
backend vivo, `peak_inflight=1`.

**El pico de demanda concurrente es 2 en TODAS las configuraciones medidas** -- llama a CONC=8
incluido. Con el coste de 257 MiB por slot, la lectura de ingenieria es directa:

- **el menor pool estable observado es 2**;
- el default de 8 ya sobreaprovisiona x4 (2,1 GB de RAM fijada para una demanda de 2);
- 32 y 64 slots (8,4 y 16,6 GB) no compran nada en ninguna carga de este banco;
- **la perilla cara es el techo `SLOT_CAP_MB`, no el numero de slots**, y el arnes las movia
  sin distinguirlas.

**Decision de arquitectura (fase 17): A, pool estatico provisionado.** Descartadas B
(crecimiento acotado) y C (fallback por presion) porque ninguna carga medida genera la presion
que justificaria su complejidad: cero esperas, cero rechazos y pico 2 en todos los puntos.
Implementarlas seria anadir complejidad de lifetime contra un problema que no se ha observado.

## 6. Clasificacion del estado

| | |
|---|---|
| **Implementado y probado** | provisioning del backend propagado y verificado; validacion de configuracion (6 casos); instrumentacion de ocupacion, admisiones y causas de rechazo; observabilidad de acks rechazados; inyeccion ABA determinista (`hold_ack`); matriz (fallo x ablacion) de 8 celdas; recovery en las dos direcciones; fairness en 4 puntos; estres en 7 puntos; unitarios del selector (6 grupos); AddressSanitizer sobre el frontend; cerrojo y reintento en el reinicio del backend |
| **Parcialmente implementado** | contrapresion: existe (sondeo con plazo de 30 s y fallback a AM) pero **nunca se ejercito**, porque ninguna carga saturo el pool |
| **No implementado** | estados explicitos de conexion (`Initializing/Active/Draining/Failed/Closed`); despertar explicito de waiters en disconnect; metricas por conexion; comprobacion `waiter_thread_is_not_progress_thread`; ThreadSanitizer; inyeccion especifica para la guarda de **epoch**; ablacion `pointer_keyed` |
| **Hipotesis no verificada** | que la excepcion de `acquire_rx_slot` pueda matar el backend; que la contencion del sondeo importe con varios hilos; la causa del abort intermitente (1 de 9) |

### 6.1 Criterios de aceptacion, uno a uno

| criterio | estado |
|---|---|
| causa exacta del colapso verificada | **parcial**: descartadas 3 causas por medida; la raiz sigue abierta |
| % de operaciones que pasan a RMA medido | si: 0,004 % escalar / 1,31 % cuadrantes |
| pico de slots vivos medido | si: **2** en las 6 configuraciones |
| menor pool estable encontrado | si: **2** |
| memoria de 8/16/24/32/64 medida | si: lineal, 257 MiB/slot |
| 32 slots reverificado; 64 comparado con 32 | si: ninguno compra nada sobre 2 |
| N=1 sin regresion | si |
| N=8 estable con configuracion suficiente | si |
| pool insuficiente no mata el backend | si (3 configuraciones) |
| sin errores secundarios masivos | si |
| sin esperas indefinidas | si: 0 esperas en todos los puntos |
| sin sondeo agresivo significativo | **no comprobado**: el sondeo existe pero nunca se entro en el |
| generation probada de forma determinista | **si** (`hold_ack`) |
| epoch probada de forma determinista | **no**: falta la inyeccion especifica |
| pointer reuse probada | **no** en esta campana |
| ack rancio no libera una generacion nueva | si: `gen_mismatch=1`, `ack_on_free=0` |
| epoch retirado no se reclama antes de tiempo | **no comprobado** |
| disconnect limpia o retira correctamente | si, en las dos direcciones |
| backend acepta conexion nueva tras fallo | si: 1 562 y 1 540 transferencias |
| tests unitarios pasan | si: 6 grupos |
| tests de estres pasan | si: 7 puntos, cero corrupcion |
| tests de inyeccion pasan | si: 8 celdas + 3 ABA |
| el codigo compila | si |
| los guiones reproducen los resultados | si (§7) |
| se conservan los crudos | si: `docs/gusto_raw_2026-08-02/`, incluidas las versiones fallidas |
| la documentacion coincide con los resultados | si |

## 7. Reproduccion

```bash
# backend con pool explicito (dpu-01)
GVIRTUS_RMA_MIN_BYTES=8192 GVS_SLOTS=8 GVS_SLOT_MIN_MB=4 GVS_SLOT_CAP_MB=32 \
  GVS_PREALLOC=1 bash ~/reset_backend_pool.sh

# matriz de fallos y ABA determinista
bash faults2.sh ; bash aba.sh
# pool insuficiente y tasa del abort
bash b34.sh
# footprint host+GPU
bash footprint2.sh
```

Crudos en `night/*.csv` y `night/*/*.log`. Working tree preservado en `~/preserve_0802/`;
cada fichero editado conserva `.bak_gusto`, `.bak_periodic`, `.bak_ackobs`, `.bak_hold`.
**Cero commits.**
