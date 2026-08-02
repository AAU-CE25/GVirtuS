# Paper-ready: provisioning del pool RMA y colocación semántica

Todos los números vienen de corridas de 2026-08-01/02 sobre el banco descrito en
`SOLIDIFICATION_BASELINE.md`. Crudos en `docs/gusto_raw_2026-08-02/`. **Ningún número está
estimado, extrapolado ni redondeado a conveniencia.**

---

## 1. La causa raíz, y por qué importa metodológicamente

El resultado que motivó este trabajo —«8 slots colapsan, 32 lo arreglan, luego umbral y pool
son un par acoplado»— **no midió lo que creía medir**.

El pool de slots que consume H2D pertenece al **backend**: `send_rma_setup` empaqueta sus
`rx_slots` y el cliente hace `ucp_put` sobre ellos. El arnés fijaba `GVIRTUS_RMA_SLOTS` en el
contenedor del **frontend**, y el script de reinicio del backend no propagaba esa variable. Los
tres brazos (8, 32, 64) corrieron con el mismo pool por defecto del backend —**2 slots de
128 KiB**— y con la misma línea en el log:

```
[GVS] RMA fast path declined: message 3170196 B exceeds the peer's largest slot 131072 B
[GVS] rma_setup: epoch 5 arrived, 0/2 slots in flight
```

Es decir: las transferencias de 3,17 MB de llama **nunca tomaron el camino RMA en ningún
brazo**. Los tres midieron el camino de mensajes activos. Por eso los tres dieron 591,6.

> **Lección transferible**: una perilla de configuración fijada en el lado que no la lee produce
> un experimento que corre, no falla, y no mide nada. El antídoto que se implementó no es una
> convención sino código: el script de reinicio **verifica el entorno efectivo dentro del
> contenedor** antes de devolver el control, y el transporte **imprime la configuración
> efectiva del pool** al construirlo.

## 2. Umbral → demanda: ×310 en fracción admitida, ×1 en ocupación

Con el backend provisionado de verdad (8 slots × 4 MiB), llama-7B, CONC=8, ventana 45 s:

| política | reservas RMA | operaciones totales | **fracción admitida a RMA** | **pico de slots vivos** | esperas | rechazos |
|---|---:|---:|---:|---:|---:|---:|
| escalar 4 MiB | 65 | 1 531 381 | 0,00424 % | **2** de 8 | 0 | 0 |
| cuadrantes | 17 097 | 1 300 310 | 1,31484 % | **2** de 8 | 0 | 0 |

Bajar el umbral multiplica la **fracción admitida** por **310** (1,31484 / 0,00424). El **pico de
slots simultáneamente vivos no se mueve**: 2 en ambas políticas.

> **Por qué la fracción y no el conteo.** El cociente de conteos absolutos da ×263, y remedido en
> otra ventana daba ×329, porque los dos brazos **no ejecutaron el mismo número de operaciones**
> (1 531 381 frente a 1 300 310) y el conteo crece con la ventana y con el caudal. La fracción
> admitida es invariante a las dos cosas. Se publica **×310 y `rma_pct`, siempre con la ventana
> declarada (45 s)**; el ×263 de conteos no se publica.

La razón es estructural y no accidental: la carga es petición/respuesta síncrona, y el ACK de
consumo del backend libera el slot antes de que haga falta el siguiente. **El umbral gobierna
cuántas operaciones entran al camino RMA; la ocupación del pool la gobierna la concurrencia,
que es una dimensión independiente.**

> Formulación defendible: *lowering the threshold increases the fraction of transfers admitted
> to the RMA path and therefore the number of RMA operations; it does not, in a synchronous
> request/response workload, increase the number of simultaneously live slots.*
> **No** se afirma que bajar el suelo ×512 multiplique la demanda de slots: los contadores lo
> desmienten.

## 3. La demanda pico es 2 en todas las configuraciones medidas

| carga | conexiones | slots | dispatch | reservas | **pico** | esperas |
|---|---:|---:|---|---:|---:|---:|
| llama-7B CONC=8 | 1 | 8 | asíncrono | 17 097 | **2** | 0 |
| `stress4` | 1 | 2 | asíncrono | 6 562 | **2** | 0 |
| `stress4` | 1 | 4 | asíncrono | 6 455 | **2** | 0 |
| `stress4` | 1 | 8 | asíncrono | 6 278 | **2** | 0 |
| `stress4` | 1 | 8 | síncrono | 6 287 | **2** | 0 |
| `rma_checksum` ×4 | 4 | 1 | síncrono | 192 | **1** | 0 |

Dos hechos estructurales explican la última fila y hay que enunciarlos juntos:

1. **El pool es por conexión** (`UcxCommunicator.cpp:1406`, «Per-connection AM state, RX pool,
   and worker mutex — no sharing»). Cuatro clientes son cuatro pools independientes: añadir
   clientes **no** crea contención de slots.
2. Por tanto la única vía de contención es concurrencia **dentro** de una conexión. Se forzó
   con `GVIRTUS_ASYNC_DISPATCH=1` y el pico siguió siendo 2.

**En ninguna configuración de este banco el pool llegó a ser el cuello de botella**: cero
esperas, cero rechazos y cero fallbacks en los seis puntos.

## 4. El coste del provisioning es lineal y lo fija el techo, no el tráfico

RSS del contenedor del backend en reposo, `SLOT_CAP_MB=256`, `PREALLOC=1`, tráfico real 4 MiB:

| slots | RSS (MiB) | incremento |
|---:|---:|---:|
| 8 | 2 183 | — |
| 16 | 4 240 | +2 057 |
| 24 | 6 297 | +2 057 |
| 32 | 8 355 | +2 058 |
| 64 | 16 588 | +8 233 |

**257 MiB de memoria host fijada por slot**, y ese número es el **techo**
`GVIRTUS_RMA_SLOT_CAP_MB`, no el tamaño del tráfico: con `PREALLOC=1`, `rma_slot_cap_for(0)`
devuelve el techo porque todavía no hay evidencia de tráfico. El pool se reserva entero al
arrancar, antes de que ninguna transferencia lo justifique.

Consecuencias, y son las dos accionables del artículo:

- **«Subir el número de slots» no es gratis**: de 8 a 32 son 6,2 GB de RAM fijada; a 64,
  16,6 GB. En un backend multi-inquilino es memoria que se le quita al resto.
- **La perilla cara es el techo, no el número.** Con el techo ajustado al tráfico (4 MiB), los
  mismos 64 slots costarían ~260 MiB en total, un factor 64 menos. El arnés movía las dos sin
  distinguirlas.

Con demanda pico 2 y coste 257 MiB/slot, **el default de 8 ya sobreaprovisiona ×4**.

## 5. El colapso no era el pool

En la única corrida que abortó: `peak_inflight=2` de 8, `peak_waiters=0`, `waited=0`, y los
cuatro contadores de rechazo a cero. **El backend sobrevivió** —contenedor y listener
intactos—; quien murió fue el **frontend**:

```
ggml_backend_cuda_buffer_get_tensor → cudaStreamSynchronize((cudaStream_t)0x2)
CUDA error: invalid argument → ggml_abort()
```

`0x2` es `cudaStreamPerThread`. Pila: `server_slot::prompt_save` → `state_seq_get_data` →
`llama_io_write_host`, un D2H de estado de caché de KV.

Los **248 021 «fallos»** del registro original son **consecuencia, no causa**: con el proceso
muerto el generador de carga recibe conexión rechazada de inmediato, lo que produce ~5 500
fallos/s en una ventana de 45 s. Un agotamiento de pool habría producido esperas de 30 s y
decenas de fallos.

**Frecuencia: 1 de 9** corridas con la misma configuración (control escalar: 0 de 5). **No se
da una tasa**: con un solo evento el intervalo va de «muy raro» a «uno de cada dos», y 1/9
frente a 0/5 no separa las políticas. Descartado por medida: agotamiento de pool, muerte del
backend, y admisión de D2H (subir sus umbrales a 8 MiB deja las reservas en 17 073 frente a
17 097). **Causa raíz: abierta.**

## 6. Los mecanismos de lifetime, demostrados por ablación

La celda que demuestra que un mecanismo es **necesario** es (fallo inyectado × protección
retirada). Una variante degradada que nadie ataca no falla; un ataque contra el protocolo
completo tampoco.

### 6.1 Generación (guarda ABA)

Inyección determinista `hold_ack`: el ACK se aplica con normalidad **y se guarda una copia**
que se re-entrega cuando ese mismo slot vuelve a estar en vuelo con otra generación. La
condición ABA existe por construcción, no por temporización.

| variante | reservas | `gen_mismatch` | **`ack_on_free`** | corrupción |
|---|---:|---:|---:|---:|
| control | 96 | 0 | 0 | 0 |
| protocolo completo | 96 | **1** | **0** | 0 |
| **sin guarda de generación** | 96 | 1 | **1** | 0 |

**La firma que separa las variantes es `ack_on_free`.** Con la guarda, el ACK rancio se rechaza
y el slot **sigue en vuelo**; el ACK legítimo lo libera después. Sin la guarda, el ACK rancio
**libera un slot vivo**, y el ACK legítimo posterior encuentra el slot ya libre: liberación
prematura, observable y reproducible.

**Sin corrupción en ninguna de las dos**, y hay que decirlo así: con transferencias síncronas
de 4 MiB el backend ya había terminado de consumir. **La violación del invariante es real y
está medida; que se materialice en corrupción depende de la carga. Ausencia de corrupción no
es prueba de correctitud.**

### 6.2 Generación bajo ACK obsoleto

| variante | reservas | `gen_mismatch` | comportamiento |
|---|---:|---:|---|
| protocolo completo | **7** | 1 | rechaza, el slot no vuelve a libre, **degrada al camino AM** |
| sin guarda | **96** | 96 | sigue por RMA con **96 violaciones** del protocolo |

Es el modo de fallo deseable de una protección: prefiere perder rendimiento antes que liberar
un buffer que el par sigue usando.

### 6.3 Lo que la ablación NO demuestra

Con ACK duplicado o retrasado y carga síncrona, `ack_on_free` es alto y `gen_mismatch` cero:
al duplicado lo rechaza la comprobación de **estado**, no la de generación, y quitar la guarda
no cambia una sola cifra. **La ventana ABA no se construye con un retardo fijo**; que se abra
o no depende del reloj. De ahí la necesidad de `hold_ack`.

## 7. Comportamiento con configuración inválida o pool insuficiente

Validación de provisioning, seis casos contra el backend real:

| caso | resultado | listener |
|---|---|---|
| válido (8 × 32 MB) | acepta e imprime `total=513.0 MiB` | en pie |
| `SLOTS=0` | **rechazado** | en pie |
| `SLOTS=abc` | **rechazado** | en pie |
| `SLOTS=99999` | **rechazado** (fuera de [1,1024]) | en pie |
| 32 × 256 MB con presupuesto 100 MiB | **rechazado** (pide 16 388 MiB) | en pie |
| 8 × 32 MB con presupuesto 8 GiB | acepta | en pie |

**En los cuatro rechazos el listener sigue en pie**: una configuración inválida no construye el
pool y el transporte se queda en el camino AM. Es degradación, no fallo.

Pool deliberadamente corto (1, 2 y 4 slots con 4 clientes): cero corrupción, cero esperas,
backend vivo en los tres.

## 8. Decisión de arquitectura

**Pool estático provisionado.** Se descartan crecimiento acotado y admisión sensible a presión
porque **ninguna carga medida genera la presión que justificaría su complejidad de lifetime**:
cero esperas, cero rechazos y pico 2 en los seis puntos. Implementarlas sería añadir estados de
epoch y ventanas de carrera contra un problema que no se ha observado.

**Recomendación de dimensionado**: pool de 4 slots (pico medido 2, con margen ×2) y techo
ajustado al percentil alto del tráfico de la carga, no al default de 1025 MB. Para llama-7B:
4 slots × 4 MiB ≈ 16 MiB, frente a los 2,1 GB del default de 8 × 256 MB.

## 9. Lo que este trabajo NO establece

1. **La causa del abort intermitente** (1 de 9). Tres hipótesis descartadas por medida; la viva
   —interacción de `cudaStreamPerThread` con el reenvío de streams— no está comprobada.
2. **Una ganancia end-to-end de la política de colocación.** llama a CONC=8 da paridad
   (591,6 frente a 591,6): está limitada por el plano de control, no por la colocación. La
   carga que lo mostraría —genuinamente limitada por datos entre 8 KiB y 4 MiB— falta.
3. **Corrupción bajo ablación**: el checksum no la ve en esta carga. Solo se afirma la
   violación del invariante, que sí está contada.
4. **El régimen de contención del pool**: no se ha alcanzado en ninguna configuración, así que
   nada de lo dicho aquí describe el comportamiento bajo saturación real.

## 10. Tabla propuesta para el artículo

| | escalar 4 MiB | cuadrantes |
|---|---:|---:|
| operaciones admitidas a RMA | 65 (0,004 %) | 17 097 (1,31 %) |
| pico de slots simultáneamente vivos | 2 | 2 |
| esperas de reserva | 0 | 0 |
| goodput (llama-7B, CONC=8) | 591,6 | 591,6 |

**Pie propuesto**: *Lowering the placement threshold admits 263× more transfers to the RDMA
path without changing peak slot occupancy: in a synchronous request/response workload the
consumption acknowledgement returns a slot before the next transfer needs one. Threshold and
pool capacity are therefore independent axes for this class of workload, and end-to-end
throughput is unchanged because the workload is control-plane bound.*

## 11. Reproducción

```bash
# backend con pool explícito y verificación del entorno efectivo
GVIRTUS_RMA_MIN_BYTES=8192 GVS_SLOTS=8 GVS_SLOT_MIN_MB=4 GVS_SLOT_CAP_MB=32 \
  GVS_PREALLOC=1 bash ~/reset_backend_pool.sh

bash faults2.sh      # matriz (fallo × ablación)
bash aba.sh          # ABA determinista con hold_ack
bash b3b.sh          # contención con dispatch asíncrono
bash footprint2.sh   # coste de memoria por número de slots
bash f11b.sh         # recovery en las dos direcciones
bash f12.sh          # fairness de admisión
bash f14.sh          # estrés + AddressSanitizer
bash tests/gusto/test_rma_policy.sh   # unitarios del selector
```
