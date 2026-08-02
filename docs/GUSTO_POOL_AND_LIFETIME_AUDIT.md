# Gusto — auditoria de pool, slots, registros y lifetime (2026-08-02)

Todo lo marcado **[medido]** se ejecuto; **[codigo]** es lectura del fuente en el commit
`649d07b`; **[hipotesis]** es lo que aun no esta comprobado y se dice como tal.

---

## 1. Flujo H2D completo (frontend → backend)

`cudaMemcpy(H2D)` → `CudaRt_memory.cpp` → `Frontend::Execute` → `UcxCommunicator::WriteIov`.

1. **Decision** — `WriteIov`, `UcxCommunicator.cpp:3883-3896` **[codigo]**
   - El payload es **el fragmento iov mas grande**, no la suma: los otros son cabecera de
     sobre y argumentos marshalled, e incluirlos clasificaria mal una transferencia pequena
     con argumentos voluminosos.
   - La direccion es **estructural**: `WriteIov` *es* el PUT del cliente (H2D). No hay flag.
   - `pinned` sale de `HostMemoryIsPinned`, un mapa de intervalos local del frontend
     (`cudaHostAlloc`/`cudaFreeHost`). **No** de `cudaPointerGetAttributes`, que en el frontend
     esta remotada y costaria una RPC para ahorrar una RPC.
   - `prefer_rma(h2d=true, pinned, total)` aplica la tabla de `RmaPolicy.h`.

2. **Reserva** — `WriteIovRma`, `:3157-3284` **[codigo]**
   - Elige un slot `Free` **que quepa**: tomar el primero libre e ignorar la capacidad hacia
     que un slot infradimensionado en un indice bajo mandara toda transferencia grande al
     camino eager durante el resto de la conexion.
   - `Free → InFlight`, `++generation`, y se etiqueta con `make_slot_tag(remote_epoch_, gen)`.
   - Si no hay slot: **poll con `yield` y `ucp_worker_progress`, deadline 30 s**, no
     `condition_variable`. El comentario del codigo explica por que: en un cliente de un solo
     hilo, ESE hilo es el unico que progresa el worker, asi que esperar en la cv es esperar
     una notificacion que nadie va a entregar. **Ver §12: esto es correcto para 1 hilo y
     problematico para 8.**

3. **Transferencia** — `ucp_put_nbx` al slot remoto (host, o shadow GPU si el par lo anuncio),
   seguido de un AM `RmaPosted` diminuto.

4. **Consumo** — backend: el handler AM marca el slot `rma_origin=true`,
   `rma_generation=tag`, `in_use=true` **[codigo `:840-940`]**.

5. **Liberacion** — `release_rx_slot` **[codigo `:2231-2250`]**: `in_use=false` y, si era
   `rma_origin`, `send_slot_consumed(idx, gen)` **fuera** del mutex del pool (evita inversion
   de orden con `worker_mutex_`).

6. **Cierre del ciclo** — `release_remote_slot` **[codigo `:2261-2327`]**: guarda de epoch,
   guarda de generacion, exige `state == InFlight`, y solo entonces `InFlight → Free`.

**El evento que libera el slot es el ACK de consumo del backend, NO la compleccion local del
`ucp_put`.** Una compleccion local solo dice que la NIC escribio; no dice que la aplicacion
remota haya terminado de leer.

## 2. Flujo D2H

`GetFromRemoteGpu` — GET iniciado por el cliente. La compleccion CUDA ocurre en el backend y
el frontend hace el GET. Umbrales de admision distintos y **mas altos** que H2D, por una razon
medida y documentada en `RmaPolicy.h:23-26`: por debajo de 1 MiB el signo se invierte
(16 KiB ×0,38; 64 KiB ×0,44) porque el GET registra su destino por llamada y ese coste fijo no
amortiza.

## 3. Maquina de estados de un slot remoto (cliente)

```
        WriteIovRma reserva                    SlotConsumed(epoch,gen) valido
  Free ──────────────────────▶ InFlight ─────────────────────────────────▶ Free
   ▲                              │
   │                              │ ack con epoch != remote_epoch_   → descartado (+contador)
   │                              │ ack con generacion != actual     → descartado (+contador)
   │                              │ ack cuando state == Free         → ignorado
   │                              │
   └──────────────────────────────┘  RAII: si WriteIovRma sale por error tras reservar,
                                     el slot vuelve a Free (si no, la contrapresion se atasca)
```

Solo dos estados explicitos (`RemoteSlot::State`). No hay estado `Reserved` separado de
`InFlight`: la reserva y la publicacion ocurren bajo el mismo `rma_state_mu_`.

## 4. Maquina de estados de un slot local (servidor)

`PinnedSlot` no tiene enum; su estado es la combinacion de `in_use`, `rma_persistent`,
`rma_origin`, `rma_retired` y `rma_epoch` **[codigo `:128-167`]**.

- `rma_persistent` — del pool RMA. **Nunca** se entrega a un AM eager entrante: el par puede
  estar a mitad de un `ucp_put` sin que el servidor lo sepa.
- `rma_retired` — publicado bajo un epoch superado. No se anuncia ni se entrega a nadie; se
  libera un epoch mas tarde en `retire_and_free_locked`.

## 5. Lifecycle de un epoch

`materialise_rma_pool` → `init_rx_pool` → `send_rma_setup` (**incrementa el epoch**; publica
solo slots vivos). En el cliente, un `RmaSetup` que llega con transferencias en vuelo **no se
instala**: se aparca (`rma_swap_pending_`, contador `rma_swap_parked_count_`) y se aplica en
`release_remote_slot` cuando la ultima transferencia drena. Mientras esta aparcado,
`try_pick` devuelve `false` y no se entregan slots de la layout saliente.

## 6. Maquina de estados de la conexion

**[codigo]** No existe un enum de estado de conexion. Lo que hay son banderas dispersas:
`running_`, `endpoint_failed_`, `rma_setup_received_`, `rma_put_capable_`,
`rma_pool_requested_`, `rma_pool_ready_`. **La fase 11 pide `Initializing/Active/Draining/
Failed/Closed` explicitos; hoy NO existen.** → *no implementado*.

## 7. Tabla de ownership

| recurso | lo crea | dueno | quien lo referencia | evento que permite liberarlo | hilo que libera |
|---|---|---|---|---|---|
| `PinnedSlot.addr` (host pinned) | backend, `init_rx_pool`/`acquire_rx_slot` | `rx_pool_` de esa conexion | AM handler, consumidor, PUT del par | `retire_and_free_locked` un epoch despues | hilo del consumidor (`Read`) |
| `PinnedSlot.gpu_addr` (shadow) | backend, si GPUDirect **y** la conexion probo trafico device-destined | idem | peer-DMA de la NIC | idem | idem |
| `ucp_mem_h` (registro) | `map_slot_to_ucp` | el slot | UCX | al liberar el slot | idem |
| `RemoteSlot.rkey` | cliente, `handle_rma_setup_am` | `remote_slots_` | `WriteIovRma` | swap de layout → `retired_rkeys_`, drenado en `drain_retired_rkeys` desde el hilo llamante | hilo de la aplicacion |
| generacion de slot | cliente, en la reserva | el slot | el ACK que vuelve | — | — |
| epoch del pool | backend, `send_rma_setup` | la conexion | descriptores publicados | cuando no quedan outstanding | `Read` |

## 8. Respuestas explicitas a las preguntas de la fase 1

1. **¿Los ocho slots eran globales o por conexion?**
   **Por conexion.** `accepted->rx_pool_ = std::make_shared<RxPool>()` con el comentario
   «Per-connection AM state, RX pool, and worker mutex — no sharing»
   (`UcxCommunicator.cpp:1402-1406`) **[codigo]**.
   Y llama-server abre **una sola** conexion GVirtuS: 3 muestras de `ss` bajo carga dan 1
   **[medido]**. Sus 8 peticiones concurrentes son **hilos sobre una conexion**, no 8
   conexiones. La frase «eight concurrent connections contending for a pool of 8 slots» que
   circulaba es incorrecta.

2. **¿Que tamano tiene cada slot?**
   `rma_slot_cap_for(hint)` **[codigo `:1839-1880`]**: el payload observado mas grande, redondeado
   a potencia de dos, mas 64 KiB de holgura de framing, acotado entre un suelo
   (`GVIRTUS_RMA_SLOT_MIN_MB`, o el suelo RMA si no se define) y un techo
   (`GVIRTUS_RMA_SLOT_CAP_MB`, default 1025 MB). **No** es una constante.
   Medido con llama y el banco tal cual estaba: **131072 B (128 KiB)**. Forzando
   `GVS_SLOT_MIN_MB=4`: 4 MiB + holgura **[medido]**.

3. **¿Cada slot incluye host, GPU o ambas?**
   Host **siempre**. Shadow GPU **solo** si GPUDirect esta activo *y* la conexion ha demostrado
   mover bulk con destino device (`NoteDeviceDestinedPayload`), o con
   `GVIRTUS_RMA_GPU_SHADOW_EAGER=1`. El comentario del codigo lo justifica: el shadow duplica
   el coste en memoria de GPU y una conexion que nunca hace peer-DMA se lo estaba quitando a
   los demas inquilinos.

4. **¿Cuanta memoria consume un pool de 8/16/24/32/64?**
   Formula **[codigo]**: `N × (cap + 64 KiB) × (1 + shadow)`. Pendiente de **medir** en GPU y
   host por configuracion → §11 de `GUSTO_POOL_PROVISIONING.md`. *No se da una cifra calculada
   como si fuera medida.*

5. **¿Que operacion mantiene un slot ocupado?**
   Desde la reserva en `WriteIovRma` hasta el `SlotConsumed` del backend. Incluye el `ucp_put`,
   el AM `RmaPosted`, el tiempo de cola en el backend y el consumo por la aplicacion.

6. **¿Que evento exacto libera el slot?**
   `release_remote_slot(server_idx, tag)` con **epoch valido**, **generacion coincidente** y
   `state == InFlight`. Cualquier otro ack se descarta y se cuenta.

7. **¿Puede un slot permanecer ocupado esperando un ACK?** **Si, por diseno.** Es exactamente
   el invariante que hace segura la reutilizacion.

8. **¿La politica escalar enviaba menos operaciones a RMA?** → §9 **[medido]**.

9. **¿El factor de aumento de demanda esta medido o inferido?** Ahora **medido**
   (`admit_rma`/`admit_am`). Antes solo se podia inferir, y la afirmacion «bajar el umbral ×512
   multiplica la demanda ×512» no estaba sostenida por ningun contador.

10. **¿El backend crashea, aborta, pierde conexion o queda bloqueado?**
    **Ninguna de las cuatro. El backend sobrevive** [medido]: contenedor `Up`, listener en
    `25.25.25.2:32222` intacto despues del fallo. **Quien muere es el frontend.**

11. **¿Cual es el primer error causal?** → §9.

## 9. El colapso: que es realmente

### 9.1 Lo que NO es **[medido]**

- **No es agotamiento del pool.** En la corrida que aborta: `peak_inflight=2` sobre 8 slots,
  `peak_waiters=0`, `waited=0`, `wait_us_avg=0.05`, y `decline_capacity=decline_timeout=
  decline_swap=decline_epfail=0`. El pool no se acerco a saturarse.
- **No es muerte del backend.** Sigue vivo y escuchando.
- **No se reproduce con el pool inerte.** Con el banco tal como estaba (backend en su default
  de 2×128 KiB, RMA nunca admitido) el brazo quadrant+8 dio 3 corridas STABLE seguidas,
  591,6 / 571,7 / 591,6, cero fallos.

### 9.2 Lo que si es **[medido]**

Con el pool del backend realmente provisionado (8 slots × 4 MiB) y politica quadrant, el
**frontend aborta**:

```
ggml-cuda.cu:795  ggml_backend_cuda_buffer_get_tensor
cudaStreamSynchronize((cudaStream_t)0x2)
CUDA error: invalid argument
→ ggml_abort()  → proceso muerto
```

Pila: `server_slot::prompt_save` → `llama_context::state_seq_get_data` → `llama_io_write_host`
→ `ggml_backend_cuda_buffer_get_tensor`. Es decir: **un D2H** del guardado de cache de prompt.

`(cudaStream_t)0x2` es `cudaStreamPerThread`, el pseudo-handle de PTDS.

Los 248 021 «fallos» del registro original son **consecuencia**, no causa: con el proceso
muerto, el generador de carga recibe conexion rechazada de forma inmediata, lo que da ~5 500
fallos/s en la ventana de 45 s. Un agotamiento de pool habria dado esperas de 30 s y decenas
de fallos, no cientos de miles.

### 9.3 La hipotesis del D2H, REFUTADA

Se probo que la politica quadrant, al bajar el umbral **D2H** a 1-2 MiB, admitia el guardado de
KV cache al camino RMA/GET que scalar dejaba fuera. Tres brazos identicos salvo la admision:

| brazo | H2D | D2H | prediccion | **resultado** |
|---|---|---|---|---|
| A escalar | >=4 MiB | >=4 MiB | sobrevive | **sobrevive** (3/3 STABLE) |
| B cuadrantes | >=8 KiB | >=1-2 MiB | muere | **sobrevive** (3/3 STABLE) |
| C cuadrantes + D2H a 8 MiB | >=8 KiB | >=8 MiB | sobrevive | **sobrevive** (3/3 STABLE) |

**La prediccion fallo**: B no murio. Y el contador lo explica: subir el umbral D2H a 8 MiB deja
las reservas practicamente iguales (**17 073 frente a 17 097**), luego lo que llama admite al
camino RMA es **H2D**, y mover el umbral D2H no cambia el regimen. La hipotesis del D2H queda
descartada por medida.

### 9.4 Frecuencia y estado

**1 de 9** corridas de cuadrantes con RMA activo; control escalar **0 de 5**. **No se da una
tasa**: con un solo evento el intervalo va de "muy raro" a "uno de cada dos", y 1/9 frente a
0/5 no separa las politicas con estos tamanos.

Descartado por medida: agotamiento del pool, muerte del backend, admision de D2H.
**Causa raiz: abierta.** La hipotesis viva -- interaccion de cudaStreamPerThread con el
reenvio de streams del frontend -- no esta comprobada y no se presenta como conclusion.

## 10. Riesgos de deadlock

| riesgo | estado |
|---|---|
| Inversion `rx_pool_->mu` ↔ `worker_mutex_` | **mitigado en codigo**: `send_slot_consumed` se llama fuera del mutex del pool, con comentario explicito |
| Inversion `rma_state_mu_` ↔ `worker_mutex_` | **mitigado**: `WriteIovRma` suelta `rma_state_mu_` antes de tomar `worker_mutex_` para progresar |
| Esperar un ACK que solo puede entregar el propio hilo | **evitado a proposito** con poll+progress en vez de cv, para el caso de 1 hilo. **Con 8 hilos introduce un problema distinto**, §12 |
| Layout aparcada que no drena nunca | acotado por el deadline de 30 s → cae a eager |

## 11. Riesgos de use-after-free, doble liberacion, ack rancio

| riesgo | proteccion | ¿ejercitada? |
|---|---|---|
| ACK duplicado libera dos veces | exige `state == InFlight` | contador `rma_ack_gen_mismatch_count_`; **fase 8** lo fuerza con `GVS_FAULT=dup_ack` |
| ACK de generacion vieja libera un slot reasignado (ABA) | guarda de generacion | idem, con `delay_ack` |
| ACK de un layout reemplazado | guarda de epoch, `rma_ack_dropped_epoch_count_` | **fase 9** |
| Registro que sobrevive a su allocation CUDA | invalidacion por identidad de allocation | **fase 10**; ya existe la ablacion `pointer_keyed` |
| `rkey` destruido con puts en vuelo | `retired_rkeys_` + `drain_retired_rkeys` desde el hilo llamante | — |
| Referencia a `PinnedSlot` invalidada al crecer el pool | `std::deque`, no `vector`, **con comentario que lo explica** | — |

## 12. Donde una falta de slot se convierte en error, y donde puede matar algo

1. `WriteIovRma:3247` — no hay slot → **`return 0`** → el llamante cae al camino IOV/AM.
   Es un fallback correcto, no un error. **Coste: hasta 30 s de poll antes de rendirse.**
2. `acquire_rx_slot:2191,2212` — `alloc_pinned_host` devuelve nullptr →
   **`throw std::runtime_error`**. Esto ocurre en el camino del handler AM. Una excepcion que
   escape de un callback de UCX es la via mas plausible de matar el backend.
   **[hipotesis, no comprobada]** → fase 6.
3. El poll de la reserva llama a `ucp_worker_progress` bajo `worker_mutex_` **desde cada hilo
   que espera**. Con 8 hilos de peticion sobre una conexion, eso es contencion sobre el mutex
   del worker exactamente cuando hace falta progresar para recibir los ACKs que liberan slots.
   **[hipotesis, no comprobada]** — no se ha observado en las corridas de hoy porque el pool
   nunca llego a saturarse (`waited=0`).

## 13. Clasificacion

| | |
|---|---|
| **Implementado y probado** | ownership explicito de slot, ACK exacto con epoch+generacion, guarda ABA, deque anti-invalidacion, aparcado de layout, fallback a eager, instrumentacion F3 |
| **Implementado, pendiente de ejercitar** | guardas de epoch y generacion bajo fallo inyectado (la infra existe, los tests deterministas no) |
| **Parcialmente implementado** | provisioning (configurable, pero sin validacion de rangos ni budget, y el banco no lo propagaba); contrapresion (existe, pero por poll con deadline de 30 s) |
| **No implementado** | estados explicitos de conexion; despertar de waiters en disconnect; metricas por conexion; comprobacion `waiter_thread_is_not_progress_thread` |
| **Hipotesis no verificada** | que la excepcion de `acquire_rx_slot` pueda matar el backend; que la contencion del poll importe con 8 hilos |
