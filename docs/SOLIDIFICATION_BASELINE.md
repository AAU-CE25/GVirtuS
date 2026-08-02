# Gusto — solidification baseline (2026-08-02)

Estado del repositorio y del banco de pruebas ANTES de tocar nada, para que cualquier medida
posterior sea atribuible. Todo lo de aqui esta verificado ejecutandolo, no leido de notas.

## 1. Repositorio

| campo | valor |
|---|---|
| host | `es-dpu-02` (frontend) |
| ruta | `/home/student.aau.dk/ll33pq/GVirtuS` |
| rama | `exp/lazy-pool-v2` |
| commit | `649d07bae10946047cb7f612c1eeacb3d890b444` |
| ficheros trackeados modificados | 7 |
| sin trackear relevantes | `include/gvirtus/communicators/RmaPolicy.h`, `AblationGate.h`, `examples/rmatest/{sweep_bench,stress4}.cu` |

Modificados: `include/gvirtus/communicators/Communicator.h`,
`plugins/cudart/frontend/CudaRt_memory.cpp`, `src/backend/Process.cpp`,
`src/communicators/Result.cpp`, `src/communicators/ucx/UcxCommunicator.{cpp,h}`,
`src/frontend/Frontend.cpp`.

**El trabajo de la politica de colocacion NO esta commiteado.** Vive en el working tree.

### Preservacion

Antes de cualquier edicion (`~/preserve_0802/` en dpu-02):

| fichero | contenido |
|---|---|
| `working_tree_649d07b.diff` | `git diff` completo, 786 lineas |
| `untracked_headers.tgz` | RmaPolicy.h, AblationGate.h, sweep_bench.cu, stress4.cu |
| `git_status.txt`, `HEAD.txt` | estado exacto |

Ademas, cada fichero editado despues conserva su copia: `.bak_gusto`, `.bak_periodic`.
No se ha ejecutado `git reset`, ni `git checkout` destructivo, ni commits.

## 2. Topologia

```
   es-dpu-02  (frontend, L40S propia)          es-dpu-01  (backend, L40S)
   llama-server / cliente GVirtuS   ────────▶  gvirtus-ll33pq
   contenedor cudf_gvirtus_dyncudf             listener 25.25.25.2:32222
```

El backend **recompila desde `./src` en cada arranque del contenedor**, asi que un cambio de
fuente exige sincronizar a los DOS hosts. El frontend no: hay que reconstruir y copiar la `.so`
a mano (ver §4).

## 3. Hardware y pila (medido hoy)

| | dpu-02 (frontend) | dpu-01 (backend) |
|---|---|---|
| GPU | NVIDIA L40S, 46068 MiB | NVIDIA L40S, 46068 MiB |
| driver | 560.35.05 | (mismo contenedor CUDA 12.6) |
| kernel | 5.15.0-100-generic | — |
| CPU / RAM | 64 nucleos / 251 GiB | — |
| UCX | 1.17.0 (`/lib/libucs.so.0`) | idem |
| NIC | `mlx5_1:1` / `ens1f1np1`, GID index 3 | idem |
| transporte | `UCX_TLS=rc_mlx5,ud_mlx5,tcp,self`, wireup por TCP | idem |
| imagen | `ll33pq/cudf_gvirtus_dyncudf:cuda12.6` | `ll33pq/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04` |

GPUDirect: **habilitado en el backend** (`probe OK`), **deshabilitado en el frontend** (no
tiene GPU visible en ese contenedor; el probe falla y cae a slots host). Es lo esperado para
este reparto de roles, no un fallo.

## 4. Procedimiento de build (verificado hoy, 2 ciclos)

```bash
# frontend, en dpu-02
cd ~/GVirtuS/build
CPATH=$HOME/lz4inc LIBRARY_PATH=$HOME/lz4shim make gvirtus-communicators-ucx -j8
cp build/libgvirtus-communicators-ucx.so lib/libgvirtus-communicators-ucx.so   # NO lo hace make
```

Tres trampas confirmadas:

1. **`make` deja las `.so` en `build/`, no en `lib/`.** El contenedor monta `lib/`. Sin la copia
   se sigue midiendo la libreria vieja mientras `make` dice «Built target».
2. `~/lz4shim/liblz4.so` es un symlink a `/usr/lib/x86_64-linux-gnu/liblz4.so.1` y `~/lz4inc/`
   lleva cabeceras sacadas del contenedor: no hay `liblz4-dev` en el host.
3. `gvirtus-plugin-cudart` no compila en el host (necesita GL del backend). El shim de frontend
   es el target `cudart`.

Backend: `bash ~/reset_backend_pool.sh` (nuevo, §5) o `~/reset_backend_fault.sh`.

## 5. Configuracion: lo que el banco NO propagaba

`reset_backend_fault.sh` propaga `GVS_FAULT`, `GVIRTUS_RMA_MIN_BYTES`,
`GVIRTUS_UCX_PROGRESS_TIMEOUT_MS` y `BACKEND_CONFIG`. **No propaga el provisioning del pool.**

Consecuencia medida (2026-08-02), y es la razon de ser de este documento: el pool que consume
H2D vive en el **backend** (`send_rma_setup` empaqueta sus `rx_slots` y el cliente hace
`ucp_put` sobre ellos). Poniendo `GVIRTUS_RMA_SLOTS` en el contenedor del **frontend**, como
hacian los arneses, el backend seguia anunciando su default de **2 slots de 128 KiB** y toda
transferencia de 3,17 MB de llama caia a AM eager con

```
[GVS] RMA fast path declined: message 3170196 B exceeds the peer's largest slot 131072 B
```

Es decir: **un barrido de slots hecho desde el frontend mide el camino AM, no el pool.**

Se anade `~/reset_backend_pool.sh` en dpu-01, que propaga `GVIRTUS_RMA_SLOTS`,
`GVIRTUS_RMA_SLOT_CAP_MB`, `GVIRTUS_RMA_SLOT_MIN_MB` y `GVIRTUS_RMA_PREALLOC`, y **verifica
el env efectivo dentro del contenedor** antes de devolver el control. Construye los `-e` solo
para las variables definidas: pasar `-e VAR=` DEFINE la variable vacia, que no es lo mismo que
no pasarla en codigo que mira `getenv() != NULL`.

Comprobacion de que ahora si se ejercita el camino:

```
[GVS] rma_setup: epoch 2 arrived, 0/8 slots in flight -> install now   # 8, no 2
(ninguna linea "RMA fast path declined")
```

## 6. Instrumentacion anadida (F3)

Antes solo habia contadores de coste de reserva y de acks, y se emitian **solo en el teardown**
— que con `docker rm -f` (SIGKILL) no llega a ejecutarse nunca. Se anade:

- ocupacion instantanea, **pico** y **media ponderada por tiempo** (no por evento);
- waiters actuales y pico;
- admisiones por camino (`admit_rma`, `admit_am`, `rma_pct`) y `rma_fellback`;
- rechazos **por causa**: `decline_capacity`, `decline_timeout`, `decline_swap`, `decline_epfail`;
- emision **periodica** (`GUSTO_RMA_METRICS_MS`, default 5000 ms, 0 = off) desde los puntos
  seguros ya existentes, mas la de teardown.

Formato parseable de una linea:

```
GUSTO_METRIC tag=periodic slots_total=8 peak_inflight=2 avg_inflight=0.085 peak_waiters=0
  reservations=4911 waited=0 wait_us_avg=0.05 admit_rma=4911 admit_am=74333 rma_pct=6.20
  rma_fellback=0 decline_capacity=0 decline_timeout=0 decline_swap=0 decline_epfail=0
  ack_sent=0 ack_gen_mismatch=0 ack_epoch_dropped=0 parked=0
```

Coste: un reloj monotono por **transicion de estado de slot**, no por operacion.

## 7. Tests y arneses existentes

| que | donde |
|---|---|
| smoke de APIs CUDA | `tests/*.cu` (cudart, cudadr, cublas, cudnn, cufft, curand, cusolver, cusparse, nvrtc) |
| especificos de RMA | `examples/rmatest/`: `concgrow`, `d2h_only`, `dst_realloc`, `growtest`, `pageable_d2h`, `rma3x64`, `rma_checksum`, `rma_srcprov`, `rma_verdict`, `src_realloc`, `stress4`, `sweep_bench` |
| inyeccion de fallos | `GVS_FAULT` (`dup_ack`, `stale_ack`, `delay_ack`) y `GVS_ABLATE` (`no_generation`, `no_epoch`, `pointer_keyed`) en `AblationGate.h` |
| build con sanitizers | `build_asan/` ya existe |
| carga end-to-end | `~/rep_run.sh` (repeticiones + veredicto STABLE/UNSTABLE), `~/results/summary.csv` |
| auditoria previa | `docs/P2_rma_round6_slot_lifetime_audit.md` |

## 8. Baseline numerico

llama-7B (mistral-7b-q4), CONC=8, prompts UNIQUE, ventana 45 s, warm 12 s, 1 conexion GVirtuS:

| config | goodput | fallos | estado |
|---|---:|---:|---|
| quadrant, pool del backend por defecto (2×128 KiB ⇒ **RMA nunca admitido**) | 591,6 | 0 | STABLE |
| quadrant, `GVIRTUS_RMA_SLOTS=32` en el frontend (**misma cosa**: inerte) | 571,7 | 0 | STABLE |
| quadrant, backend provisionado 8×4 MiB (**RMA si admitido**) | — | — | **el frontend ABORTA**, ver §9 |

Las dos primeras filas son la misma configuracion efectiva para H2D; la diferencia 591,6 vs
571,7 es ruido entre corridas.

## 9. Estado conocido al empezar

- El colapso original (182,0, 248 021 fallos) **no se reproduce** con el pool inerte: 3 corridas
  STABLE seguidas.
- Con el pool realmente activo, el **frontend** (`llama-server`) aborta en
  `ggml_backend_cuda_buffer_get_tensor` → `cudaStreamSynchronize((cudaStream_t)0x2)` →
  `CUDA error: invalid argument`. **El backend sobrevive** (contenedor y listener intactos).
- En esa corrida el pool NO estaba saturado: `peak_inflight=2` de 8, `peak_waiters=0`,
  `waited=0`, todos los `decline_*` a cero.

Eso descarta el agotamiento del pool como causa y deja como hipotesis principal la **admision
de D2H** al camino RMA, que la politica quadrant baja a 1–2 MiB y scalar mantenia en 4 MiB.
El experimento que lo aisla esta en `docs/GUSTO_POOL_AND_LIFETIME_AUDIT.md` §9.
