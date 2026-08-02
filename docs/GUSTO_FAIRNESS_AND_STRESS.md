# Fairness de admisión y pruebas de estrés

**Estado**: ambas ejecutadas. Fairness sin desequilibrio; estrés sin corrupción; AddressSanitizer
con fugas **aisladas y caracterizadas como no escalables**.

## 1. Fairness

La claim se limita a **admisión y progreso del pool**. No se afirma nada sobre planificación de
kernels CUDA, que no controla el transporte.

Como el pool es **por conexión**, el reparto de slots es justo por construcción. Lo que sí podía
haber starvation es en el backend: un proceso, un hilo por conexión, una GPU. Clientes idénticos,
ventana de 45 s:

| clientes | slots | transferencias por cliente | lento/rápido | **Jain** | esperas | veredicto |
|---:|---:|---|---:|---:|---:|---|
| 2 | 8 | 3518, 3567 | 0,986 | **1,0000** | 0 | justo |
| 4 | 8 | 3504, 3564, 3558, 3601 | 0,973 | **0,9999** | 0 | justo |
| 8 | 8 | 3514, 3524, 3561, 3598, 3621, 3594, 3677, 3717 | 0,945 | **0,9997** | 0 | justo |
| 4 | 2 | 3588, 3576, 3597, 3637 | 0,983 | **1,0000** | 0 | justo |

**Ninguna conexión queda por debajo del 94,5 % de la más rápida**, y el pool corto (2 slots con
4 clientes) no empeora el reparto — coherente con que cada cliente tiene su propio pool.

No se implementó FIFO de waiters, round-robin ni cuota por conexión: **no hay starvation que
corregir**, y no hubo una sola espera en ninguno de los cuatro puntos.

## 2. Estrés determinista

`rma_checksum` genera su carga a partir del índice de transferencia, así que la misma
configuración reproduce los mismos bytes: no hace falta semilla externa.

| tamaño | reps | slots | transferencias | h2d FAIL | d2h FAIL | % a RMA | backend |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 KiB | 200 | 8 | 200 | 0 | 0 | 12,44 | vivo |
| 1 MiB | 150 | 8 | 150 | 0 | 0 | 12,42 | vivo |
| 4 MiB | 100 | 8 | 100 | 0 | 0 | 12,38 | vivo |
| 16 MiB | 40 | 8 | 40 | 0 | 0 | 12,20 | vivo |
| 64 MiB | 20 | 8 | 20 | 0 | 0 | 11,90 | vivo |
| 4 MiB | 100 | **2** | 100 | 0 | 0 | 12,38 | vivo |
| 4 MiB | 100 | **32** | 100 | 0 | 0 | 12,38 | vivo |

**Cero corrupción en los siete puntos**, con verificación doble (checksum calculado en el
dispositivo y relectura completa por D2H), cubriendo tres órdenes de magnitud de tamaño y
pools de 2 a 32. El porcentaje admitido a RMA es estable (~12,4 %) porque el resto de
operaciones son control y metadatos.

## 3. AddressSanitizer

`build_asan/` no tenía el módulo UCX: su caché de CMake era anterior a que ese target existiera.
Reconfigurado y reconstruido (2,6 MB frente a 192 KB de la librería normal, consistente con la
instrumentación).

**Resultado: 16 852 bytes en 9 asignaciones.** Las nueve, con su traza:

| asignación | bytes | objetos |
|---|---:|---:|
| `Frontend::GetFrontend` (singleton) | 232 | 1 |
| `__cudaRegisterFatBinary` / `...End` | 160 | 2 |
| `CudaUtil::MarshalHostPointer` | 76 | 4 |
| `CudaUtil::MarshalFatCudaBinary` | 16 384 | 2 |

**Ninguna está en el camino de slots RMA.** Y la prueba que lo decide: con **160**
transferencias en vez de 40, el informe es **idéntico — 16 852 bytes en 9 asignaciones**. No
escalan con el trabajo, luego son asignaciones de arranque que viven hasta el fin del proceso
(singleton del frontend y registro de fatbins), no una fuga del plano de datos.

Las 160 transferencias pasan **ambos** checkpoints bajo el sanitizer.

**Lo que esto no dice**: no se ha ejecutado ThreadSanitizer, y ASan cubre el **frontend**, no el
backend. Un informe limpio en un lado no dice nada del otro.


---

## Actualización 2026-08-02 (tarde) — saturación sostenida y ThreadSanitizer

Los dos huecos que este documento declaraba están cerrados en
`GUSTO_CONTINUACION_2026-08-02.md` §2 y §3:

- **Saturación**: 96,7 % de ocupación sostenida con 47 % de las reservas esperando; con el ack
  retrasado 35 s se agota el plazo de 30 s, se cuenta `decline_timeout` y **degrada a AM sin
  colgarse**. El control sin fallo tiene *las mismas* 150 esperas pero de 0,84 µs: la contención
  es de la carga, el coste lo fija el consumidor.
- **ThreadSanitizer**: el build no estaba roto; lo que fallaba era la corrida (hooks de memoria
  de UCX, y que TSan no admite `LD_PRELOAD` sobre un binario sin instrumentar). Encontró **3
  carreras de datos y 4 inversiones de orden de cerrojos**, todas arregladas; después, 4 corridas
  consecutivas limpias.

Donde este documento dice *«no se ha ejecutado ThreadSanitizer»*, ya no es cierto.
