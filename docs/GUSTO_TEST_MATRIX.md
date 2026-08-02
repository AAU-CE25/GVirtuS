# Matriz de pruebas

Índice de todo lo que se ejecuta, qué demuestra y dónde están los crudos
(`docs/gusto_raw_2026-08-02/`).

## Unitarios (sin hardware)

| prueba | qué cubre | estado |
|---|---|---|
| `tests/gusto/test_rma_policy.sh` | selector: tabla de cuadrantes, bordes del umbral por ambos lados, asimetría de dirección, metadata, tamaño 0, overrides de entorno, valor inválido | **6 grupos, todos pasan** |

Cada política corre en su **propio proceso**: `rma_policy()` cachea el modo en un `static` a
propósito, y esa inmutabilidad es en sí una propiedad deseable.

## Configuración (contra el backend real)

| caso | esperado | resultado |
|---|---|---|
| válido | acepta e imprime total | ✅ |
| `SLOTS=0`, `SLOTS=abc`, `SLOTS=99999` | rechaza, listener en pie | ✅ |
| sobre presupuesto | rechaza | ✅ |
| bajo presupuesto | acepta | ✅ |

Crudos: `cfg_validation2.csv`.

## Lifetime (fallo × ablación)

| celda | resultado |
|---|---|
| control | limpio, 0 rechazos |
| `dup_ack` × {completo, sin generación} | rechazado por **estado**; la guarda de generación no interviene |
| `delay_ack` × {completo, sin generación, sin epoch} | ídem |
| `stale_ack` × completo | rechaza y **degrada a AM** (7 reservas) |
| `stale_ack` × sin generación | 96 violaciones contadas |
| **`hold_ack` × completo** | **ABA rechazada**, `on_free=0` |
| **`hold_ack` × sin generación** | **libera un slot vivo**, `on_free=1` |

Crudos: `faults2.csv`, `aba.csv`, más `aba_v1_autobloqueada.csv` y `aba_v2_sin_entrega.csv`,
que se conservan porque documentan dos formas de que una inyección no ejercite nada.

## Pool y provisioning

| prueba | crudos |
|---|---|
| coste de memoria 8/16/24/32/64 | `footprint2.csv` |
| pool insuficiente, 4 clientes | `undersized.csv` |
| contención con dispatch asíncrono | `undersized_async.csv` |
| admisiones por política | en `GUSTO_METRIC` de las corridas de llama |

## Recovery

| dirección | resultado | crudos |
|---|---|---|
| muere el backend | sale en 1 s, rc=1, sin crash, teardown limpio, backend recupera | `f11b/f11b.csv` |
| muere el cliente (SIGKILL) | backend escucha, GPU 435→435, cliente nuevo OK | ídem |

## Estrés y sanitizers

`f14.sh`: barrido de tamaños 64 KiB → 64 MiB × pools 2/8/32, y AddressSanitizer sobre el
comunicador del frontend. Crudos en `f14/`.

## Fairness

`f12.sh`: N ∈ {2,4,8} clientes, cociente lento/rápido e índice de Jain. Crudos en `f12/`.
La claim se limita a **admisión y progreso del pool**; no se afirma nada sobre planificación
de kernels CUDA.

## Lo que NO se ejecuta

- ThreadSanitizer sobre los componentes de CPU aislables.
- Ablación `pointer_keyed` (reutilización de dirección virtual) en esta campaña.
- Inyección específica para la guarda de **epoch** (descriptor de un epoch retirado que llega
  tras la reclamación).
- Timeout durante la publicación del pool; ACK y callback de CUDA tardíos tras disconnect.
