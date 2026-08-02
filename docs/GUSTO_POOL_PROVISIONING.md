# Provisioning del pool RMA

**Estado**: implementado y probado. Validación de configuración probada contra el backend real
en seis casos. Recomendación de dimensionado derivada de medidas, no de criterio.

## 1. Quién es dueño del pool

El pool que consume **H2D** vive en el **backend**: `send_rma_setup` empaqueta los rkeys de sus
`rx_slots` y el cliente hace `ucp_put` sobre ellos. El RMA es **bidireccional**: el cliente
anuncia también su propio pool (`UcxCommunicator.cpp:1539`) para que el backend le escriba las
respuestas grandes. De ahí se sigue lo único que hay que recordar:

> **`GVIRTUS_RMA_SLOTS` en el frontend NO dimensiona el pool que usa H2D.** Fijarla ahí y medir
> H2D es medir otra cosa. Costó una campaña entera; ver §1 de `PAPER_READY_POOL_RESULTS.md`.

El pool es **por conexión**, no global (`UcxCommunicator.cpp:1402-1406`). N clientes son N pools
independientes.

## 2. Variables

| variable | lado | efecto | default |
|---|---|---|---|
| `GVIRTUS_RMA_SLOTS` | backend (para H2D) | número de slots del pool | 2 |
| `GVIRTUS_RMA_SLOT_CAP_MB` | backend | **techo** de capacidad por slot | 1025 |
| `GVIRTUS_RMA_SLOT_MIN_MB` | backend | suelo de capacidad por slot | el suelo RMA |
| `GVIRTUS_RMA_PREALLOC` | backend | construir al conectar en vez de bajo demanda | 0 |
| `GUSTO_RMA_HOST_POOL_BUDGET_BYTES` | backend | presupuesto; si no cabe, no se construye | sin límite |
| `GUSTO_RMA_MAX_SLOTS` | backend | cota superior admitida | 1024 |

La capacidad efectiva por slot es `rma_slot_cap_for(hint)`: el mayor payload observado
redondeado a potencia de dos más 64 KiB de holgura, acotado entre suelo y techo. **Con
`PREALLOC=1` no hay evidencia todavía y devuelve el techo**, que es de dónde sale el coste de
§3.

## 3. Coste medido

RSS del contenedor del backend en reposo, `SLOT_CAP_MB=256`, `PREALLOC=1`:

| slots | 8 | 16 | 24 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| RSS (MiB) | 2 183 | 4 240 | 6 297 | 8 355 | 16 588 |

**Lineal, 257 MiB por slot**, y ese 257 es el **techo**, no el tráfico (4 MiB en la medida).
El pool es memoria host fijada (`cudaHostAlloc`): **no aparece en `nvidia-smi`**. La primera
medida miraba sólo la GPU y por eso daba cero; hubo que rehacerla mirando el RSS.

## 4. Validación

`gusto_validate_pool_cfg()` corre **antes de reservar nada** y comprueba rango, capacidad no
nula, desbordamiento de `slots × (cap + shadow)` y presupuesto. Si algo falla **no construye el
pool** y el transporte se queda en el camino AM: degradación, no fallo. Imprime siempre la
configuración efectiva:

```
[GUSTO CFG] pool efectivo: slots=8 cap=32.1 MiB shadow=si total=513.0 MiB
```

Seis casos verificados contra el backend real (`gusto_raw_2026-08-02/cfg_validation2.csv`);
en los cuatro rechazos **el listener sigue en pie**.

## 5. Recomendación

Demanda pico medida = **2 slots** en todas las cargas del banco. Con 257 MiB por slot:

- **4 slots** (pico ×2 de margen) y **techo ajustado al tráfico**, no al default de 1025 MB.
- Para llama-7B: 4 × 4 MiB ≈ **16 MiB**, frente a los 2,1 GB de un pool de 8 × 256 MB.
- **La perilla cara es el techo**, no el número de slots.

## 6. Limitaciones

- No se ha alcanzado el régimen de saturación en ninguna configuración: nada de lo anterior
  describe el comportamiento bajo contención real.
- El shadow GPU es condicional y su asignación no está caracterizada: a 64 slots aparecieron
  11,9 GB de GPU que no aparecen a 32 ni por debajo. **Observado, no explicado.**
