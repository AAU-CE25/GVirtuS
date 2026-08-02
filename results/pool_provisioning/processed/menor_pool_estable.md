# Menor pool estable por (politica, clientes)

Estable = cero fallos de checksum, cero rechazos por timeout y backend vivo en
TODAS las repeticiones del punto.

| politica | clientes | menor pool estable | pico observado | esperas |
|---|---:|---:|---:|---:|
| quadrant | 1 | **4** | 1 | 0 |
| quadrant | 2 | **4** | 1 | 0 |
| quadrant | 4 | **4** | 1 | 0 |
| quadrant | 8 | **4** | 1 | 0 |
| scalar | 1 | **4** | 1 | 0 |
| scalar | 2 | **4** | 1 | 0 |
| scalar | 4 | **4** | 1 | 0 |
| scalar | 8 | **4** | 1 | 0 |
