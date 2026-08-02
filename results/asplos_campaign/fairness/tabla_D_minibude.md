cuanto de servicio (avg_ms en N=1, mediana):
  baremetal          295.29 ms
  baremetal_mps      295.28 ms
  tcp                295.91 ms
  ucx_rdma           295.34 ms
  ucx_gpudirect      295.35 ms

escrito /home/student.aau.dk/ll33pq/GVirtuS/results/asplos_campaign/fairness/tabla_D_minibude_por_tenant.csv (352 filas tenant-corrida)

## Reparto del cuanto de servicio, N=8, MiniBUDE

| sistema | corridas | mult. min | mult. mediana | mult. max | iteraciones servidas sin esperar (de 80) | mayor parada (s) |
|---|---:|---:|---:|---:|---:|---:|
| baremetal | 5 | 1.01 | 7.95 | 7.97 | **5** de 400 | 2.06 |
| baremetal_mps | 5 | 1.00 | 7.95 | 7.99 | **7** de 400 | 2.06 |
| ucx_gpudirect | 15 | 1.00 | 3.50 | 16.00 | **405** de 1200 | 4.43 |
| ucx_rdma | 14 | 1.00 | 3.00 | 10.00 | **364** de 1120 | 2.66 |
| tcp | 5 | 1.00 | 3.99 | 11.98 | **68** de 400 | 3.25 |

## Tabla D — una cohorte, tenant a tenant (run_ucx_gpudirect_n8_r1)

| tenant | trabajo ofrecido | iters | slowdown | progreso norm. | multiplicadores por iteracion | mayor parada (s) | correccion |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | poses=65536;prot=938;lig=26 | 10 | 2.375 | 0.421 | `1.0 1.0 1.0 1.0 3.0 2.0 3.0 5.0 1.0 3.0` | 1.181 | valid=true;max_diff_%=0.014 |
| 2 | poses=65536;prot=938;lig=26 | 10 | 1.001 | 0.9992 | `1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0` | 0.001 | valid=true;max_diff_%=0.014 |
| 3 | poses=65536;prot=938;lig=26 | 10 | 4.874 | 0.2051 | `3.0 4.0 2.0 6.0 5.0 6.0 10.0 6.0 3.0 1.0` | 2.658 | valid=true;max_diff_%=0.014 |
| 4 | poses=65536;prot=938;lig=26 | 10 | 2.5 | 0.4 | `5.0 3.0 4.0 4.0 2.0 1.0 4.0 3.0 1.0 1.0` | 1.181 | valid=true;max_diff_%=0.014 |
| 5 | poses=65536;prot=938;lig=26 | 10 | 4.125 | 0.2425 | `4.0 4.0 1.0 8.0 6.0 6.0 4.0 2.0 3.0 3.0` | 2.067 | valid=true;max_diff_%=0.014 |
| 6 | poses=65536;prot=938;lig=26 | 10 | 3.25 | 0.3077 | `3.0 3.0 4.0 3.0 5.0 1.0 5.0 5.0 1.0 2.0` | 1.181 | valid=true;max_diff_%=0.014 |
| 7 | poses=65536;prot=938;lig=26 | 10 | 3.624 | 0.2759 | `3.0 4.0 2.0 5.0 3.0 2.0 7.0 4.0 1.0 5.0` | 1.772 | valid=true;max_diff_%=0.014 |
| 8 | poses=65536;prot=938;lig=26 | 10 | 1.75 | 0.5714 | `2.0 2.0 1.0 2.0 1.0 2.0 1.0 1.0 1.0 5.0` | 1.181 | valid=true;max_diff_%=0.014 |

## El mismo corte en el control nativo (run_baremetal_n8_r1)

| tenant | slowdown | progreso norm. | multiplicadores por iteracion |
|---:|---:|---:|---|
| 1 | 7.831 | 0.1277 | `5.0 7.9 8.0 8.0 8.0 8.0 8.0 8.0 8.0 7.0` |
| 2 | 7.953 | 0.1257 | `2.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0` |
| 3 | 7.951 | 0.1258 | `1.0 2.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0` |
| 4 | 7.831 | 0.1277 | `6.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 7.0` |
| 5 | 7.831 | 0.1277 | `7.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 7.0` |
| 6 | 7.83 | 0.1277 | `4.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 7.0` |
| 7 | 7.829 | 0.1277 | `3.0 7.9 8.0 8.0 8.0 8.0 8.0 8.0 8.0 7.0` |
| 8 | 7.829 | 0.1277 | `2.0 7.9 8.0 8.0 8.0 8.0 8.0 8.0 8.0 7.0` |
