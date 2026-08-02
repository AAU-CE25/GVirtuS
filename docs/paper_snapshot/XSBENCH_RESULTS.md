---
title: "XSBench -- escalado por contención, cuatro sistemas, y equidad por tenant"
date: "2026-07-27, corregido 2026-08-01, equidad añadida 2026-08-02"
geometry: margin=2.3cm
fontsize: 10pt
---

**App:** XSBench CUDA (proxy Monte-Carlo de transporte de neutrones), `-s large -m event -G
nuclide`, **`-l 6e8` lookups**. **Métrica:** tiempo de pared de la cohorte de N tenants
concurrentes, y --desde 2026-08-02-- **Mlookups/s y runtime por tenant**. **Sistemas:**
baremetal = XSBench nativo en la L40S local de dpu-02 - tcp / rdma / rdma\_zc / ucx = GVirtuS
contra el backend de dpu-01. `ucx` = GPUDirect activo; `rdma_zc` = GPUDirect desactivado **en
el backend** (activarlo en el frontend no lo desactiva: pool, shadow de GPU y GET de D2H viven
en el backend). Todas las corridas son posteriores a la recompilación del frontend del
2026-07-26 04:24.

> ## AVISO: El `exit_code` de XSBench no significa nada
>
> **0 de 238 corridas a 6e8 salen con código 0.** XSBench lleva checksums de verificación
> fijados por tamaño de problema, válidos solo para el número de lookups por defecto de ese
> tamaño; ni 6e8 ni 1.25e9 lo son, así que imprime `WARNING - INAVALID CHECKSUM!` y devuelve
> un código distinto de cero incluso cuando la simulación termina bien. **La señal correcta es
> la línea `Lookups/s`.**

**Por qué 6e8 y no 1.25e9.** `GridInit.cu:66` reserva `lookups x sizeof(unsigned long)` para
el array de verificación: 1.25e9 lookups son **10 GB por instancia**, así que 4 tenants caben
en la tarjeta de 46 GB y 8 no. Cada punto N=8 a 1.25e9 era una cohorte parcial mientras el
arnés seguía imprimiendo `done=8/8`; el makespan seguía entonces al número de supervivientes
con correlación perfecta (4->108 s, 5->135 s, 6->160 s). A 6e8 cada instancia necesita 4,8 GB y
caben las ocho. *(`-s small` no lo habría arreglado: el preajuste de tamaño cambia las
dimensiones de la malla, no el array de verificación, que escala solo con `-l`.)*

# 1. Tiempo de pared de la cohorte

Reconstruido de los crudos por cliente. Cohorte válida = los N clientes presentes **y** los N
con línea `Runtime:`; el `done=N/N` del arnés no basta, como demostró la experiencia a 1.25e9.

| N | baremetal | ucx (GPUDirect) | rdma\_zc | rdma | tcp |
|---|---:|---:|---:|---:|---:|
| 1 | **12,88** | 13,94 | 13,76 | 13,99 | 14,02 |
| 2 | **25,02** | 27,08 | 26,58 | 27,34 | 27,63 |
| 4 | **50,23** | 52,64 | 52,26 | 53,17 | 52,59 |
| 8 | **98,39** | 103,89 | 103,48 | 103,98 | **no completa** |

Mediana del makespan de cohorte, definido `max(t_end) - min(t_start)` sobre los N clientes.
Cohortes válidas por celda: 4--10. Recomputado el 2026-08-02 desde
`~/xsbench_campaign/results/xsbench/<arm>/N<n>/sync/seed*/client*/status_raw.json`; coincide
con el bloque de corrección del 2026-08-01 en todas las celdas salvo `ucx` N=8 (103,89 frente
a 103,62, distinto conjunto de repeticiones).

> **Una tabla anterior de este documento daba baremetal N=8 = 114,92 s y TCP N=8 = 100,89 s.
> Era incorrecta y se ha eliminado.** No debe reconciliarse: invertía la conclusión, porque
> hacía perder a baremetal frente a los brazos remotos desde N=2, y hacía a TCP el más rápido
> a N=8 cuando en realidad no completa. Los datos válidos son los de arriba.

**El transporte es irrelevante para esta carga.** Los cuatro brazos remotos caen dentro de un
2 % entre sí a cada N, TCP incluido hasta N=4, y `ucx` frente a `rdma_zc` difieren por debajo
del 0,5 %. XSBench mueve su malla a la GPU **una vez** y luego ejecuta 6x10^8 lookups, así que
la transferencia son milisegundos contra un makespan de ~100 s y GPUDirect no tiene nada que
acelerar. **Es un resultado nulo, y útil:** acota dónde *no* aplica la ventaja del camino de
datos.

**Corrección:** todos los brazos reportan checksums de verificación idénticos a cada N (408237
a 6e8), así que el remoting es exacto bit a bit aquí.

**El sobrecoste del remoting es plano en la escala:** 8,2 % a N=1, 8,2 % a N=2, 4,8 % a N=4,
5,6 % a N=8 para `ucx` frente a baremetal. No se degrada con los tenants.

# 2. La fila de MPS: retirada por falta de respaldo

Este documento contenía una sección que afirmaba que **MPS acelera la ejecución nativa un
13--15 % desde N=2**, con baremetal+MPS en 25,11 / 50,15 / 98,40 s. **Se retira**, por dos
razones independientes:

1. **Su línea de comparación era la tabla obsoleta.** El «gana» del 13--15 % se calculaba contra
   baremetal sin MPS en 29,53 / 58,05 / 114,92. Contra los valores correctos --25,02 / 50,23 /
   98,39-- el margen desaparece: **-0,4 % / +0,2 % / 0,0 %**. Para XSBench, MPS no aporta nada
   medible.
2. **No existe ningún crudo que la respalde.** Los valores 25,11 / 50,15 / 98,40 aparecen solo
   en este documento: no hay árbol `baremetal_mps` bajo `~/xsbench_campaign/`, ni fila en
   ningún CSV. No son reconstruibles.

Y con ello cae también la premisa que la sección usaba para justificarse: *«baremetal sin MPS
pierde contra todos los brazos remotos desde N=2, que es lo contrario de lo que debería hacer
una GPU local»*. Con los datos correctos **baremetal gana a cada N sin MPS** (25,02 < 26,58 <
27,08 < 27,34 < 27,63). No había nada que explicar.

**Esto es específico de XSBench.** El resultado de MPS para **llama** (`RESULTS.md` §8b, n=3)
y para **miniBUDE** (árbol `run_baremetal_mps_*` en `~/mb_campaign/`) sí tiene crudos y se
sostiene por separado. La regla de reportar ambas líneas base sigue en pie; lo que se retira
es la fila de XSBench.

# 3. Equidad por tenant

Lo que el paquete anterior no traía. Reconstruido de los `stdout.log` por cliente, que **sí
están en el servidor**, en `~/xsbench_campaign/results/xsbench/<arm>/N<n>/sync/seed*/client*/`.
Datos por tenant en `XSBench_fairness_por_tenant.csv` (**306 filas**).

El progreso normalizado usa el `Runtime:` interno de XSBench --la simulación, sin el arranque--
contra la mediana de N=1 del mismo sistema. Jain se calcula **por cohorte** y se resume con la
mediana entre cohortes.

| sistema | N | cohortes | **Jain** | **lento/rápido** | Mlookups/s peor | mediana | mejor |
|---|---:|---:|---:|---:|---:|---:|---:|
| baremetal | 2 | 5 | **1,0000** | **1,000** | 24,9 | 24,9 | 24,9 |
| baremetal | 4 | 5 | **1,0000** | **1,000** | 12,2 | 12,2 | 12,2 |
| baremetal | 8 | 4 | **1,0000** | **1,001** | 6,2 | 6,2 | 6,2 |
| ucx | 2 | 5 | 0,9007 | 1,994 | 24,1 | 36,1 | **48,1** |
| ucx | 4 | 5 | 0,9309 | 1,992 | 12,1 | 16,2 | 24,1 |
| ucx | 8 | 5 | **0,6613** | **5,982** | 8,0 | 12,0 | **47,7** |
| rdma\_zc | 2 | 5 | 0,9010 | 1,992 | 24,5 | 36,7 | **48,8** |
| rdma\_zc | 4 | 5 | 0,8049 | 2,973 | 16,2 | 20,2 | **48,0** |
| rdma\_zc | 8 | 5 | **0,7163** | **4,848** | 9,6 | 12,1 | **46,5** |
| rdma | 8 | 5 | **0,6933** | **4,882** | 9,7 | 14,1 | **48,5** |
| tcp | 2 | 7 | 0,9009 | 1,992 | 24,3 | 36,2 | **48,2** |
| tcp | 4 | 7 | 0,9309 | 1,995 | 12,2 | 16,2 | 24,3 |

**El nativo reparte por igual.** Los N tenants obtienen el mismo rendimiento hasta el tercer
decimal. Su tasa en solitario es 50,7 Mlookups/s, y el reparto observado --24,9 / 12,2 / 6,2--
queda algo por debajo del ideal 50,7/N --25,4 / 12,7 / 6,3--: hay una pérdida de eficiencia del
2 % con los tenants, pero **la reparte entre todos por igual**.

**Los brazos remotos no.** A N=8 el mejor tenant de `ucx` obtiene **47,7 Mlookups/s, el 96 %
de su tasa en solitario (49,6): corre casi como si estuviera solo en la tarjeta** -- mientras el
peor obtiene 8,0, un sexto. Es el mismo
patrón que MiniBUDE, y aparece **también en TCP**, así que no es del camino de datos.

**A N=2 la firma es exacta:** lento/rápido = 1,99 en los cuatro brazos remotos, con el mejor
tenant en el 97 % de su tasa en solitario (48,1 de 49,6) y el peor a la mitad. Eso no es contención: es **serialización**.

**Advertencia sobre qué métrica mira uno.** Con **tiempo de pared** la desigualdad casi
desaparece (lento/rápido 1,15--1,41 a N=8) porque las cohortes empiezan y terminan juntas; con
el **runtime interno** es de 4,8 a 6,0x. La diferencia entre ambas dice que los tenants
reparten distinto el tiempo entre arranque y cómputo, y **el makespan de cohorte no puede
mostrarlo**. Un documento que solo reporte makespan concluirá, erróneamente, que el reparto es
justo.

Contexto completo, controles y la descomposición por iteración del mismo fenómeno en MiniBUDE:
`FAIRNESS_RESULTS.md`.

# 4. Densidad de tenants bajo sobresuscripción deliberada -- incompleto

Ejecutado a **1.25e9** a propósito, de modo que 8 tenants piden 80 GB de una tarjeta de 46 GB
y algunos tienen que morir. La métrica son los supervivientes, no el makespan.

| brazo | supervivientes de 8 (5 reps) | media |
|---|---|---:|
| baremetal | 4, 4, 4, 4, 4 | **4,0** |
| ucx (GPUDirect) | 5, 6, 5, 4, 6 | **5,2** |
| rdma\_zc | *brazo nulo* | -- |
| TCP | 4, 4, 4, 0, 0 | *inutilizable* |

Baremetal es perfectamente reproducible en exactamente 4; `ucx` acomoda **un 30 % más de
tenants**, consistente con que un único contexto CUDA compartido deja más memoria de
dispositivo para los propios tenants -- el mismo mecanismo que el ahorro de ~460 MiB/tenant de
`RESULTS.md` §8.

**Dos brazos no son utilizables y el experimento no debe citarse como comparación a cuatro.**
La primera corrida de `rdma_zc` registró `GPUDirect=enabled` en el backend, así que era un
duplicado de `ucx` y no un brazo con GPUDirect desactivado; la repetición devolvió 2/8
supervivientes una vez y luego ceros. TCP produjo dos repeticiones con cero supervivientes, sin
diagnosticar. **Baremetal frente a ucx es el único par que este experimento sostiene.**

# 5. Salvedad de la línea base

Baremetal corre en la L40S de **dpu-02** mientras los brazos remotos usan la de **dpu-01**. Es
una referencia válida de corrección y una referencia *relativa* válida de escalado, pero no una
línea base absoluta de velocidad. Que baremetal sea el más rápido a N=1 (12,88 frente a 13,94)
argumenta contra una diferencia de hardware como motor de los resultados a N>=2, pero no la
elimina.

**El árbol crudo está parcialmente sobrescrito.** El experimento de densidad se ejecutó a
1.25e9 reutilizando los mismos directorios `seed*/`, así que
`~/xsbench_campaign/results/xsbench/<arm>/N<n>/` contiene ahora una mezcla de ambos tamaños.
Quien rederive números del árbol --en vez de de las tablas-- debe filtrar por `lookups` en
`seed_raw.json` o promediará dos cargas distintas. Las tablas de este documento están
filtradas.

**El arnés nunca arma el backend.** `run_seed_xs.sh` fija banderas solo del lado cliente
(`GVIRTUS_GPUDIRECT`, `GVIRTUS_RMA_ZEROCOPY`); la bandera del frontend por sí sola no desactiva
GPUDirect. Armar el backend por brazo y comprobar la línea `[GVS] GPUDirect=` antes de fiarse
de la etiqueta de un brazo.

Crudos: `~/xsbench_campaign/results/xsbench/<arm>/N<n>/sync/seed*/` (stdout por cliente,
`status_raw.json`, `seed_raw.json`) y `~/xsbench_campaign/results/.attempts/` (instantánea de
cada intento, incluidos los rechazados). Empaquetado: `XSBench_fairness_por_tenant.csv`,
`XSBench_filtered_v2.csv`.
