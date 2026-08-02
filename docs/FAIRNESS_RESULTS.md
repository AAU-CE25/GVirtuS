---
title: "Fairness multi-tenant: auditoría y reconstrucción"
subtitle: "Todos los workloads, todos los sistemas, por tenant y por corrida"
date: "2026-08-02"
geometry: margin=2.3cm
fontsize: 10pt
---

# 0. Qué es esto y qué no

Auditoría completa de la evaluación de fairness del proyecto. **No** parte de los índices de
Jain publicados: los recalcula desde los crudos por tenant, con controles, y comprueba si la
desigualdad observada la introduce Gusto o la explican la demanda, el número de muestras, las
llegadas Poisson, el desfase de arranque o la variabilidad natural.

Regla aplicada sin excepción: **ningún dato inventado ni rellenado**. Donde una métrica no es
calculable se dice qué falta.

Artefactos, todos en `results/asplos_campaign/fairness/` del repositorio:

| fichero | contenido |
|---|---|
| `tenants_canonico.csv` | **3688 filas**, una por tenant y corrida, de los cuatro workloads de trabajo fijo |
| `fairness_trabajo_fijo_por_corrida.csv` | 493 cohortes, Jain por corrida (nunca agrupado) |
| `fairness_trabajo_fijo_resumen.csv` | mediana entre corridas por (workload, sistema, N, modo) |
| `llama_fairness_por_corrida.csv` | 39 corridas de serving, con línea nula por permutación |
| `llama_fairness_por_tenant.csv` | 169 filas tenant-corrida de serving |
| `tabla_D_minibude_por_tenant.csv` | 352 filas con el timeline por iteración |
| `scripts/` | los seis programas que generan todo lo anterior |

# 1. Tabla A -- auditoría de métricas

| # | métrica | fórmula actual | problema | fórmula corregida | impacto medido |
|---|---|---|---|---|---|
| A1 | `goodput` (llama) | `tokens(comp)/WINDOW` con `comp` filtrado a `tc  en  [t_meas, t_end+TIMEOUT]` (`bench.py:118,132`) | cuenta finalizaciones de una ventana de **55 s** y divide entre **30 s** | recortar a `tc <= t_end` | **x1,76** en `slo_ucx_n8_l2.0`: 277,3 -> **157,9 t/s**. El **cociente entre brazos sobrevive** (comparten denominador); el valor absoluto no es una tasa de régimen permanente |
| A2 | `jain` (llama) | Jain sobre **tokens por tenant** (`bench.py:125-126`) | la magnitud es servicio absoluto, no normalizado por demanda; con llegadas Poisson un tenant recibe hasta **7,0x** más peticiones que otro | Jain sobre `completed_i/offered_i` | los Jain de 0,719--0,804 en N=6/8/10 pasan a **1,0000** exacto. Eran artefacto de demanda |
| A3 | `jain` sobre SLO bajo saturación | Jain sobre `slo_5s_fraction` | con casi todos los tenants a cero, el índice mide inanición y la informa como igualdad | suprimir si `no_nulos < N/2` o media `< 5 %` | 12 celdas suprimidas; en ellas 1/4--5/8 tenants con atención no nula y media 0,005--0,017 |
| A4 | Jain sobre runtime (trabajo fijo) | *tentación* de aplicar Jain a `duration_s` | un runtime mayor es **peor** servicio: el índice premia lo contrario de lo que mide | Jain sobre `progreso = t_solo/t_concurrente` | no llegó a publicarse; se documenta para que no se haga |
| A5 | agrupación de repeticiones | clave `(sistema, N, modo, semilla)` | dos árboles hermanos con la misma semilla se funden en **una** cohorte de 2N tenants de campañas distintas | añadir `cohort_path` a la clave | inventaba un desfase de arranque de **3162 s**; las cohortes reales tienen **0,0 s** |
| A6 | árboles duplicados | `experiments/babelstream/results_stale/` | duplicado obsoleto no excluido, se fusionaba con `results/` | etiquetar, no borrar | causa raíz de A5 |
| A7 | extracción de MiniBUDE | glob `tenant_*.log` | los brazos remotos escriben `t<i>.log` **y** `tenant_<i>.log`; los baremetal solo el primero | glob `t*.log` con deduplicación | **borraba el brazo de control entero** (0 filas nativas) |
| A8 | `exit_code` de XSBench | usado como señal de éxito | el checksum está fijado al número de lookups por defecto; el código no significa nada | usar la línea `Lookups/s` | ya documentado en campañas previas; confirmado aquí |
| A9 | modo `stagger` | mezclado con `sync` en los resúmenes | desfase de arranque **de 14,0 s por diseño** | separar siempre por modo | Jain cae a 0,957--0,99 y lento/rápido sube a 1,24--1,76 **solo por el desfase** |

**Defectos en los datos, no en las fórmulas.** XSBench TCP N=8: los 64 tenants sin línea
`Runtime:` y algunos con `duration_s = 0,0`; 8 cohortes descartadas y **contadas**, no
silenciadas. CloverLeaf no conserva su figura de mérito (escribe a `clover.out`, que no viaja
en el artefacto): solo hay tiempo de pared. MiniBUDE tiene `epoch_s` con resolución de **1
segundo** frente a corridas de 2,4--12 s.

# 2. Tabla B -- fairness de progreso en trabajo fijo

Modo `sync`, trabajo idéntico verificado por los parámetros de entrada, Jain sobre progreso
normalizado calculado **por cohorte** y resumido con la mediana entre corridas.

| workload | sistema | N | corr. | Jain | peor slowdown | mediana slowdown | **lento/rápido** | clasif. |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| MiniBUDE | nativo | 8 | 5 | 0,9998 | 7,95 | 7,83 | **1,032** | -- |
| MiniBUDE | nativo+MPS | 8 | 5 | 1,0000 | 7,96 | 7,95 | **1,016** | -- |
| MiniBUDE | **Gusto GPUDirect** | 8 | 15 | **0,696** | 4,87 | 3,75 | **4,871** | **A** |
| MiniBUDE | UCX host RMA | 8 | 14 | 0,746 | 4,31 | 3,22 | **4,311** | **A** |
| MiniBUDE | TCP | 8 | 5 | 0,669 | 4,87 | 3,93 | **4,619** | **A** |
| XSBench | nativo | 8 | 6 | 1,0000 | 7,60 | 7,60 | 1,002 | -- |
| XSBench | Gusto AM | 8 | 12 | 0,9804 | 8,67 | 7,67 | **1,409** | **B** |
| XSBench | Gusto GPUDirect | 8 | 8 | 0,9960 | 7,55 | 7,51 | 1,148 | **B** |
| XSBench | UCX host RMA | 8 | 5 | 0,9965 | 7,43 | 6,43 | 1,163 | **B** |
| BabelStream | nativo | 8 | 5 | 1,0000 | 7,81 | 7,80 | 1,007 | -- |
| BabelStream | Gusto AM | 8 | 5 | 0,9999 | 6,60 | 6,54 | 1,034 | **D** |
| BabelStream | UCX host RMA | 8 | 5 | 1,0000 | 6,63 | 6,61 | 1,015 | **D** |
| CloverLeaf | nativo | 8 | 5 | 1,0000 | 8,60 | 8,59 | 1,005 | -- |
| CloverLeaf | nativo+MPS | 8 | 5 | 1,0000 | 8,71 | 8,70 | 1,004 | -- |
| CloverLeaf | Gusto AM | 8 | 5 | 1,0000 | 8,40 | 8,39 | 1,019 | **D** |
| CloverLeaf | UCX host RMA | 8 | 5 | 1,0000 | 8,40 | 8,38 | 1,015 | **D** |

Clasificación: **A** evidencia fuerte de desigualdad introducida por Gusto - **B** señal
compatible, magnitud modesta - **C** explicada por demanda o ruido - **D** sin diferencia -
**E** experimento inválido.

**XSBench TCP N=8 -> E** (datos inutilizables, ver §1).

Detalle que sostiene la clasificación A: el efecto aparece **igual en TCP**, de modo que
**no es del camino de datos RMA**. Y el nativo con MPS --que consolida contextos igual que
Gusto-- **no lo reproduce** (1,016), así que tampoco es la consolidación de contexto por sí
sola: es cómo el backend **ordena** el trabajo de conexiones concurrentes.

# 3. Tabla C -- fairness de servicio en llama, normalizada por demanda

Las tres repeticiones de cada celda estaban **concatenadas** en el JSONL (`bench.py` abre en
modo *append* y comparten `label`). Se segmentaron por conteos acumulados de `summary.csv`;
la suma coincide **exactamente** en las 10 etiquetas comprobadas, con tramos no uniformes
como `[310, 306, 307]`.

## 3.1 Régimen estable

| sistema | N | **desbalance de demanda** | Jain compl. | Jain SLO 5 s | compl. peor--mejor |
|---|---:|---:|---:|---:|---|
| Gusto GPUDirect | 8 | 4,0x | **1,0000** | **1,0000** | 1,00--1,00 |
| Gusto GPUDirect | 10 | **7,0x** | **1,0000** | **1,0000** | 1,00--1,00 |
| nativo | 8 | 4,0x | 1,0000 | 0,9982 | 1,00--1,00 |

Un tenant recibió **siete veces más peticiones que otro** por el sorteo de llegadas, y aun
así todos completaron el 100 % de lo suyo.

## 3.2 Saturación, contra la línea nula de permutación

| sistema | N | Jain compl. observado | nulo p50 | **p** | compl. peor--mejor | SLO 5 s |
|---|---:|---:|---:|---:|---|---|
| Gusto | 8 | 0,957 / 0,968 / 0,930 | 0,934 / 0,932 / 0,933 | **0,77 / 0,90 / 0,48** | 0,15--0,34 | 0,00--0,03 |
| nativo | 8 | 0,960 x3 | 0,898 / 0,899 / 0,901 | **0,93 / 0,94 / 0,93** | 0,13--0,23 | 0,00--0,03 |

La línea nula baraja las **etiquetas de tenant** conservando la demanda observada: no supone
llegadas multinomiales ni tiempos de servicio iguales, que es lo que la forma cerrada
`n/(n+k-1)` da por hecho y aquí **no se verifica**. El Jain observado cae dentro de la nula en
**todas** las celdas.

## 3.3 Comparación emparejada y equivalencia

13 celdas casadas por (N, lambda, repetición):

| magnitud | dif. media (Gusto - nativo) | IC95 bootstrap | veredicto |
|---|---:|---|---|
| Jain de completion fraction | **-0,0027** | **[-0,0079, +0,0009]** | **EQUIVALENTES** (TOST, margen declarado ±0,05) |
| completion fraction media | **+0,0764** | **[+0,0426, +0,1122]** | **excluye el cero: Gusto sirve más** |

A N=2, lambda=2 la diferencia es de **+17,3 puntos** (0,9706 frente a 0,7975).

Sobre la potencia, sin adornos: con 13 pares el TOST es débil. Un IC que **cabe** en el margen
es evidencia de equivalencia; uno que no cupiera **no** sería evidencia de diferencia.

**Clasificación llama -> C**: la diferencia se explica por demanda y ruido experimental.

# 4. Tabla D -- evidencia por tenant del caso más fuerte

El cuanto de servicio es el tiempo de iteración en solitario, **prácticamente idéntico en los
cinco sistemas** (295,28--295,91 ms), de modo que los multiplicadores son comparables.

| sistema | corridas | mult. mediana | mult. máx | **iteraciones servidas sin esperar** | mayor parada |
|---|---:|---:|---:|---:|---:|
| nativo | 5 | **7,95** | 7,97 | **5 de 400** | 2,06 s |
| nativo+MPS | 5 | **7,95** | 7,99 | **7 de 400** | 2,06 s |
| Gusto GPUDirect | 15 | **3,50** | **16,00** | **405 de 1200 (34 %)** | 4,43 s |
| UCX host RMA | 14 | 3,00 | 10,00 | 364 de 1120 (33 %) | 2,66 s |
| TCP | 5 | 3,99 | 11,98 | 68 de 400 (17 %) | 3,25 s |

Una cohorte, tenant a tenant (`run_ucx_gpudirect_n8_r1`, trabajo idéntico, arranques dentro
de 1 s):

| tenant | slowdown | multiplicadores por iteración |
|---:|---:|---|
| **2** | **1,001** | `1 1 1 1 1 1 1 1 1 1` |
| **3** | **4,874** | `3 4 2 6 5 6 **10** 6 3 1` |

Y el mismo corte en el control nativo (`run_baremetal_n8_r1`): todos los tenants a `8` en casi
todas las iteraciones. Las únicas excepciones son la primera o segunda columna, donde **se lee
el orden de incorporación**: `5 7,9 8 8...`, `2 8 8...`, `1 2 8 8...`.

# 5. Figuras

| fichero | qué muestra |
|---|---|
| `figures/fig1_jain_vs_N.pdf` | Jain de progreso normalizado frente a N, cuatro workloads, cinco sistemas |
| `figures/fig2_lento_rapido_vs_N.pdf` | ratio lento/rápido frente a N -- la métrica que discrimina |
| `figures/fig3_minibude_cuanto.pdf` | **la figura causal**: multiplicador de cuanto por tenant e iteración, nativo frente a Gusto |
| `figures/fig4_llama_por_tenant.pdf` | fracción completada por tenant en llama, régimen útil y saturación |

# 6. Conclusión

## 6.1 Las cinco frases

| frase | veredicto | evidencia |
|---|---|---|
| 1. «Gusto preserva la eficiencia agregada pero no garantiza progreso justo por tenant» | **supported** | MiniBUDE N=8: mediana de slowdown 3,75 frente a 7,83, Jain 0,696 frente a 0,9998 |
| 2. «Con trabajo fijo igual, algunos tenants corren cerca de su ritmo de cliente único mientras otros sufren slowdown múltiple» | **supported** | tenant 2 a x1,001 con multiplicador 1 en las diez iteraciones, junto a tenant 3 a x4,874, misma cohorte, en 3 repeticiones |
| 3. «Los Jain de llama están dominados por llegadas estocásticas y no establecen unfairness del planificador» | **supported** | demanda desigual hasta 7,0x; Jain normalizado 1,0000 en todo el régimen estable; bajo saturación p = 0,48--0,96 contra la nula de permutación |
| 4. «La limitación de fairness depende del workload, no es universal» | **supported** | 4,87x en MiniBUDE frente a 1,03x en BabelStream y CloverLeaf, mismo N, mismo sistema |
| 5. «Un planificador con fairness es ortogonal a la contribución del contrato semántico» | **unsupported** | es una afirmación de diseño; ningún dato de esta auditoría la sostiene ni la refuta |

## 6.2 Los tres resultados de fairness más sólidos

1. **En trabajo fijo, Gusto reparte el cuanto de servicio de forma marcadamente desigual y el
   nativo no** -- 4,87x frente a 1,03x de lento/rápido a N=8, con 15 corridas y control nativo
   y nativo+MPS. El mecanismo es visible por iteración, no inferido.
2. **En serving, Gusto y el nativo son estadísticamente equivalentes en equidad** (IC95 de la
   diferencia pareada de Jain dentro de ±0,05) **y Gusto sirve significativamente más**
   (+0,0764, IC95 excluye el cero).
3. **Los índices de Jain publicados sobre throughput por tenant eran artefacto de demanda.**
   Normalizados, valen 1,0000 exacto incluso con un desbalance de llegadas de 7,0x.

## 6.3 Los tres mayores problemas metodológicos

1. **El denominador de `goodput`** cuenta 55 s de finalizaciones y divide entre 30 s (x1,76).
2. **Jain sobre servicio absoluto** en presencia de demanda desigual -- invalida todo índice de
   equidad publicado para llama.
3. **Las repeticiones se concatenan en el JSONL** sin separador, de modo que cualquier análisis
   por corrida era imposible sin reconstruirlas. Y en el análisis de trabajo fijo, agrupar por
   semilla en vez de por ruta de cohorte **fusiona campañas distintas**.

## 6.4 Qué retirar y qué conservar

**Retirar del paper:**

- Cualquier índice de Jain calculado sobre throughput o tokens por tenant (llama). Sustituir
  por fracciones normalizadas por demanda.
- El índice de Jain de SLO en los puntos saturados: ahí mide inanición, no equidad.
- El valor absoluto de goodput bajo saturación como si fuera una tasa (277,3 / 302,9 t/s), o
  declarar la ventana real de 55 s.
- El **x1,42** multi-tenant: es el máximo de tres corridas. La media de las tres es **x1,37**.

**Conservar:**

- El cociente entre brazos bajo saturación: sobrevive al cambio de denominador.
- El **x1,37** frente a nativo por defecto y la **paridad al 97--99 % frente a nativo+MPS**.
- Las cifras de trabajo fijo: cohortes completas, trabajo verificado, arranque coordinado a
  0,0 s en el control.

## 6.5 La afirmación más fuerte y honesta para ASPLOS

> *El remoting de API consolida el trabajo multi-tenant en un solo contexto CUDA, lo que
> mejora tanto el rendimiento agregado como la fracción de trabajo servida --en llama, +0,076
> de fracción completada frente al nativo, con IC95 que excluye el cero-- **sin degradar la
> equidad de servicio**, que es estadísticamente equivalente a la del nativo. Ese mismo
> mecanismo, sin embargo, **no garantiza progreso equitativo con trabajo fijo**: en MiniBUDE a
> ocho tenants, el 34 % de las iteraciones se sirven sin espera alguna mientras otras encolan
> tras hasta diez cuantos, de modo que un tenant termina a su ritmo de cliente único y otro
> cinco veces más lento, donde el nativo y el nativo+MPS reparten a 1,03x. La limitación es de
> ordenación, no del camino de datos: aparece igual sobre TCP.*

## 6.6 El experimento mínimo que falta

**Uno, y es barato.** Instrumentar el backend para registrar, por conexión, el instante de
entrada en cola y el de despacho de cada RPC, y repetir MiniBUDE a N=8. Eso separaría las tres
hipótesis que los datos actuales no distinguen:

- orden de llegada de las conexiones (¿el tenant favorecido es siempre el que conecta antes?);
- monopolización de un hilo o *stream* del backend;
- bloqueo en cabeza de cola en el despachador.

Sin esa traza puedo demostrar **que** el reparto es desigual y **cuánto**, pero no **por qué**.

**Segundo hueco, declarado:** la campaña de llama **no tiene marca de tiempo por petición**
(se añadió el 2026-08-02, después). Por eso no son reconstruibles las finalizaciones dentro de
ventana frente a las del drenaje por tenant, la primera finalización, ni el intervalo máximo
sin progreso. Un timeline de serving exige volver a correr con el `bench.py` ya parcheado.
