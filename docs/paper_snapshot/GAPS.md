---
title: "Huecos del artefacto -- qué falta, por qué importa y el experimento mínimo"
date: "2026-08-02"
geometry: margin=2.3cm
fontsize: 10pt
---

Lista viva de lo que el paquete **no** contiene. Cada entrada dice qué falta, qué afirmación
bloquea, si es reconstruible desde el servidor o exige volver a medir, y el experimento mínimo
que la cierra. Nada de esto se estima ni se rellena.

# 1. Curva de capacidad bajo SLO por N -- **falta, exige medir**

**Qué hay.** Tres regímenes, y ninguno da la curva:

| lo que existe | por qué no sirve para capacidad |
|---|---|
| carga baja: N=1,2,4,8 (+6,10) con lambda **total** = 1 | la demanda total no crece con N, así que el goodput queda clavado en 128 t/s para todo N. Sirve para memoria y para equidad estable, no para capacidad |
| sobrecarga por tenant: N=2 lambda=2 - N=4 lambda=4 - N=8 lambda=8, native/MPS/Gusto, n=3 | todos los puntos son **UNSTABLE**; el SLO alcanzado es 0--3 % y el 77 % de las peticiones muere en el deadline de 25 s |
| saturación de un solo servidor: lambda={1,3,4,6}, 10 semillas, native/Gusto/TCP | es N=1: no dice nada sobre reparto entre tenants |

Hay tres corridas sueltas de Gusto a N=8 con lambda=2, pero **sin control nativo emparejado**: una
colapsa y dos quedan en una región intermedia. No construyen una curva.

**Qué bloquea.** La afirmación que de verdad interesa --*cuánta capacidad útil bajo SLO conserva
el sistema al crecer los tenants*-- no es calculable. Concretamente:

    max SLO-goodput  sujeto a  TTFT p95 < 1 s  y  timeouts = 0

**Experimento mínimo.** N  en  {2,4,8} x lambda  en  {0,25 - 0,5 - 0,75 - 1,0 - 1,5 - 2,0} (total),
**native y Gusto emparejados**, >=3 semillas. La rodilla está estimada en lambda ~= goodput/NPRED ~=
300/128 ~= 2,4 req/s, de ahí la resolución concentrada por debajo.

**Ya está construido y sin correr:** `~/mt_slo_sweep.sh` (mantiene los pods vivos entre puntos
de lambda, aleatoriza el orden dentro de la repetición y derriba entre repeticiones) y
`~/run_slo_grid.sh` (una fila de la rejilla por N). Coste estimado: ~2 h por repetición
completa de la rejilla, dominado por el arranque de los pods.

**Además, `bench.py` ya está parcheado** para emitir métricas de **ventana estricta**
(`goodput_strict`, `slogp_strict_5s`, `pct_slo_strict_*`, `slo_min_tenant_*`) en el sidecar
`.meta.jsonl`, de modo que la curva saldrá ya sin el defecto del denominador.

# 2. Huella de memoria de Native+MPS por tenant -- **falta, exige medir**

**Qué hay.** Huella por tenant de contextos nativos independientes y de Gusto: 4 950 MiB
nativos frente a 4 487 remotos a N=8, es decir ~463 MiB ahorrados por tenant.

**Qué falta.** La tercera columna: **Native+MPS**. Y es justo la que decide el argumento,
porque MPS consolida contextos igual que hace el backend por construcción. Sin ella, el ahorro
de memoria se atribuye al remoting cuando podría deberse solo a la consolidación de contexto --
que es exactamente la distinción que el control de MPS estableció para el rendimiento.

**Experimento mínimo.** Repetir el punto de memoria a N=1,2,4,8 con el demonio MPS activo,
muestreando `nvidia-smi --query-gpu=memory.used` igual que en los otros dos brazos. El arnés
existe: `~/mt_pods_bm_mps.sh N LAMBDA WINDOW on|off`. Recordatorio operativo: los clientes
deben correr con **el mismo uid que el demonio** (`--user $(id -u):$(id -g)` y `--ipc=host`) o
MPS no arranca servidor y la carga corre acelerada en silencio.

# 3. Causa del reparto desigual -- **falta, exige instrumentar**

Está demostrado **que** el reparto es desigual y **cuánto** (`FAIRNESS_RESULTS.md`): a N=8 en
MiniBUDE, el 34 % de las iteraciones se sirven sin espera alguna mientras otras encolan tras
hasta diez cuantos. **Por qué**, no.

Los datos actuales no distinguen tres hipótesis:

- el orden de conexión (¿el tenant favorecido es siempre el que conecta antes?);
- monopolización de un hilo o *stream* del backend;
- bloqueo en cabeza de cola en el despachador.

**Experimento mínimo.** Registrar en el backend, por conexión, el instante de entrada en cola y
el de despacho de cada RPC; repetir MiniBUDE a N=8. Sin esa traza no se puede separar espera de
backend, espera de GPU y tiempo de RPC.

# 4. Timeline de serving por tenant -- **no reconstruible, exige volver a medir**

La campaña multi-tenant de llama es del 2026-07-26 y **no lleva marca de tiempo por petición**
(los campos `tc_rel_s` / `tarr_rel_s` se añadieron el 2026-08-02). Por eso **no** son
reconstruibles, por tenant: finalizaciones dentro de ventana frente a las del drenaje, primera
finalización, e intervalo máximo sin progreso.

Lo que **sí** se reconstruyó y está empaquetado: fracciones de servicio normalizadas por
demanda, atención de SLO por tenant, y la línea nula por permutación
(`llama_fairness_por_tenant.csv`, `llama_fairness_por_corrida.csv`).

Cualquier repetición futura ya sale con los campos, porque el parche está aplicado.

# 5. Huecos de datos, cerrados como tales

| hueco | estado |
|---|---|
| **CloverLeaf sin figura de mérito** | escribe a `clover.out`, que no viaja en el artefacto. Solo hay tiempo de pared por tenant. **No reconstruible sin volver a correr.** |
| **XSBench TCP N=8** | los 64 tenants sin línea `Runtime:`, algunos con `duration_s = 0,0`. 8 cohortes descartadas y **contadas**. Clasificado **E**. |
| **Fila de MPS de XSBench** | los valores 25,11 / 50,15 / 98,40 no tienen crudo en ningún árbol ni CSV, y su línea de comparación era la tabla obsoleta. **Retirada**, ver `XSBENCH_RESULTS.md` §2. |
| **Equidad por tenant de XSBench** | **cerrado el 2026-08-02.** Los `stdout.log` por cliente sí estaban en el servidor; reconstruido en `XSBench_fairness_por_tenant.csv` (306 filas). Ya no hace falta repetir corridas. |
| **Árbol crudo de XSBench parcialmente sobrescrito** | el experimento de densidad a 1.25e9 reutilizó los directorios `seed*/`. Filtrar por `lookups` antes de rederivar. Las tablas publicadas ya están filtradas. |

# 6. Qué hay que empaquetar en el próximo tar

Ficheros producidos el 2026-08-02 que **no** estaban en el paquete anterior:

```
FAIRNESS_RESULTS.md / .pdf              el documento de la auditoría
tenants_canonico.csv                    3688 filas por tenant, cuatro workloads
fairness_trabajo_fijo_por_corrida.csv   493 cohortes
fairness_trabajo_fijo_resumen.csv       mediana entre cohortes
llama_fairness_por_corrida.csv          39 corridas, con línea nula por permutación
llama_fairness_por_tenant.csv           169 filas tenant-corrida
tabla_D_minibude_por_tenant.csv         352 filas, timeline por iteración
XSBench_fairness_por_tenant.csv         306 filas por tenant
tabla_C_cierre.md / tabla_D_minibude.md salidas reproducibles de los análisis
figures/fig1..fig4 (.pdf y .png)        las cuatro figuras
GAPS.md / .pdf                          este documento
```

Y en el repositorio, bajo control de versiones:
`results/asplos_campaign/fairness/` (datos + los seis scripts que los generan),
`results/asplos_campaign/figures/`, `docs/FAIRNESS_RESULTS.md`.

**`~/paper` no está bajo control de versiones.** Las copias durables viven en el repositorio;
el tar debe construirse desde ahí, no desde `~/paper`, o volverá a perder trazabilidad.
