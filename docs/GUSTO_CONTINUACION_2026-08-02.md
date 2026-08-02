# Continuación de la solidificación del pool RMA — 2026-08-02 (tarde)

Cierra los seis puntos que `docs/HANDOFF_2026-08-02.md` §7 dejó abiertos. Misma convención:
**[medido]** se ejecutó y tiene crudo en `docs/gusto_raw_2026-08-02/`; **[código]** es lectura de
fuente y se dice como tal; **[abierto]** no está demostrado.

Cero commits. Un `.bak_*` por edición. Librerías instaladas en `lib/` verificadas por md5
contra `build/`.

---

## 1. Guarda de epoch: se ejercita, y la conclusión fácil era falsa

El fallo `epoch_ack` no llegaba a armarse porque exigía 20 acks. La causa real es más fina: con
la política escalar (suelo 4 MiB) **los acks sólo empiezan después del primer regrow**, y ya no
hay un segundo cambio de epoch que dispare el replay. Se hizo elegible el ack de armado
(`GVS_FAULT_ARM`, por defecto 20) y se armó en el **primero**, que es el único cuya generación
puede colisionar: las generaciones reinician a 0 en cada layout, así que sólo un ack de
generación 1 casa con un slot recién reasignado.

**[medido]** `growtest 8 MiB → 32 MiB`, pool perezoso, `GVS_FAULT_ARM=1`:

| brazo | `ack_held` | `ack_released` | **`ack_epoch_dropped`** | `ack_on_free` | `gen_mismatch` | `ack_applied` |
|---|---:|---:|---:|---:|---:|---:|
| protocolo completo | 1 | 1 | **1** | 0 | 0 | 61 |
| `GVS_ABLATE=no_epoch` | 1 | 1 | **0** | 0 | 0 | 61 |

La guarda **corre y descarta el ack** de un layout ya sustituido, determinísticamente. Eso cierra
el punto del handoff.

### 1.1 Pero quitarla no rompe nada, y hay que decir por qué

Con la guarda quitada **no cambia ninguna otra cifra**: el ack viejo no encuentra a quién
aplicarse. Se añadió una traza de los `server_idx` anunciados en cada epoch (`[GVS IDX]`) y un
inyector `epoch_ack_idx` que re-entrega el ack retenido **sólo** si además coincide el
`server_idx`, o sea exactamente la combinación que haría daño.

**[medido]** 186 oportunidades evaluadas, **0 entregas**, en tres cargas:

| carga | pool | índices anunciados por epoch | replays evaluados | entregas |
|---|---|---|---:|---:|
| `epochgrow` 5 fases | perezoso | `[1-8] [9-16] [17-24] [25-32]` | 95 | 0 |
| `growtest` | `PREALLOC=1` | `[0-7] → [9-16]` | 62 | 0 |
| **cuDF ETL** | `PREALLOC=1` | **`[0-7] → [0-7] → [9-16]`** | 29 | 0 |

### 1.2 Retractación de mi propia conclusión intermedia

Escrito y luego refutado el mismo día: *«los `server_idx` anunciados son estrictamente
crecientes, luego la guarda es inalcanzable»*. **Falso.** La tercera fila lo desmiente: cuDF
anuncia **`[0-7]` en el epoch 1 y otra vez `[0-7]` en el epoch 2**. Ocurre cuando
`ensure_rma_pool()` vuelve a anunciar sin reconstruir (el pool ya estaba provisionado), y
entonces el mismo índice vive en dos epochs con las generaciones reiniciadas: es exactamente el
estado contra el que existe la guarda.

Lo que **sí** se sostiene, y es más estrecho:

> El estado peligroso —índice repetido entre epochs consecutivos— **es alcanzable y se ha
> observado**. La combinación que produce daño —un ack de generación *g* del epoch *N* aplicado
> a un slot vivo con el mismo índice y la misma generación en el epoch *N+1*— **no la produjo
> ninguna de las tres cargas medidas** (0 de 186), porque los epochs que comparten índices son
> los que no tienen tráfico entre medias. La guarda es **necesaria como invariante del
> protocolo, no ejercitada por las cargas de este banco**.

Casi se publica «defensa en profundidad, no necesidad». Lo habría sostenido un experimento
—95 evaluaciones, 0 entregas— que sólo cubría re-anuncios **con** renumeración. La carga que
faltaba era la que no crece.

---

## 2. Régimen de saturación sostenida

La concurrencia sólo existe **dentro** de una conexión: GVirtuS mapea una conexión por hilo, así
que más hilos son más pools, no más presión. La vía es el despachador asíncrono (`concgrow`,
`GVIRTUS_ASYNC_DISPATCH=1`, 32 transferencias de 8 MiB en vuelo, 10 rondas) con el ack del
backend retrasado a propósito (`slow_ack`).

**[medido]**

| celda | pool | retardo del ack | ocupación media | esperas | espera media | rechazos | corrupción | pared |
|---|---:|---:|---:|---:|---:|---|---|---:|
| S1 | 2 slots | 50 ms | 1,404 / 2 (70 %) | 0 | — | 0 | 0/320 | 18,32 s |
| S2 | 1 slot | 50 ms | **0,967 / 1 (96,7 %)** | **150 / 320 (47 %)** | 23,5 ms | 0 | 0/320 | 18,25 s |
| S3 | 1 slot | 35 s | 1,000 / 1 (100 %) | 1 | **30 s → `decline_timeout`** | 1 → AM | 0/4 | 106,31 s |
| S4 (control) | 1 slot | ninguno | 0,474 / 1 | **150 / 320 (47 %)** | **0,84 µs** | 0 | 0/320 | **2,33 s** |

Tres lecturas, y la tercera es la que importa:

1. **Se entró de verdad en saturación**: 96,7 % de ocupación sostenida y casi la mitad de las
   reservas esperando. Nada de esto se había ejercitado (el handoff anterior tenía *una* espera).
2. **Bajo saturación degrada, no se cuelga.** S3 agota el plazo de 30 s, cuenta el rechazo por
   `decline_timeout`, cae al camino AM y **termina con los datos correctos**.
3. **El coste de la contención no lo fija el pool, lo fija el consumidor.** S4 tiene *las mismas
   150 esperas* que S2 —el patrón de contención es de la carga— pero cuestan **0,84 µs en vez de
   23,5 ms**, y la corrida entera baja de 18,25 s a 2,33 s. Y doblar el pool (S1 frente a S2) no
   compró caudal: 18,32 vs 18,25 s.

---

## 3. ThreadSanitizer: el build nunca estuvo roto

El handoff decía «el build falla y no capturé el error». **Reconfigurado limpio con la receta de
`build_asan` cambiando `address` por `thread`, compila todo**: comunicador, frontend y `cudart`.
Lo que fallaba era la **corrida**, por dos motivos que no se habían separado:

1. **UCX instala hooks de memoria** que chocan con el sanitizer → SIGSEGV antes de la primera
   transferencia. Se desactivan con `UCX_MEM_EVENTS=n UCX_MEM_MMAP_RELOC=n
   UCX_MEM_MALLOC_HOOKS=n UCX_MEM_MALLOC_RELOC=n`.
2. **TSan no admite `LD_PRELOAD` sobre un ejecutable sin instrumentar** (a diferencia de ASan).
   Hay que compilar el banco: `nvcc --cudart shared -g -Xcompiler -fsanitize=thread ... -ltsan`,
   y aun así poner `libtsan.so.0` **primero** en `LD_PRELOAD`.

**[medido]** `stress4 h2d_d2h`, 4 hilos, 75 339 transferencias: **3 carreras de datos + 4
inversiones de orden de cerrojos**. Tras los arreglos de §4: **4 corridas consecutivas con 0
avisos**, y sin regresión (79 119 y 70 002 transferencias, 0 malas).

---

## 4. Cuatro defectos reales, arreglados y verificados

### D1 — El mapa `tid → Frontend` se leía sin el cerrojo que protege las inserciones
`Frontend::Init` hacía `mpFrontends->find(tid)` **cuatro veces fuera de `gFrontendMutex`**
mientras otros hilos insertaban dentro de él. Un `find()` concurrente con el rebalanceo de un
árbol rojo-negro es comportamiento indefinido. Señalado por TSan en `stl_tree.h:780`.
**Arreglo**: resolver el puntero una vez dentro del cerrojo. No cambia qué objeto se elige.

### D2 — `EndpointFactory::ind_endpoint` sin sincronizar
Contador estático con lectura-modificación-escritura desde N hilos. **Arreglo**: `std::atomic<int>`
más cerrojo en la pareja leer-modificar-escribir.

### D3 — Inversión de orden de cerrojos entre nuestro mutex y el worker de UCX
`progress_am_rndv()` tomaba `am_state_->mutex` y **dentro** llamaba a `ucp_request_free()` y
`release_rx_slot()`, que entran en UCX y toman su cerrojo de worker. El callback AM de UCX hace
lo contrario: entra con el cerrojo del worker tomado y ahí pide `am_state_->mutex`. Dos hilos,
un orden cada uno, bloqueo mutuo. TSan lo marcaba 4 veces por corrida.
**Arreglo**: dentro del cerrojo sólo se toca nuestra estructura; lo que entra en UCX se aplaza.
Tras el arreglo: **0 inversiones**.

### D4 — Lectura fuera de rango en el arranque concurrente *(el más grave)*
Los cuatro `Endpoint_{Ucx,Tcp,Rdma,Hybrid}::from_json` indexaban
`j["communicator"][EndpointFactory::index()]` con el contador **en crudo**, mientras
`get_endpoint()` sí le aplica el módulo. Con N hilos abriendo conexión a la vez el contador vale
hasta N−1 y el array tiene un elemento. `nlohmann::operator[](size_type)` sobre un json const
**no comprueba límites**: con índices pequeños cae dentro del mismo bloque del heap y no se nota.

Pila de ASan: `heap-buffer-overflow`, READ of size 1, `Endpoint_Ucx.cpp:66`.

**[medido]** con `firsttouch` (N hilos con barrera antes de su **primera** llamada CUDA):

| hilos | antes del arreglo | después |
|---:|---|---|
| 2, 4, 8, 16 | 0 caídas / 2 cada uno | — |
| 24 | **2 caídas / 2** | 0 / 5 |
| 32 | **7 caídas / 8** | 0 / 8 |
| 64 | — | 0 / 5 |

**9 caídas en 10 corridas** con ≥24 hilos, **0 en 18** después. (Corrige un recuento mío previo
de «6 en 7» y «0 en 15»: había sumado mal las dos ramas del A/B, que a esa altura corrían el
**mismo** binario porque el arreglo todavía no estaba puesto.) Arreglo: el mismo módulo que ya
aplica la fábrica, y `.at()` en vez de `operator[]` para que un índice malo sea una excepción y
no una lectura silenciosa.

### Dos defectos del arnés, de la familia «perilla en el lado que no la lee»
- **`GVS_FAULT_MS` no se propagaba al backend.** `slow_ack` corría **siempre a 50 ms**, dijera lo
  que dijera la variable. Sin esto la celda S3 de §2 era inalcanzable.
- **Un banner que mentía**: `hold_ack`/`epoch_ack`/`slow_ack` disparaban
  `[GVS FAULT] valor no reconocido; sin inyeccion` en mitad de corridas **con** inyección.

---

## 5. cuDF

**[medido]** `cudf_etl`, 8 192 000 filas, 524 MB por batch, memoria host paginable, brazo
`hostrma`, pool de 8 slots × 32 MiB con `PREALLOC=1`.

| | medianas por rep | mediana |
|---|---|---:|
| hoy (instrumentado + los 4 arreglos) | 0,4171 · 0,4132 · 0,4197 | **0,4171 s** |
| `c14_pag` (30 jul, pre-instrumentación) | 0,4128 · 0,4118 · 0,4062 | **0,4118 s** |

**+1,29 %**. Los rangos de tres reps son adyacentes y no se solapan (por 0,4 ms), así que la
diferencia es pequeña pero consistente. **Confound declarado**: es una comparación entre días, y
hoy el host del backend arrastra 44 muestreadores huérfanos de campañas anteriores (~12 % de CPU)
que no existían durante `c14`. No se afirma que el +1,29 % sea el coste de la instrumentación;
se afirma que es la cota superior de una medida no controlada. La medida controlada sigue siendo
la anterior: +0,14 % por RPC de 64 B.

**Datos nuevos del pool sobre cuDF**, que era la carga sin medir:

- `peak_inflight` = **2 de 8** con N=1 **y** con N=8. La demanda pico de 2 se confirma en la
  única carga que faltaba.
- `waited=0`, `decline_timeout=0`. `rma_pct` = **0,10 %**: cuDF es abrumadoramente RPC de
  control pequeño.
- `decline_capacity` = 16 (N=1) y 48 (N=8): hay transferencias por encima del techo de 32 MiB.
- **N=8: 8/8 clientes, 56 registros, sin errores.**
- **El cuelgue al instalar el epoch 3 no se reprodujo**: 4 epochs instalados, corrida completa.
  No se declara arreglado —no se buscó su causa— pero con este código no aparece.

---

## 6. Normalización del titular del paper

`PAPER_READY_POOL_RESULTS.md` §2 anunciaba **«×263 en admisiones»**, y el handoff ya avisaba de
que ese número «depende de la ventana» (remedido ×329). La razón es que **×263 es un cociente de
conteos absolutos** entre dos brazos que no ejecutaron el mismo número de operaciones
(1 531 381 frente a 1 300 310).

El estadístico independiente de la ventana es la **fracción admitida**:

| política | reservas RMA | operaciones | fracción admitida |
|---|---:|---:|---:|
| escalar 4 MiB | 65 | 1 531 381 | 0,00424 % |
| cuadrantes | 17 097 | 1 300 310 | 1,31484 % |

**×310** (1,31484 / 0,00424). Publicar ese cociente y `rma_pct` **con la ventana declarada
(45 s)**; el ×263 de conteos no se publica.

---

## 7. Lo que queda

| # | qué | estado |
|---|---|---|
| 1 | **Causa del abort de llama** | **[medido, y con el límite dicho]**. **26 corridas, 0 abortos, 0 fallos de arranque.** P(0 abortos \| tasa original 1/9) = **4,68 %**, así que la tasa original se rechaza al 5 % — pero **por 0,23 puntos**: la cota superior al 95 % es **10,88 %** frente al 11,11 % puntual. Y ese 1/9 era **un solo evento**, con un intervalo propio de ~1,6 %–79 %. Lo defendible: **la tasa real está por debajo del 10,9 % con 95 % de confianza**; no que el defecto esté arreglado, porque ningún arreglo de hoy apuntaba a él |
| 2 | **Carreras de destrucción al salir** | **[abierto]**. Una corrida de TSan reportó 38 avisos, todos en teardown (`~Buffer`, `~LD_Lib`, refcuentas de `shared_ptr`, `ucs_config_parser_cleanup`) más un SEGV en log4cplus. No se ha repetido en 4 corridas posteriores. No probado ausente |
| 3 | **`GetFrontend` devuelve un objeto que no es el registrado** | **[código, leído entero]**. `GetFrontend` hace `new Frontend()`, llama a `Init()`, e `Init()` crea **otro** `Frontend` y lo registra; el `insert` posterior es un no-op porque la clave ya existe, y se devuelve el primero. **No rompe** porque los métodos que importan ignoran `this` y vuelven a resolver por el mapa (`Frontend::Prepare` hace `mpFrontends->find(tid)->second->...`). Coste: un `Frontend` filtrado por hilo, y una trampa latente — el día que un método use `this` de verdad, operará sobre el objeto equivocado |
| 3b | **`Frontend::Prepare()` repite el patrón de D1 en el camino caliente** | **[código]**. Hace **dos `mpFrontends->find(tid)` sin cerrojo**, y corre **antes de cada RPC**. Es la misma clase de carrera que D1, sobre el mapa que recibe inserciones durante el arranque. **TSan no la marcó** en las corridas finales porque para entonces ya no había inserciones: la ventana es sólo el arranque concurrente. Arreglo recomendado: resolver el puntero del hilo **una vez** en un `thread_local`, que además quita dos búsquedas por RPC |
| 4 | **log4cplus se reconfigura por hilo** | **[medido]** indirectamente: `BasicConfigurator::configure()` corre en cada `Frontend::Init`. Es el punto donde TSan revienta |

## 8. Dónde está el crudo

`docs/gusto_raw_2026-08-02/`: `epoch_*.log` (4), `sat_S*.log` (4), `tsan_{1_prefix,2_tras_arreglo_carreras,3_final}.log`, `firsttouch_{pre,post}_arreglo.log`.

Bancos nuevos: `examples/rmatest/epochgrow.cu`, `examples/rmatest/firsttouch.cu`,
`examples/rmatest/stress4_tsan`. Arneses: `~/gusto_{epoch,epochgrow,sat,tsan,firsttouch_ab}_run.sh`.

**Regla que sigue vigente: antes de citar cualquier número, pide su crudo.**

---

## 9. No-regresión: ¿hay que remedir la campaña?

**No.** Los cuatro arreglos no mueven ninguna medida publicada. Mismos bancos, mismas métricas,
mismo pool (8 slots × 32 MiB, min 4 MiB, `PREALLOC=1`, `MIN_BYTES=8192`) y el mismo backend
reiniciado justo antes, que es como corrió el control de la mañana
(`docs/gusto_raw_2026-08-02/{control,rpclat2}`).

| banco | métrica | hoy | control de la mañana | delta | rangos |
|---|---|---:|---:|---:|---|
| miniBUDE (compute-bound) | `avg_ms` | 295,340 | 295,350 | **−0,00 %** | solapan |
| `rma_checksum` 4 MiB | GB/s H2D | 16,602 | 16,472 | **+0,79 %** | no solapan — **hoy más rápido** |
| `rma_checksum` | fallos | 0 | 0 | — | — |
| `rpclat` 64 B | `h2d_us` | 14,791 | 14,715 | +0,52 % | solapan |
| `rpclat` 64 B | `d2h_us` | 16,055 | 15,956 | +0,62 % | solapan |
| `rpclat` 64 B | `ctl_us` | 0,009 | 0,009 | 0,00 % | idénticos |
| llama-7B CONC=8 | goodput | **591,6** (mediana de 26) | 591,6 | **+0,00 %** | ver abajo |
| cuDF batch | latencia | 0,4171 s | 0,4118 s (30 jul) | +1,29 % | comparación entre días, confound declarado |

Las dos diferencias que no solapan van **a favor**: `rma_checksum` es un 0,79 % más rápido, lo
cual es coherente con haber sacado `ucp_request_free()` de dentro del mutex (D3), aunque con
tres reps no se afirma causalidad. Las de `rpclat` (+0,5 %) caen dentro de la dispersión propia
del control, cuya primera rep es siempre la más lenta (arranque en frío) en las dos tandas.

### El goodput de llama merece una nota, y una corrección

Goodput = `completadas × 128 / 45 s`: una métrica derivada de un **entero**, con cuanto
`128/45 = 2,84 t/s = 0,48 %`.

Con las **12** primeras corridas salía **591,6 exacto y desviación 0,00**, y escribí que eso
acotaba la variación real por debajo del 0,48 %. **Con 26 corridas eso ya no se sostiene**:

| goodput | completadas | corridas |
|---:|---:|---:|
| 591,6 | 208 | **25** |
| 568,9 | 200 | **1** |

La mediana sigue clavada en **591,6**, idéntica a la publicada, pero la dispersión real llega a
**−3,8 %** una vez de cada 26. La afirmación correcta es: *la mediana no se mueve, y el banco
tiene un valor atípico de −3,8 % con frecuencia ~1/26*, no *la variación está por debajo del
0,48 %*. La lección es la de siempre en esta campaña: **cero varianza en n=12 era una propiedad
de la muestra, no del sistema**.

### Dos errores míos en el primer intento de esta misma tabla

Se registran porque el primer intento produjo dos «regresiones» que no existían:

1. **`rma_checksum` a 64 MiB frente a un control de 4 MiB.** 64 MiB no cabe en un slot de
   32 MiB, así que las 60 transferencias caían a AM: 6,85 GB/s frente a 16,47. Parecía −58 % y
   era **comparar RMA contra AM**. El contador lo decía en el mismo log (`decline_capacity=60`
   frente a `admit_rma=60` del control) — mirar el contador antes que el cociente lo habría
   atajado en el acto.
2. **miniBUDE con el binario equivocado** (`cuda-bude` en vez de `cuda-bude-gvirtus`): abortaba
   con `cudaErrorNotSupported` y el extractor devolvía `NA` **sin que el guion se quejara**.

Y un tercero que no llegó a producir un número falso porque se comprobó antes: se iba a hacer el
A/B de librerías con `LD_LIBRARY_PATH`, y **el comunicador no se resuelve por ahí** — se abre con
`dlopen` sobre `$GVIRTUS_HOME/lib/libgvirtus-communicators-<proto>.so`, ruta absoluta. Las dos
ramas habrían cargado la misma librería: la trampa del build no emparejado, otra vez.
