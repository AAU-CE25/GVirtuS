# Recovery, disconnect y limpieza

**Estado**: auditado, un defecto corregido, ambas direcciones probadas.

## 1. Posición de diseño: fallar limpio, no resucitar

**No se intenta recuperar un cliente caído, y es deliberado.** Si el frontend aborta, el
proceso desapareció y no hay nada que recuperar. Si muere el backend, se fue con él todo el
estado remoto: asignaciones de dispositivo, streams, módulos, contexto. Reconectar en silencio
devolvería resultados de un contexto que ya no existe — **éxito falso**, peor que un error.

Lo que sí se exige: fallo **acotado**, error **explícito**, recursos **reclamados**, y que el
backend **vuelva a aceptar** conexiones.

## 2. Dirección A: muere el backend

Cliente a mitad de vuelo (1 562 transferencias completadas), `docker kill` del backend:

| medida | resultado |
|---|---|
| segundos hasta que el cliente sale | **1** |
| código de salida | 1 (error, no éxito) |
| SIGSEGV / doble liberación / corrupción | **no** |
| `GUSTO_METRIC tag=teardown` emitido | **sí** (los destructores corrieron) |
| efecto de `GVIRTUS_UCX_PROGRESS_TIMEOUT_MS` | **ninguno** (1 s con y sin) |
| backend tras reiniciar | escucha |

Que el timeout no cambie nada importa: **el fallo lo detecta el transporte**, no el plazo de
liveness. Eso **refuta** la preocupación del comentario del propio código (`:724-731`) de que
las comprobaciones de liveness están «desactivadas de hecho» — al menos para este modo de fallo.

## 3. Dirección B: muere el cliente

| modo | backend | GPU antes → después | hilos | cliente nuevo |
|---|---|---|---:|---|
| SIGKILL | escucha | 435 → **435 MiB** | 27 | **1 562 transferencias** |

Sin fuga de memoria de GPU, sin acumulación de hilos, y una conexión nueva funciona de
inmediato.

## 4. El defecto que sí había: el bind tras muerte abrupta

Tras `docker kill`, un reinicio del backend podía fallar con

```
sock.c:495 UCX ERROR bind(fd=44 addr=25.25.25.2:32222) failed
```

dejando el contenedor **`Up` sin listener** — el modo de fallo más traicionero del banco,
porque `docker ps` dice que todo va bien.

**Causa**: dos invocaciones del reinicio solapadas. La segunda mata un backend que todavía
compila desde `./src` y arranca otro; compiten por el puerto.

**Corregido** en `reset_backend_pool.sh`, tres cambios:

1. **Cerrojo `flock -w 300`** para serializar reinicios, con el descriptor cerrado en los hijos
   (`9>&-`) para que ningún contenedor herede el cerrojo y lo retenga tras morir el script —
   fallo ya sufrido antes en este banco.
2. **Esperar a que el contenedor desaparezca** (`docker inspect`), no sólo a que el puerto salga
   de `LISTEN`: `ss -ltn` no ve sockets en otros estados.
3. **Un reintento** si no hay listener y el log muestra un fallo de bind, distinguiéndolo de un
   fallo de compilación, que no se arregla reintentando.

Verificado: dos reinicios simultáneos se serializan y ambos acaban con listener.

**Rectificación**: el veredicto `BACKEND_NO_RECUPERA` de la primera tanda era en parte artefacto
del arnés, que llamaba al reinicio dos veces por caso. Con un solo reinicio por caso, el
resultado es `FALLO_ACOTADO_Y_RECUPERA(1s)`.

## 5. Lo que no está implementado

- **Estados explícitos de conexión** (`Initializing/Active/Draining/Failed/Closed`). Hoy hay
  banderas dispersas: `running_`, `endpoint_failed_`, `rma_setup_received_`, `rma_put_capable_`,
  `rma_pool_requested_`, `rma_pool_ready_`.
- **Despertar explícito de waiters en disconnect**: no hace falta hoy porque el reservador
  sondea y comprueba `endpoint_failed_` en cada vuelta, pero es frágil ante un cambio futuro a
  espera por variable de condición.
- Casos no probados: timeout durante la publicación del pool, ACK tardío tras disconnect,
  callback de CUDA tardío, y reutilización de identificadores por una conexión nueva.
