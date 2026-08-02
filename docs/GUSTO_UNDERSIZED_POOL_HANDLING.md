# Comportamiento con el pool insuficiente

**Estado**: auditado y probado. **No se implementó ningún mecanismo nuevo**, porque las medidas
no lo justifican; se explica por qué abajo.

## 1. Qué hace hoy el código

`WriteIovRma` (`UcxCommunicator.cpp:3157-3284`) reserva un slot **libre y que quepa**. Si no
hay:

1. Sondea con `yield` + `ucp_worker_progress`, con **plazo de 30 s**. No usa la variable de
   condición, y el comentario del código explica por qué: en un cliente de un solo hilo, ese
   hilo es el único que progresa el worker, así que esperar en la cv sería esperar una
   notificación que nadie va a entregar.
2. Al vencer el plazo devuelve `0` y el llamante **cae al camino AM eager**.

Es decir: **contrapresión acotada y luego fallback**, nunca error al usuario ni espera
indefinida.

## 2. Qué se midió

| configuración | reservas | pico | esperas | rechazos | corrupción | backend |
|---|---:|---:|---:|---:|---:|---|
| 1 slot, 4 clientes | 192 | 1 | 0 | 0 | 0 | vivo |
| 2 slots, 4 clientes | 192 | 1 | 0 | 0 | 0 | vivo |
| 4 slots, 4 clientes | 192 | 1 | 0 | 0 | 0 | vivo |
| 2 slots, 1 cliente, dispatch asíncrono | 6 562 | 2 | 0 | 0 | 0 | vivo |
| 4 slots, ídem | 6 455 | 2 | 0 | 0 | 0 | vivo |
| 8 slots, ídem | 6 278 | 2 | 0 | 0 | 0 | vivo |

**Con 1 slot y 4 clientes no hay contención**, y el motivo es estructural: el pool es **por
conexión**. Cuatro clientes son cuatro pools de un slot cada uno. Añadir clientes no puede
saturar un pool.

La única vía es concurrencia **dentro** de una conexión. Se forzó con
`GVIRTUS_ASYNC_DISPATCH=1` y **el pico siguió en 2**, incluso con el pool a 2.

## 3. Por qué no se implementó nada nuevo

La fase pedía elegir entre espera acotada, contrapresión, fallback a AM, fallback al camino
seguro, o error explícito. **Los dos primeros ya existen y el tercero también.** Y sobre todo:
en ninguna de las seis configuraciones se alcanzó el régimen que justificaría más mecanismo —
cero esperas, cero rechazos, cero fallbacks.

Añadir crecimiento dinámico o admisión sensible a presión sería introducir estados de epoch y
ventanas de carrera **contra un problema que no se ha observado**.

## 4. Lo que sí quedó pendiente y es real

- **El sondeo llama a `ucp_worker_progress` bajo `worker_mutex_` desde cada hilo que espera.**
  Con varios hilos de petición sobre una conexión eso es contención sobre el mutex del worker
  justo cuando hace falta progresar para recibir los ACKs que liberan slots. **Hipótesis no
  verificada**: no se pudo observar porque el pool nunca se saturó.
- `acquire_rx_slot` lanza `std::runtime_error` si falla la reserva de memoria fijada, y eso
  ocurre en el camino del handler de AM. Una excepción que escape de un callback de UCX es la
  vía más plausible de matar el backend. **Hipótesis no verificada.**
- No existe la comprobación `waiter_thread_is_not_progress_thread` que pedía la fase.
