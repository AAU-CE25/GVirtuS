# Inyección de fallos sobre el protocolo de vida de los slots

**Estado**: implementado y probado. Una inyección nueva (`hold_ack`) y dos contadores nuevos.

## 1. El principio

La celda que demuestra que un mecanismo es **necesario** es (fallo inyectado × protección
retirada). Una variante degradada que nadie ataca no falla, y un ataque contra el protocolo
completo tampoco: ninguna de las dos por separado demuestra nada.

| eje | variable | valores |
|---|---|---|
| protección retirada | `GVS_ABLATE` | `full`, `no_generation`, `no_epoch`, `pointer_keyed` |
| fallo inyectado | `GVS_FAULT` | `none`, `dup_ack`, `stale_ack`, `delay_ack`, **`hold_ack`** |

**Cada inyección vive en un lado concreto y ponerla en el otro la deja inerte**:

| inyección | función | lado |
|---|---|---|
| `dup_ack`, `stale_ack` | `send_slot_consumed` (`:2351,2361`) | **servidor** |
| `delay_ack`, `hold_ack` | `release_remote_slot` (`:2270`) | **cliente** |

## 2. `hold_ack`: la condición ABA por construcción

Con retardo fijo, que la ventana ABA se abra depende del reloj, y un test que depende del reloj
no puede dar veredicto. `hold_ack` la construye: el ACK se aplica con normalidad **y se guarda
una copia**, que `WriteIovRma` re-entrega en cuanto ese mismo slot vuelve a estar en vuelo con
otra generación.

Costó tres versiones, y los dos fallos son instructivos:

1. **Retener el ACK** se auto-bloquea: si no se aplica, el slot queda `InFlight` para siempre,
   nunca se reasigna, y la copia nunca se entrega (`released=0`, reservas paradas en 7).
2. **Armar en el primer ACK** no dispara: ese ACK viene de una transferencia de arranque contra
   la layout **inicial**; luego el pool crece, cambia el epoch y el régimen estable usa otro
   slot. La traza lo dijo: `armado slot=0 vs reasignado slot=9`, 190 veces.
3. **Armar en un ACK tardío** (el número 20), con la layout ya estable: funciona.

## 3. Resultados

### Guarda de generación bajo ABA determinista

| variante | reservas | `gen_mismatch` | **`ack_on_free`** | corrupción | backend |
|---|---:|---:|---:|---:|---|
| control | 96 | 0 | 0 | 0 | vivo |
| completo | 96 | **1** | **0** | 0 | vivo |
| sin generación | 96 | 1 | **1** | 0 | vivo |

La firma es `ack_on_free`: con la guarda el ACK rancio se rechaza y el slot **sigue en vuelo**;
sin ella **libera un slot vivo** y el ACK legítimo posterior cae en un slot ya libre.

### Guarda de generación bajo ACK obsoleto

| variante | reservas | `gen_mismatch` | comportamiento |
|---|---:|---:|---|
| completo | **7** | 1 | rechaza, no libera, **degrada a AM** |
| sin generación | **96** | 96 | sigue por RMA con 96 violaciones |

### Lo que la matriz NO demuestra

Con `dup_ack` o `delay_ack` sobre carga síncrona: `ack_on_free` alto, `gen_mismatch` cero, y
**quitar la guarda no cambia una sola cifra**. Al duplicado lo rechaza la comprobación de
**estado**, no la de generación.

## 4. Un hueco de instrumentación que hubo que tapar antes

`release_remote_slot` sólo contaba `(state==InFlight && generación≠actual)`. Un ACK duplicado
que llega con el slot ya libre **no incrementaba nada**, así que «la guarda lo rechazó» y «el
fallo no se ejercitó» daban los dos cero. Añadidos `ack_on_free` y `ack_applied`.

## 5. Limitaciones

- **Cero corrupción en todas las celdas.** Con transferencias síncronas de 4 MiB el backend ya
  había terminado de consumir. La violación del invariante está medida; que se materialice
  depende de la carga. **Ausencia de corrupción no es prueba de correctitud.**
- La ablación `no_epoch` no tiene todavía una inyección que construya su condición específica
  (descriptor de un epoch retirado que llega tras la reclamación). **Pendiente.**
- `pointer_keyed` no se ha ejercitado en esta campaña.


---

## Actualización 2026-08-02 (tarde) — la guarda de epoch ya está demostrada

Lo que aquí quedaba sin ejercitar (`epoch_ack` no llegaba a armarse) está cerrado en
`GUSTO_CONTINUACION_2026-08-02.md` §1:

- La guarda **corre y descarta** el ack de un layout sustituido: `ack_epoch_dropped=1`,
  `ack_on_free=0`, frente a `ack_epoch_dropped=0` con `GVS_ABLATE=no_epoch`.
- Quitarla **no cambia ninguna otra cifra** en las cargas medidas, y ahí hay una retractación
  registrada: los `server_idx` **no** son estrictamente crecientes (cuDF anuncia `[0-7]` en dos
  epochs seguidos), así que el estado peligroso **sí es alcanzable**. Lo que no se produjo en
  ninguna carga es el daño: **0 entregas en 186 oportunidades**.
- Nuevo mando: `GVS_FAULT_ARM` (en qué ack se arma; 20 por defecto) y variante `epoch_ack_idx`
  (re-entrega sólo si además coincide el `server_idx`).
