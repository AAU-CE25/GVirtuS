#ifndef GVIRTUS_COMMUNICATORS_ABLATIONGATE_H
#define GVIRTUS_COMMUNICATORS_ABLATIONGATE_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Ablacion de CORRECTITUD del protocolo de vida de los slots RMA.
// ---------------------------------------------------------------------------
// El protocolo protege la reutilizacion de un slot con tres mecanismos apilados. El
// argumento de que los tres hacen falta solo es convincente si se puede ENSENAR que
// quitando cada uno aparece un fallo concreto y reproducible. Esta puerta compila las
// variantes degradadas para poder medirlas:
//
//   variante         mecanismo retirado          fallo esperado
//   ---------------  --------------------------  ------------------------------------
//   pointer_keyed    invalidacion del registro    mapeo obsoleto / corrupcion silenciosa
//                    (cache por direccion)        cuando el asignador recicla una direccion
//   no_generation    contador de generacion       ABA: un ack viejo libera un slot que ya
//                                                 se reasigno a otra operacion
//   no_epoch         epoch del layout             un descriptor de un layout ya sustituido
//                                                 libera un slot del layout NUEVO
//   full (defecto)   ninguno                      cero fallos
//
// Se controla en tiempo de EJECUCION, no de compilacion, para que las cuatro variantes se
// midan con el MISMO binario: si cada una necesitase su propia build, cualquier diferencia
// observada podria venir del compilador y no del mecanismo retirado.
//
// La puerta grita por stderr la primera vez que se consulta con una variante activa. Una
// ablacion de correctitud encendida por descuido produce numeros de rendimiento que parecen
// legitimos, y este proyecto ya tiene un caso de un parche que siguio activo 11 h despues
// de que su revert fallara en silencio.

namespace gvirtus {
namespace communicators {

enum class Ablation {
    Full = 0,        // protocolo completo
    PointerKeyed,    // cache de registro por direccion, sin invalidacion en free
    NoGeneration,    // sin guarda ABA
    NoEpoch,         // sin guarda de epoch
    NoEpochGen,      // sin NINGUNA de las dos: la unica celda en la que un ack viejo puede
                     // llegar a liberar un slot vivo, porque la guarda de generacion esta
                     // DETRAS de la de epoch y atrapa lo que aquella deja pasar.
};

inline Ablation ablation_mode() {
    static const Ablation m = []() {
        const char *v = std::getenv("GVS_ABLATE");
        if (v == nullptr || v[0] == '\0' || std::strcmp(v, "full") == 0) return Ablation::Full;
        if (std::strcmp(v, "pointer_keyed") == 0) {
            std::fprintf(stderr, "[GVS ABLATE] *** registration cache keyed BY ADDRESS, with no "
                                 "invalidation on free -- deliberately DEFECTIVE variant\n");
            return Ablation::PointerKeyed;
        }
        if (std::strcmp(v, "no_generation") == 0) {
            std::fprintf(stderr, "[GVS ABLATE] *** generation (ABA) guard DISABLED "
                                 "-- deliberately DEFECTIVE variant\n");
            return Ablation::NoGeneration;
        }
        if (std::strcmp(v, "no_epoch_gen") == 0) {
            std::fprintf(stderr, "[GVS ABLATE] *** epoch AND generation guards DISABLED "
                                 "-- deliberately DEFECTIVE variant\n");
            return Ablation::NoEpochGen;
        }
        if (std::strcmp(v, "no_epoch") == 0) {
            std::fprintf(stderr, "[GVS ABLATE] *** epoch guard DISABLED "
                                 "-- deliberately DEFECTIVE variant\n");
            return Ablation::NoEpoch;
        }
        std::fprintf(stderr, "[GVS ABLATE] unrecognised value '%s'; using the full protocol\n", v);
        return Ablation::Full;
    }();
    return m;
}

inline bool ablated(Ablation a) {
    const Ablation m = ablation_mode();
    // La variante combinada tiene que responder que SI a las dos guardas por separado: cada
    // punto del codigo pregunta por la suya, y sin esto la celda "las dos fuera" apagaria
    // solo una y mediria lo mismo que la celda simple.
    if (m == Ablation::NoEpochGen)
        return a == Ablation::NoEpoch || a == Ablation::NoGeneration;
    return m == a;
}

// ---------------------------------------------------------------------------
// Inyeccion de fallos (complemento de la ablacion).
// ---------------------------------------------------------------------------
// La ablacion APAGA protecciones; esto ENCIENDE la condicion adversa que cada proteccion
// existe para sobrevivir. Hacen falta las dos para sostener el argumento: una variante
// degradada que nadie ataca no falla, y un ataque contra el protocolo completo tampoco --
// asi que ninguna de las dos por separado demuestra que el mecanismo sea NECESARIO. La
// celda que lo demuestra es (fallo inyectado x proteccion retirada).
//
//   dup_ack     el mismo SlotConsumed se envia dos veces. Contra el protocolo completo lo
//               descarta la guarda de generacion; sin ella libera un slot ya reasignado.
//   stale_ack   el ack se acuna con una generacion anterior a la vigente.
//   delay_ack   el ack se retrasa GVS_FAULT_MS milisegundos, dando tiempo a que el cliente
//               reutilice el slot antes de que llegue. Combinado con un crecimiento del pool
//               es lo que produce el descriptor tardio que motiva el campo epoch.
//
// Igual que la ablacion: se anuncia por stderr, porque una inyeccion de fallos olvidada
// encendida produce "fallos" que se atribuirian al codigo bajo prueba.
enum class Fault {
    None = 0,
    DupAck,
    StaleAck,
    DelayAck,
};

inline Fault fault_mode() {
    static const Fault m = []() {
        const char *v = std::getenv("GVS_FAULT");
        if (v == nullptr || v[0] == '\0' || std::strcmp(v, "none") == 0) return Fault::None;
        if (std::strcmp(v, "dup_ack") == 0) {
            std::fprintf(stderr, "[GVS FAULT] *** SlotConsumed deliberately DUPLICATED\n");
            return Fault::DupAck;
        }
        if (std::strcmp(v, "stale_ack") == 0) {
            std::fprintf(stderr, "[GVS FAULT] *** SlotConsumed with a STALE generation\n");
            return Fault::StaleAck;
        }
        if (std::strcmp(v, "delay_ack") == 0) {
            std::fprintf(stderr, "[GVS FAULT] *** SlotConsumed deliberately DELAYED\n");
            return Fault::DelayAck;
        }
        // hold_ack / epoch_ack / epoch_ack_idx / slow_ack son inyecciones REALES, solo que
        // las lee UcxCommunicator.cpp directamente y no esta enumeracion. Sin esta rama el
        // banner anuncia "sin inyeccion" en mitad de una corrida con inyeccion.
        if (std::strcmp(v, "hold_ack") == 0 || std::strcmp(v, "epoch_ack") == 0 ||
            std::strcmp(v, "epoch_ack_idx") == 0 || std::strcmp(v, "slow_ack") == 0)
            return Fault::None;
        std::fprintf(stderr, "[GVS FAULT] unrecognised value '%s'; no injection\n", v);
        return Fault::None;
    }();
    return m;
}

inline bool faulting(Fault f) { return fault_mode() == f; }

inline int fault_delay_ms() {
    static const int v = []() {
        const char *e = std::getenv("GVS_FAULT_MS");
        const int n = (e && e[0]) ? std::atoi(e) : 0;
        return (n > 0) ? n : 50;
    }();
    return v;
}

}  // namespace communicators
}  // namespace gvirtus

#endif  // GVIRTUS_COMMUNICATORS_ABLATIONGATE_H
