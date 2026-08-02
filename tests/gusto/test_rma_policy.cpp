// F13 — tests unitarios del selector de colocacion (RmaPolicy.h).
//
// El selector es header-only y de funciones puras, asi que se puede probar de verdad y no
// por integracion. Cada politica necesita su PROPIO proceso: rma_policy() cachea el modo en
// un static, a proposito, asi que no se puede cambiar dentro de una misma ejecucion. El
// driver (test_rma_policy.sh) invoca este binario una vez por politica.
//
// Uso: GVIRTUS_RMA_POLICY=<scalar|quadrant|oracle> ./test_rma_policy <esperado>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include "gvirtus/communicators/RmaPolicy.h"

using gvirtus::communicators::prefer_rma;
using gvirtus::communicators::quadrant_threshold;
using gvirtus::communicators::scalar_floor_bytes;

static int fallos = 0, total = 0;

static void check(bool cond, const char *nombre) {
    ++total;
    if (!cond) { ++fallos; std::printf("  FAIL  %s\n", nombre); }
    else       { std::printf("  ok    %s\n", nombre); }
}

// El umbral se comprueba en sus DOS bordes: uno por debajo debe rechazar y el valor exacto
// debe aceptar. Comprobar solo un punto no distingue >= de >.
static void borde(bool h2d, bool pinned, size_t umbral, const char *nombre) {
    std::string a = std::string(nombre) + " : umbral-1 -> AM";
    std::string b = std::string(nombre) + " : umbral   -> RMA";
    check(!prefer_rma(h2d, pinned, umbral - 1, 0), a.c_str());
    check( prefer_rma(h2d, pinned, umbral,     0), b.c_str());
}

int main(int argc, char **argv) {
    const char *esperado = (argc > 1) ? argv[1] : "scalar";
    std::printf("== politica esperada: %s ==\n", esperado);

    if (std::strcmp(esperado, "quadrant") == 0) {
        // La tabla del articulo: H2D pinned 8 KiB, H2D pageable 1 MiB,
        //                        D2H pinned 1 MiB, D2H pageable 2 MiB.
        check(quadrant_threshold(true,  true ) == (8ull << 10),  "tabla H2D pinned   = 8 KiB");
        check(quadrant_threshold(true,  false) == (1ull << 20),  "tabla H2D pageable = 1 MiB");
        check(quadrant_threshold(false, true ) == (1ull << 20),  "tabla D2H pinned   = 1 MiB");
        check(quadrant_threshold(false, false) == (2ull << 20),  "tabla D2H pageable = 2 MiB");
        borde(true,  true,  8ull << 10, "H2D pinned");
        borde(true,  false, 1ull << 20, "H2D pageable");
        borde(false, true,  1ull << 20, "D2H pinned");
        borde(false, false, 2ull << 20, "D2H pageable");
        // La asimetria de direccion es el argumento central del diseno: a 64 KiB pinned,
        // H2D debe ir por RMA y D2H NO. Si esto falla, la politica no es de cuadrantes.
        check( prefer_rma(true,  true, 64ull << 10, 0), "asimetria: 64 KiB pinned H2D -> RMA");
        check(!prefer_rma(false, true, 64ull << 10, 0), "asimetria: 64 KiB pinned D2H -> AM");
        // Metadata / control: cualquier cosa diminuta va por AM en las cuatro celdas.
        check(!prefer_rma(true,  true,  64, 0), "metadata 64 B H2D pinned   -> AM");
        check(!prefer_rma(true,  false, 64, 0), "metadata 64 B H2D pageable -> AM");
        check(!prefer_rma(false, true,  64, 0), "metadata 64 B D2H pinned   -> AM");
        check(!prefer_rma(false, false, 64, 0), "metadata 64 B D2H pageable -> AM");
        // Tamano cero: nunca RMA (hubo un cuelgue historico con size=0).
        check(!prefer_rma(true, true, 0, 0), "tamano 0 -> AM");
    } else if (std::strcmp(esperado, "scalar") == 0) {
        const size_t f = scalar_floor_bytes();
        check(f == (4ull << 20), "suelo escalar por defecto = 4 MiB");
        // El escalar IGNORA direccion y tipo de memoria: ese es justamente su defecto.
        borde(true,  true,  f, "escalar H2D pinned");
        borde(true,  false, f, "escalar H2D pageable");
        borde(false, true,  f, "escalar D2H pinned");
        borde(false, false, f, "escalar D2H pageable");
        // A 64 KiB pinned H2D el escalar manda a AM donde el cuadrante manda a RMA: es la
        // diferencia que separa las dos politicas y la que hay que poder afirmar.
        check(!prefer_rma(true, true, 64ull << 10, 0), "escalar: 64 KiB pinned H2D -> AM");
        check(!prefer_rma(true, true, 1ull << 20, 0),  "escalar: 1 MiB pinned H2D  -> AM");
    } else if (std::strcmp(esperado, "oracle") == 0) {
        check( prefer_rma(true,  true,  8ull << 10, 0), "oraculo H2D pinned 8 KiB   -> RMA");
        check(!prefer_rma(true,  true,  (8ull << 10) - 1, 0), "oraculo H2D pinned 8K-1 -> AM");
        check( prefer_rma(false, false, 2ull << 20, 0), "oraculo D2H pageable 2 MiB -> RMA");
        check(!prefer_rma(false, false, (2ull << 20) - 1, 0), "oraculo D2H pageable 2M-1 -> AM");
    } else if (std::strcmp(esperado, "override") == 0) {
        // Los umbrales son overridables por entorno para poder retunear en otra fabrica sin
        // recompilar. El driver arranca con GVIRTUS_RMA_MIN_H2D_PINNED=16384.
        check(quadrant_threshold(true, true) == 16384, "override H2D pinned = 16384");
        borde(true, true, 16384, "override H2D pinned");
        // Las otras tres celdas NO deben moverse.
        check(quadrant_threshold(true,  false) == (1ull << 20), "override no toca H2D pageable");
        check(quadrant_threshold(false, true ) == (1ull << 20), "override no toca D2H pinned");
    } else if (std::strcmp(esperado, "floor_override") == 0) {
        check(scalar_floor_bytes() == 65536, "override del suelo escalar = 64 KiB");
        borde(true, true, 65536, "suelo escalar override");
    } else if (std::strcmp(esperado, "desconocida") == 0) {
        // Un valor no reconocido debe caer a escalar, no abortar ni elegir algo al azar.
        check(scalar_floor_bytes() == (4ull << 20), "valor invalido -> suelo escalar");
        check(!prefer_rma(true, true, 64ull << 10, 0), "valor invalido se comporta como escalar");
    }

    std::printf("== %d/%d ok, %d fallos ==\n", total - fallos, total, fallos);
    return fallos == 0 ? 0 : 1;
}
