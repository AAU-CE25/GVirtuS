/*
 * gVirtuS -- huella de contenido de un fatbin, para deduplicar su transferencia.
 *
 * POR QUE
 * -------
 * RAPIDS registra 374 fatbins al importar cuDF: 246,8 MiB. Con N clientes eso son N
 * copias del MISMO contenido cruzando la red hacia el mismo backend (a N=8, casi 2 GiB).
 * Un fatbin es contenido inmutable: dos con el mismo contenido producen el mismo modulo.
 *
 * Se indexa por contenido, no por direccion, asi que --a diferencia de la cache de
 * registros RDMA-- NO hay ciclo de vida que rastrear ni invalidacion que resolver. Esa
 * es la diferencia que la hace segura.
 *
 * POR QUE 128 BITS Y NO 64
 * ------------------------
 * Una colision haria cargar el modulo EQUIVOCADO, con resultados silenciosamente
 * erroneos: exactamente la clase de fallo que esta campana lleva persiguiendo. Con 64
 * bits y unos pocos miles de modulos la probabilidad es despreciable en teoria, pero el
 * coste de equivocarse es un resultado mal calculado sin ningun error visible. 128 bits
 * mas la longitud exacta como parte de la clave lo lleva a lo imposible en la practica,
 * y cuesta lo mismo recorrer los bytes una vez.
 *
 * FNV-1a con dos bases distintas: sin dependencias externas y suficiente para contenido
 * que no es adversario.
 */

#ifndef GVIRTUS_COMMON_FATBIN_HASH_H
#define GVIRTUS_COMMON_FATBIN_HASH_H

#include <cstddef>
#include <cstdint>

namespace gvirtus {
namespace common {

struct FatbinKey {
    std::uint64_t len;
    std::uint64_t h1;
    std::uint64_t h2;

    bool operator==(const FatbinKey &o) const {
        return len == o.len && h1 == o.h1 && h2 == o.h2;
    }
    bool operator<(const FatbinKey &o) const {
        if (len != o.len) return len < o.len;
        if (h1 != o.h1) return h1 < o.h1;
        return h2 < o.h2;
    }
};

inline FatbinKey fatbin_key(const void *data, std::size_t len) {
    const std::uint8_t *p = static_cast<const std::uint8_t *>(data);
    std::uint64_t a = 1469598103934665603ULL;   /* FNV-1a offset basis */
    std::uint64_t b = 14695981039346656037ULL;  /* segunda base, independiente */
    for (std::size_t i = 0; i < len; i++) {
        a ^= p[i];
        a *= 1099511628211ULL;
        b ^= static_cast<std::uint64_t>(p[i]) + i;  /* mezcla la posicion: distingue
                                                     * permutaciones que FNV-1a solo
                                                     * podria confundir */
        b *= 1099511628211ULL;
    }
    FatbinKey k;
    k.len = static_cast<std::uint64_t>(len);
    k.h1 = a;
    k.h2 = b;
    return k;
}

}  // namespace common
}  // namespace gvirtus

#endif  // GVIRTUS_COMMON_FATBIN_HASH_H
