/*
 * gVirtuS -- envio diferido de fatbins.
 *
 * EL PROBLEMA, MEDIDO
 * -------------------
 * `import cudf` registra 374 fatbins y los manda TODOS por la red al arrancar: 246,8 MiB
 * y 716 ms, mas otros 246,8 MiB en __cudaRegisterFatBinaryEnd. Nativo hace el mismo
 * registro en local a coste despreciable, de ahi que el import cueste 2,891 s sobre
 * GVirtuS frente a 1,004 s nativo. Una consulta de cuDF usa un punado de kernels: la
 * inmensa mayoria de esos modulos no se llegan a usar nunca.
 *
 * POR QUE NO SIRVE CUDA_MODULE_LOADING=LAZY
 * ----------------------------------------
 * Medido: 3,004 vs 3,056 s, es decir nada. Esa carga perezosa es del DRIVER y ocurre
 * DESPUES de que el fatbin haya cruzado la red. El desperdicio esta antes, en el envio.
 * Por eso hay que diferir aqui y no alli.
 *
 * COMO
 * ----
 * __cudaRegisterFatBinary sigue analizando el fatbin en LOCAL (hace falta: de ahi salen
 * las tablas de parametros de kernel que usa el lanzamiento), pero no lo envia: se anota
 * como pendiente y se devuelve el mismo handle de siempre. Esto es posible porque el
 * handle que devuelve GVirtuS es el propio puntero del cliente, no uno del backend, asi
 * que diferir no cambia ninguna identidad.
 *
 * RegisterFunction / RegisterVar se encolan igual, y ademas anotan que hostFun pertenece
 * a que fatbin. En el primer cudaLaunchKernel que referencia una de esas funciones se
 * envia el fatbin de esa familia y sus registros, en el mismo orden que tendrian sin
 * diferir, y despues se lanza.
 *
 * SEGURIDAD
 * ---------
 * Cualquier ruta que pueda necesitar un modulo sin pasar por un lanzamiento --las de
 * simbolos (cudaMemcpyToSymbol, cudaGetSymbolAddress, ...) y cudaFuncGetAttributes--
 * llama a flush_all() antes de nada. Es conservador: si no sabemos a que fatbin toca,
 * se envian todos y se pierde la ganancia, pero nunca se ejecuta contra un backend que
 * no conoce el modulo.
 *
 * Apagado por defecto: GVIRTUS_LAZY_FATBIN=1.
 */

#ifndef GVIRTUS_CUDART_LAZYFATBIN_H
#define GVIRTUS_CUDART_LAZYFATBIN_H

#include <cstddef>
#include <vector_types.h>

namespace gvirtus_lazyfat {

bool enabled();

void note_fatbin(void **handle, void *bin);
void note_fatbin_end(void **handle, void *bin);

/* Devuelven false si el handle NO estaba diferido: el llamante debe entonces enviar
 * por la via normal. Sin esto, una registracion cuyo fatbin no anotamos se perderia en
 * silencio y su kernel fallaria con cudaErrorInvalidDeviceFunction. */
bool note_function(void **handle, const void *hostFun, char *deviceFun, const char *deviceName,
                   int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

bool note_var(void **handle, char *hostVar, char *deviceAddress, const char *deviceName, int ext,
              int size, int constant, int global);

/* Envia el fatbin duenno de hostFun si aun no se ha enviado. Barato y sin cerrojos en el
 * caso comun (ya enviado), porque esta en el camino de cada lanzamiento. */
void ensure_for_hostfun(const void *hostFun);

/* Envia todo lo pendiente. Para rutas que no sabemos atribuir a un fatbin concreto. */
void flush_all();

/* Diagnostico (GVIRTUS_LAZY_FATBIN_STATS=1): cuantos se enviaron de cuantos. */
void report_stats();

}  // namespace gvirtus_lazyfat

#endif  // GVIRTUS_CUDART_LAZYFATBIN_H
