/*
 * gVirtuS -- cache local de consultas del Driver API.
 *
 * POR QUE
 * -------
 * Un run del ETL de cuDF con 5 batches emite 846 RPC que son consultas puras:
 * cuCtxGetDevice (565), cuDeviceGetAttribute (113), cuCtxGetCurrent (112) y
 * cudaStreamIsCapturing (56). Son ~169 por batch, y cada una cuesta el suelo de RPC
 * --medido en 5-13 us de mediana-- para devolver un valor que no ha cambiado.
 * Responderlas en el cliente ahorra ~3,1 ms por batch, un 8 % del hueco de 38 ms.
 *
 * QUE SE CACHEA Y POR QUE ES SEGURO
 * ---------------------------------
 * cuDeviceGetAttribute: los atributos de un dispositivo son INMUTABLES durante la vida
 *   del proceso. Se cachea por (atributo, dispositivo) sin necesidad de invalidacion.
 *
 * cuCtxGetCurrent / cuCtxGetDevice: dependen del contexto en curso, que SI puede
 *   cambiar. Se cachean y se invalidan desde todas las llamadas que pueden cambiarlo, y
 *   todas pasan por este frontend: cuCtxSetCurrent, cuCtxPushCurrent(_v2),
 *   cuCtxPopCurrent(_v2), cuCtxCreate, cuCtxDestroy, cuDevicePrimaryCtxRetain,
 *   cuDevicePrimaryCtxRelease y cuDevicePrimaryCtxReset.
 *
 * QUE NO SE CACHEA
 * ----------------
 * cudaStreamIsCapturing depende del estado de captura del stream, que cambia sin que
 * nosotros lo veamos necesariamente. Queda fuera a proposito: son 56 de las 846.
 *
 * Apagado por defecto (GVIRTUS_LOCAL_QUERY_CACHE=1) para poder medir el A/B con
 * binarios identicos, como el resto de cambios de esta campana.
 */

#ifndef GVIRTUS_CUDADR_QUERYCACHE_H
#define GVIRTUS_CUDADR_QUERYCACHE_H

#include <cuda.h>

namespace gvirtus_drqcache {

bool enabled();

/* Atributos de dispositivo: inmutables, sin invalidacion. */
bool attr_get(int attrib, int dev, int *out);
void attr_put(int attrib, int dev, int value);

/* Contexto en curso y su dispositivo. */
bool ctx_get_current(CUcontext *out);
void ctx_put_current(CUcontext v);
bool ctx_get_device(CUdevice *out);
void ctx_put_device(CUdevice v);

/* La llama TODA operacion que pueda cambiar el contexto en curso. */
void ctx_invalidate();

}  // namespace gvirtus_drqcache

#endif  // GVIRTUS_CUDADR_QUERYCACHE_H
