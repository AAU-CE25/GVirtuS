// asyncfault.cu -- ¿nombra el registro la operacion que dejo el error?
//
// Provoca un fallo ASINCRONO en un stream y luego sincroniza. En CUDA el sync devuelve el
// error del trabajo anterior, asi que el programa ve el fallo en el sync -- exactamente la
// forma del abort de llama. Lo que se comprueba aqui no es que el sync falle (eso ya se sabe),
// sino que el backend imprima QUE operacion lo causo.
//
// El fallo se provoca con un kernel que escribe fuera de su asignacion: el lanzamiento
// devuelve exito y el error aparece despues.
//
// *** AVISO OPERATIVO: ESTE TEST ENVENENA EL BACKEND. ***
// El error que provoca (717, "operation not supported on global/shared address space") es
// STICKY a nivel de CONTEXTO CUDA. El backend comparte su contexto entre conexiones, asi que
// despues de correr esto TODO cliente posterior falla con 717 hasta que se reinicia el
// backend. Me paso: mi siguiente medida de coste salio con cero transferencias y tarde en ver
// que la causa era el test anterior, no la instrumentacion.
//   -> correr SIEMPRE seguido de `bash ~/reset_backend_gdps.sh` en dpu-01.
//   -> no correrlo en medio de una campana.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void desborda(int *p, size_t n) {
    // Escritura muy lejos del final: fallo de direccion en tiempo de ejecucion.
    p[n + (size_t)1024 * 1024 * 64] = 1;
}

int main() {
    cudaStream_t s;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return 2;
    int *d = nullptr;
    if (cudaMalloc((void **)&d, 1024) != cudaSuccess) return 2;

    // Algo de trafico legitimo antes, para que el anillo tenga contexto que mostrar.
    int host[256];
    for (int i = 0; i < 4; ++i)
        cudaMemcpyAsync(d, host, 1024, cudaMemcpyHostToDevice, s);

    desborda<<<1, 1, 0, s>>>(d, 256);
    const cudaError_t lanz = cudaGetLastError();
    printf("launch rc=%d (%s)\n", (int)lanz, cudaGetErrorString(lanz));

    const cudaError_t sinc = cudaStreamSynchronize(s);
    printf("sync   rc=%d (%s)\n", (int)sinc, cudaGetErrorString(sinc));
    printf("SUMMARY async_fault_surfaced_at_sync=%s\n", sinc != cudaSuccess ? "yes" : "no");
    return sinc != cudaSuccess ? 0 : 1;   // 0 = el experimento hizo lo que pretendia
}
