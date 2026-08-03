// graphvis2.cu -- el LIMITE del contrato de visibilidad D2H.
//
// graphvis.cu cubre los cinco puntos de observacion EXPLICITOS. Queda un sexto camino legal en
// CUDA: una operacion sincrona sobre el stream legacy sincroniza implicitamente los streams
// "blocking", asi que un grafo lanzado en un stream blocking puede volverse observable sin que
// el programa llame a ninguna de las cinco. Si no se recoge ahi, el contrato hay que
// enunciarlo con esa condicion, no en general.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); \
  return 2; } }while(0)
static const size_t N = 4096;
static const unsigned long long SENT = 0xDEADBEEFCAFEBABEULL;
__global__ void fill(unsigned long long *d, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = i;
}
int main() {
    unsigned long long *d = nullptr, *h = nullptr, *otro_d = nullptr, *otro_h = nullptr;
    CK(cudaMalloc((void **)&d, N * sizeof(*d)));
    CK(cudaHostAlloc((void **)&h, N * sizeof(*h), cudaHostAllocDefault));
    CK(cudaMalloc((void **)&otro_d, 1024));
    CK(cudaHostAlloc((void **)&otro_h, 1024, cudaHostAllocDefault));

    // Stream BLOCKING (cudaStreamDefault), no nonBlocking: es la condicion que hace que el
    // stream legacy lo sincronice.
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamDefault));

    CK(cudaMemcpyAsync(h, d, N * sizeof(*d), cudaMemcpyDeviceToHost, s));  // calentar el pool
    CK(cudaStreamSynchronize(s));

    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    fill<<<(N + 255) / 256, 256, 0, s>>>(d, N);
    CK(cudaMemcpyAsync(h, d, N * sizeof(*d), cudaMemcpyDeviceToHost, s));
    cudaGraph_t g; CK(cudaStreamEndCapture(s, &g));
    cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));

    for (size_t i = 0; i < N; ++i) h[i] = SENT;
    CK(cudaGraphLaunch(ge, s));
    // El unico punto de observacion: una copia SINCRONA en el stream legacy. No se llama a
    // ninguna de las cinco.
    CK(cudaMemcpy(otro_h, otro_d, 1024, cudaMemcpyDeviceToHost));

    int estado = (h[0] == SENT && h[N - 1] == SENT) ? 1 : (h[0] == 0 && h[N - 1] == N - 1 ? 0 : 2);
    printf("observation_point,verdict,detail\n");
    printf("legacy-stream implicit sync,%s,h[0]=0x%llx h[%zu]=0x%llx\n",
           estado == 0 ? "VISIBLE" : (estado == 1 ? "STALE" : "WRONG"),
           (unsigned long long)h[0], N - 1, (unsigned long long)h[N - 1]);
    printf("SUMMARY points=1 failures=%d %s\n", estado ? 1 : 0, estado ? "FAIL" : "PASS");

    CK(cudaDeviceSynchronize());
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(s));
    CK(cudaFree(d)); CK(cudaFreeHost(h)); CK(cudaFree(otro_d)); CK(cudaFreeHost(otro_h));
    return estado ? 1 : 0;
}
