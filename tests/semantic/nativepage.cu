// nativepage.cu -- NATIVE control arm. Question: during an open stream capture,
// does cudaMemcpyAsync H2D from *pageable* host memory invalidate the capture,
// and does it depend on the size?  This is exactly what the backend's
// gvs_capture::stage_host does (std::malloc + cudaMemcpyAsync H2D).
// No GVirtuS involved: real cudart, real GPU.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static const char* run(size_t nb, int pinned) {
    void *h = nullptr, *d = nullptr;
    cudaStream_t s = nullptr; cudaGraph_t g = nullptr;
    if (pinned) cudaHostAlloc(&h, nb, cudaHostAllocDefault);
    else        h = std::malloc(nb);
    cudaMalloc(&d, nb);
    cudaStreamCreate(&s);
    cudaGetLastError();
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, s);
    cudaError_t e = cudaStreamEndCapture(s, &g);
    const char *name = cudaGetErrorName(e);
    if (e == cudaSuccess && g) cudaGraphDestroy(g);
    cudaFree(d);
    if (pinned) cudaFreeHost(h); else std::free(h);
    cudaStreamDestroy(s);
    cudaGetLastError();
    return name;
}

int main() {
    cudaFree(0);
    size_t sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 65536, 1048576};
    std::printf("%10s  %-34s %-34s\n", "bytes", "pageable(malloc)", "pinned(cudaHostAlloc)");
    for (size_t n : sizes) {
        const char *a = run(n, 0);
        const char *b = run(n, 1);
        std::printf("%10zu  %-34s %-34s\n", n, a, b);
    }
    return 0;
}
