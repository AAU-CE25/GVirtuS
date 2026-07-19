// test_memcpyasync_phase1.cu
//
// Phase-1 async cudaMemcpyAsync test: exercises the newly-async paths
//   * small H2D (< 56 KB -> AM path -> fire-and-forget)
//   * cudaMemsetAsync, cudaLaunchKernel (already async)
//   * D2D (pointer-only -> fire-and-forget)
// followed by a real cudaStreamSynchronize barrier and a synchronous D2H
// readback, and verifies the numeric result is bit-identical to the sync path.
//
// Buffers are deliberately small (N=8192 floats = 32 KB) so the H2D copy stays
// on the AM path and takes the async route. Runs in well under a second.
//
// Build (frontend container): nvcc -O2 --cudart shared -arch=sm_89 \
//     test_memcpyasync_phase1.cu -o test_memcpyasync_phase1
// Run:  GVIRTUS_ASYNC_DISPATCH={0,1} ./test_memcpyasync_phase1
// Exit 0 = PASS.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "FAIL: CUDA error at %s:%d - %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(_e));                         \
            return 2;                                                          \
        }                                                                      \
    } while (0)

__global__ void saxpy(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int   N       = 8192;         // 32 KB per float buffer (< 56 KB AM cutoff)
    const float A       = 2.0f;
    const int   THREADS = 256;
    const int   BLOCKS  = (N + THREADS - 1) / THREADS;
    const int   ITERS   = 3;

    const char *mode = std::getenv("GVIRTUS_ASYNC_DISPATCH");
    printf("test_memcpyasync_phase1: GVIRTUS_ASYNC_DISPATCH=%s, N=%d (%zu B)\n",
           (mode ? mode : "(unset)"), N, N * sizeof(float));

    CHECK(cudaSetDevice(0));

    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_z = (float *)malloc(N * sizeof(float));
    if (!h_x || !h_z) { fprintf(stderr, "FAIL: host alloc\n"); return 2; }
    for (int i = 0; i < N; ++i) h_x[i] = 1.0f;

    float *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK(cudaMalloc(&d_z, N * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // --- all of these should be fire-and-forget under async=1 ---
    CHECK(cudaMemcpyAsync(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice, stream)); // small H2D
    CHECK(cudaMemsetAsync(d_y, 0, N * sizeof(float), stream));                            // memset
    for (int it = 0; it < ITERS; ++it)
        saxpy<<<BLOCKS, THREADS, 0, stream>>>(N, A, d_x, d_y);                            // launch
    CHECK(cudaMemcpyAsync(d_z, d_y, N * sizeof(float), cudaMemcpyDeviceToDevice, stream));// D2D

    // --- real sync barrier, then synchronous D2H readback ---
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(h_z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    const float expected = A * 1.0f * ITERS;
    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(h_z[i] - expected) > 1e-4f) {
            if (bad < 5) fprintf(stderr, "  mismatch at %d: got %f expected %f\n",
                                 i, h_z[i], expected);
            ++bad;
        }

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_x)); CHECK(cudaFree(d_y)); CHECK(cudaFree(d_z));
    free(h_x); free(h_z);

    if (bad) { fprintf(stderr, "FAIL: %d/%d wrong (expected %f)\n", bad, N, expected); return 1; }
    printf("PASS: all %d elements == %f\n", N, expected);
    return 0;
}
