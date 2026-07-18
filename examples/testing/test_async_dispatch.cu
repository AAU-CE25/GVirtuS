// test_async_dispatch.cu
//
// Small, fast correctness test for the GVirtuS async dispatcher
// (GVIRTUS_ASYNC_DISPATCH). Exercises the fire-and-forget allowlist
// (cudaLaunchKernel, cudaMemsetAsync, cudaEventRecord, cudaStreamWaitEvent)
// followed by real synchronization points, and verifies the numeric result is
// bit-identical to what the fully-synchronous path produces.
//
// Contract being validated:
//   * async (no-response) stream-ordered calls still execute in program order;
//   * a cudaStreamSynchronize / cudaMemcpy after them observes their effects;
//   * results are correct whether GVIRTUS_ASYNC_DISPATCH is 0 or 1.
//
// Deliberately tiny (N = 1<<20 = 4 MB/buffer, few iters) so a remote run over
// GVirtuS finishes in well under a second — safe to wrap in a short `timeout`.
//
// Build (frontend container, links GVirtuS stubs via --cudart shared):
//   nvcc -O2 --cudart shared -arch=sm_89 test_async_dispatch.cu -o test_async_dispatch
// Run (async ON):  GVIRTUS_ASYNC_DISPATCH=1 ./test_async_dispatch
// Run (async OFF): GVIRTUS_ASYNC_DISPATCH=0 ./test_async_dispatch
//
// Exit code 0 = PASS, non-zero = FAIL.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "FAIL: CUDA error at %s:%d — %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(_e));                         \
            return 2;                                                          \
        }                                                                      \
    } while (0)

__global__ void saxpy(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int   N       = 1 << 20;      // 1M floats = 4 MB
    const float A        = 2.0f;
    const int   THREADS  = 256;
    const int   BLOCKS   = (N + THREADS - 1) / THREADS;
    const int   ITERS    = 3;           // apply saxpy a few times

    const char *mode = std::getenv("GVIRTUS_ASYNC_DISPATCH");
    printf("test_async_dispatch: GVIRTUS_ASYNC_DISPATCH=%s, N=%d\n",
           (mode ? mode : "(unset)"), N);

    CHECK(cudaSetDevice(0));

    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_y = (float *)malloc(N * sizeof(float));
    if (!h_x || !h_y) { fprintf(stderr, "FAIL: host alloc\n"); return 2; }
    for (int i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 0.0f; }

    float *d_x = nullptr, *d_y = nullptr;
    CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK(cudaMalloc(&d_y, N * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t evt;
    CHECK(cudaEventCreate(&evt));

    // H2D uses synchronous cudaMemcpy (H2D async is intentionally NOT on the
    // async allowlist in v1). d_y is zeroed on-device via the async allowlist.
    CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemsetAsync(d_y, 0, N * sizeof(float), stream));   // async (allowlist)

    // Fire-and-forget path: ITERS kernel launches + an event record, all
    // stream-ordered, none waited on inline when async is enabled.
    for (int it = 0; it < ITERS; ++it) {
        saxpy<<<BLOCKS, THREADS, 0, stream>>>(N, A, d_x, d_y);  // async (allowlist)
    }
    CHECK(cudaEventRecord(evt, stream));                        // async (allowlist)
    CHECK(cudaStreamWaitEvent(stream, evt, 0));                 // async (allowlist)

    // Real synchronization point — must observe all the async work above and,
    // if any async op failed on the backend, surface that error here.
    CHECK(cudaStreamSynchronize(stream));

    // D2H readback (synchronous) — proves the device state is correct.
    CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // A stray error must not be latched (async errors reconcile at sync).
    CHECK(cudaGetLastError());

    const float expected = A * 1.0f * ITERS;  // y = A*x summed ITERS times, x=1
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_y[i] - expected) > 1e-4f) {
            if (bad < 5)
                fprintf(stderr, "  mismatch at %d: got %f expected %f\n", i,
                        h_y[i], expected);
            ++bad;
        }
    }

    CHECK(cudaEventDestroy(evt));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);

    if (bad) {
        fprintf(stderr, "FAIL: %d/%d elements wrong (expected %f)\n", bad, N,
                expected);
        return 1;
    }
    printf("PASS: all %d elements == %f\n", N, expected);
    return 0;
}
