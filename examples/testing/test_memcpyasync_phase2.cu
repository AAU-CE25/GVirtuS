// test_memcpyasync_phase2.cu
//
// Phase-2 async large-H2D flow-control stress/correctness test.
// Issues K async large H2D copies (each 1 MB, above the RMA slot threshold) to
// K DISTINCT device buffers, each filled with a distinct value, then a single
// cudaStreamSynchronize barrier and per-buffer D2H verification.
//
// If the RMA slot flow control were wrong (frontend reusing a remote slot the
// backend hasn't consumed), some buffers would be corrupted with another
// iteration's bytes. Correct flow control => every buffer reads back its own
// value regardless of how many slots the backend advertises.
//
// Build: nvcc -O2 --cudart shared -arch=sm_89 test_memcpyasync_phase2.cu -o t
// Run:   GVIRTUS_ASYNC_DISPATCH={0,1} ./t
// Exit 0 = PASS.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "FAIL: CUDA error at %s:%d - %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(_e));                         \
            return 2;                                                          \
        }                                                                      \
    } while (0)

int main() {
    const int N = 262144;   // 1 MB per float buffer (>> 64 KB RMA threshold)
    const int K = 24;       // distinct buffers / async uploads

    const char *mode = std::getenv("GVIRTUS_ASYNC_DISPATCH");
    printf("test_memcpyasync_phase2: GVIRTUS_ASYNC_DISPATCH=%s, K=%d x %zu B\n",
           (mode ? mode : "(unset)"), K, N * sizeof(float));

    CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::vector<float *> d(K, nullptr);
    std::vector<float *> h(K, nullptr);
    for (int k = 0; k < K; ++k) {
        CHECK(cudaMalloc(&d[k], N * sizeof(float)));
        h[k] = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; ++i) h[k][i] = (float)(k + 1);  // distinct per buffer
    }

    // Fire K async large H2D copies; flow control keeps them async until the
    // slot ring would wrap, then transparently demotes to sync to drain.
    for (int k = 0; k < K; ++k)
        CHECK(cudaMemcpyAsync(d[k], h[k], N * sizeof(float), cudaMemcpyHostToDevice, stream));

    CHECK(cudaStreamSynchronize(stream));

    int bad = 0;
    float *chk = (float *)malloc(N * sizeof(float));
    for (int k = 0; k < K; ++k) {
        CHECK(cudaMemcpy(chk, d[k], N * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i)
            if (chk[i] != (float)(k + 1)) {
                if (bad < 8)
                    fprintf(stderr, "  buffer %d elem %d: got %f expected %d\n",
                            k, i, chk[i], k + 1);
                ++bad;
                break;  // one report per buffer
            }
    }
    free(chk);

    for (int k = 0; k < K; ++k) { CHECK(cudaFree(d[k])); free(h[k]); }
    CHECK(cudaStreamDestroy(stream));

    if (bad) { fprintf(stderr, "FAIL: %d/%d buffers corrupted\n", bad, K); return 1; }
    printf("PASS: all %d buffers intact\n", K);
    return 0;
}
