// test_memcpyasync_phase3.cu
//
// Phase-3 deferred D2H correctness + stream-ordering test.
//   * dst buffers are pinned (cudaMallocHost) so the deferred path engages.
//   * A deferred D2H must capture device state AT ITS POSITION in the stream,
//     i.e. BEFORE a later kernel that overwrites the source. If deferral broke
//     ordering, the copy would observe the post-kernel value.
//
// Sequence (all on one stream):
//   fill(d, 5) ; D2H h1 <- d (deferred, expect 5) ; fill(d, 99) ;
//   D2H h2 <- d (deferred, expect 99) ; cudaStreamSynchronize ; verify.
//
// Build: nvcc -O2 --cudart shared -arch=sm_89 test_memcpyasync_phase3.cu -o t3
// Run:   GVIRTUS_ASYNC_DISPATCH={0,1} ./t3    (0 = sync path, same result)
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

__global__ void fill(float *x, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

static int check_buf(const float *h, int n, float expected, const char *name) {
    for (int i = 0; i < n; ++i)
        if (fabsf(h[i] - expected) > 1e-4f) {
            fprintf(stderr, "  %s elem %d: got %f expected %f\n", name, i, h[i], expected);
            return 1;
        }
    return 0;
}

int main() {
    const int N = 8192;
    const int T = 256, B = (N + T - 1) / T;
    const char *mode = std::getenv("GVIRTUS_ASYNC_DISPATCH");
    printf("test_memcpyasync_phase3: GVIRTUS_ASYNC_DISPATCH=%s, N=%d\n",
           (mode ? mode : "(unset)"), N);

    CHECK(cudaSetDevice(0));
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));

    float *d = nullptr;
    CHECK(cudaMalloc(&d, N * sizeof(float)));
    float *h1 = nullptr, *h2 = nullptr;
    CHECK(cudaMallocHost(&h1, N * sizeof(float)));   // pinned -> deferred-eligible
    CHECK(cudaMallocHost(&h2, N * sizeof(float)));

    fill<<<B, T, 0, s>>>(d, N, 5.0f);
    CHECK(cudaMemcpyAsync(h1, d, N * sizeof(float), cudaMemcpyDeviceToHost, s));  // deferred, expect 5
    fill<<<B, T, 0, s>>>(d, N, 99.0f);
    CHECK(cudaMemcpyAsync(h2, d, N * sizeof(float), cudaMemcpyDeviceToHost, s));  // deferred, expect 99

    CHECK(cudaStreamSynchronize(s));   // drains both deferred D2H into h1/h2

    int bad = 0;
    bad += check_buf(h1, N, 5.0f, "h1(pre-kernel)");
    bad += check_buf(h2, N, 99.0f, "h2(post-kernel)");

    CHECK(cudaFreeHost(h1)); CHECK(cudaFreeHost(h2));
    CHECK(cudaFree(d)); CHECK(cudaStreamDestroy(s));

    if (bad) { fprintf(stderr, "FAIL: %d buffer(s) wrong (ordering or data)\n", bad); return 1; }
    printf("PASS: deferred D2H correct and stream-ordered\n");
    return 0;
}
