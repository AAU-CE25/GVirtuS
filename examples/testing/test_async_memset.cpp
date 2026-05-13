#include <cstdio>
#include <cstdlib>
#include <cstddef>

extern "C" {
    typedef int cudaError_t;

    struct CUstream_st;
    typedef CUstream_st* cudaStream_t;

    cudaError_t cudaStreamCreate(cudaStream_t* pStream);
    cudaError_t cudaStreamDestroy(cudaStream_t stream);
    cudaError_t cudaStreamSynchronize(cudaStream_t stream);

    cudaError_t cudaMalloc(void** devPtr, size_t size);
    cudaError_t cudaFree(void* devPtr);

    cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);
}

#define cudaSuccess 0

#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error at %s:%d: error code %d\n", \
                         __FILE__, __LINE__, err);                      \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

int main() {
    const size_t N = 1024 * 1024;

    cudaStream_t stream = nullptr;
    void *d = nullptr;

    std::printf("Creating stream...\n");
    CHECK(cudaStreamCreate(&stream));

    std::printf("Allocating device memory...\n");
    CHECK(cudaMalloc(&d, N));

    std::printf("Before cudaMemsetAsync 1\n");
    CHECK(cudaMemsetAsync(d, 0, N, stream));
    std::printf("After cudaMemsetAsync 1\n");

    std::printf("Before cudaMemsetAsync 2\n");
    CHECK(cudaMemsetAsync(d, 1, N, stream));
    std::printf("After cudaMemsetAsync 2\n");

    std::printf("Before cudaStreamSynchronize\n");
    CHECK(cudaStreamSynchronize(stream));
    std::printf("After cudaStreamSynchronize\n");

    CHECK(cudaFree(d));
    CHECK(cudaStreamDestroy(stream));

    std::printf("OK\n");
    return 0;
}
