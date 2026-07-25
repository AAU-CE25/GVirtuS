// rma_checksum.cu — discriminate WHERE a large-transfer payload first goes wrong.
//
// Every previous validation read the destination back with a full-size cudaMemcpy D2H,
// which travels a different path (client-initiated RDMA GET) from the H2D put under
// test. A failure was therefore attributed to H2D by assumption, never by measurement.
//
// This checks the same destination buffer two independent ways per transfer:
//
//   Checkpoint B  a checksum computed ON THE DEVICE by a kernel, returned as 8 bytes.
//                 If the H2D landed correctly, this is correct -- it never uses the
//                 large D2H path.
//   Checkpoint C  the conventional full-size D2H readback, checksummed on the host.
//
// Interpretation:
//   B wrong                -> the payload is wrong in device memory: H2D / RMA / shadow
//   B right, C wrong       -> device memory is fine: the bug is in the large D2H path
//   both wrong, same value -> consistent but wrong data, i.e. H2D wrote the wrong bytes
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

using namespace std::chrono;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// FNV-1a over 64-bit words, reduced with atomics. Deterministic and order independent
// because addition is commutative over the per-word hashes.
__global__ void checksum_kernel(const unsigned long long *data, size_t words,
                                unsigned long long *out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long acc = 0;
    for (; i < words; i += stride) {
        unsigned long long h = 1469598103934665603ULL;
        unsigned long long v = data[i];
        for (int b = 0; b < 8; ++b) {
            h ^= (v >> (b * 8)) & 0xFF;
            h *= 1099511628211ULL;
        }
        acc += h ^ (unsigned long long)i;
    }
    atomicAdd(out, acc);
}

static unsigned long long host_checksum(const unsigned long long *data, size_t words) {
    unsigned long long acc = 0;
    for (size_t i = 0; i < words; ++i) {
        unsigned long long h = 1469598103934665603ULL;
        unsigned long long v = data[i];
        for (int b = 0; b < 8; ++b) { h ^= (v >> (b * 8)) & 0xFF; h *= 1099511628211ULL; }
        acc += h ^ (unsigned long long)i;
    }
    return acc;
}

int main(int argc, char **argv) {
    const size_t sz    = (argc > 1) ? (size_t)atoll(argv[1]) : (64ull << 20);
    const int    reps  = (argc > 2) ? atoi(argv[2]) : 16;
    const size_t words = sz / 8;

    unsigned long long *h = nullptr, *back = nullptr, *d = nullptr, *d_sum = nullptr;
    CK(cudaHostAlloc((void **)&h,    sz, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&back, sz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, sz));
    CK(cudaMalloc((void **)&d_sum, sizeof(unsigned long long)));

    printf("transfer,h2d_ms,h2d_GBps,expected,device_ck,host_ck,B_device,C_host\n");
    for (int t = 0; t < reps; ++t) {
        // unique deterministic payload per transfer
        for (size_t i = 0; i < words; ++i)
            h[i] = (unsigned long long)(t + 1) * 0x9E3779B97F4A7C15ULL + i;
        const unsigned long long expect = host_checksum(h, words);

        CK(cudaDeviceSynchronize());
        auto t0 = steady_clock::now();
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        auto t1 = steady_clock::now();

        // Checkpoint B: checksum computed on the device, only 8 bytes come back
        unsigned long long zero = 0, dev_ck = 0;
        CK(cudaMemcpy(d_sum, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        checksum_kernel<<<256, 256>>>(d, words, d_sum);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&dev_ck, d_sum, sizeof(dev_ck), cudaMemcpyDeviceToHost));

        // Checkpoint C: the conventional full-size D2H readback
        std::memset(back, 0, sz);
        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));
        const unsigned long long host_ck = host_checksum(back, words);

        double ms = duration<double, std::milli>(t1 - t0).count();
        printf("%d,%.3f,%.3f,%llu,%llu,%llu,%s,%s\n", t + 1, ms,
               (double)sz / (ms / 1000.0) / 1e9,
               expect, dev_ck, host_ck,
               dev_ck  == expect ? "pass" : "FAIL",
               host_ck == expect ? "pass" : "FAIL");
        fflush(stdout);
    }

    CK(cudaFree(d)); CK(cudaFree(d_sum));
    CK(cudaFreeHost(h)); CK(cudaFreeHost(back));
    return 0;
}
