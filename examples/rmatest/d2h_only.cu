// d2h_only.cu — decisive discriminator: corrupt the D2H path with NO large H2D at all.
//
// Every prior harness (rma3x64, rma_srcprov) did:  H2D(pattern t) -> D2H -> verify.
// A mismatch was attributed to H2D by assumption. But the observable is always the
// D2H readback buffer, and "got == want - 31" is equally consistent with the D2H
// returning the PREVIOUS readback's bytes.
//
// This removes the H2D entirely. The device buffer is filled BY A KERNEL, so the
// large-RMA/slot path is never exercised (a kernel launch RPC is a few hundred bytes).
// The only large transfer per iteration is the D2H readback.
//
//   corruption here  -> the defect is in the D2H path; H2D / RMA slots are exonerated
//   clean here       -> the defect really is on the H2D side; look elsewhere
//
// Pattern is identical to the other harnesses so the signature is directly comparable:
//   byte[k] = (char)(t*31 + (k>>12))   =>  got == want - 31 means "previous iteration".
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

using namespace std::chrono;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// Fill the whole buffer on the device. Byte k gets (char)(t*31 + (k>>12)), matching
// the host-side pattern used by rma3x64 / rma_srcprov.
__global__ void fill_pattern(char *d, size_t n, int t) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) d[i] = (char)(t * 31 + (int)(i >> 12));
}

int main(int argc, char **argv) {
    const size_t sz   = (argc > 1) ? (size_t)atoll(argv[1]) : (64ull << 20);
    const int    reps = (argc > 2) ? atoi(argv[2]) : 16;

    char *back = nullptr, *d = nullptr;
    CK(cudaHostAlloc((void **)&back, sz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, sz));

    printf("transfer,d2h_ms,d2h_GBps,bad_samples,first_bad_off,got,want,delta\n");
    int failed = 0;
    for (int t = 0; t < reps; ++t) {
        // Device-side fill: no large H2D, no RMA slot, nothing on the request path.
        fill_pattern<<<256, 256>>>(d, sz, t);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        std::memset(back, 0, sz);
        auto t0 = steady_clock::now();
        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));
        auto t1 = steady_clock::now();

        long nbad = 0; size_t first = (size_t)-1; int got = 0, want = 0;
        for (size_t k = 0; k < sz; k += 4096) {
            int w = (char)(t * 31 + (int)(k >> 12));
            if (back[k] != (char)w) {
                if (first == (size_t)-1) { first = k; got = back[k]; want = (char)w; }
                ++nbad;
            }
        }
        failed += (nbad > 0);

        double ms = duration<double, std::milli>(t1 - t0).count();
        if (nbad)
            printf("%d,%.3f,%.3f,%ld,%zu,%d,%d,%d\n", t + 1, ms,
                   (double)sz / (ms / 1000.0) / 1e9, nbad, first, got, want, got - want);
        else
            printf("%d,%.3f,%.3f,0,-,-,-,-\n", t + 1, ms,
                   (double)sz / (ms / 1000.0) / 1e9);
        fflush(stdout);
    }
    fprintf(stderr, "SUMMARY d2h_only failed_transfers=%d of %d\n", failed, reps);
    CK(cudaFree(d)); CK(cudaFreeHost(back));
    return 0;
}
