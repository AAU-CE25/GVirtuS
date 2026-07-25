// rma3x64.cu — three consecutive 64 MB H2D transfers over ONE persistent connection.
//
// The bandwidth sweep conflates two things: it opens a connection, walks sizes upward,
// and the lazy pool trigger fires somewhere in the middle. This isolates the question
// the brief actually asks: after the pool is built and re-advertised, does a LATER
// transfer on the SAME connection take the RMA path?
//
// Transfer #1 is expected to be staged (the trigger only fires once the backend has
// already received it). #2 and #3 are the ones that matter.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

using namespace std::chrono;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

int main(int argc, char **argv) {
    const size_t sz = (argc > 1) ? (size_t)atoll(argv[1]) : (64ull << 20);
    const int reps  = (argc > 2) ? atoi(argv[2]) : 3;

    char *h = nullptr, *d = nullptr, *back = nullptr;
    CK(cudaHostAlloc((void **)&h, sz, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&back, sz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, sz));

    printf("transfer,bytes,ms,GBps,check\n");
    for (int i = 0; i < reps; ++i) {
        // distinct payload per transfer so a stale-slot bug shows up as a mismatch
        for (size_t k = 0; k < sz; k += 4096) h[k] = (char)(i * 31 + (k >> 12));
        h[sz - 1] = (char)(0xA0 + i);

        CK(cudaDeviceSynchronize());
        auto t0 = steady_clock::now();
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        auto t1 = steady_clock::now();

        // read it back and verify, so a fast-but-wrong path cannot pass
        std::memset(back, 0, sz);
        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));
        bool ok = true;
        size_t first_bad = (size_t)-1; int got = 0, want = 0; long nbad = 0;
        for (size_t k = 0; k < sz; k += 4096) {
            if (back[k] != (char)(i * 31 + (k >> 12))) {
                if (first_bad == (size_t)-1) {
                    first_bad = k; got = back[k]; want = (char)(i * 31 + (k >> 12));
                }
                ++nbad; ok = false;
            }
        }
        if (back[sz - 1] != (char)(0xA0 + i)) {
            if (first_bad == (size_t)-1) {
                first_bad = sz - 1; got = back[sz - 1]; want = (char)(0xA0 + i);
            }
            ++nbad; ok = false;
        }
        if (!ok) {
            fprintf(stderr,
                "MISMATCH transfer=%d first_bad_off=%zu (%.2f%% into buffer) "
                "got=%d want=%d bad_samples=%ld of %zu\n",
                i + 1, first_bad, 100.0 * (double)first_bad / (double)sz,
                got, want, nbad, sz / 4096 + 1);
        }

        double ms = duration<double, std::milli>(t1 - t0).count();
        printf("%d,%zu,%.3f,%.3f,%s\n", i + 1, sz, ms,
               (double)sz / (ms / 1000.0) / 1e9, ok ? "pass" : "FAIL");
        fflush(stdout);
    }

    CK(cudaFree(d));
    CK(cudaFreeHost(h));
    CK(cudaFreeHost(back));
    return 0;
}
