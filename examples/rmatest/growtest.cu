// growtest.cu — exercise the on-demand slot pool: build small, then force a regrow.
//
// One connection, three phases:
//   A  small transfers  -> the pool materialises sized to THEM, not to the env ceiling
//   B  large transfers  -> first one cannot fit, falls back to eager (slow), the server
//                          observes the size and regrows + re-advertises, so the ones
//                          after it must come back at fast-path bandwidth
//   C  small again      -> must still work against the grown pool
//
// What to look for: B[0] slow, B[1..] fast. If B never recovers, the regrow or the
// re-advertisement is broken. If any check fails, the epoch/quiesce handoff is unsafe.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

using namespace std::chrono;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static int g_tag = 0;

static void phase(const char *name, size_t sz, int reps, char *h, char *back, char *d) {
    for (int i = 0; i < reps; ++i) {
        const int t = ++g_tag;
        for (size_t k = 0; k < sz; k += 4096) h[k] = (char)(t * 31 + (int)(k >> 12));
        h[sz - 1] = (char)(0xA0 + (t & 0x3f));

        CK(cudaDeviceSynchronize());
        auto t0 = steady_clock::now();
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        auto t1 = steady_clock::now();

        std::memset(back, 0, sz);
        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));

        long nbad = 0; size_t first = (size_t)-1;
        for (size_t k = 0; k < sz; k += 4096)
            if (back[k] != (char)(t * 31 + (int)(k >> 12))) {
                if (first == (size_t)-1) first = k;
                ++nbad;
            }
        if (back[sz - 1] != (char)(0xA0 + (t & 0x3f))) ++nbad;

        double ms = duration<double, std::milli>(t1 - t0).count();
        printf("%s,%d,%zu,%.3f,%.3f,%ld,%s\n", name, i, sz, ms,
               (double)sz / (ms / 1000.0) / 1e9, nbad, nbad ? "FAIL" : "pass");
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    const size_t small = (argc > 1) ? (size_t)atoll(argv[1]) : (8ull << 20);
    const size_t big   = (argc > 2) ? (size_t)atoll(argv[2]) : (64ull << 20);
    const int    reps  = (argc > 3) ? atoi(argv[3]) : 6;
    const size_t maxsz = big > small ? big : small;

    char *h = nullptr, *back = nullptr, *d = nullptr;
    CK(cudaHostAlloc((void **)&h,    maxsz, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&back, maxsz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, maxsz));

    printf("phase,i,bytes,ms,GBps,bad,check\n");
    phase("A_small", small, reps, h, back, d);
    phase("B_big",   big,   reps, h, back, d);
    phase("C_small", small, 3,    h, back, d);

    CK(cudaFree(d)); CK(cudaFreeHost(h)); CK(cudaFreeHost(back));
    return 0;
}
