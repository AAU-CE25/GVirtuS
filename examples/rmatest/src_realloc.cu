// src_realloc.cu — the H2D twin of dst_realloc.cu.
//
// WriteIovRma caches the ucp_mem_h of a large H2D SOURCE buffer keyed by its address
// (>= 2 MB device, >= 16 MB host). Nothing used to invalidate that on free, so an
// application that frees a source buffer and allocates another at the same address had
// the NIC read the OLD mapping -- the same defect proven on the D2H destination by
// dst_realloc.cu.
//
// Each iteration allocates a FRESH source, fills it with a unique pattern, H2Ds it,
// verifies, and frees it. The address is printed so recycling is visible.
//
//   mode pinned   : cudaHostAlloc / cudaFreeHost   -> the free IS visible to GVirtuS,
//                                                     so this must stay cached AND correct
//   mode pageable : malloc / free                  -> the free is INVISIBLE, so the
//                                                     transport must not cache it at all
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

using namespace std::chrono;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

int main(int argc, char **argv) {
    const size_t sz   = (argc > 1) ? (size_t)atoll(argv[1]) : (64ull << 20);
    const int    reps = (argc > 2) ? atoi(argv[2]) : 8;
    const bool   pin  = (argc > 3) ? (std::strcmp(argv[3], "pageable") != 0) : true;

    char *d = nullptr, *back = nullptr;
    CK(cudaMalloc((void **)&d, sz));
    CK(cudaHostAlloc((void **)&back, sz, cudaHostAllocDefault));  // stable, not under test

    printf("transfer,src_addr,ms,GBps,bad,check\n");
    long total_bad = 0;
    for (int t = 1; t <= reps; ++t) {
        char *h = nullptr;
        if (pin) CK(cudaHostAlloc((void **)&h, sz, cudaHostAllocDefault));
        else     h = (char *)std::malloc(sz);
        for (size_t k = 0; k < sz; k += 4096) h[k] = (char)(t * 31 + (int)(k >> 12));
        h[sz - 1] = (char)(0xA0 + (t & 0x3f));

        CK(cudaDeviceSynchronize());
        auto t0 = steady_clock::now();
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        auto t1 = steady_clock::now();

        std::memset(back, 0, sz);
        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));

        long bad = 0;
        for (size_t k = 0; k < sz; k += 4096)
            if (back[k] != (char)(t * 31 + (int)(k >> 12))) ++bad;
        if (back[sz - 1] != (char)(0xA0 + (t & 0x3f))) ++bad;
        total_bad += bad;

        double ms = duration<double, std::milli>(t1 - t0).count();
        printf("%d,%p,%.3f,%.3f,%ld,%s\n", t, (void *)h, ms,
               (double)sz / (ms / 1000.0) / 1e9, bad, bad ? "FAIL" : "pass");
        fflush(stdout);

        if (pin) CK(cudaFreeHost(h)); else std::free(h);
    }
    fprintf(stderr, "SUMMARY src_realloc mode=%s bad=%ld %s\n",
            pin ? "pinned" : "pageable", total_bad, total_bad ? "FAIL" : "PASS");
    CK(cudaFree(d)); CK(cudaFreeHost(back));
    return total_bad ? 1 : 0;
}
