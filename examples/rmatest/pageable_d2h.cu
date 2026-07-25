// pageable_d2h.cu — is a large cudaMemcpy D2H into PAGEABLE host memory correct?
//
// Every harness in examples/rmatest reads back into cudaHostAlloc'd (pinned) memory.
// concgrow.cu originally used a std::vector, and its large readbacks came back as
// zeros -- which was misread as a fire-and-forget H2D failure until swapping the
// readback to pinned made it pass. Pageable destinations are the common case in real
// applications, so if that path is broken it matters more than the thing it was
// mistaken for.
//
// Same transfer both ways, only the destination's pinned-ness differs.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static long verify(const char *p, size_t n, int t) {
    long bad = 0;
    for (size_t k = 0; k < n; k += 4096)
        if (p[k] != (char)(t * 31 + (int)(k >> 12))) ++bad;
    return bad;
}

int main(int argc, char **argv) {
    const size_t sz   = (argc > 1) ? (size_t)atoll(argv[1]) : (64ull << 20);
    const int    reps = (argc > 2) ? atoi(argv[2]) : 6;

    char *h = nullptr, *d = nullptr, *pinned = nullptr;
    CK(cudaHostAlloc((void **)&h, sz, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&pinned, sz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, sz));
    char *pageable = (char *)std::malloc(sz);
    if (!pageable) { fprintf(stderr, "malloc failed\n"); return 1; }

    printf("transfer,pinned_bad,pageable_bad,verdict\n");
    long total_pinned = 0, total_pageable = 0;
    for (int t = 1; t <= reps; ++t) {
        for (size_t k = 0; k < sz; k += 4096) h[k] = (char)(t * 31 + (int)(k >> 12));
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());

        std::memset(pinned, 0, sz);
        CK(cudaMemcpy(pinned, d, sz, cudaMemcpyDeviceToHost));
        long bp = verify(pinned, sz, t);

        std::memset(pageable, 0, sz);
        CK(cudaMemcpy(pageable, d, sz, cudaMemcpyDeviceToHost));
        long bg = verify(pageable, sz, t);

        total_pinned += bp; total_pageable += bg;
        printf("%d,%ld,%ld,%s\n", t, bp, bg,
               (bp == 0 && bg == 0) ? "both ok"
               : (bp == 0 ? "PAGEABLE BROKEN" : "BOTH BROKEN"));
        fflush(stdout);
    }
    fprintf(stderr, "SUMMARY pinned_bad=%ld pageable_bad=%ld %s\n",
            total_pinned, total_pageable,
            (total_pinned == 0 && total_pageable == 0) ? "PASS" : "FAIL");
    CK(cudaFree(d)); CK(cudaFreeHost(h)); CK(cudaFreeHost(pinned));
    std::free(pageable);
    return (total_pinned || total_pageable) ? 1 : 0;
}
