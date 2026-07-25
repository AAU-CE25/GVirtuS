// dst_realloc.cu — free the D2H destination and allocate a new one at the same address.
//
// GetFromRemoteGpu caches the client-side ucp_mem_h for the D2H destination in
// client_dst_regs_, keyed by the buffer's ADDRESS, and never invalidates it when the
// application frees that buffer. An application that frees a destination and allocates
// another one -- which the allocator will happily place at the same address -- then gets
// an RDMA GET issued against a registration that describes the OLD mapping.
//
// This is the same hazard as the H2D user_memh_cache, where it is currently papered over
// by size thresholds (2 MB device / 16 MB host, commit 360d473) rather than fixed.
//
// Each iteration: allocate, H2D a unique pattern, D2H back into the fresh buffer, verify,
// free. The address is printed so recycling is visible.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

int main(int argc, char **argv) {
    const size_t sz   = (argc > 1) ? (size_t)atoll(argv[1]) : (64ull << 20);
    const int    reps = (argc > 2) ? atoi(argv[2]) : 6;
    const bool   pin  = (argc > 3) && std::strcmp(argv[3], "pinned") == 0;

    char *h = nullptr, *d = nullptr;
    CK(cudaHostAlloc((void **)&h, sz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, sz));

    printf("transfer,dst_addr,bad,check\n");
    long total_bad = 0;
    for (int t = 1; t <= reps; ++t) {
        for (size_t k = 0; k < sz; k += 4096) h[k] = (char)(t * 31 + (int)(k >> 12));
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());

        // FRESH destination every iteration, then freed -- the allocator reuses the
        // address, which is exactly what the address-keyed registration cache assumes
        // cannot happen.
        char *back = nullptr;
        if (pin) CK(cudaHostAlloc((void **)&back, sz, cudaHostAllocDefault));
        else     back = (char *)std::malloc(sz);
        std::memset(back, 0, sz);

        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));

        long bad = 0;
        for (size_t k = 0; k < sz; k += 4096)
            if (back[k] != (char)(t * 31 + (int)(k >> 12))) ++bad;
        total_bad += bad;
        printf("%d,%p,%ld,%s\n", t, (void *)back, bad, bad ? "FAIL" : "pass");
        fflush(stdout);

        if (pin) CK(cudaFreeHost(back)); else std::free(back);
    }
    fprintf(stderr, "SUMMARY dst_realloc mode=%s bad=%ld %s\n",
            pin ? "pinned" : "pageable", total_bad, total_bad ? "FAIL" : "PASS");
    CK(cudaFree(d)); CK(cudaFreeHost(h));
    return total_bad ? 1 : 0;
}
