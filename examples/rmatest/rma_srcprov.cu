// rma_srcprov.cu — split the "previous transfer" signature into its two possible causes.
//
// The observed signature (got == want - 31) says the corrupted byte holds the value the
// PREVIOUS transfer put at that offset. Two independent mechanisms predict exactly that:
//
//   (a) destination-side: the receiver's slot still holds the previous transfer's bytes
//   (b) source-side:      the reused, registered host source buffer still held the
//                         previous transfer's bytes when the NIC read it
//
// Every experiment so far reused one source buffer for every transfer, so it could not
// tell them apart. This runs the identical workload two ways:
//
//   mode=reuse : one source buffer, rewritten per transfer   (what all prior runs did)
//   mode=fresh : a distinct source buffer per transfer, written once, never rewritten
//
// If corruption follows `reuse` and vanishes under `fresh`, the defect is on the source
// side (registration/lifetime/visibility of CPU stores to a re-registered buffer) and
// every destination-slot hypothesis is wrong. If it persists under `fresh`, the source is
// exonerated and the destination slot is implicated.
//
// Payload encodes provenance directly: byte = (char)(transfer*31 + page_index), so a
// mismatch of exactly -31 identifies the immediately preceding transfer unambiguously.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

using namespace std::chrono;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

int main(int argc, char **argv) {
    const size_t sz   = (argc > 1) ? (size_t)atoll(argv[1]) : (64ull << 20);
    const int    reps = (argc > 2) ? atoi(argv[2]) : 16;
    const bool   fresh = (argc > 3) && (std::strcmp(argv[3], "fresh") == 0);

    // fresh mode: one dedicated, never-rewritten source buffer per transfer
    std::vector<char *> srcs(fresh ? reps : 1, nullptr);
    for (size_t s = 0; s < srcs.size(); ++s) CK(cudaHostAlloc((void **)&srcs[s], sz, cudaHostAllocDefault));

    char *back = nullptr, *d = nullptr;
    CK(cudaHostAlloc((void **)&back, sz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, sz));

    // In fresh mode fill every source up front, so no CPU store to a source happens
    // anywhere near its transfer.
    if (fresh)
        for (int t = 0; t < reps; ++t) {
            for (size_t k = 0; k < sz; k += 4096) srcs[t][k] = (char)(t * 31 + (k >> 12));
            srcs[t][sz - 1] = (char)(0xA0 + t);
        }

    printf("mode,transfer,ms,GBps,bad_samples,first_bad_off,got,want,delta\n");
    int total_bad = 0;
    for (int t = 0; t < reps; ++t) {
        char *h = fresh ? srcs[t] : srcs[0];
        if (!fresh) {
            for (size_t k = 0; k < sz; k += 4096) h[k] = (char)(t * 31 + (k >> 12));
            h[sz - 1] = (char)(0xA0 + t);
        }

        CK(cudaDeviceSynchronize());
        auto t0 = steady_clock::now();
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        auto t1 = steady_clock::now();

        std::memset(back, 0, sz);
        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));

        long nbad = 0; size_t first = (size_t)-1; int got = 0, want = 0;
        for (size_t k = 0; k < sz; k += 4096) {
            int w = (char)(t * 31 + (k >> 12));
            if (back[k] != (char)w) {
                if (first == (size_t)-1) { first = k; got = back[k]; want = (char)w; }
                ++nbad;
            }
        }
        if (back[sz - 1] != (char)(0xA0 + t)) {
            if (first == (size_t)-1) { first = sz - 1; got = back[sz - 1]; want = (char)(0xA0 + t); }
            ++nbad;
        }
        total_bad += (nbad > 0);

        double ms = duration<double, std::milli>(t1 - t0).count();
        if (nbad)
            printf("%s,%d,%.3f,%.3f,%ld,%zu,%d,%d,%d\n", fresh ? "fresh" : "reuse",
                   t + 1, ms, (double)sz / (ms / 1000.0) / 1e9, nbad, first, got, want, got - want);
        else
            printf("%s,%d,%.3f,%.3f,0,-,-,-,-\n", fresh ? "fresh" : "reuse",
                   t + 1, ms, (double)sz / (ms / 1000.0) / 1e9);
        fflush(stdout);
    }
    fprintf(stderr, "SUMMARY mode=%s failed_transfers=%d of %d\n",
            fresh ? "fresh" : "reuse", total_bad, reps);
    return 0;
}
