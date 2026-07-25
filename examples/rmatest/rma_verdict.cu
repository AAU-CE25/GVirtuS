// rma_verdict.cu — settle H2D-vs-D2H attribution without perturbing the race.
//
// Identical timeline to rma_srcprov(reuse): fill host, H2D, sync, memset, big D2H, verify.
// The ONLY addition happens AFTER a mismatch has already been detected, so nothing about
// the racing window changes:
//
//   re-read the very same offsets straight out of device memory with a SMALL cudaMemcpy
//   D2H (well under the 4 MB GPUDirect threshold, so it takes the legacy host-staged
//   path: a real device->host copy, which CUDA defines as returning only once complete).
//
// Interpretation, per failing sample:
//   dev == want  -> device memory is CORRECT. The big D2H handed back stale bytes.
//                   => the defect is in the D2H readback path; H2D/RMA slots exonerated.
//   dev == got   -> device memory really holds the previous transfer's byte.
//                   => the defect is in the H2D path, as assumed for five rounds.
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

    char *h = nullptr, *back = nullptr, *d = nullptr;
    CK(cudaHostAlloc((void **)&h,    sz, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&back, sz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, sz));

    int failed = 0, verdict_d2h = 0, verdict_h2d = 0, verdict_other = 0;
    printf("transfer,bad_samples,first_off,got,want\n");
    for (int t = 0; t < reps; ++t) {
        for (size_t k = 0; k < sz; k += 4096) h[k] = (char)(t * 31 + (int)(k >> 12));

        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());

        std::memset(back, 0, sz);
        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));

        std::vector<size_t> bad;
        for (size_t k = 0; k < sz; k += 4096)
            if (back[k] != (char)(t * 31 + (int)(k >> 12))) bad.push_back(k);

        if (bad.empty()) { printf("%d,0,-,-,-\n", t + 1); fflush(stdout); continue; }
        ++failed;
        printf("%d,%zu,%zu,%d,%d\n", t + 1, bad.size(), bad[0], back[bad[0]],
               (char)(t * 31 + (int)(bad[0] >> 12)));

        // ---- post-mortem: what does DEVICE memory actually hold at those offsets? ----
        for (size_t j = 0; j < bad.size() && j < 8; ++j) {
            const size_t k = bad[j];
            char dev = 0;
            // 1-byte D2H: below every GPUDirect threshold -> legacy synchronous path.
            CK(cudaMemcpy(&dev, d + k, 1, cudaMemcpyDeviceToHost));
            const char want = (char)(t * 31 + (int)(k >> 12));
            const char got  = back[k];
            const char *v = (dev == want) ? "DEVICE_OK__D2H_RETURNED_STALE"
                          : (dev == got)  ? "DEVICE_STALE__H2D_AT_FAULT"
                                          : "DEVICE_THIRD_VALUE";
            if (dev == want) ++verdict_d2h; else if (dev == got) ++verdict_h2d; else ++verdict_other;
            fprintf(stderr, "VERDICT t=%d off=%zu big_d2h=%d device=%d want=%d -> %s\n",
                    t + 1, k, (int)got, (int)dev, (int)want, v);
        }
        fflush(stdout);
    }

    fprintf(stderr,
            "SUMMARY failed_transfers=%d of %d | samples: device_ok(D2H at fault)=%d "
            "device_stale(H2D at fault)=%d other=%d\n",
            failed, reps, verdict_d2h, verdict_h2d, verdict_other);
    CK(cudaFree(d)); CK(cudaFreeHost(h)); CK(cudaFreeHost(back));
    return 0;
}
