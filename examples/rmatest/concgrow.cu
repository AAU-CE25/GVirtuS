// concgrow.cu — drive a slot-pool REGROW while transfers are still IN FLIGHT.
//
// This is the case growtest.cu cannot reach. A serialized client never has a slot in
// flight when a new RmaSetup lands, so handle_rma_setup_am always takes the "install
// immediately" branch and the quiesce path -- park the layout, keep WriteIovRma from
// issuing, install when the last transfer drains -- never executes. The same goes for
// the shared free-list race: an eager AM can only collide with an RMA put if one is
// outstanding while the other is issued.
//
// GVirtuS maps one connection per THREAD (Frontend::GetFrontend keys on tid), so extra
// threads would just make extra connections. The only way to get several RMA writes in
// flight on ONE connection is the async dispatcher: with GVIRTUS_ASYNC_DISPATCH=1 a
// large cudaMemcpyAsync H2D stays fire-and-forget while a free remote slot exists.
//
// Shape:
//   A  burst of SMALL async H2D          -> pool materialises small, slots go InFlight
//   B  BIG async H2D issued WITHOUT sync -> does not fit, falls back to eager, the
//                                           server regrows and re-advertises while
//                                           A's slots are still unacknowledged
//                                           => the parked/quiesce branch
//   C  more BIG async H2D                -> must run against the installed new layout
//   throughout: tiny SYNCHRONOUS copies  -> eager AMs interleaved with live RMA puts
//                                           => the acquire_rx_slot / rma_persistent race
//
// Every destination is verified after a single sync at the end. Each transfer has its
// own source buffer and its own device buffer, so any mixing is unambiguous.
//
// Run with:  GVIRTUS_ASYNC_DISPATCH=1 GVIRTUS_RMA_SLOTS=4 ... /ex/rmatest/concgrow
// Look for:  "[GVS] rma_setup: epoch N PARKED" on stderr. No PARKED line means the
//            quiesce branch still was not reached and this test proved nothing about it.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

using namespace std::chrono;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

struct Xfer {
    char  *src;
    char  *dev;
    size_t bytes;
    int    tag;
};

static void fill(char *p, size_t n, int tag) {
    for (size_t k = 0; k < n; k += 4096) p[k] = (char)(tag * 31 + (int)(k >> 12));
    p[n - 1] = (char)(0xA0 + (tag & 0x3f));
}

static long check(const char *p, size_t n, int tag) {
    long bad = 0;
    for (size_t k = 0; k < n; k += 4096)
        if (p[k] != (char)(tag * 31 + (int)(k >> 12))) ++bad;
    if (p[n - 1] != (char)(0xA0 + (tag & 0x3f))) ++bad;
    return bad;
}

int main(int argc, char **argv) {
    const size_t small  = (argc > 1) ? (size_t)atoll(argv[1]) : (8ull  << 20);
    const size_t big    = (argc > 2) ? (size_t)atoll(argv[2]) : (64ull << 20);
    const int    nsmall = (argc > 3) ? atoi(argv[3]) : 6;
    const int    nbig   = (argc > 4) ? atoi(argv[4]) : 6;
    const int    rounds = (argc > 5) ? atoi(argv[5]) : 3;
    // argv[6] == "interleave": alternate small/big instead of all-smalls-then-all-bigs,
    // so a regrow can land while slots that FIT the old layout are still in flight.
    const bool interleave = (argc > 6) && std::strcmp(argv[6], "interleave") == 0;

    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Tiny buffer for the interleaved synchronous copies (well under the 4 MB RMA
    // floor, so each one is an eager AM landing in the same RX pool the RMA puts
    // target).
    const size_t tiny = 64 * 1024;
    char *h_tiny = nullptr, *d_tiny = nullptr;
    CK(cudaHostAlloc((void **)&h_tiny, tiny, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d_tiny, tiny));
    std::memset(h_tiny, 0x5A, tiny);

    int tag = 0;
    long total_bad = 0;
    int  total_xfers = 0;

    for (int r = 0; r < rounds; ++r) {
        std::vector<Xfer> xs;

        auto add = [&](size_t bytes) {
            Xfer x{};
            x.bytes = bytes;
            x.tag   = ++tag;
            CK(cudaHostAlloc((void **)&x.src, bytes, cudaHostAllocDefault));
            CK(cudaMalloc((void **)&x.dev, bytes));
            fill(x.src, bytes, x.tag);
            xs.push_back(x);
        };
        for (int i = 0; i < nsmall; ++i) add(small);
        for (int i = 0; i < nbig;   ++i) add(big);

        auto t0 = steady_clock::now();

        if (!interleave) {
            // A: burst of small async H2D. No sync -- these stay in flight.
            for (int i = 0; i < nsmall; ++i) {
                CK(cudaMemcpyAsync(xs[i].dev, xs[i].src, xs[i].bytes,
                                   cudaMemcpyHostToDevice, stream));
                // Interleave an eager AM while RMA puts are outstanding.
                if ((i & 1) == 0)
                    CK(cudaMemcpy(d_tiny, h_tiny, tiny, cudaMemcpyHostToDevice));
            }

            // B+C: big transfers issued with the small ones still unacknowledged. The
            // first cannot fit, so it goes eager and triggers the regrow.
            for (int i = 0; i < nbig; ++i) {
                CK(cudaMemcpyAsync(xs[nsmall + i].dev, xs[nsmall + i].src,
                                   xs[nsmall + i].bytes, cudaMemcpyHostToDevice, stream));
                if ((i & 1) == 0)
                    CK(cudaMemcpy(d_tiny, h_tiny, tiny, cudaMemcpyHostToDevice));
            }
        } else {
            // INTERLEAVED. All-smalls-then-all-bigs cannot reach the quiesce branch:
            // AM messages on one endpoint arrive in order, so every SlotConsumed the
            // server emitted while consuming the smalls precedes the regrow's RmaSetup,
            // and the client has drained by the time the new layout lands (observed:
            // "epoch 3 arrived, 0/2 slots in flight").
            //
            // Alternating means smalls -- which DO fit the current slots and so DO take
            // them -- are still outstanding when a big declines for capacity and
            // triggers the regrow. That is the only shape in which a new advertisement
            // can arrive with a slot genuinely InFlight.
            const int n = (nsmall > nbig) ? nsmall : nbig;
            for (int i = 0; i < n; ++i) {
                if (i < nsmall)
                    CK(cudaMemcpyAsync(xs[i].dev, xs[i].src, xs[i].bytes,
                                       cudaMemcpyHostToDevice, stream));
                if (i < nbig)
                    CK(cudaMemcpyAsync(xs[nsmall + i].dev, xs[nsmall + i].src,
                                       xs[nsmall + i].bytes, cudaMemcpyHostToDevice,
                                       stream));
            }
        }

        CK(cudaStreamSynchronize(stream));
        CK(cudaDeviceSynchronize());
        auto t1 = steady_clock::now();

        // Verify every destination.
        // PINNED readback buffer. A pageable std::vector here would confound the
        // experiment: the D2H fast path is gated on the destination being pinned
        // (gvirtus_is_pinned), so a pageable destination exercises a different path
        // than the one every other harness in examples/rmatest uses.
        char *back = nullptr;
        CK(cudaHostAlloc((void **)&back, (big > small ? big : small), cudaHostAllocDefault));

        long round_bad = 0;
        for (auto &x : xs) {
            std::memset(back, 0, x.bytes);
            CK(cudaMemcpy(back, x.dev, x.bytes, cudaMemcpyDeviceToHost));
            long bad = check(back, x.bytes, x.tag);
            if (bad) {
                // Decode WHOSE data is actually here. byte[k] = tag*31 + (k>>12), so
                // (got - (k>>12)) / 31 recovers the writing transfer's tag when the
                // buffer holds another transfer's payload. Report the raw bytes too so
                // "all zeros / never written" is distinguishable from "wrong transfer".
                size_t first = (size_t)-1;
                for (size_t k = 0; k < x.bytes; k += 4096)
                    if (back[k] != (char)(x.tag * 31 + (int)(k >> 12))) { first = k; break; }
                int got = (first == (size_t)-1) ? 0 : (unsigned char)back[first];
                int want = (first == (size_t)-1) ? 0
                         : (unsigned char)(char)(x.tag * 31 + (int)(first >> 12));
                int delta = got - want;
                int guess_tag = -1;
                for (int t = 1; t <= 64; ++t)
                    if ((unsigned char)(char)(t * 31 + (int)(first >> 12)) == got) { guess_tag = t; break; }
                fprintf(stderr,
                        "MISMATCH round=%d tag=%d bytes=%zu bad_samples=%ld "
                        "first_off=%zu got=%d want=%d delta=%d looks_like_tag=%d\n",
                        r, x.tag, x.bytes, bad, first, got, want, delta, guess_tag);
            }
            round_bad += bad;
        }

        double ms = duration<double, std::milli>(t1 - t0).count();
        size_t moved = (size_t)nsmall * small + (size_t)nbig * big;
        printf("round=%d xfers=%d moved=%zuB issue_ms=%.2f agg_GBps=%.2f bad=%ld %s\n",
               r, (int)xs.size(), moved, ms, (double)moved / (ms / 1000.0) / 1e9,
               round_bad, round_bad ? "FAIL" : "pass");
        fflush(stdout);

        total_bad += round_bad;
        total_xfers += (int)xs.size();

        CK(cudaFreeHost(back));
        for (auto &x : xs) { CK(cudaFree(x.dev)); CK(cudaFreeHost(x.src)); }
    }

    fprintf(stderr, "SUMMARY concgrow transfers=%d bad_samples=%ld %s\n",
            total_xfers, total_bad, total_bad ? "FAIL" : "PASS");
    CK(cudaFree(d_tiny)); CK(cudaFreeHost(h_tiny));
    return total_bad ? 1 : 0;
}
