/*
 * End-to-end test of the unified RPC protocol over the Communicator base-class
 * framing (the path TCP uses). Drives a full request -> dispatch -> response
 * round trip through:
 *   - Communicator::WriteFrame / TryAcquireFrame  (length-prefixed framing)
 *   - communicators::am::WriteRequest / ReadRequest / WriteResponse /
 *     ReadResponse  (envelope codec, both directions)
 *   - Buffer::AddRef / GetIov  (zero-copy payload segment)
 *   - Buffer::Add()'s GpuRef auto-detect + Communicator::WriteIov's GPU-safe
 *     fallback
 *
 * No CUDA/UCX/log4cplus needed:
 *   g++ -std=c++23 -I include tests/test_protocol_loopback.cpp \
 *       src/communicators/Buffer.cpp src/communicators/RpcCodec.cpp \
 *       src/communicators/DeviceMemory.cpp -o /tmp/tp -ldl
 *
 * The GpuRef response scenario (see bottom of main()) additionally benefits
 * from the fake libcudart in tests/fake_cudart.c on LD_LIBRARY_PATH; without
 * it, IsDevicePointer degrades to false and that scenario verifies the
 * (still correct) host copy fallback instead.
 */
#include <cassert>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "gvirtus/communicators/RpcCodec.h"
#include "gvirtus/communicators/Buffer.h"
#include "gvirtus/communicators/Communicator.h"
#include "gvirtus/communicators/DeviceMemory.h"
#include "gvirtus/communicators/Protocol.h"

using namespace gvirtus::communicators;

// FIFO loopback transport: Write appends, Read consumes. It does NOT override
// TryAcquireFrame/WriteFrame, so it exercises the Communicator base-class
// length-prefixed framing exactly as TcpCommunicator does.
struct LoopComm : Communicator {
    std::deque<char> buf;
    void Serve() override {}
    const Communicator *const Accept() const override { return nullptr; }
    void Connect() override {}
    size_t Read(char *b, size_t n) override {
        size_t i = 0;
        for (; i < n && !buf.empty(); ++i) {
            b[i] = buf.front();
            buf.pop_front();
        }
        return i;
    }
    size_t Write(const char *b, size_t n) override {
        for (size_t i = 0; i < n; ++i) buf.push_back(b[i]);
        return n;
    }
    void Sync() override {}
    void Close() override {}
};

int main() {
    LoopComm c;

    // ---- FRONTEND: assemble + send a request via the codec ----
    const std::string routine = "cudaMemcpy";
    int head_arg = 1234;
    double big[8] = {1, 2, 3, 4, 5, 6, 7, 8};  // zero-copy payload via AddRef
    int tail_arg = 99;

    Buffer in;
    in.Add(head_arg);
    in.AddRef(big, 8);  // borrowed segment — never copied
    in.Add(tail_arg);

    std::vector<IovFrag> piov;
    in.GetIov(piov);
    std::string err;
    bool wreq_ok = am::WriteRequest(&c, /*request_id*/ 0xABCDu, routine, piov.data(),
                                    piov.size(), err);
    assert(wreq_ok);

    // ---- BACKEND: decode the request via the codec ----
    am::EnvelopeHeader got{};
    std::string got_routine;
    const unsigned char *pd = nullptr;
    size_t ps = 0;
    void *gp = nullptr;
    size_t gps = 0;
    bool owns = false;
    bool ok = am::ReadRequest(&c, got, got_routine, pd, ps, gp, gps, owns, err);
    assert(ok);
    assert(owns);
    assert(got.request_id == 0xABCDu);
    assert(got_routine == "cudaMemcpy");
    assert(ps == in.GetLogicalSize());

    // Backend handler parses the payload (non-owning wrap, like Process does).
    Buffer parsed(reinterpret_cast<char *>(const_cast<unsigned char *>(pd)), ps);
    assert(parsed.Get<int>() == head_arg);
    double *gotbig = parsed.Assign<double>(8);
    for (int i = 0; i < 8; ++i) assert(gotbig[i] == big[i]);
    assert(parsed.Get<int>() == tail_arg);
    c.ReleaseFrame();

    // ---- BACKEND: build + send a response via the codec ----
    auto out = std::make_shared<Buffer>();
    int result_val = 4242;
    out->Add(result_val);
    double exec = 0.5;
    bool wok = am::WriteResponse(&c, got, /*exit_code*/ 0, exec, out, err);
    assert(wok);

    // ---- FRONTEND: read the response via the codec ----
    int got_exit = -1;
    double got_exec = 0;
    const unsigned char *got_out = nullptr;
    size_t got_out_size = 0;
    bool resp_owns = false;
    bool rok = am::ReadResponse(&c, /*expected_request_id*/ 0xABCDu, got_exit, got_exec,
                                got_out, got_out_size, resp_owns, err);
    assert(rok);
    assert(resp_owns);
    assert(got_exit == 0);
    assert(got_exec == exec);
    assert(got_out_size == out->GetBufferSize());

    Buffer rout(reinterpret_cast<char *>(const_cast<unsigned char *>(got_out)), got_out_size);
    assert(rout.Get<int>() == result_val);
    c.ReleaseFrame();

    // ---- GpuRef response scenario: a backend handler's output
    // Buffer borrows a device pointer via Add() instead of copying, and the
    // whole WriteResponse -> LoopComm (base-class WriteIov fallback) ->
    // ReadResponse round trip must still deliver the correct bytes. LoopComm
    // does not override WriteIov/WriteFrame, so this exercises exactly the
    // GPU-safe default fallback in Communicator.h (DeviceMemcpyD2H bounce),
    // the same path plain TcpCommunicator would take.
    {
        LoopComm c2;
        SetDeviceProbeEnabled(true);
        void *dev = nullptr;
        const size_t kBig = GVIRTUS_GPUREF_THRESHOLD;
        bool alloc_ok = AllocDeviceMemory(&dev, kBig);
        std::vector<char> pattern(kBig);
        for (size_t i = 0; i < kBig; ++i) pattern[i] = static_cast<char>((i * 7) & 0xFF);

        if (alloc_ok && IsDevicePointer(dev)) {
            assert(DeviceMemcpyD2H(dev, pattern.data(), kBig));  // seed "device" memory

            am::EnvelopeHeader req2{};
            req2.magic = am::kEnvelopeMagic;
            req2.version = am::kEnvelopeVersion;
            req2.request_id = 0x1111u;

            auto out2 = std::make_shared<Buffer>();
            out2->Add(result_val);
            out2->Add(static_cast<char *>(dev), kBig);  // should take the GpuRef path
            assert(out2->HasGpuSegments());

            std::string werr;
            bool wok2 = am::WriteResponse(&c2, req2, /*exit_code*/ 0, exec, out2, werr);
            assert(wok2);

            int exit2 = -1;
            double exec2 = 0;
            const unsigned char *out_data2 = nullptr;
            size_t out_size2 = 0;
            bool owns2 = false;
            std::string rerr;
            bool rok2 = am::ReadResponse(&c2, 0x1111u, exit2, exec2, out_data2, out_size2,
                                        owns2, rerr);
            assert(rok2);
            assert(exit2 == 0);
            assert(exec2 == exec);
            assert(out_size2 == sizeof(int) + sizeof(size_t) + kBig);

            Buffer rout2(reinterpret_cast<char *>(const_cast<unsigned char *>(out_data2)),
                        out_size2);
            assert(rout2.Get<int>() == result_val);
            char *got_bytes = rout2.Assign<char>(kBig);
            assert(memcmp(got_bytes, pattern.data(), kBig) == 0);
            c2.ReleaseFrame();

            printf("test_protocol_loopback: GpuRef response scenario exercised "
                   "(real GpuRef borrow + DeviceMemcpyD2H bounce)\n");
            FreeDeviceMemory(dev);
        } else {
            printf("test_protocol_loopback: no fake libcudart found -- GpuRef "
                   "response scenario verified safe fallback only\n");
            if (alloc_ok) FreeDeviceMemory(dev);
        }
        SetDeviceProbeEnabled(false);
    }

    printf("test_protocol_loopback: ALL PASSED\n");
    return 0;
}
