/*
 * End-to-end test of the unified RPC protocol over the Communicator base-class
 * framing (the path TCP uses). Drives a full request -> dispatch -> response
 * round trip through:
 *   - Communicator::WriteFrame / TryAcquireFrame  (length-prefixed framing)
 *   - communicators::am::ReadRequest / WriteResponse  (envelope codec)
 *   - Buffer::AddRef / GetIov  (zero-copy payload segment)
 *
 * No CUDA/UCX/log4cplus needed:
 *   g++ -std=c++23 -I include tests/test_protocol_loopback.cpp \
 *       src/communicators/Buffer.cpp src/communicators/AmProtocol.cpp -o /tmp/tp
 */
#include <cassert>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "gvirtus/communicators/AmProtocol.h"
#include "gvirtus/communicators/Buffer.h"
#include "gvirtus/communicators/Communicator.h"
#include "gvirtus/communicators/UcxAmProtocol.h"

using namespace gvirtus::communicators;
namespace ux = gvirtus::communicators::ucxam;

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

    // ---- FRONTEND: assemble a request like Frontend::Execute does ----
    const char *routine = "cudaMemcpy";
    int head_arg = 1234;
    double big[8] = {1, 2, 3, 4, 5, 6, 7, 8};  // zero-copy payload via AddRef
    int tail_arg = 99;

    Buffer in;
    in.Add(head_arg);
    in.AddRef(big, 8);  // borrowed segment — never copied
    in.Add(tail_arg);

    ux::EnvelopeHeader rh{};
    rh.magic = ux::kEnvelopeMagic;
    rh.version = ux::kEnvelopeVersion;
    rh.message_type = static_cast<uint16_t>(ux::MessageType::Request);
    rh.header_size = sizeof(ux::EnvelopeHeader);
    rh.request_id = 0xABCDu;
    rh.routine_size = std::strlen(routine);
    rh.payload_size = in.GetLogicalSize();

    std::vector<struct iovec> iov;
    iov.push_back(iovec{static_cast<void *>(&rh), sizeof(rh)});
    iov.push_back(iovec{const_cast<char *>(routine), std::strlen(routine)});
    std::vector<struct iovec> piov;
    in.GetIov(piov);
    iov.insert(iov.end(), piov.begin(), piov.end());
    c.WriteFrame(iov.data(), iov.size());  // base-class length-prefixed frame

    // ---- BACKEND: decode the request via the codec ----
    ux::EnvelopeHeader got{};
    std::string got_routine;
    const unsigned char *pd = nullptr;
    size_t ps = 0;
    void *gp = nullptr;
    size_t gps = 0;
    bool owns = false;
    std::string err;
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
    bool wok = am::WriteResponse(&c, got, /*exit_code*/ 0, exec, out, nullptr, 0, err);
    assert(wok);

    // ---- FRONTEND: read the response frame ----
    const unsigned char *fr = nullptr;
    size_t frs = 0;
    assert(c.TryAcquireFrame(fr, frs));
    ux::EnvelopeHeader resp{};
    assert(frs >= sizeof(resp));
    std::memcpy(&resp, fr, sizeof(resp));
    assert(resp.magic == ux::kEnvelopeMagic);
    assert(resp.message_type == static_cast<uint16_t>(ux::MessageType::Response));
    assert(resp.request_id == 0xABCDu);
    assert(resp.status_code == 0);

    const unsigned char *p = fr + sizeof(resp);
    double got_exec = 0;
    std::memcpy(&got_exec, p, sizeof(double));
    p += sizeof(double);
    assert(got_exec == exec);
    size_t out_size = 0;
    std::memcpy(&out_size, p, sizeof(size_t));
    p += sizeof(size_t);
    assert(out_size == out->GetBufferSize());

    Buffer rout(reinterpret_cast<char *>(const_cast<unsigned char *>(p)), out_size);
    assert(rout.Get<int>() == result_val);
    c.ReleaseFrame();

    printf("test_protocol_loopback: ALL PASSED\n");
    return 0;
}
