/*
 * End-to-end test of the unified RPC protocol over the Communicator base-class
 * framing (the path TCP uses). Drives a full request -> dispatch -> response
 * round trip through:
 *   - Communicator::WriteFrame / TryAcquireFrame  (length-prefixed framing)
 *   - communicators::am::WriteRequest / ReadRequest / WriteResponse /
 *     ReadResponse  (envelope codec, both directions)
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

    std::vector<struct iovec> piov;
    in.GetIov(piov);
    std::string err;
    bool wreq_ok = am::WriteRequest(&c, /*request_id*/ 0xABCDu, routine, piov.data(),
                                    piov.size(), in.GetLogicalSize(), err);
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
    bool wok = am::WriteResponse(&c, got, /*exit_code*/ 0, exec, out, nullptr, 0, err);
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

    printf("test_protocol_loopback: ALL PASSED\n");
    return 0;
}
