/*
 * Standalone correctness test for Buffer's IoV (scatter/gather) extensions.
 *
 * Verifies the core invariant of the clean-sheet IoV Buffer: AddRef() (the
 * zero-copy, borrowed-pointer path) produces BYTE-IDENTICAL wire output to
 * the legacy Add<T>(ptr, n) copy path, so the receiver and all backend
 * handlers parse it unchanged.
 *
 * Framework-free (asserts + main) so it builds and runs without CUDA/UCX:
 *   g++ -std=c++23 -I include -I src \
 *       tests/test_buffer_iov.cpp src/communicators/Buffer.cpp -o /tmp/tbuf
 */
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gvirtus/communicators/Buffer.h"

using namespace gvirtus::communicators;

// Minimal Communicator that captures everything written (Write + the base
// WriteIov fallback, which concatenates then calls Write).
struct MemComm : Communicator {
    std::string out;
    void Serve() override {}
    const Communicator *const Accept() const override { return nullptr; }
    void Connect() override {}
    size_t Read(char *, size_t) override { return 0; }
    size_t Write(const char *b, size_t n) override {
        out.append(b, n);
        return n;
    }
    void Sync() override {}
    void Close() override {}
};

static std::string concat_iov(const std::vector<struct iovec> &iov) {
    std::string s;
    for (const auto &v : iov) s.append(static_cast<const char *>(v.iov_base), v.iov_len);
    return s;
}

int main() {
    const int head = 7, tail = 9;
    double payload[4] = {1.5, 2.5, 3.5, 4.5};

    // --- 1. No segments: fast path parity ---
    {
        Buffer b;
        int x = 42;
        b.Add(x);
        assert(!b.HasSegments());
        assert(b.GetLogicalSize() == b.GetBufferSize());
        std::vector<struct iovec> iov;
        b.GetIov(iov);
        assert(iov.size() == 1);
        assert(iov[0].iov_len == b.GetBufferSize());
        assert(memcmp(iov[0].iov_base, b.GetBuffer(), b.GetBufferSize()) == 0);
    }

    // --- 2. AddRef ≡ Add: byte-identical wire body ---
    Buffer a;  // all-inline reference
    a.Add(head);
    a.Add(payload, 4);
    a.Add(tail);

    Buffer b;  // segmented (zero-copy)
    b.Add(head);
    b.AddRef(payload, 4);
    b.Add(tail);

    assert(b.HasSegments());
    assert(b.GetLogicalSize() == a.GetBufferSize());

    std::vector<struct iovec> iov;
    b.GetIov(iov);
    std::string body = concat_iov(iov);
    assert(body.size() == a.GetBufferSize());
    assert(memcmp(body.data(), a.GetBuffer(), a.GetBufferSize()) == 0);

    // --- 3. Receiver round-trips the segmented buffer's bytes ---
    {
        std::string wire = body;  // what landed on the wire
        Buffer r(wire.data(), wire.size());  // non-owning receive wrap
        assert(r.Get<int>() == head);
        double *got = r.Assign<double>(4);
        for (int i = 0; i < 4; ++i) assert(got[i] == payload[i]);
        assert(r.Get<int>() == tail);
    }

    // --- 4. Dump() framing parity (size prefix + body) ---
    {
        MemComm ca, cb;
        a.Dump(&ca);
        b.Dump(&cb);
        assert(ca.out == cb.out);  // identical wire incl. the size_t prefix
        size_t framed = 0;
        memcpy(&framed, cb.out.data(), sizeof(size_t));
        assert(framed == a.GetBufferSize());
        assert(cb.out.size() == sizeof(size_t) + a.GetBufferSize());
    }

    // --- 5. Multiple AddRef: ordering across N external segments ---
    {
        char A[3] = {'a', 'b', 'c'};
        short B[2] = {10, 20};
        Buffer m1;
        m1.Add(head);
        m1.Add(A, 3);
        m1.Add(B, 2);
        m1.Add(tail);
        Buffer m2;
        m2.Add(head);
        m2.AddRef(A, 3);
        m2.AddRef(B, 2);
        m2.Add(tail);
        std::vector<struct iovec> mv;
        m2.GetIov(mv);
        std::string mbody = concat_iov(mv);
        assert(mbody.size() == m1.GetBufferSize());
        assert(memcmp(mbody.data(), m1.GetBuffer(), m1.GetBufferSize()) == 0);

        // --- 6. Reset clears segment state ---
        m2.Reset();
        assert(!m2.HasSegments());
        assert(m2.GetLogicalSize() == 0);
    }

    // --- 7. Null AddRef matches Add(null): writes a 0 size prefix only ---
    {
        Buffer n1;
        n1.Add((double *)nullptr, 4);
        Buffer n2;
        n2.AddRef((double *)nullptr, 4);
        assert(!n2.HasSegments());
        assert(n2.GetLogicalSize() == n1.GetBufferSize());
    }

    printf("test_buffer_iov: ALL PASSED\n");
    return 0;
}
