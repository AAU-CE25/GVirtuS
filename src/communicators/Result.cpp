#include "gvirtus/communicators/Result.h"

using gvirtus::communicators::Result;

Result::Result(int exit_code) {
    mExitCode = exit_code;
    mpOutputBuffer = NULL;
}

Result::Result(int exit_code, const std::shared_ptr<gvirtus::communicators::Buffer> output_buffer) {
    mExitCode = exit_code;
    mpOutputBuffer = (output_buffer);
}

int Result::GetExitCode() { return mExitCode; }

std::shared_ptr<gvirtus::communicators::Buffer> Result::GetOutputBuffer() const {
    return mpOutputBuffer;
}

void Result::Dump(Communicator *c) {
    c->Write((char *)&mExitCode, sizeof(int));
    c->Write(reinterpret_cast<const char *>(&mTimeTaken), sizeof(mTimeTaken));
    if (mpOutputBuffer != NULL)
        mpOutputBuffer->Dump(c);
    else {
        size_t size = 0;
        c->Write((char *)&size, sizeof(size_t));
        c->Sync();
    }
}

void Result::TimeTaken(double time_taken) { mTimeTaken = time_taken; }

double Result::TimeTaken() const { return mTimeTaken; }

void Result::SetGpuPayload(void *gpu_addr, std::size_t size) {
    mGpuPayload = gpu_addr;
    mGpuPayloadSize = size;
}

void *Result::GetGpuPayload() const { return mGpuPayload; }

std::size_t Result::GetGpuPayloadSize() const { return mGpuPayloadSize; }


// ---------------------------------------------------------------------------
// Frame drain hook. Defined here so it lives in libgvirtus-communicators, which
// the cudart backend plugin already resolves symbols from. See Communicator.h.
#include <atomic>

namespace gvirtus {
namespace communicators {

static std::atomic<FrameDrainFn> g_frame_drain_hook{nullptr};

void SetFrameDrainHook(FrameDrainFn fn) { g_frame_drain_hook.store(fn, std::memory_order_release); }

void RunFrameDrainHook() {
    FrameDrainFn fn = g_frame_drain_hook.load(std::memory_order_acquire);
    if (fn != nullptr) fn();
}

// Registration-lifetime hooks. See Communicator.h for why these exist.
static std::atomic<RegistrationInvalidateFn> g_reg_invalidate_hook{nullptr};
static std::atomic<RegistrationCacheableFn> g_reg_cacheable_hook{nullptr};

void SetRegistrationInvalidateHook(RegistrationInvalidateFn fn) {
    g_reg_invalidate_hook.store(fn, std::memory_order_release);
}

void RunRegistrationInvalidate(const void *addr) {
    if (addr == nullptr) return;
    RegistrationInvalidateFn fn = g_reg_invalidate_hook.load(std::memory_order_acquire);
    if (fn != nullptr) fn(addr);
}

void SetRegistrationCacheableHook(RegistrationCacheableFn fn) {
    g_reg_cacheable_hook.store(fn, std::memory_order_release);
}

// Default when no frontend installed a hook: NOT cacheable. Failing closed keeps a
// transport that cannot be told about frees from caching anything by accident.
bool RegistrationCacheable(const void *addr, size_t len) {
    RegistrationCacheableFn fn = g_reg_cacheable_hook.load(std::memory_order_acquire);
    return (fn != nullptr) && fn(addr, len);
}

static std::atomic<ConnectionCleanupFn> g_conn_cleanup_hook{nullptr};

void SetConnectionCleanupHook(ConnectionCleanupFn fn) {
    g_conn_cleanup_hook.store(fn, std::memory_order_release);
}

void RunConnectionCleanup() {
    ConnectionCleanupFn fn = g_conn_cleanup_hook.load(std::memory_order_acquire);
    if (fn != nullptr) fn();
}

}  // namespace communicators
}  // namespace gvirtus
