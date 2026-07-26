/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   Buffer.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 18 13:16:46 2009
 *
 * @brief  Buffer marshals/unmarshals the arguments and return values of
 * every intercepted CUDA call between frontend and backend. Callers pack
 * typed values or pointers in with Add()/AddMarshal()/AddString() and read
 * them back with Get()/Assign()/AssignAll() on the other side.
 *
 * Large bulk payloads don't have to be copied into Buffer's own arena to be
 * sent: AddRef() records a borrowed HOST pointer as a zero-copy segment
 * instead, and Add<T>(ptr, n) itself additionally auto-detects a DEVICE
 * (CUDA) pointer and records that as a zero-copy segment too, transparently
 * — the same call plugins already make for every bulk payload
 * (out->Add<char>(ptr, n)), no separate method to opt in. GetIov() emits
 * the resulting ordered, memory-kind-tagged fragment list for
 * Communicator::WriteIov()/WriteFrame() to actually put on the wire: a
 * GPUDirect-capable transport can peer-DMA a device fragment straight off
 * the GPU, everything else bounces it through host memory first
 * (DeviceMemory.h). Auto-detection only ever activates above a size
 * threshold and only when the process has confirmed GPUDirect is usable
 * (DeviceProbeEnabled(), see DeviceMemory.h) — outside that window a
 * pointer handed to Add() must be ordinary host memory, since the fallback
 * path is a plain memmove.
 */

#pragma once

#include <cxxabi.h>
#include <execinfo.h>
#include <gvirtus/common/gvirtus-type.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include <sys/uio.h>

#include "Communicator.h"
#include "DeviceMemory.h"

#define BLOCK_SIZE 4096

// Threshold below which Add()'s device-pointer check is skipped even when
// GPUDirect is active — mirrors CudaRtHandler_memory.cpp's own D2H
// threshold. Below this size, one cudaPointerGetAttributes call costs more
// than the small host copy it would avoid.
#define GVIRTUS_GPUREF_THRESHOLD (4u * 1024u * 1024u)

static void printStacktrace() {
    void *callstack[128];
    int frames = backtrace(callstack, 128);
    char **symbols = backtrace_symbols(callstack, frames);
    std::cerr << "------ STACK TRACE ------" << std::endl;
    for (int i = 0; i < frames; ++i) {
        std::cerr << symbols[i] << std::endl;
    }
    std::cerr << "-------------------------" << std::endl;
    free(symbols);
}

namespace gvirtus::communicators {
/**
 * Buffer is a general purpose for marshalling and unmarshalling data. It's used
 * for exchanging data beetwen Frontend and Backend. It has the functionality to
 * be created starting from an input stream and to be sent over an output
 * stream.
 */

template <typename T>
std::string demangled_type_name() {
    int status = 0;
    std::unique_ptr<char, void (*)(void *)> demangled(
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status), std::free);
    return (status == 0) ? demangled.get() : typeid(T).name();
}

template <typename T>
constexpr std::size_t safe_sizeof() {
    if constexpr (std::is_void_v<T>) {
        return 1;
    } else if constexpr (std::is_function_v<T>) {
        throw std::runtime_error("safe_sizeof<T> cannot be used with function types");
    } else {
        return sizeof(T);
    }
}

class Buffer {
   public:
    Buffer(size_t initial_size = 0, size_t block_size = BLOCK_SIZE);
    Buffer(const Buffer &orig);
    Buffer(std::istream &in);
    Buffer(char *buffer, size_t buffer_size, size_t block_size = BLOCK_SIZE);
    virtual ~Buffer();

    template <class T>
    void Add(T item) {
        if ((mLength + safe_sizeof<T>()) >= mSize) {
            mSize = ((mLength + safe_sizeof<T>()) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                throw std::runtime_error("Buffer::Add(item): Can't reallocate memory.");
        }
        memmove(mpBuffer + mLength, (char *)&item, safe_sizeof<T>());
        mLength += safe_sizeof<T>();
        mBackOffset = mLength;
    }

    // Auto-detects whether `item` is CUDA device/managed memory and, if so,
    // borrows it as a zero-copy GpuRef segment instead of copying — the
    // SAME call plugins already make for every bulk payload
    // (out->Add<char>(ptr, n)); no separate method is exposed to plugin
    // code. The decision is entirely local to this overload:
    //   1. DeviceProbeEnabled() — one atomic load. False on every frontend
    //      process and every backend where GPUDirect didn't probe OK, which
    //      is the overwhelming common case, so nothing else below runs.
    //   2. size >= GVIRTUS_GPUREF_THRESHOLD — tiny buffers aren't worth a
    //      real driver call even when GPUDirect is active.
    //   3. IsDevicePointer(item) — the one real cudaPointerGetAttributes
    //      call, paid only when (1) and (2) both hold.
    // When all three hold, `item` is recorded as SegKind::GpuRef (mirrors
    // AddRef()'s HostRef bookkeeping exactly) and NOT memmove'd — the bytes
    // stay on the GPU until Communicator::GetIov()'s consumer (UCX RMA or
    // the GPU-safe WriteIov fallback) decides how to move them.
    //
    // Lifetime contract (same one AddRef() already imposes on borrowed host
    // pointers): `item` must remain valid until the message has actually
    // been transmitted, not merely until this function returns. A handler
    // that stages into a temporary/scratch device buffer must give it a
    // lifetime that outlives the call (e.g. thread_local), never a
    // function-scope allocation freed at return.
    template <class T>
    void Add(T *item, size_t n = 1) {
        if (item == NULL) {
            Add((size_t)0);
            return;
        }
        size_t size = safe_sizeof<T>() * n;
        Add(size);  // length prefix -> inline arena (mpBuffer), same as always

        // Function-pointer T (e.g. marshaling a callback like cuOccupancy's
        // blockSizeToDynamicSMemSize or cudnnSetCallback's debug hook) can
        // never be CUDA device memory, and a function pointer can't be
        // static_cast to const void* at all — skip the probe entirely at
        // compile time for non-object T, falling through to the plain copy
        // below exactly as before this auto-detect existed.
        if constexpr (std::is_object_v<T>) {
            if (DeviceProbeEnabled() && size >= GVIRTUS_GPUREF_THRESHOLD &&
                IsDevicePointer(static_cast<const void *>(item))) {
                mSegments.push_back(Segment{SegKind::Inline, mInlineConsumed, nullptr,
                                            mLength - mInlineConsumed});
                mInlineConsumed = mLength;
                mSegments.push_back(
                    Segment{SegKind::GpuRef, 0, static_cast<const void *>(item), size});
                mExternalBytes += size;
                return;  // NO memmove — bytes stay on the GPU.
            }
        }

        // Unchanged legacy path: host memory (or GPUDirect inactive/below
        // threshold) — copy in exactly as before.
        if ((mLength + size) >= mSize) {
            mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                throw std::runtime_error("Buffer::Add(item, n): Can't reallocate memory.");
        }
        memmove(mpBuffer + mLength, (char *)item, size);
        mLength += size;
        mBackOffset = mLength;
    }

    // Bulk-append raw bytes with NO size prefix (unlike Add<T*>(item, n)
    // which prepends an 8-byte size header). Used by Frontend.cpp to flush
    // a large response payload (e.g. 64MB cudaMemcpy D2H) into the output
    // buffer in a single memcpy. Replacing this with the per-byte Add<char>
    // loop costs ~67M function calls + repeated reallocs = ~1.3s for 64MB;
    // this is ~3ms instead.
    void AppendBytes(const char *src, size_t n) {
        if (n == 0 || src == NULL) return;
        if ((mLength + n) >= mSize) {
            mSize = ((mLength + n) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                throw std::runtime_error("Buffer::AppendBytes: Can't reallocate memory.");
        }
        std::memcpy(mpBuffer + mLength, src, n);
        mLength += n;
        mBackOffset = mLength;
    }

    // ===== IoV (scatter / gather) extensions =====
    // Append a BORROWED host pointer as a zero-copy segment. Writes the same
    // [size_t length][bytes] wire layout that Add<T>(ptr, n) does, but the
    // data bytes are NOT copied into the internal arena — they are recorded
    // as an external segment and emitted, in order, by GetIov(). The caller
    // MUST keep `ptr` valid until the buffer has been sent (WriteIov + Sync).
    // Transports without scatter concatenate lazily in Dump(), producing
    // byte-identical wire output at the cost of one copy.
    template <class T>
    void AddRef(const T *ptr, size_t n = 1) {
        if (ptr == NULL) {
            Add((size_t)0);
            return;
        }
        size_t bytes = safe_sizeof<T>() * n;
        Add(bytes);  // length prefix -> inline arena (mpBuffer)
        mSegments.push_back(Segment{SegKind::Inline, mInlineConsumed, nullptr,
                                    mLength - mInlineConsumed});
        mInlineConsumed = mLength;
        mSegments.push_back(
            Segment{SegKind::HostRef, 0, static_cast<const void *>(ptr), bytes});
        mExternalBytes += bytes;
    }

    // True iff any borrowed external segment (host OR device) has been
    // recorded.
    bool HasSegments() const { return !mSegments.empty(); }

    // True iff any borrowed segment is device-resident (SegKind::GpuRef) —
    // i.e. Add()'s auto-detect actually borrowed a GPU pointer rather than
    // copying. Used by Result/Process/RpcCodec to decide whether the
    // GPU-aware send path is needed at all; not called by plugins.
    bool HasGpuSegments() const {
        for (const auto &s : mSegments)
            if (s.kind == SegKind::GpuRef) return true;
        return false;
    }

    // Total bytes that will go on the wire: inline arena + external segments.
    // Equals GetBufferSize() when no AddRef segments are present.
    size_t GetLogicalSize() const { return mLength + mExternalBytes; }

    // Emit the ordered, memory-kind-tagged fragment list for
    // Communicator::WriteIov()/WriteFrame(). With no external segments this
    // is a single host fragment spanning the arena. Called only by
    // communicator-layer code (Buffer::Dump, RpcCodec, UcxRma) — never by
    // plugins.
    void GetIov(std::vector<IovFrag> &out) const;

    template <class T>
    void AddConst(const T item) {
        if ((mLength + safe_sizeof<T>()) >= mSize) {
            mSize = ((mLength + safe_sizeof<T>()) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                throw std::runtime_error("Buffer::AddConst(item): Can't reallocate memory.");
        }
        memmove(mpBuffer + mLength, (char *)&item, safe_sizeof<T>());
        mLength += safe_sizeof<T>();
        mBackOffset = mLength;
    }

    template <class T>
    void AddConst(const T *item, size_t n = 1) {
        if (item == NULL) {
            Add((size_t)0);
            return;
        }
        size_t size = safe_sizeof<T>() * n;
        Add(size);
        if ((mLength + size) >= mSize) {
            mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                throw std::runtime_error("Buffer::AddConst(item, n): Can't reallocate memory.");
        }
        memmove(mpBuffer + mLength, (char *)item, size);
        mLength += size;
        mBackOffset = mLength;
    }

    void AddString(const char *s) {
        size_t size = strlen(s) + 1;
        Add(size);
        Add(s, size);
    }

    template <class T>
    void AddMarshal(T item) {
        Add((gvirtus::common::pointer_t)item);
    }

    template <class T>
    void Read(Communicator *c) {
        auto required_size = mLength + safe_sizeof<T>();
        if (required_size >= mSize) {
            mSize = (required_size / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                throw std::runtime_error("Buffer::Read(*c) Can't reallocate memory.");
        }
        c->Read(mpBuffer + mLength, safe_sizeof<T>());
        mLength += safe_sizeof<T>();
        mBackOffset = mLength;
    }

    template <class T>
    void Read(Communicator *c, size_t n = 1) {
        auto required_size = mLength + safe_sizeof<T>() * n;
        if (required_size >= mSize) {
            mSize = (required_size / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                throw std::runtime_error("Buffer::Read(*c, n): Can't reallocate memory.");
        }
        c->Read(mpBuffer + mLength, safe_sizeof<T>() * n);
        mLength += safe_sizeof<T>() * n;
        mBackOffset = mLength;
    }

    template <class T>
    T Get() {
        if (mOffset + safe_sizeof<T>() > mLength) {
            printStacktrace();
            throw std::runtime_error(std::string("Buffer::Get(): Can't read any ") +
                                     demangled_type_name<T>());
        }
        T result = *((T *)(mpBuffer + mOffset));
        mOffset += safe_sizeof<T>();
        return result;
    }

    template <class T>
    T BackGet() {
        if (mBackOffset - safe_sizeof<T>() > mLength)
            throw std::runtime_error(std::string("Buffer::BackGet(): Can't read ") +
                                     demangled_type_name<T>());
        T result = *((T *)(mpBuffer + mBackOffset - safe_sizeof<T>()));
        mBackOffset -= safe_sizeof<T>();
        return result;
    }

    template <class T>
    T *Get(size_t n) {
        if (Get<size_t>() == 0) return NULL;
        if (mOffset + safe_sizeof<T>() * n > mLength)
            throw std::runtime_error(std::string("Buffer::Get(n): Can't read  ") +
                                     demangled_type_name<T>());
        T *result = new T[n];
        memmove((char *)result, mpBuffer + mOffset, safe_sizeof<T>() * n);
        mOffset += safe_sizeof<T>() * n;
        return result;
    }

    template <class T>
    T *Delegate(size_t n = 1) {
        size_t size = safe_sizeof<T>() * n;
        Add(size);
        if ((mLength + size) >= mSize) {
            mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
                throw std::runtime_error("Buffer::Delegate(n): Can't reallocate memory.");
        }
        T *dst = (T *)(mpBuffer + mLength);
        mLength += size;
        mBackOffset = mLength;
        return dst;
    }

    template <class T>
    T *Assign(size_t n = 1) {
        if (Get<size_t>() == 0) return NULL;
        if (mOffset + safe_sizeof<T>() * n > mLength) {
            throw std::runtime_error(std::string("Buffer::Assign(n): Can't read  ") +
                                     demangled_type_name<T>());
        }
        T *result = (T *)(mpBuffer + mOffset);
        mOffset += safe_sizeof<T>() * n;
        return result;
    }

    template <class T>
    T *AssignAll() {
        size_t size = Get<size_t>();
        if (size == 0) return NULL;
        size_t n = size / safe_sizeof<T>();
        if (mOffset + safe_sizeof<T>() * n > mLength)
            throw std::runtime_error(std::string("Buffer::AssignAll(): Can't read ") +
                                     demangled_type_name<T>());
        T *result = (T *)(mpBuffer + mOffset);
        mOffset += safe_sizeof<T>() * n;
        return result;
    }

    char *AssignString() {
        size_t size = Get<size_t>();
        return Assign<char>(size);
    }

    template <class T>
    T *BackAssign(size_t n = 1) {
        if (mBackOffset - safe_sizeof<T>() * n > mLength)
            throw std::runtime_error(std::string("Buffer::BackAssign(n): Can't read ") +
                                     demangled_type_name<T>());
        T *result = (T *)(mpBuffer + mBackOffset - safe_sizeof<T>() * n);
        mBackOffset -= safe_sizeof<T>() * n + sizeof(size_t);
        return result;
    }

    template <class T>
    T GetFromMarshal() {
        return (T)Get<gvirtus::common::pointer_t>();
    }

    inline bool Empty() { return mOffset == mLength; }

    void Reset();
    void Reset(Communicator *c);
    const char *const GetBuffer() const;
    size_t GetBufferSize() const;
    void Dump(Communicator *c) const;

    // Optional GPU-backed payload on the READ/input side. When set, the
    // trailing portion of the LOGICAL message lives on the GPU at
    // `gpu_addr` rather than in mpBuffer. GPU-aware handlers (e.g. cudaMemcpy
    // HostToDevice in CudaRtHandler_memory) detect this and route the
    // payload via cudaMemcpyDeviceToDevice instead of HostToDevice — saving
    // the backend a D2H-consolidate-then-H2D copy pair. Set by Process.cpp
    // when constructing the INPUT Buffer from a frame whose PooledMsg
    // carries gpu_data != null.
    //
    // This is deliberately separate from the WRITE-side SegKind::GpuRef
    // segment model below. That model represents "a pointer THIS Buffer
    // instance is about to send may be device memory, borrowed via Add()."
    // This pair represents "the Buffer instance the wire just handed ME
    // already has a GPU-resident tail, attached by the transport before any
    // handler touched it." Folding the read side into the same segment
    // model too is worthwhile future work, but touches every
    // Assign/AssignAll call site across every plugin — out of scope here
    // since it can't be validated without a real UCX+GPU stack.
    void SetGpuPayload(void *gpu_addr, std::size_t size);
    void *GetGpuPayload() const;
    std::size_t GetGpuPayloadSize() const;

   private:
    // ---- IoV segment model (write side only) ----
    // GpuRef realises the extensibility this enum was designed for: a
    // borrowed DEVICE pointer, recorded by Add()'s auto-detect (see above)
    // exactly the way AddRef() records a borrowed HOST pointer as HostRef.
    enum class SegKind { Inline, HostRef, GpuRef };
    struct Segment {
        SegKind kind;
        size_t offset;    // Inline: byte offset into mpBuffer
        const void *ptr;  // HostRef/GpuRef: borrowed source pointer
        size_t len;
    };

    size_t mBlockSize;
    size_t mSize;
    size_t mLength;
    size_t mOffset;
    size_t mBackOffset;
    char *mpBuffer;
    bool mOwnBuffer;
    void *mGpuPayload = nullptr;
    std::size_t mGpuPayloadSize = 0;

    // Ordered fragments. Empty => purely contiguous (fast path, original
    // behaviour). Populated only by AddRef().
    std::vector<Segment> mSegments;
    size_t mInlineConsumed = 0;  // arena bytes already captured into an Inline seg
    size_t mExternalBytes = 0;   // total borrowed bytes recorded
};
}  // namespace gvirtus::communicators