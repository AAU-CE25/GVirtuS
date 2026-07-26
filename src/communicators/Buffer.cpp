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
 * @file   Buffer.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 18 13:16:46 2009
 *
 * @brief
 *
 *
 */
// #define DEBUG
#include "gvirtus/communicators/Buffer.h"

using namespace std;
using gvirtus::communicators::Buffer;

Buffer::Buffer(size_t initial_size, size_t block_size) {
    mSize = initial_size;
    mBlockSize = block_size;
    mLength = 0;
    mOffset = 0;
    mpBuffer = NULL;
    mOwnBuffer = true;
    if (mSize == 0) mSize = 0;
    if ((mSize = (mSize / mBlockSize) * mBlockSize) == 0) mSize = mBlockSize;
    if ((mpBuffer = (char *)malloc(mSize)) == NULL) throw runtime_error("Can't allocate memory.");
    mBackOffset = mLength;
}

Buffer::Buffer(const Buffer &orig) {
    mBlockSize = orig.mBlockSize;
    mLength = orig.mLength;
    mSize = orig.mLength;
    mOffset = orig.mOffset;
    mLength = orig.mLength;
    mOwnBuffer = true;
    if ((mpBuffer = (char *)malloc(mSize)) == NULL) throw runtime_error("Can't allocate memory.");
    memmove(mpBuffer, orig.mpBuffer, mLength);
    mBackOffset = mLength;
    mSegments = orig.mSegments;
    mInlineConsumed = orig.mInlineConsumed;
    mExternalBytes = orig.mExternalBytes;
}

Buffer::Buffer(istream &in) {
    in.read((char *)&mSize, sizeof(size_t));
    mBlockSize = BLOCK_SIZE;
    mLength = mSize;
    mOffset = 0;
    mOwnBuffer = true;
    if ((mpBuffer = (char *)malloc(mSize)) == NULL) throw runtime_error("Can't allocate memory.");
    in.read(mpBuffer, mSize);
    mBackOffset = mLength;
}

Buffer::Buffer(char *buffer, size_t buffer_size, size_t block_size) {
    mSize = buffer_size;
    mBlockSize = block_size;
    mLength = mSize;
    mOffset = 0;
    mpBuffer = buffer;
    mOwnBuffer = false;
    mBackOffset = mLength;
}

Buffer::~Buffer() {
    if (mOwnBuffer) free(mpBuffer);
}

void Buffer::Reset() {
    mLength = 0;
    mOffset = 0;
    mBackOffset = 0;
    mSegments.clear();
    mInlineConsumed = 0;
    mExternalBytes = 0;
}

void Buffer::Reset(Communicator *c) {
    c->Read((char *)&mLength, sizeof(size_t));
#ifdef DEBUG
    cout << "Read " << mLength << " bytes from the buffer" << endl;
#endif
    mOffset = 0;
    mBackOffset = mLength;
    // A buffer reconstructed from the wire is always contiguous.
    mSegments.clear();
    mInlineConsumed = 0;
    mExternalBytes = 0;
    if (mLength >= mSize) {
        mSize = (mLength / mBlockSize + 1) * mBlockSize;
        if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
            throw runtime_error("Can't reallocate memory.");
    }

    c->Read(mpBuffer, mLength);
}

const char *const Buffer::GetBuffer() const { return mpBuffer; }

size_t Buffer::GetBufferSize() const { return mLength; }

void Buffer::GetIov(std::vector<IovFrag> &out) const {
    out.clear();
    if (mSegments.empty()) {
        // Fast path: one contiguous, host fragment (original behaviour).
        if (mLength > 0)
            out.push_back(IovFrag{static_cast<void *>(mpBuffer), mLength, false});
        return;
    }
    out.reserve(mSegments.size() + 1);
    for (const auto &s : mSegments) {
        if (s.kind == SegKind::Inline) {
            if (s.len > 0)
                out.push_back(
                    IovFrag{static_cast<void *>(mpBuffer + s.offset), s.len, false});
        } else if (s.kind == SegKind::HostRef) {
            // Borrowed HOST bytes, never copied.
            out.push_back(IovFrag{const_cast<void *>(s.ptr), s.len, false});
        } else {  // SegKind::GpuRef: borrowed DEVICE bytes, never copied.
            out.push_back(IovFrag{const_cast<void *>(s.ptr), s.len, true});
        }
    }
    // Trailing inline run written after the last AddRef/Add(GpuRef).
    if (mInlineConsumed < mLength)
        out.push_back(IovFrag{static_cast<void *>(mpBuffer + mInlineConsumed),
                              mLength - mInlineConsumed, false});
}

void Buffer::SetGpuPayload(void *gpu_addr, std::size_t size) {
    mGpuPayload = gpu_addr;
    mGpuPayloadSize = size;
}

void *Buffer::GetGpuPayload() const { return mGpuPayload; }

std::size_t Buffer::GetGpuPayloadSize() const { return mGpuPayloadSize; }

void Buffer::Dump(Communicator *c) const {
    // Frame the message with its LOGICAL size (inline arena + any borrowed
    // external segments), then write the body. With no AddRef segments this
    // is the original single contiguous Write(). With segments we hand the
    // ordered fragments to WriteIov(): scatter-capable transports (UCX) send
    // them in place, others fall back to one concatenation — identical wire
    // bytes either way.
    size_t total = GetLogicalSize();
    c->Write((char *)&total, sizeof(size_t));
    if (mSegments.empty()) {
        c->Write(mpBuffer, mLength);
    } else {
        std::vector<IovFrag> iov;
        GetIov(iov);
        c->WriteIov(iov.data(), iov.size());
    }
    c->Sync();
}