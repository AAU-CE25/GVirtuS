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
 * @file   Result.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 18 13:23:56 2009
 *
 * @brief
 *
 *
 */

#pragma once

#include "Buffer.h"

namespace gvirtus::communicators {
/**
 * Result is used to store the results of a CUDA Runtime routine.
 */
class Result {
   public:
    Result(int exit_code);
    Result(int exit_code, const std::shared_ptr<Buffer> output_buffer);

    virtual ~Result() = default;
    int GetExitCode();
    std::shared_ptr<Buffer> GetOutputBuffer() const;

    void Dump(Communicator *c);

    void TimeTaken(double time_taken);
    double TimeTaken() const;

    // Optional zero-copy GPU payload (GVIRTUS_GPUDIRECT path). When set, the
    // OutputBuffer contains only the protocol prefix (size_t count). The
    // actual `count` bytes live on the GPU at `gpu_addr`. The UCX-AM
    // response writer (Process.cpp::write_ucx_am_response) detects this and
    // routes the GPU buffer as a separate iov entry registered with
    // UCS_MEMORY_TYPE_CUDA in WriteIovRma. Frontend receives the data
    // contiguously into its host RX slot via peer-DMA.
    void SetGpuPayload(void *gpu_addr, std::size_t size);
    void *GetGpuPayload() const;
    std::size_t GetGpuPayloadSize() const;

   private:
    int mExitCode;
    std::shared_ptr<Buffer> mpOutputBuffer;
    double mTimeTaken = 0;
    void *mGpuPayload = nullptr;
    std::size_t mGpuPayloadSize = 0;
};
}  // namespace gvirtus::communicators
