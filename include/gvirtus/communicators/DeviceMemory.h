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
 */

/**
 * @file DeviceMemory.h
 *
 * Transport-agnostic CUDA device-memory primitives, shared by Buffer (the
 * marshaller), the base Communicator (the default, non-UCX write path), and
 * UcxGpu.cpp (which forwards to these for backward compatibility).
 *
 * These were originally UCX-specific helpers in
 * src/communicators/ucx/UcxGpu.cpp (ucx_internal::is_gpu_pointer and the
 * "Device resolver" dlsym block). They are promoted here, unchanged in
 * behaviour, because Buffer::Add() and Communicator::WriteIov() need to
 * classify/bounce device pointers WITHOUT Buffer or Communicator including
 * any UCX header — Buffer is the general-purpose marshaller used by every
 * plugin and every transport, not just UCX.
 *
 * Design invariants preserved from UcxGpu.cpp:
 *   - No static link to libcudart. Every CUDA entry point is resolved via
 *     dlopen/dlsym on first use (std::call_once), so host-only builds and
 *     dev containers without a GPU driver still link and run.
 *   - IsDevicePointer() short-circuits on the atomic "device probe enabled"
 *     flag BEFORE calling cudaPointerGetAttributes, so on every frontend
 *     process and every backend where GPUDirect didn't probe OK, classifying
 *     a pointer costs one atomic load and nothing else — no driver call, no
 *     RPC (the frontend's libcudart is GVirtuS's own shim, which would
 *     otherwise remote cudaPointerGetAttributes as an RPC; the short-circuit
 *     is what avoids that storm).
 */
#pragma once

#include <cstddef>

namespace gvirtus::communicators {

// Process-wide gate: true iff GVIRTUS_GPUDIRECT=1 was set AND UCX's startup
// probe (cudaMalloc + ucp_mem_map(CUDA)) succeeded. Single source of truth
// shared by ucx_internal::{set_,}gpudirect_enabled (thin forwarders) and by
// Buffer::Add()'s auto-detect gate.
void SetDeviceProbeEnabled(bool on);
bool DeviceProbeEnabled();

// Classify a pointer as CUDA device/managed memory. Returns false on NULL,
// on host/unregistered memory, when GPUDirect is not active
// (DeviceProbeEnabled() == false — checked FIRST, before any driver call),
// or if cudaPointerGetAttributes is unavailable. This is the exact logic of
// the former ucx_internal::is_gpu_pointer.
bool IsDevicePointer(const void *p);

// dlsym'd cudaMemcpy(..., cudaMemcpyDeviceToHost). Returns false if libcudart
// is unavailable or the copy failed. This is net new (no equivalent existed
// in UcxGpu.cpp) — it is the one place a non-RDMA-capable transport safely
// bounces a GpuRef Buffer segment through host memory instead of a raw
// memcpy (which would corrupt/crash on a device pointer).
bool DeviceMemcpyD2H(void *dst_host, const void *src_gpu, std::size_t n);

// Thin wrappers around the same dlsym'd cudaMalloc/cudaFree used internally
// by IsDevicePointer's resolver, exposed so UcxGpu.cpp's alloc_gpu_slot /
// free_gpu_slot / probe_gpudirect can reuse one resolver instead of keeping
// a second copy of the dlopen logic.
bool AllocDeviceMemory(void **p, std::size_t n);
void FreeDeviceMemory(void *p);

}  // namespace gvirtus::communicators
