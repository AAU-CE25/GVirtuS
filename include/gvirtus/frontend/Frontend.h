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
 * @file   Frontend.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 12:57:11 2009
 *
 * @brief
 *
 *
 */

#pragma once

#include <gvirtus/common/LD_Lib.h>
#include <gvirtus/communicators/Buffer.h>
#include <gvirtus/communicators/Communicator.h>

#include <map>
#include <cstdint>

namespace gvirtus::frontend {
/**
 * Frontend is the object used by every cuda routine wrapper for requesting the
 * execution to the backend.
 *
 * Frontend is a singleton, the single instance can be retrived through the
 * static member getInstance().
 *
 * For requesting the execution of a cuda routine to the backend the wrapper has
 * to:
 * -# retrieve the Frontend instance.
 * -# prepare the execution using the Prepare() method.
 * -# add the input parameters in the correct order with the Add...() methods.
 * -# requests the execution of a named routine with Execute() method.
 * -# check if the execution has been executed successfully with Success().
 * -# retrieve the output parameters with the Get...() methods.
 *
 * Note that every pointer is assumed to be an output parameter. Every output
 * parameter must be retrieved otherwise the Frontend will be left in a dirty
 * status.
 */
class Frontend {
   public:
    virtual ~Frontend();

    /**
     * Retrieves the single instance of the Frontend class.
     *
     * @param register_var
     *
     * @return The instance of the Frontend class.
     */
    static Frontend *GetFrontend(communicators::Communicator *c = NULL);

    /**
     * Requests the execution of the CUDA RunTime routine with the arguments
     * marshalled in the input_buffer.
     * input_buffer is an optional parameter: if it isn't provided any then
     * frontend will use the internal one.
     *
     * @param routine the name of the routine to execute.
     * @param input_buffer the buffer containing the parameters of the routine.
     */
    void Execute(const char *routine, const communicators::Buffer *input_buffer = NULL);

    /**
     * Asynchronous / fire-and-forget dispatch (GVIRTUS_ASYNC_DISPATCH). Sends
     * the request without waiting for a response; only honoured on the UCX
     * Active-Message transport (falls back to a synchronous Execute otherwise).
     * Intended solely for stream-ordered, output-less CUDA calls (e.g.
     * cudaLaunchKernel); any error is reconciled by the backend onto the next
     * synchronous call, matching CUDA's deferred async-error semantics.
     *
     * @param routine the name of the routine to execute.
     * @param input_buffer the buffer containing the parameters of the routine.
     */
    void ExecuteAsync(const char *routine, const communicators::Buffer *input_buffer = NULL);

    /**
     * Deferred device-to-host copy (Phase 3 async dispatcher). Sends the D2H
     * request in stream order but does NOT block for the reply; the destination
     * `dst` (count bytes) is filled from the reply at the next synchronization
     * point (drained at the start of the next synchronous Execute). Only the
     * caller's pinned host buffers may be deferred (pageable dst must stay
     * synchronous). On non-UCX transports this degrades to a synchronous copy
     * that fills `dst` immediately. The frontend owns writing into `dst`.
     */
    void ExecuteDeferredD2H(const char *routine, void *dst, size_t count,
                            const communicators::Buffer *input_buffer = NULL);

    /**
     * Prepares the Frontend for the execution. This method _must_ be called
     * before any requests of execution or any method for adding parameters for
     * the next execution.
     */
    void Prepare();

    inline communicators::Buffer *GetInputBuffer() { return mpInputBuffer.get(); }

    inline communicators::Buffer *GetOutputBuffer() { return mpOutputBuffer.get(); }

    // Fase 4 - zero-copy D2H: caller pre-registers a destination so the
    // response handler in Execute() writes the big output payload directly
    // there, skipping the AppendBytes(64MB) -> mpOutputBuffer staging and
    // the subsequent memmove on the user side. Honoured only when the
    // response Buffer is exactly [size_t prefix == count][count bytes];
    // otherwise Execute() falls back to the standard AppendBytes path and
    // DirectOutputConsumed() stays false so the caller can still memmove.
    inline void SetOutputDestination(void *dst, size_t count) {
        mDirectOutputDst      = dst;
        mDirectOutputCount    = count;
        mDirectOutputConsumed = false;
    }
    inline void ClearOutputDestination() {
        mDirectOutputDst      = nullptr;
        mDirectOutputCount    = 0;
        mDirectOutputConsumed = false;
    }
    inline bool DirectOutputConsumed() const { return mDirectOutputConsumed; }

    // Fase 5 - zero-copy variant of AddHostPointerForArguments. Writes ONLY
    // the size_t length prefix into mpInputBuffer; records ptr as a direct
    // chunk that Execute() will splice into the WriteIov iov at the current
    // buffer offset. Caller must NOT mutate ptr until Execute() returns.
    // Pattern (matches AddHostPointerForArguments wire format):
    //     [size_t = n*sizeof(T)] [n*sizeof(T) bytes of ptr]
    //
    // Only the UCX path in Execute() honours the iov-split. Plain-TCP /
    // HybridCommunicator go through input_buffer->Dump() which serializes
    // mpInputBuffer as-is and would send the truncated buffer (header-only,
    // missing the user payload) — backend AssignAll<char> then throws
    // "Can't read char" and cudaMemcpy returns cudaErrorMemoryAllocation(2).
    // Fall back to the legacy in-buffer marshal in that case so non-UCX
    // transports keep working; UCX retains the zero-copy fast path.
    template <class T>
    void AddHostPointerForArgumentsDirect(const T *ptr, size_t n = 1) {
        const bool ucx =
            _communicator && _communicator->obj_ptr() &&
            _communicator->obj_ptr()->to_string() == "ucxcommunicator";
        if (!ucx) {
            mpInputBuffer->Add<T>(const_cast<T *>(ptr), n);
            return;
        }
        if (ptr == nullptr) {
            mpInputBuffer->Add((size_t)0);
            return;
        }
        const size_t bytes = sizeof(T) * n;
        mpInputBuffer->Add(bytes);  // size_t prefix only
        mDirectInputBufferOffset = mpInputBuffer->GetBufferSize();
        mDirectInputSrc          = ptr;
        mDirectInputBytes        = bytes;
    }
    inline bool HasDirectInput()  const { return mDirectInputSrc != nullptr; }
    inline void ClearDirectInput()      {
        mDirectInputSrc          = nullptr;
        mDirectInputBytes        = 0;
        mDirectInputBufferOffset = 0;
    }

    inline communicators::Buffer *GetLaunchBuffer() { return mpLaunchBuffer.get(); }

    /**
     * Returns the exit code of the last execution request.
     *
     * @return the exit code of the last execution request.
     */
    int GetExitCode() { return mExitCode; }

    inline bool initialized() { return mpInitialized; }  // should be commented

    /**
     * Checks if the latest execution had been completed successfully.
     *
     * @return True if the last execution had been completed successfully.
     */
    bool Success(int success_value = 0) { return mExitCode == success_value; }

#if 0
  /**
   * Adds a scalar variabile as an input parameter for the next execution
   * request.
   *
   * @param var the variable to add as a parameter.
   */
  template <class T>void AddVariableForArguments(T var) {
      mpInputBuffer->Add(var);
  }

  /**
   * Adds a string (array of char(s)) as an input parameter for the next
   * execution request.
   *
   * @param s the string to add as a parameter.
   */
  void AddStringForArguments(const char *s) {
      mpInputBuffer->AddString(s);
  }

  /**
   * Adds, marshalling it, an host pointer as an input parameter for the next
   * execution request.
   * The optional parameter n is usefull when adding an array: with n is
   * possible to specify the length of the array in terms of elements.
   *
   * @param ptr the pointer to add as a parameter.
   * @param n the length of the array, if ptr is an array.
   */
  template <class T>void AddHostPointerForArguments(T *ptr, size_t n = 1) {
      mpInputBuffer->Add(ptr, n);
  }

  /**
   * Adds a device pointer as an input parameter for the next execution
   * request.
   *
   * @param ptr the pointer to add as a parameter.
   */
  void AddDevicePointerForArguments(const void *ptr) {
      mpInputBuffer->Add((uint64_t) ptr);
  }

  /**
   * Adds a symbol, a named variable, as an input parameter for the next
   * execution request.
   *
   * @param symbol the symbol to add as a parameter.
   */
  void AddSymbolForArguments(const char *symbol) {
      AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
      AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
  }

  template <class T>T GetOutputVariable() {
      return mpOutputBuffer->Get<T>();
  }

  /**
   * Retrives an host pointer from the output parameters of the last execution
   * request.
   * The optional parameter n is usefull when retriving an array: with n is
   * possible to specify the length of the array in terms of elements.
   *
   * @param n the length of the array.
   *
   * @return the pointer from the output parameters.
   */
  template <class T> T * GetOutputHostPointer(size_t n = 1) {
      return mpOutputBuffer->Assign<T>(n);
  }

  /**
   * Retrives a device pointer from the output parameters of the last
   * execution request.
   *
   * @return the pointer to the device memory.
   */
  void * GetOutputDevicePointer() {
      return (void *) mpOutputBuffer->Get<uint64_t>();
  }

  /**
   * Retrives a string, array of chars, from the output parameters of the last
   * execution request.
   *
   * @return the string from the output parameters.
   */
  char * GetOutputString() {
      return mpOutputBuffer->AssignString();
  }

  inline Buffer * GetLaunchBuffer() {
      return mpLaunchBuffer;
  }
#endif

   private:
    /**
     * Dispatch mode for ExecuteInternal.
     *   Sync         — send and block for the reply (default).
     *   FireAndForget— send, no reply expected (UCX AM only).
     *   DeferredD2H  — send, reply collected later (drained at next sync).
     */
    enum class DispatchMode { Sync, FireAndForget, DeferredD2H };

    /**
     * Shared implementation of Execute()/ExecuteAsync()/ExecuteDeferredD2H().
     * For DeferredD2H, d2h_dst/d2h_count record where the reply payload must be
     * written when the pending copy is drained.
     */
    void ExecuteInternal(const char *routine, const communicators::Buffer *input_buffer,
                         DispatchMode mode, void *d2h_dst = nullptr, size_t d2h_count = 0);

    /**
     * Read and complete all outstanding DeferredD2H replies (in-order FIFO
     * guarantees they precede any later synchronous reply). Called at the start
     * of a synchronous receive so a sync call never mis-reads a deferred reply.
     */
    void DrainPendingD2H();

    /**
     * Constructs a new Frontend. It creates and sets also the Communicator to
     * use obtaining the information from the configuration file which path is
     * setted at compile time.
     */
    void Init(communicators::Communicator *c);
    std::shared_ptr<
        common::LD_Lib<communicators::Communicator, std::shared_ptr<communicators::Endpoint>>>
        _communicator;
    std::shared_ptr<communicators::Buffer> mpInputBuffer;
    std::shared_ptr<communicators::Buffer> mpOutputBuffer;

    // See SetOutputDestination(). Per-frontend (i.e. per-thread) state.
    void  *mDirectOutputDst       = nullptr;
    size_t mDirectOutputCount     = 0;
    bool   mDirectOutputConsumed  = false;

    // Fase 5 - zero-copy H2D send: when set, Execute()'s iov construction
    // splits the input_buffer iov entry into three:
    //   [input_buffer[0..mDirectInputBufferOffset]]
    //   [mDirectInputSrc .. mDirectInputBytes]
    //   [input_buffer[mDirectInputBufferOffset..end]]
    // so the 64MB user payload travels straight from caller memory into
    // sendmsg/RMA without the Add<T>(ptr,n) memcpy into mpInputBuffer.
    // Per-thread, cleared automatically at the end of Execute().
    const void *mDirectInputSrc          = nullptr;
    size_t      mDirectInputBytes        = 0;
    size_t      mDirectInputBufferOffset = 0;
    std::shared_ptr<communicators::Buffer> mpLaunchBuffer;

    // Phase 3 async dispatcher: outstanding deferred D2H copies, keyed by AM
    // request_id, recording where the reply payload must land. Drained at the
    // next synchronous receive.
    struct PendingD2H {
        void *dst;
        size_t count;
    };
    std::map<std::uint64_t, PendingD2H> mPendingD2H;

    // Phase 2 async dispatcher: count of fire-and-forget large-H2D copies that
    // have consumed an RMA remote slot since the last drain (any synchronous
    // response). Bounds outstanding async RMA sends to the slot count so a
    // remote slot is never reused before the backend consumed it.
    size_t mAsyncRmaInflight = 0;

    int mExitCode;
    static std::map<pthread_t, Frontend *> *mpFrontends;
    bool mpInitialized;

    uint64_t mRoutinesExecuted = 0;
    uint64_t mDataSent = 0;
    uint64_t mDataReceived = 0;
    double mSendingTime = 0.0;
    double mReceivingTime = 0.0;
    double mRoutineExecutionTime = 0.0;
};
}  // namespace gvirtus::frontend
