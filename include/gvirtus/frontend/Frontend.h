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

/*
 * Frontend — per-thread RPC stub extended for UCX zero-copy transfers.
 *
 * UCX communicator additions (Phases 4-5):
 *
 *   SetOutputDestination() / DirectOutputConsumed()  — Fase 4 (D2H zero-copy):
 *     caller pre-registers a destination buffer. Execute()'s response handler
 *     writes the big output payload directly there from the pinned RX-pool
 *     frame, skipping the intermediate mpOutputBuffer staging.
 *
 *   AddHostPointerForArgumentsDirect()  — Fase 5 (H2D zero-copy): records the
 *     caller's buffer pointer for splice into WriteIov's iov array. The 64 MB
 *     payload travels straight from user memory into RMA without memcpy into
 *     mpInputBuffer. Falls back to standard marshal for non-UCX transports.
 *
 *   Reentrancy guard (mpInitialized)  — prevents recursive RPC during UCX's
 *     libuct_cuda module init (dlopen fires cu* calls through the frontend
 *     shim before Connect() completes).
 *
 * Optimization phases: 4 (gather-send, frame receive), 5 (zero-copy H2D/D2H)
 */
#pragma once

#include <gvirtus/common/LD_Lib.h>
#include <gvirtus/communicators/Buffer.h>
#include <gvirtus/communicators/Communicator.h>

#include <map>

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

    // Zero-copy variant of AddHostPointerForArguments. Records `ptr` as a
    // BORROWED segment in the Buffer (Buffer::AddRef) instead of copying it:
    // the same [size_t len][bytes] wire layout is produced, but the data is
    // referenced in place and emitted by Buffer::GetIov() at send time.
    // Caller must NOT mutate `ptr` until Execute() returns. Works on every
    // transport — the UCX path sends the fragments via WriteIov (true
    // zero-copy); other transports concatenate once in Buffer::Dump(), giving
    // identical wire bytes. No transport gate, no Frontend-side splice state.
    template <class T>
    void AddHostPointerForArgumentsDirect(const T *ptr, size_t n = 1) {
        mpInputBuffer->AddRef<T>(ptr, n);
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
      AddStringForArguments(symbol);
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

    // Zero-copy H2D send is now handled entirely inside the input Buffer via
    // Buffer::AddRef / Buffer::GetIov (see AddHostPointerForArgumentsDirect) —
    // no Frontend-side splice state required.
    std::shared_ptr<communicators::Buffer> mpLaunchBuffer;

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
