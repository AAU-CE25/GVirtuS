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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"

using namespace std;

/*Create a stream.*/
extern "C" CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(Flags);
    CudaDrFrontend::Execute("cuStreamCreate");
    if (CudaDrFrontend::Success()) *phStream = (CUstream)(CudaDrFrontend::GetOutputDevicePointer());
    return CudaDrFrontend::GetExitCode();
}

/*Destroys a stream.*/
extern "C" CUresult cuStreamDestroy(CUstream hStream) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStream);
    CudaDrFrontend::Execute("cuStreamDestroy");
    return CudaDrFrontend::GetExitCode();
}

/*Determine status of a compute stream.*/
extern "C" CUresult cuStreamQuery(CUstream hStream) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStream);
    CudaDrFrontend::Execute("cuStreamQuery");
    return CudaDrFrontend::GetExitCode();
}

/*Wait until a stream's tasks are completed.*/
extern "C" CUresult cuStreamSynchronize(CUstream hStream) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStream);
    CudaDrFrontend::Execute("cuStreamSynchronize");
    return CudaDrFrontend::GetExitCode();
}

/*Make a stream wait on an event.*/
extern "C" CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStream);
    CudaDrFrontend::AddDevicePointerForArguments((void *)hEvent);
    CudaDrFrontend::AddVariableForArguments(Flags);
    CudaDrFrontend::Execute("cuStreamWaitEvent");
    return CudaDrFrontend::GetExitCode();
}

/*Create a stream with the given priority.*/
extern "C" CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::AddVariableForArguments(priority);
    CudaDrFrontend::Execute("cuStreamCreateWithPriority");
    if (CudaDrFrontend::Success()) *phStream = (CUstream)(CudaDrFrontend::GetOutputDevicePointer());
    return CudaDrFrontend::GetExitCode();
}

/*Query the priority of a given stream.*/
extern "C" CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStream);
    CudaDrFrontend::Execute("cuStreamGetPriority");
    if (CudaDrFrontend::Success()) *priority = CudaDrFrontend::GetOutputVariable<int>();
    return CudaDrFrontend::GetExitCode();
}

/*Query the flags of a given stream.*/
extern "C" CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStream);
    CudaDrFrontend::Execute("cuStreamGetFlags");
    if (CudaDrFrontend::Success()) *flags = CudaDrFrontend::GetOutputVariable<unsigned int>();
    return CudaDrFrontend::GetExitCode();
}

/*Query the context associated with a stream.*/
extern "C" CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStream);
    CudaDrFrontend::Execute("cuStreamGetCtx");
    if (CudaDrFrontend::Success()) *pctx = (CUcontext)(CudaDrFrontend::GetOutputDevicePointer());
    return CudaDrFrontend::GetExitCode();
}
