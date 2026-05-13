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

#include <cstdint>
#include <gvirtus/common/Encoder.h>

#include <fstream>
#include <sstream>

#include "CudaDr.h"

using namespace std;

using gvirtus::common::Encoder;

/*Load a module's data. */
// Determine the size in bytes of a CUDA module image. CuPy/RAPIDS pass:
//   - cubin: ELF64 binary starting with \\x7fELF
//   - fatbin: starts with magic 0x*ed55ba* (\\x50\\xed\\x55\\xba little-endian)
//   - PTX: null-terminated text (legacy)
// strlen() truncates at the first internal NUL byte for binaries; this caused
// cuModuleLoadData to return CUDA_ERROR_INVALID_IMAGE (200).
static size_t gvirtus_module_image_size(const void *image) {
    if (image == nullptr) return 0;
    const unsigned char *b = (const unsigned char *)image;

    // ELF64 detection
    if (b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F') {
        uint64_t e_shoff = *(const uint64_t *)(b + 0x28);
        uint16_t e_shentsize = *(const uint16_t *)(b + 0x3a);
        uint16_t e_shnum = *(const uint16_t *)(b + 0x3c);
        size_t end = e_shoff + (size_t)e_shentsize * (size_t)e_shnum;
        // Walk section headers and find max(sh_offset + sh_size) for non-NOBITS
        for (uint16_t i = 0; i < e_shnum; i++) {
            const unsigned char *sh = b + e_shoff + (size_t)i * e_shentsize;
            uint32_t sh_type   = *(const uint32_t *)(sh + 0x04);
            uint64_t sh_offset = *(const uint64_t *)(sh + 0x18);
            uint64_t sh_size   = *(const uint64_t *)(sh + 0x20);
            if (sh_type != 8 /*SHT_NOBITS*/ && sh_offset + sh_size > end) {
                end = sh_offset + sh_size;
            }
        }
        return end;
    }

    // CUDA fatbin v1 detection (magic 0x*ed55ba* in NVIDIA fatbin header)
    if (b[0] == 0x50 && b[1] == 0xed && b[2] == 0x55 && b[3] == 0xba) {
        uint16_t headerSize = *(const uint16_t *)(b + 6);
        uint64_t dataSize   = *(const uint64_t *)(b + 8);
        return (size_t)headerSize + (size_t)dataSize;
    }

    // Fallback: assume null-terminated PTX text
    return strlen((const char *)image) + 1;
}

extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    CudaDrFrontend::Prepare();
    size_t image_size = gvirtus_module_image_size(image);
    // Use HostPointer with explicit size — backend reads via AssignString (size+bytes).
    CudaDrFrontend::AddVariableForArguments(image_size);
    CudaDrFrontend::AddHostPointerForArguments((char *)image, image_size);
    CudaDrFrontend::Execute("cuModuleLoadData");
    if (CudaDrFrontend::Success()) *module = (CUmodule)(CudaDrFrontend::GetOutputDevicePointer());
    return CudaDrFrontend::GetExitCode();
}

/*Returns a function handle*/
extern "C" CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *)name);
    CudaDrFrontend::AddDevicePointerForArguments((void *)hmod);
    CudaDrFrontend::Execute("cuModuleGetFunction");
    void *tmp;
    if (CudaDrFrontend::Success()) {
        tmp = (CUfunction)(CudaDrFrontend::GetOutputDevicePointer());
        *hfunc = (CUfunction)tmp;
    }
    return CudaDrFrontend::GetExitCode();
}

/*Returns a global pointer from a module.*/
extern "C" CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                      const char *name) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *)name);
    CudaDrFrontend::AddDevicePointerForArguments((void *)hmod);
    CudaDrFrontend::Execute("cuModuleGetGlobal");
    if (CudaDrFrontend::Success()) {
        *dptr = (CUdeviceptr)(CudaDrFrontend::GetOutputDevicePointer());
        *bytes = (size_t)(CudaDrFrontend::GetOutputDevicePointer());
    }
    return CudaDrFrontend::GetExitCode();
}

/*Returns a handle to a texture-reference.*/
extern "C" CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *)name);
    CudaDrFrontend::AddDevicePointerForArguments((void *)hmod);
    CudaDrFrontend::Execute("cuModuleGetTexRef");
    if (CudaDrFrontend::Success()) {
        *pTexRef = (CUtexref)(CudaDrFrontend::GetOutputDevicePointer());
    }
    return CudaDrFrontend::GetExitCode();
}

/*Load a module's data with options.*/
extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                                       CUjit_option *options, void **optionValues) {
    CudaDrFrontend::Prepare();
    char *tmp;
    char *tmp2;
    CudaDrFrontend::AddVariableForArguments(numOptions);
    CudaDrFrontend::AddHostPointerForArguments(options, numOptions);
    CudaDrFrontend::AddStringForArguments((char *)image);
    for (unsigned int i = 0; i < numOptions; i++) {
        CudaDrFrontend::AddHostPointerForArguments(&optionValues[i]);
    }
    CudaDrFrontend::Execute("cuModuleLoadDataEx");
    if (CudaDrFrontend::Success()) {
        *module = (CUmodule)(CudaDrFrontend::GetOutputDevicePointer());
        int len_str;
        for (unsigned int i = 0; i < numOptions - 1; i++) {
            switch (options[i]) {
                case CU_JIT_INFO_LOG_BUFFER:
                    len_str = CudaDrFrontend::GetOutputVariable<int>();
                    if (len_str > 0) {
                        tmp = CudaDrFrontend::GetOutputHostPointer<char>(len_str);
                        strcpy((char *)*(optionValues + i), tmp);
                    }
                    break;
                case CU_JIT_ERROR_LOG_BUFFER:
                    tmp2 = (CudaDrFrontend::GetOutputString());
                    strcpy((char *)*(optionValues + i), tmp2);
                    break;
                default:
                    *(optionValues + i) =
                        (void *)(CudaDrFrontend::GetOutputHostPointer<unsigned int>());
            }
        }
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    Encoder *encoder = new Encoder();
    std::ostringstream res;
    std::ifstream input("helloWorldDriverAPI.ptx", std::ios::binary);
    encoder->Encode(input, res);

    std::string moduleLoad = res.str();

    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *)fname);
    CudaDrFrontend::AddStringForArguments(moduleLoad.c_str());
    CudaDrFrontend::Execute("cuModuleLoad");
    if (CudaDrFrontend::Success()) *module = (CUmodule)(CudaDrFrontend::GetOutputDevicePointer());
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *)fatCubin);
    CudaDrFrontend::Execute("cuModuleLoadFatBinary");
    if (CudaDrFrontend::Success()) *module = (CUmodule)(CudaDrFrontend::GetOutputDevicePointer());
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuModuleUnload(CUmodule hmod) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((char *)hmod);
    CudaDrFrontend::Execute("cuModuleUnload");
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                                 void **optionValues, CUlinkState *stateOut) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(numOptions);
    CudaDrFrontend::AddHostPointerForArguments(options, numOptions);
    for (unsigned int i = 0; i < numOptions; i++) {
        CudaDrFrontend::AddHostPointerForArguments(&optionValues[i]);
    }
    CudaDrFrontend::Execute("cuLinkCreate");
    if (CudaDrFrontend::Success()) {
        *stateOut = (CUlinkState)(CudaDrFrontend::GetOutputDevicePointer());
    }
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size,
                                  const char *name, unsigned int numOptions, CUjit_option *options,
                                  void **optionValues) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)state);
    CudaDrFrontend::AddVariableForArguments(type);
    CudaDrFrontend::AddHostPointerForArguments(data, size);
    CudaDrFrontend::AddVariableForArguments(size);
    CudaDrFrontend::AddStringForArguments((char *)name);
    CudaDrFrontend::AddVariableForArguments(numOptions);
    CudaDrFrontend::AddHostPointerForArguments(options, numOptions);
    for (unsigned int i = 0; i < numOptions; i++) {
        CudaDrFrontend::AddHostPointerForArguments(&optionValues[i]);
    }
    CudaDrFrontend::Execute("cuLinkAddData");
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,
                                  unsigned int numOptions, CUjit_option *options,
                                  void **optionValues) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(state);
    CudaDrFrontend::AddVariableForArguments(type);
    CudaDrFrontend::AddStringForArguments(path);
    CudaDrFrontend::AddVariableForArguments(numOptions);
    CudaDrFrontend::AddHostPointerForArguments(options, numOptions);
    for (unsigned int i = 0; i < numOptions; i++) {
        CudaDrFrontend::AddHostPointerForArguments(&optionValues[i]);
    }
    CudaDrFrontend::Execute("cuLinkAddFile");
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(state);
    CudaDrFrontend::Execute("cuLinkComplete");
    if (CudaDrFrontend::Success()) {
        *cubinOut = CudaDrFrontend::GetOutputDevicePointer();
        *sizeOut = *(CudaDrFrontend::GetOutputHostPointer<size_t>());
    }
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuLinkDestroy(CUlinkState state) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(state);
    CudaDrFrontend::Execute("cuLinkDestroy");
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hfunc);
    CudaDrFrontend::AddVariableForArguments(attrib);
    CudaDrFrontend::AddVariableForArguments(value);
    CudaDrFrontend::Execute("cuFuncSetAttribute");
    return CudaDrFrontend::GetExitCode();
}