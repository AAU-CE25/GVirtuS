/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Ting-Hui Cheng <tinghc@es.aau.dk>,
 *             Department of Electronic Systems, Aalborg University, Denmark
 * 
 * cuGetProcAddress fix: Returns pointers to LOCAL GVirtuS stub functions,
 * NOT backend addresses. This is essential for libraries that use dlsym/
 * cuGetProcAddress to dynamically resolve CUDA functions (e.g., libcudf.so).
 */

#include "CudaDr.h"
#include <unordered_map>
#include <string>
#include <cstring>

using namespace std;

// Forward declarations of all CUDA driver API functions implemented in GVirtuS
// These are defined in other CudaDr_*.cpp files
extern "C" {
    // Initialization
    CUresult cuInit(unsigned int);
    CUresult cuDriverGetVersion(int*);
    
    // Device management
    CUresult cuDeviceGet(CUdevice*, int);
    CUresult cuDeviceGetAttribute(int*, CUdevice_attribute, CUdevice);
    CUresult cuDeviceGetCount(int*);
    CUresult cuDeviceGetName(char*, int, CUdevice);
    CUresult cuDeviceGetProperties(CUdevprop*, CUdevice);
    CUresult cuDeviceComputeCapability(int*, int*, CUdevice);
    CUresult cuDeviceTotalMem(size_t*, CUdevice);
    CUresult cuDeviceCanAccessPeer(int*, CUdevice, CUdevice);
    CUresult cuDevicePrimaryCtxGetState(CUdevice, unsigned int*, int*);
    CUresult cuDevicePrimaryCtxRelease(CUdevice);
    CUresult cuDevicePrimaryCtxRelease_v2(CUdevice);
    CUresult cuDevicePrimaryCtxReset(CUdevice);
    CUresult cuDevicePrimaryCtxReset_v2(CUdevice);
    CUresult cuDevicePrimaryCtxRetain(CUcontext*, CUdevice);
    
    // Context management
    CUresult cuCtxAttach(CUcontext*, unsigned int);
    CUresult cuCtxCreate(CUcontext*, unsigned int, CUdevice);
    CUresult cuCtxDestroy(CUcontext);
    CUresult cuCtxDetach(CUcontext);
    CUresult cuCtxDisablePeerAccess(CUcontext);
    CUresult cuCtxEnablePeerAccess(CUcontext, unsigned int);
    CUresult cuCtxGetCurrent(CUcontext*);
    CUresult cuCtxGetDevice(CUdevice*);
    CUresult cuCtxGetLimit(size_t*, CUlimit);
    CUresult cuCtxPopCurrent(CUcontext*);
    CUresult cuCtxPopCurrent_v2(CUcontext*);
    CUresult cuCtxPushCurrent(CUcontext);
    CUresult cuCtxPushCurrent_v2(CUcontext);
    CUresult cuCtxSetCurrent(CUcontext);
    CUresult cuCtxSetLimit(CUlimit, size_t);
    CUresult cuCtxSynchronize();
    
    // Memory management
    CUresult cuMemAlloc(CUdeviceptr*, size_t);
    CUresult cuMemAllocHost(void**, size_t);
    CUresult cuMemAllocPitch(CUdeviceptr*, size_t*, size_t, size_t, unsigned int);
    CUresult cuMemFree(CUdeviceptr);
    CUresult cuMemFreeHost(void*);
    CUresult cuMemGetAddressRange(CUdeviceptr*, size_t*, CUdeviceptr);
    CUresult cuMemGetInfo(size_t*, size_t*);
    CUresult cuMemHostAlloc(void**, size_t, unsigned int);
    CUresult cuMemHostGetDevicePointer(CUdeviceptr*, void*, unsigned int);
    CUresult cuMemHostGetFlags(unsigned int*, void*);
    CUresult cuMemcpyDtoH(void*, CUdeviceptr, size_t);
    CUresult cuMemcpyDtoHAsync(void*, CUdeviceptr, size_t, CUstream);
    CUresult cuMemcpyHtoD(CUdeviceptr, const void*, size_t);
    CUresult cuMemcpyHtoDAsync(CUdeviceptr, const void*, size_t, CUstream);
    CUresult cuMemcpyDtoD(CUdeviceptr, CUdeviceptr, size_t);
    CUresult cuMemcpyDtoDAsync(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult cuMemsetD8(CUdeviceptr, unsigned char, size_t);
    CUresult cuMemsetD16(CUdeviceptr, unsigned short, size_t);
    CUresult cuMemsetD32(CUdeviceptr, unsigned int, size_t);
    CUresult cuMemsetD32Async(CUdeviceptr, unsigned int, size_t, CUstream);
    CUresult cuMemsetD2D8(CUdeviceptr, size_t, unsigned char, size_t, size_t);
    CUresult cuMemsetD2D16(CUdeviceptr, size_t, unsigned short, size_t, size_t);
    CUresult cuMemsetD2D32(CUdeviceptr, size_t, unsigned int, size_t, size_t);
    CUresult cuPointerGetAttribute(void*, CUpointer_attribute, CUdeviceptr);
    
    // Array management  
    CUresult cuArrayCreate(CUarray*, const CUDA_ARRAY_DESCRIPTOR*);
    CUresult cuArray3DCreate(CUarray*, const CUDA_ARRAY3D_DESCRIPTOR*);
    CUresult cuArrayDestroy(CUarray);
    CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR*, CUarray);
    CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR*, CUarray);
    CUresult cuMemcpy2D(const CUDA_MEMCPY2D*);
    CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D*, CUstream);
    CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D*);
    CUresult cuMemcpy3D(const CUDA_MEMCPY3D*);
    CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D*, CUstream);
    CUresult cuMemcpyAtoA(CUarray, size_t, CUarray, size_t, size_t);
    CUresult cuMemcpyAtoD(CUdeviceptr, CUarray, size_t, size_t);
    CUresult cuMemcpyAtoH(void*, CUarray, size_t, size_t);
    CUresult cuMemcpyAtoHAsync(void*, CUarray, size_t, size_t, CUstream);
    CUresult cuMemcpyDtoA(CUarray, size_t, CUdeviceptr, size_t);
    CUresult cuMemcpyHtoA(CUarray, size_t, const void*, size_t);
    CUresult cuMemcpyHtoAAsync(CUarray, size_t, const void*, size_t, CUstream);
    
    // Virtual memory
    CUresult cuMemAddressFree(CUdeviceptr, size_t);
    CUresult cuMemAddressReserve(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult cuMemCreate(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    CUresult cuMemExportToShareableHandle(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
    CUresult cuMemGetAllocationGranularity(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
    CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
    CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    CUresult cuMemRelease(CUmemGenericAllocationHandle);
    CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    CUresult cuMemUnmap(CUdeviceptr, size_t);
    
    // Module management
    CUresult cuModuleGetFunction(CUfunction*, CUmodule, const char*);
    CUresult cuModuleGetGlobal(CUdeviceptr*, size_t*, CUmodule, const char*);
    CUresult cuModuleGetTexRef(CUtexref*, CUmodule, const char*);
    CUresult cuModuleLoad(CUmodule*, const char*);
    CUresult cuModuleLoadData(CUmodule*, const void*);
    CUresult cuModuleLoadDataEx(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
    CUresult cuModuleLoadFatBinary(CUmodule*, const void*);
    CUresult cuModuleUnload(CUmodule);
    CUresult cuLinkCreate(unsigned int, CUjit_option*, void**, CUlinkState*);
    CUresult cuLinkAddData(CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**);
    CUresult cuLinkAddFile(CUlinkState, CUjitInputType, const char*, unsigned int, CUjit_option*, void**);
    CUresult cuLinkComplete(CUlinkState, void**, size_t*);
    CUresult cuLinkDestroy(CUlinkState);
    
    // Execution
    CUresult cuFuncGetAttribute(int*, CUfunction_attribute, CUfunction);
    CUresult cuFuncSetAttribute(CUfunction, CUfunction_attribute, int);
    CUresult cuFuncSetBlockShape(CUfunction, int, int, int);
    CUresult cuFuncSetCacheConfig(CUfunction, CUfunc_cache);
    CUresult cuFuncSetSharedSize(CUfunction, unsigned int);
    CUresult cuLaunch(CUfunction);
    CUresult cuLaunchGrid(CUfunction, int, int);
    CUresult cuLaunchGridAsync(CUfunction, int, int, CUstream);
    CUresult cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
    CUresult cuLaunchKernelEx(const CUlaunchConfig*, CUfunction, void**, void**);
    CUresult cuParamSetf(CUfunction, int, float);
    CUresult cuParamSeti(CUfunction, int, unsigned int);
    CUresult cuParamSetSize(CUfunction, unsigned int);
    CUresult cuParamSetTexRef(CUfunction, int, CUtexref);
    CUresult cuParamSetv(CUfunction, int, void*, unsigned int);
    
    // Occupancy
    CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t*, CUfunction, int, int);
    CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int*, CUfunction, int, size_t);
    CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int*, CUfunction, int, size_t, unsigned int);
    CUresult cuOccupancyMaxActiveClusters(int*, CUfunction, const CUlaunchConfig*);
    CUresult cuOccupancyMaxPotentialBlockSize(int*, int*, CUfunction, void*, size_t, int);
    CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, CUfunction, void*, size_t, int, unsigned int);
    CUresult cuOccupancyMaxPotentialClusterSize(int*, CUfunction, const CUlaunchConfig*);
    
    // Stream management
    CUresult cuStreamCreate(CUstream*, unsigned int);
    CUresult cuStreamDestroy(CUstream);
    CUresult cuStreamQuery(CUstream);
    CUresult cuStreamSynchronize(CUstream);
    CUresult cuStreamWriteValue32(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
    CUresult cuStreamWriteValue32_v2(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
    
    // Event management
    CUresult cuEventCreate(CUevent*, unsigned int);
    CUresult cuEventDestroy(CUevent);
    CUresult cuEventElapsedTime(float*, CUevent, CUevent);
    CUresult cuEventQuery(CUevent);
    CUresult cuEventRecord(CUevent, CUstream);
    CUresult cuEventSynchronize(CUevent);
    
    // Texture management
    CUresult cuTexRefCreate(CUtexref*);
    CUresult cuTexRefDestroy(CUtexref);
    CUresult cuTexRefGetAddress(CUdeviceptr*, CUtexref);
    CUresult cuTexRefGetAddressMode(CUaddress_mode*, CUtexref, int);
    CUresult cuTexRefGetArray(CUarray*, CUtexref);
    CUresult cuTexRefGetFilterMode(CUfilter_mode*, CUtexref);
    CUresult cuTexRefGetFlags(unsigned int*, CUtexref);
    CUresult cuTexRefGetFormat(CUarray_format*, int*, CUtexref);
    CUresult cuTexRefSetAddress(size_t*, CUtexref, CUdeviceptr, size_t);
    CUresult cuTexRefSetAddress2D(CUtexref, const CUDA_ARRAY_DESCRIPTOR*, CUdeviceptr, size_t);
    CUresult cuTexRefSetAddressMode(CUtexref, int, CUaddress_mode);
    CUresult cuTexRefSetArray(CUtexref, CUarray, unsigned int);
    CUresult cuTexRefSetFilterMode(CUtexref, CUfilter_mode);
    CUresult cuTexRefSetFlags(CUtexref, unsigned int);
    CUresult cuTexRefSetFormat(CUtexref, CUarray_format, int);
    
    // Tensor map
    CUresult cuTensorMapEncodeTiled(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    
    // Error handling
    CUresult cuGetErrorName(CUresult, const char**);
    CUresult cuGetErrorString(CUresult, const char**);
}

// Static function pointer map - built once on first call
static unordered_map<string, void*>* getFunctionMap() {
    static unordered_map<string, void*> funcMap = {
        // Initialization
        {"cuInit", (void*)cuInit},
        {"cuDriverGetVersion", (void*)cuDriverGetVersion},
        
        // Device management
        {"cuDeviceGet", (void*)cuDeviceGet},
        {"cuDeviceGetAttribute", (void*)cuDeviceGetAttribute},
        {"cuDeviceGetCount", (void*)cuDeviceGetCount},
        {"cuDeviceGetName", (void*)cuDeviceGetName},
        {"cuDeviceGetProperties", (void*)cuDeviceGetProperties},
        {"cuDeviceComputeCapability", (void*)cuDeviceComputeCapability},
        {"cuDeviceTotalMem", (void*)cuDeviceTotalMem},
        {"cuDeviceTotalMem_v2", (void*)cuDeviceTotalMem},
        {"cuDeviceCanAccessPeer", (void*)cuDeviceCanAccessPeer},
        {"cuDevicePrimaryCtxGetState", (void*)cuDevicePrimaryCtxGetState},
        {"cuDevicePrimaryCtxRelease", (void*)cuDevicePrimaryCtxRelease},
        {"cuDevicePrimaryCtxRelease_v2", (void*)cuDevicePrimaryCtxRelease_v2},
        {"cuDevicePrimaryCtxReset", (void*)cuDevicePrimaryCtxReset},
        {"cuDevicePrimaryCtxReset_v2", (void*)cuDevicePrimaryCtxReset_v2},
        {"cuDevicePrimaryCtxRetain", (void*)cuDevicePrimaryCtxRetain},
        
        // Context management
        {"cuCtxAttach", (void*)cuCtxAttach},
        {"cuCtxCreate", (void*)cuCtxCreate},
        {"cuCtxCreate_v2", (void*)cuCtxCreate},
        {"cuCtxDestroy", (void*)cuCtxDestroy},
        {"cuCtxDestroy_v2", (void*)cuCtxDestroy},
        {"cuCtxDetach", (void*)cuCtxDetach},
        {"cuCtxDisablePeerAccess", (void*)cuCtxDisablePeerAccess},
        {"cuCtxEnablePeerAccess", (void*)cuCtxEnablePeerAccess},
        {"cuCtxGetCurrent", (void*)cuCtxGetCurrent},
        {"cuCtxGetDevice", (void*)cuCtxGetDevice},
        {"cuCtxGetLimit", (void*)cuCtxGetLimit},
        {"cuCtxPopCurrent", (void*)cuCtxPopCurrent},
        {"cuCtxPopCurrent_v2", (void*)cuCtxPopCurrent_v2},
        {"cuCtxPushCurrent", (void*)cuCtxPushCurrent},
        {"cuCtxPushCurrent_v2", (void*)cuCtxPushCurrent_v2},
        {"cuCtxSetCurrent", (void*)cuCtxSetCurrent},
        {"cuCtxSetLimit", (void*)cuCtxSetLimit},
        {"cuCtxSynchronize", (void*)cuCtxSynchronize},
        
        // Memory management
        {"cuMemAlloc", (void*)cuMemAlloc},
        {"cuMemAlloc_v2", (void*)cuMemAlloc},
        {"cuMemAllocHost", (void*)cuMemAllocHost},
        {"cuMemAllocHost_v2", (void*)cuMemAllocHost},
        {"cuMemAllocPitch", (void*)cuMemAllocPitch},
        {"cuMemAllocPitch_v2", (void*)cuMemAllocPitch},
        {"cuMemFree", (void*)cuMemFree},
        {"cuMemFree_v2", (void*)cuMemFree},
        {"cuMemFreeHost", (void*)cuMemFreeHost},
        {"cuMemGetAddressRange", (void*)cuMemGetAddressRange},
        {"cuMemGetAddressRange_v2", (void*)cuMemGetAddressRange},
        {"cuMemGetInfo", (void*)cuMemGetInfo},
        {"cuMemGetInfo_v2", (void*)cuMemGetInfo},
        {"cuMemHostAlloc", (void*)cuMemHostAlloc},
        {"cuMemHostGetDevicePointer", (void*)cuMemHostGetDevicePointer},
        {"cuMemHostGetDevicePointer_v2", (void*)cuMemHostGetDevicePointer},
        {"cuMemHostGetFlags", (void*)cuMemHostGetFlags},
        {"cuMemcpyDtoH", (void*)cuMemcpyDtoH},
        {"cuMemcpyDtoH_v2", (void*)cuMemcpyDtoH},
        {"cuMemcpyDtoHAsync", (void*)cuMemcpyDtoHAsync},
        {"cuMemcpyDtoHAsync_v2", (void*)cuMemcpyDtoHAsync},
        {"cuMemcpyHtoD", (void*)cuMemcpyHtoD},
        {"cuMemcpyHtoD_v2", (void*)cuMemcpyHtoD},
        {"cuMemcpyHtoDAsync", (void*)cuMemcpyHtoDAsync},
        {"cuMemcpyHtoDAsync_v2", (void*)cuMemcpyHtoDAsync},
        {"cuMemcpyDtoD", (void*)cuMemcpyDtoD},
        {"cuMemcpyDtoD_v2", (void*)cuMemcpyDtoD},
        {"cuMemcpyDtoDAsync", (void*)cuMemcpyDtoDAsync},
        {"cuMemcpyDtoDAsync_v2", (void*)cuMemcpyDtoDAsync},
        {"cuMemsetD8", (void*)cuMemsetD8},
        {"cuMemsetD8_v2", (void*)cuMemsetD8},
        {"cuMemsetD16", (void*)cuMemsetD16},
        {"cuMemsetD16_v2", (void*)cuMemsetD16},
        {"cuMemsetD32", (void*)cuMemsetD32},
        {"cuMemsetD32_v2", (void*)cuMemsetD32},
        {"cuMemsetD32Async", (void*)cuMemsetD32Async},
        {"cuMemsetD2D8", (void*)cuMemsetD2D8},
        {"cuMemsetD2D8_v2", (void*)cuMemsetD2D8},
        {"cuMemsetD2D16", (void*)cuMemsetD2D16},
        {"cuMemsetD2D16_v2", (void*)cuMemsetD2D16},
        {"cuMemsetD2D32", (void*)cuMemsetD2D32},
        {"cuMemsetD2D32_v2", (void*)cuMemsetD2D32},
        {"cuPointerGetAttribute", (void*)cuPointerGetAttribute},
        
        // Array management
        {"cuArrayCreate", (void*)cuArrayCreate},
        {"cuArrayCreate_v2", (void*)cuArrayCreate},
        {"cuArray3DCreate", (void*)cuArray3DCreate},
        {"cuArray3DCreate_v2", (void*)cuArray3DCreate},
        {"cuArrayDestroy", (void*)cuArrayDestroy},
        {"cuArrayGetDescriptor", (void*)cuArrayGetDescriptor},
        {"cuArrayGetDescriptor_v2", (void*)cuArrayGetDescriptor},
        {"cuArray3DGetDescriptor", (void*)cuArray3DGetDescriptor},
        {"cuArray3DGetDescriptor_v2", (void*)cuArray3DGetDescriptor},
        {"cuMemcpy2D", (void*)cuMemcpy2D},
        {"cuMemcpy2D_v2", (void*)cuMemcpy2D},
        {"cuMemcpy2DAsync", (void*)cuMemcpy2DAsync},
        {"cuMemcpy2DAsync_v2", (void*)cuMemcpy2DAsync},
        {"cuMemcpy2DUnaligned", (void*)cuMemcpy2DUnaligned},
        {"cuMemcpy2DUnaligned_v2", (void*)cuMemcpy2DUnaligned},
        {"cuMemcpy3D", (void*)cuMemcpy3D},
        {"cuMemcpy3D_v2", (void*)cuMemcpy3D},
        {"cuMemcpy3DAsync", (void*)cuMemcpy3DAsync},
        {"cuMemcpy3DAsync_v2", (void*)cuMemcpy3DAsync},
        {"cuMemcpyAtoA", (void*)cuMemcpyAtoA},
        {"cuMemcpyAtoA_v2", (void*)cuMemcpyAtoA},
        {"cuMemcpyAtoD", (void*)cuMemcpyAtoD},
        {"cuMemcpyAtoD_v2", (void*)cuMemcpyAtoD},
        {"cuMemcpyAtoH", (void*)cuMemcpyAtoH},
        {"cuMemcpyAtoH_v2", (void*)cuMemcpyAtoH},
        {"cuMemcpyAtoHAsync", (void*)cuMemcpyAtoHAsync},
        {"cuMemcpyAtoHAsync_v2", (void*)cuMemcpyAtoHAsync},
        {"cuMemcpyDtoA", (void*)cuMemcpyDtoA},
        {"cuMemcpyDtoA_v2", (void*)cuMemcpyDtoA},
        {"cuMemcpyHtoA", (void*)cuMemcpyHtoA},
        {"cuMemcpyHtoA_v2", (void*)cuMemcpyHtoA},
        {"cuMemcpyHtoAAsync", (void*)cuMemcpyHtoAAsync},
        {"cuMemcpyHtoAAsync_v2", (void*)cuMemcpyHtoAAsync},
        
        // Virtual memory
        {"cuMemAddressFree", (void*)cuMemAddressFree},
        {"cuMemAddressReserve", (void*)cuMemAddressReserve},
        {"cuMemCreate", (void*)cuMemCreate},
        {"cuMemExportToShareableHandle", (void*)cuMemExportToShareableHandle},
        {"cuMemGetAllocationGranularity", (void*)cuMemGetAllocationGranularity},
        {"cuMemImportFromShareableHandle", (void*)cuMemImportFromShareableHandle},
        {"cuMemMap", (void*)cuMemMap},
        {"cuMemRelease", (void*)cuMemRelease},
        {"cuMemSetAccess", (void*)cuMemSetAccess},
        {"cuMemUnmap", (void*)cuMemUnmap},
        
        // Module management
        {"cuModuleGetFunction", (void*)cuModuleGetFunction},
        {"cuModuleGetGlobal", (void*)cuModuleGetGlobal},
        {"cuModuleGetGlobal_v2", (void*)cuModuleGetGlobal},
        {"cuModuleGetTexRef", (void*)cuModuleGetTexRef},
        {"cuModuleLoad", (void*)cuModuleLoad},
        {"cuModuleLoadData", (void*)cuModuleLoadData},
        {"cuModuleLoadDataEx", (void*)cuModuleLoadDataEx},
        {"cuModuleLoadFatBinary", (void*)cuModuleLoadFatBinary},
        {"cuModuleUnload", (void*)cuModuleUnload},
        {"cuLinkCreate", (void*)cuLinkCreate},
        {"cuLinkCreate_v2", (void*)cuLinkCreate},
        {"cuLinkAddData", (void*)cuLinkAddData},
        {"cuLinkAddData_v2", (void*)cuLinkAddData},
        {"cuLinkAddFile", (void*)cuLinkAddFile},
        {"cuLinkAddFile_v2", (void*)cuLinkAddFile},
        {"cuLinkComplete", (void*)cuLinkComplete},
        {"cuLinkDestroy", (void*)cuLinkDestroy},
        
        // Execution
        {"cuFuncGetAttribute", (void*)cuFuncGetAttribute},
        {"cuFuncSetAttribute", (void*)cuFuncSetAttribute},
        {"cuFuncSetBlockShape", (void*)cuFuncSetBlockShape},
        {"cuFuncSetCacheConfig", (void*)cuFuncSetCacheConfig},
        {"cuFuncSetSharedSize", (void*)cuFuncSetSharedSize},
        {"cuLaunch", (void*)cuLaunch},
        {"cuLaunchGrid", (void*)cuLaunchGrid},
        {"cuLaunchGridAsync", (void*)cuLaunchGridAsync},
        {"cuLaunchKernel", (void*)cuLaunchKernel},
        {"cuLaunchKernelEx", (void*)cuLaunchKernelEx},
        {"cuParamSetf", (void*)cuParamSetf},
        {"cuParamSeti", (void*)cuParamSeti},
        {"cuParamSetSize", (void*)cuParamSetSize},
        {"cuParamSetTexRef", (void*)cuParamSetTexRef},
        {"cuParamSetv", (void*)cuParamSetv},
        
        // Occupancy
        {"cuOccupancyAvailableDynamicSMemPerBlock", (void*)cuOccupancyAvailableDynamicSMemPerBlock},
        {"cuOccupancyMaxActiveBlocksPerMultiprocessor", (void*)cuOccupancyMaxActiveBlocksPerMultiprocessor},
        {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", (void*)cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags},
        {"cuOccupancyMaxActiveClusters", (void*)cuOccupancyMaxActiveClusters},
        {"cuOccupancyMaxPotentialBlockSize", (void*)cuOccupancyMaxPotentialBlockSize},
        {"cuOccupancyMaxPotentialBlockSizeWithFlags", (void*)cuOccupancyMaxPotentialBlockSizeWithFlags},
        {"cuOccupancyMaxPotentialClusterSize", (void*)cuOccupancyMaxPotentialClusterSize},
        
        // Stream management
        {"cuStreamCreate", (void*)cuStreamCreate},
        {"cuStreamDestroy", (void*)cuStreamDestroy},
        {"cuStreamDestroy_v2", (void*)cuStreamDestroy},
        {"cuStreamQuery", (void*)cuStreamQuery},
        {"cuStreamSynchronize", (void*)cuStreamSynchronize},
        {"cuStreamWriteValue32", (void*)cuStreamWriteValue32},
        {"cuStreamWriteValue32_v2", (void*)cuStreamWriteValue32_v2},
        
        // Event management
        {"cuEventCreate", (void*)cuEventCreate},
        {"cuEventDestroy", (void*)cuEventDestroy},
        {"cuEventDestroy_v2", (void*)cuEventDestroy},
        {"cuEventElapsedTime", (void*)cuEventElapsedTime},
        {"cuEventQuery", (void*)cuEventQuery},
        {"cuEventRecord", (void*)cuEventRecord},
        {"cuEventSynchronize", (void*)cuEventSynchronize},
        
        // Texture management
        {"cuTexRefCreate", (void*)cuTexRefCreate},
        {"cuTexRefDestroy", (void*)cuTexRefDestroy},
        {"cuTexRefGetAddress", (void*)cuTexRefGetAddress},
        {"cuTexRefGetAddress_v2", (void*)cuTexRefGetAddress},
        {"cuTexRefGetAddressMode", (void*)cuTexRefGetAddressMode},
        {"cuTexRefGetArray", (void*)cuTexRefGetArray},
        {"cuTexRefGetFilterMode", (void*)cuTexRefGetFilterMode},
        {"cuTexRefGetFlags", (void*)cuTexRefGetFlags},
        {"cuTexRefGetFormat", (void*)cuTexRefGetFormat},
        {"cuTexRefSetAddress", (void*)cuTexRefSetAddress},
        {"cuTexRefSetAddress_v2", (void*)cuTexRefSetAddress},
        {"cuTexRefSetAddress2D", (void*)cuTexRefSetAddress2D},
        {"cuTexRefSetAddress2D_v2", (void*)cuTexRefSetAddress2D},
        {"cuTexRefSetAddressMode", (void*)cuTexRefSetAddressMode},
        {"cuTexRefSetArray", (void*)cuTexRefSetArray},
        {"cuTexRefSetFilterMode", (void*)cuTexRefSetFilterMode},
        {"cuTexRefSetFlags", (void*)cuTexRefSetFlags},
        {"cuTexRefSetFormat", (void*)cuTexRefSetFormat},
        
        // Tensor map
        {"cuTensorMapEncodeTiled", (void*)cuTensorMapEncodeTiled},
        
        // Error handling
        {"cuGetErrorName", (void*)cuGetErrorName},
        {"cuGetErrorString", (void*)cuGetErrorString},
    };
    return &funcMap;
}

/*
 * cuGetProcAddress - Returns pointers to LOCAL GVirtuS stub functions.
 * 
 * Applications like libcudf.so use this to dynamically resolve CUDA functions.
 * We MUST return pointers to our local stub functions (which forward to backend),
 * NOT forward this call to the backend (which would return meaningless backend addresses).
 */
extern "C" CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, 
                                    cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    fprintf(stderr, "[GVirtuS cuGetProcAddress] Looking up symbol: %s (version=%d, flags=0x%llx)\n", 
            symbol ? symbol : "NULL", cudaVersion, (unsigned long long)flags);
    
    if (symbol == nullptr || pfn == nullptr) {
        fprintf(stderr, "[GVirtuS cuGetProcAddress] ERROR: NULL argument\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Look up the symbol in our local function map
    unordered_map<string, void*>* funcMap = getFunctionMap();
    auto it = funcMap->find(symbol);
    
    if (it != funcMap->end()) {
        *pfn = it->second;
        if (symbolStatus != nullptr) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        }
        fprintf(stderr, "[GVirtuS cuGetProcAddress] FOUND: %s -> %p\n", symbol, *pfn);
        return CUDA_SUCCESS;
    }
    
    // Symbol not found in our map - not implemented in GVirtuS
    *pfn = nullptr;
    if (symbolStatus != nullptr) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    fprintf(stderr, "[GVirtuS cuGetProcAddress] NOT FOUND: %s (not implemented in GVirtuS)\n", symbol);
    return CUDA_SUCCESS;  // Return success but with null pointer and NOT_FOUND status
}

// Also implement cuGetProcAddress_v2 which is used by newer CUDA versions
extern "C" CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion,
                                        cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    return cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}


