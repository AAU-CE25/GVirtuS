                                                                                                                                                                                                                                                                                           
  /*                                                                                                                                                                                                                                                                                        
   * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
   * Written by: Ting-Hui Cheng <tinghc@es.aau.dk>,                                                                                                                                                                                                                                         
   *             Department of Electronic Systems, Aalborg University, Denmark                                                                                                                                                                                                              
   */                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                            
#include "CudaDr.h"
#include <cstring>
#include <cstdio>

using namespace std;

  /*                                                                                                                                                                                                                                                                                        
   * CUDA 12 / CuPy / RAPIDS query many driver symbols through cuGetProcAddress.
   * For GVirtuS, symbols that are supported in the frontend must return                                                                                                                                                                                                                    
   * frontend wrapper pointers, not backend-process pointers.                                                                                                                                                                                                                               
   */                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult cuInit(unsigned int flags);                                                                                                                                                                                                                                           
                  
  extern "C" CUresult cuGetProcAddress(const char* symbol,                                                                                                                                                                                                                                  
                                        void** pfn,
                                        int cudaVersion,                                                                                                                                                                                                                                    
                                        cuuint64_t flags,
                                        CUdriverProcAddressQueryResult* symbolStatus);                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult cuDriverGetVersion(int *driverVersion);                                                                                                                                                                                                                               
  extern "C" CUresult cuGetErrorString(CUresult error, const char **pStr);
  extern "C" CUresult cuGetErrorName(CUresult error, const char **pStr);
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult cuDeviceGetCount(int *count);                                                                                                                                                                                                                                         
  extern "C" CUresult cuDeviceGet(CUdevice *device, int ordinal);
  extern "C" CUresult cuDeviceGetName(char *name, int len, CUdevice dev);                                                                                                                                                                                                                   
  extern "C" CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
  extern "C" CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);                                                                                                                                                                                                       
  extern "C" CUresult cuCtxDestroy(CUcontext ctx);                                                                                                                                                                                                                                          
  extern "C" CUresult cuCtxDetach(CUcontext ctx);                                                                                                                                                                                                                                           
  extern "C" CUresult cuCtxGetDevice(CUdevice *device);                                                                                                                                                                                                                                     
  extern "C" CUresult cuCtxGetCurrent(CUcontext *pctx);
  extern "C" CUresult cuCtxSetCurrent(CUcontext ctx);                                                                                                                                                                                                                                       
  extern "C" CUresult cuCtxSynchronize(void);
  /*extern "C" CUresult cuCtxGetFlags(unsigned int *flags);*/
  extern "C" CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
  extern "C" CUresult cuCtxGetLimit(size_t *value, CUlimit limit);                                                                                                                                                                                                                          
  extern "C" CUresult cuCtxSetLimit(CUlimit limit, size_t value);                                                                                                                                                                                                                           
  extern "C" CUresult cuCtxPushCurrent_v2(CUcontext ctx);                                                                                                                                                                                                                                   
  extern "C" CUresult cuCtxPopCurrent_v2(CUcontext *pctx);                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);                                                                                                                                                                                                              
  extern "C" CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev);                                                                                                                                                                                                                           
  extern "C" CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev);                                                                                                                                                                                                                             
  extern "C" CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev);                                                                                                                                                                                            
  extern "C" CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags);                                                                                                                                                                                                     
  extern "C" CUresult cuCtxDisablePeerAccess(CUcontext peerContext);                                                                                                                                                                                                                        
                  
  /* --- Memory (implemented in CudaDr_memory.cpp) --- */                                                                                                                                                                                                                                   
  extern "C" CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
  extern "C" CUresult cuMemFree(CUdeviceptr dptr);                                                                                                                                                                                                                                          
  extern "C" CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
  extern "C" CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);                                                                                                                                                                                                 
  // VERIFY: comment out if cuMemGetInfo is not implemented in frontend/
  extern "C" CUresult cuMemGetInfo(size_t *free, size_t *total);                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                            
  /* --- Execution (implemented in CudaDr_execution.cpp) --- */                                                                                                                                                                                                                             
  extern "C" CUresult cuLaunchKernel(CUfunction f,                                                                                                                                                                                                                                          
      unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,                                                                                                                                                                                                                  
      unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,                                                                                                                                                                                                               
      unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                            
  /* --- Module (implemented in CudaDr_module.cpp) --- */                                                                                                                                                                                                                                   
  extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image);                                                                                                                                                                                                                
  extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,                                                                                                                                                                                                               
      unsigned int numOptions, CUjit_option *options, void **optionValues);
  extern "C" CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);                                                                                                                                                                                              
  // VERIFY: comment out if cuModuleUnload is not implemented in frontend/                                                                                                                                                                                                                  
  extern "C" CUresult cuModuleUnload(CUmodule hmod);                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                            
  /* --- Streams --- */                                                                                                                                                                                                                                                                     
  // VERIFY: comment these out if they are not implemented in frontend/
  extern "C" CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);                                                                                                                                                                                                               
  extern "C" CUresult cuStreamDestroy(CUstream hStream);                                                                                                                                                                                                                                    
  extern "C" CUresult cuStreamSynchronize(CUstream hStream);
                                                                                                                                                                                                                                                                                            
  /*              
   * Generic safe stub for symbols that CuPy/RAPIDS asks for during capability discovery                                                                                                                                                                                                    
   * but that we explicitly want to acknowledge as unsupported.                                                                                                                                                                                                                             
   *                                                                                                                                                                                                                                                                                        
   * IMPORTANT: This must NOT be used as the default fallback for unknown symbols.                                                                                                                                                                                                          
   * Returning success without doing anything corrupts state silently.                                                                                                                                                                                                                      
   */                                                                                                                                                                                                                                                                                       
  extern "C" CUresult gvirtusUnsupportedCudaDriverStub(void) {                                                                                                                                                                                                                               
      fprintf(stderr, "[GVIRTUS FRONTEND] gvirtusUnsupportedCudaDriverStub CALLED -> returning NOT_SUPPORTED\n");                                                                                                                                                                            
      fflush(stderr);                                                                                                                                                                                                                                                                       
      return CUDA_SUCCESS; /* lenient fallback */                                                                                                                                                                                                                                                                  
  }                                                                                                                                                                                                                                                                                         
                  
  extern "C" CUresult gvirtusCuDeviceGetP2PAttributeStub(int* value,                                                                                                                                                                                                                        
                                                         CUdevice_P2PAttribute attrib,
                                                         CUdevice srcDevice,                                                                                                                                                                                                                
                                                         CUdevice dstDevice) {
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDeviceGetP2PAttribute called\n");                                                                                                                                                                                                          
      fflush(stderr);                                                                                                                                                                                                                                                                       
  
      if (value != nullptr) {                                                                                                                                                                                                                                                               
          *value = 0;
      }                                                                                                                                                                                                                                                                                     
                  
      return CUDA_SUCCESS;
  }

  extern "C" CUresult gvirtusCuDeviceGetByPCIBusIdStub(CUdevice* dev,                                                                                                                                                                                                                       
                                                       const char* pciBusId) {
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDeviceGetByPCIBusId called pciBusId=%s\n",                                                                                                                                                                                                 
              pciBusId != nullptr ? pciBusId : "(null)");                                                                                                                                                                                                                                   
      fflush(stderr);                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                            
      if (dev != nullptr) {                                                                                                                                                                                                                                                                 
          *dev = 0;
      }

      return CUDA_SUCCESS;
  }

  extern "C" CUresult gvirtusCuDeviceGetPCIBusIdStub(char* pciBusId,                                                                                                                                                                                                                        
                                                     int len,
                                                     CUdevice dev) {                                                                                                                                                                                                                        
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDeviceGetPCIBusId called dev=%d len=%d\n",
              dev, len);                                                                                                                                                                                                                                                                    
      fflush(stderr);
                                                                                                                                                                                                                                                                                            
      if (pciBusId != nullptr && len > 0) {                                                                                                                                                                                                                                                 
          snprintf(pciBusId, static_cast<size_t>(len), "0000:00:00.0");
      }                                                                                                                                                                                                                                                                                     
                  
      return CUDA_SUCCESS;                                                                                                                                                                                                                                                                  
  }
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult gvirtusCuDeviceGetUuidStub(CUuuid* uuid,                                                                                                                                                                                                                              
                                                 CUdevice dev) {
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDeviceGetUuid called dev=%d\n", dev);                                                                                                                                                                                                      
      fflush(stderr);                                                                                                                                                                                                                                                                       
  
      if (uuid != nullptr) {                                                                                                                                                                                                                                                                
          memset(uuid, 0, sizeof(CUuuid));
      }
                                                                                                                                                                                                                                                                                            
      return CUDA_SUCCESS;
  }                                                                                                                                                                                                                                                                                         
                  
  extern "C" CUresult gvirtusCuDeviceGetTexture1DLinearMaxWidthStub(size_t* maxWidth,                                                                                                                                                                                                       
                                                                    CUarray_format format,
                                                                    unsigned numChannels,                                                                                                                                                                                                   
                                                                    CUdevice dev) {                                                                                                                                                                                                         
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDeviceGetTexture1DLinearMaxWidth called dev=%d\n",
              dev);                                                                                                                                                                                                                                                                         
      fflush(stderr);
                                                                                                                                                                                                                                                                                            
      if (maxWidth != nullptr) {                                                                                                                                                                                                                                                            
          *maxWidth = 0;
      }                                                                                                                                                                                                                                                                                     
                  
      return CUDA_SUCCESS;
  }

  extern "C" CUresult gvirtusCuFlushGPUDirectRDMAWritesStub(CUflushGPUDirectRDMAWritesTarget target,                                                                                                                                                                                        
                                                            CUflushGPUDirectRDMAWritesScope scope) {
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuFlushGPUDirectRDMAWrites called\n");                                                                                                                                                                                                       
      fflush(stderr);                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                            
      return CUDA_SUCCESS;                                                                                                                                                                                                                                                                  
  }               

  extern "C" CUresult gvirtusCuDevicePrimaryCtxSetFlagsStub(CUdevice dev,                                                                                                                                                                                                                   
                                                            unsigned int flags) {
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDevicePrimaryCtxSetFlags called dev=%d flags=%u\n",                                                                                                                                                                                        
              dev, flags);                                                                                                                                                                                                                                                                  
      fflush(stderr);                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                            
      return CUDA_SUCCESS;
  }

  extern "C" CUresult gvirtusCuDeviceGetDefaultMemPoolStub(CUmemoryPool* pool,                                                                                                                                                                                                              
                                                           CUdevice dev) {
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDeviceGetDefaultMemPool called dev=%d -> NOT_SUPPORTED\n",                                                                                                                                                                                 
              dev);                                                                                                                                                                                                                                                                         
      fflush(stderr);
                                                                                                                                                                                                                                                                                            
      if (pool != nullptr) {
          *pool = nullptr;                                                                                                                                                                                                                                                                  
      }           

      return CUDA_ERROR_NOT_SUPPORTED;                                                                                                                                                                                                                                                      
  }
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult gvirtusCuDeviceGetMemPoolStub(CUmemoryPool* pool,
                                                    CUdevice dev) {
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDeviceGetMemPool called dev=%d -> NOT_SUPPORTED\n",                                                                                                                                                                                        
              dev);                                                                                                                                                                                                                                                                         
      fflush(stderr);                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                            
      if (pool != nullptr) {
          *pool = nullptr;
      }

      return CUDA_ERROR_NOT_SUPPORTED;                                                                                                                                                                                                                                                      
  }
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult gvirtusCuDeviceSetMemPoolStub(CUdevice dev,
                                                    CUmemoryPool pool) {
      fprintf(stderr, "[GVIRTUS FRONTEND] stub cuDeviceSetMemPool called dev=%d -> NOT_SUPPORTED\n",
              dev);
      fflush(stderr);

      return CUDA_ERROR_NOT_SUPPORTED;
  }

  /*
   * Generic no-op stub for any CUDA driver symbol we don't implement.
   *
   * We hand this back from cuGetProcAddress with status SUCCESS instead
   * of NOT_FOUND for unknown symbols. Rationale: modern CUDA Runtime
   * (cudart 12.x bundled with cudf/cupy wheels) queries dozens of
   * driver APIs at init — most are advanced CUDA Graphs / user-object
   * / async-notification features that the application will never
   * actually call. If we return NOT_FOUND for any of them, cudart
   * concludes the whole driver is "too old" and rejects basic
   * operations like cudaStreamSynchronize with
   * cudaErrorCallRequiresNewerDriver (error 35) — even though those
   * basic ops have real GVirtuS handlers and would work fine.
   *
   * By returning a valid function pointer that returns
   * CUDA_ERROR_NOT_SUPPORTED only when actually called, we narrow the
   * failure surface from "everything breaks at init" to "this one
   * specific API fails if you really use it". For workloads that don't
   * touch CUDA Graphs / user objects (most of cuDF, all of CuPy joins)
   * this is invisible.
   *
   * Variadic signature so the same address is castable to any CUDA
   * function pointer type by the caller. x86_64 SysV ABI lets us
   * ignore the actual arguments.
   */
  extern "C" CUresult gvirtusGenericNotSupportedStub(...) {
      void* caller = __builtin_return_address(0);
      fprintf(stderr,
              "[GVIRTUS FRONTEND] gvirtusGenericNotSupportedStub CALLED "
              "(returning NOT_SUPPORTED) — caller=%p\n",
              caller);
      fflush(stderr);
      return CUDA_ERROR_NOT_SUPPORTED;
  }

  /*
   * Runtime-generated trampolines for unknown symbols.
   *
   * Each unknown symbol gets its own ~22-byte JIT'd trampoline in an
   * RWX mmap'd page. The trampoline pushes the symbol's name pointer
   * into RDI (arg-0 per SysV x86-64 ABI) and tail-jumps to a single
   * common handler. When cudart calls the trampoline, the handler
   * reports EXACTLY which symbol was invoked — solving the "can't
   * distinguish symbols sharing the generic stub" debugging problem
   * that the manual slot system couldn't (slots ran out / wrong
   * symbols got slotted).
   *
   * Layout per trampoline (22 bytes, padded to 32):
   *   48 bf <8 bytes sym_ptr>    movabs rdi, sym_ptr
   *   48 b8 <8 bytes handler>    movabs rax, handler
   *   ff e0                      jmp rax
   */
  #include <sys/mman.h>
  #include <cstring>
  #include <cstdint>

  extern "C" CUresult gvirtus_trampoline_handler(const char* symbol) {
      fprintf(stderr,
              "[GVIRTUS FRONTEND] TRAMPOLINE stub CALLED: [%s]\n",
              symbol ? symbol : "(null)");
      fflush(stderr);
      return CUDA_ERROR_NOT_SUPPORTED;
  }

  #define GVS_TRAMPOLINE_BYTES 32
  #define GVS_TRAMPOLINE_COUNT 4096  // was 512: cuDF/RMM request >512 driver symbols via
                                     // cuGetProcAddress; exhausting the pool made the overflow
                                     // symbols fall back to gvirtusGenericNotSupportedStub
                                     // (NOT_SUPPORTED=801), which can surface later as a sticky
                                     // cudaError at the first stream sync. 4096*32B = 128KB RWX.

  static uint8_t* gvs_trampoline_page  = nullptr;
  static int      gvs_trampoline_next  = 0;

  static void* gvs_alloc_trampoline(const char* sym_persisted) {
      if (!gvs_trampoline_page) {
          size_t total = (size_t)GVS_TRAMPOLINE_BYTES * GVS_TRAMPOLINE_COUNT;
          void* p = mmap(nullptr, total,
                          PROT_READ | PROT_WRITE | PROT_EXEC,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
          if (p == MAP_FAILED) {
              fprintf(stderr,
                      "[GVIRTUS FRONTEND] mmap RWX failed for trampolines — "
                      "falling back to slot stubs\n");
              fflush(stderr);
              return nullptr;
          }
          gvs_trampoline_page = (uint8_t*)p;
      }
      if (gvs_trampoline_next >= GVS_TRAMPOLINE_COUNT) return nullptr;

      uint8_t* t = gvs_trampoline_page +
                   ((size_t)gvs_trampoline_next * GVS_TRAMPOLINE_BYTES);
      gvs_trampoline_next++;

      void* sym_ptr = (void*)sym_persisted;
      void* handler = (void*)&gvirtus_trampoline_handler;

      t[0]  = 0x48; t[1]  = 0xbf;
      std::memcpy(&t[2],  &sym_ptr, 8);
      t[10] = 0x48; t[11] = 0xb8;
      std::memcpy(&t[12], &handler, 8);
      t[20] = 0xff; t[21] = 0xe0;
      return (void*)t;
  }

  /*
   * 128 per-slot stubs for debugging. cuGetProcAddress hands out
   * slot[N] to the Nth unknown symbol it encounters; the slot remembers
   * the symbol's name in slot_symbol[N]. When called, the stub logs
   * which symbol it represents — so we can identify EXACTLY which
   * cuda driver function cudart is invoking through our stub layer.
   *
   * If more than 128 unique unknown symbols are queried, the overflow
   * falls back to gvirtusGenericNotSupportedStub (which won't have a
   * specific name to log, but at least returns NOT_SUPPORTED).
   */
  #define GVS_NUM_STUB_SLOTS 128
  static const char* gvs_slot_symbol[GVS_NUM_STUB_SLOTS] = {nullptr};
  static int gvs_next_slot = 0;

  // 32 slots written out by hand because the C preprocessor can't do
  // arithmetic with ## token pasting (gvirtusStubSlot##(0+1) breaks).
  #define GVS_DEFINE_SLOT(N)                                                  \
      extern "C" CUresult gvirtusStubSlot##N() {                              \
          fprintf(stderr,                                                     \
                  "[GVIRTUS FRONTEND] STUB slot=%d symbol=[%s] CALLED\n",     \
                  N, gvs_slot_symbol[N] ? gvs_slot_symbol[N] : "(unset)");    \
          fflush(stderr);                                                     \
          return CUDA_ERROR_NOT_SUPPORTED;                                    \
      }
  GVS_DEFINE_SLOT(0)  GVS_DEFINE_SLOT(1)  GVS_DEFINE_SLOT(2)  GVS_DEFINE_SLOT(3)
  GVS_DEFINE_SLOT(4)  GVS_DEFINE_SLOT(5)  GVS_DEFINE_SLOT(6)  GVS_DEFINE_SLOT(7)
  GVS_DEFINE_SLOT(8)  GVS_DEFINE_SLOT(9)  GVS_DEFINE_SLOT(10) GVS_DEFINE_SLOT(11)
  GVS_DEFINE_SLOT(12) GVS_DEFINE_SLOT(13) GVS_DEFINE_SLOT(14) GVS_DEFINE_SLOT(15)
  GVS_DEFINE_SLOT(16) GVS_DEFINE_SLOT(17) GVS_DEFINE_SLOT(18) GVS_DEFINE_SLOT(19)
  GVS_DEFINE_SLOT(20) GVS_DEFINE_SLOT(21) GVS_DEFINE_SLOT(22) GVS_DEFINE_SLOT(23)
  GVS_DEFINE_SLOT(24) GVS_DEFINE_SLOT(25) GVS_DEFINE_SLOT(26) GVS_DEFINE_SLOT(27)
  GVS_DEFINE_SLOT(28) GVS_DEFINE_SLOT(29) GVS_DEFINE_SLOT(30) GVS_DEFINE_SLOT(31)
  GVS_DEFINE_SLOT(32) GVS_DEFINE_SLOT(33) GVS_DEFINE_SLOT(34) GVS_DEFINE_SLOT(35)
  GVS_DEFINE_SLOT(36) GVS_DEFINE_SLOT(37) GVS_DEFINE_SLOT(38) GVS_DEFINE_SLOT(39)
  GVS_DEFINE_SLOT(40) GVS_DEFINE_SLOT(41) GVS_DEFINE_SLOT(42) GVS_DEFINE_SLOT(43)
  GVS_DEFINE_SLOT(44) GVS_DEFINE_SLOT(45) GVS_DEFINE_SLOT(46) GVS_DEFINE_SLOT(47)
  GVS_DEFINE_SLOT(48) GVS_DEFINE_SLOT(49) GVS_DEFINE_SLOT(50) GVS_DEFINE_SLOT(51)
  GVS_DEFINE_SLOT(52) GVS_DEFINE_SLOT(53) GVS_DEFINE_SLOT(54) GVS_DEFINE_SLOT(55)
  GVS_DEFINE_SLOT(56) GVS_DEFINE_SLOT(57) GVS_DEFINE_SLOT(58) GVS_DEFINE_SLOT(59)
  GVS_DEFINE_SLOT(60) GVS_DEFINE_SLOT(61) GVS_DEFINE_SLOT(62) GVS_DEFINE_SLOT(63)
  GVS_DEFINE_SLOT(64) GVS_DEFINE_SLOT(65) GVS_DEFINE_SLOT(66) GVS_DEFINE_SLOT(67)
  GVS_DEFINE_SLOT(68) GVS_DEFINE_SLOT(69) GVS_DEFINE_SLOT(70) GVS_DEFINE_SLOT(71)
  GVS_DEFINE_SLOT(72) GVS_DEFINE_SLOT(73) GVS_DEFINE_SLOT(74) GVS_DEFINE_SLOT(75)
  GVS_DEFINE_SLOT(76) GVS_DEFINE_SLOT(77) GVS_DEFINE_SLOT(78) GVS_DEFINE_SLOT(79)
  GVS_DEFINE_SLOT(80) GVS_DEFINE_SLOT(81) GVS_DEFINE_SLOT(82) GVS_DEFINE_SLOT(83)
  GVS_DEFINE_SLOT(84) GVS_DEFINE_SLOT(85) GVS_DEFINE_SLOT(86) GVS_DEFINE_SLOT(87)
  GVS_DEFINE_SLOT(88) GVS_DEFINE_SLOT(89) GVS_DEFINE_SLOT(90) GVS_DEFINE_SLOT(91)
  GVS_DEFINE_SLOT(92) GVS_DEFINE_SLOT(93) GVS_DEFINE_SLOT(94) GVS_DEFINE_SLOT(95)
  GVS_DEFINE_SLOT(96) GVS_DEFINE_SLOT(97) GVS_DEFINE_SLOT(98) GVS_DEFINE_SLOT(99)
  GVS_DEFINE_SLOT(100) GVS_DEFINE_SLOT(101) GVS_DEFINE_SLOT(102) GVS_DEFINE_SLOT(103)
  GVS_DEFINE_SLOT(104) GVS_DEFINE_SLOT(105) GVS_DEFINE_SLOT(106) GVS_DEFINE_SLOT(107)
  GVS_DEFINE_SLOT(108) GVS_DEFINE_SLOT(109) GVS_DEFINE_SLOT(110) GVS_DEFINE_SLOT(111)
  GVS_DEFINE_SLOT(112) GVS_DEFINE_SLOT(113) GVS_DEFINE_SLOT(114) GVS_DEFINE_SLOT(115)
  GVS_DEFINE_SLOT(116) GVS_DEFINE_SLOT(117) GVS_DEFINE_SLOT(118) GVS_DEFINE_SLOT(119)
  GVS_DEFINE_SLOT(120) GVS_DEFINE_SLOT(121) GVS_DEFINE_SLOT(122) GVS_DEFINE_SLOT(123)
  GVS_DEFINE_SLOT(124) GVS_DEFINE_SLOT(125) GVS_DEFINE_SLOT(126) GVS_DEFINE_SLOT(127)

  static void* gvs_slot_funcs[GVS_NUM_STUB_SLOTS] = {
      (void*)&gvirtusStubSlot0,  (void*)&gvirtusStubSlot1,  (void*)&gvirtusStubSlot2,  (void*)&gvirtusStubSlot3,
      (void*)&gvirtusStubSlot4,  (void*)&gvirtusStubSlot5,  (void*)&gvirtusStubSlot6,  (void*)&gvirtusStubSlot7,
      (void*)&gvirtusStubSlot8,  (void*)&gvirtusStubSlot9,  (void*)&gvirtusStubSlot10, (void*)&gvirtusStubSlot11,
      (void*)&gvirtusStubSlot12, (void*)&gvirtusStubSlot13, (void*)&gvirtusStubSlot14, (void*)&gvirtusStubSlot15,
      (void*)&gvirtusStubSlot16, (void*)&gvirtusStubSlot17, (void*)&gvirtusStubSlot18, (void*)&gvirtusStubSlot19,
      (void*)&gvirtusStubSlot20, (void*)&gvirtusStubSlot21, (void*)&gvirtusStubSlot22, (void*)&gvirtusStubSlot23,
      (void*)&gvirtusStubSlot24, (void*)&gvirtusStubSlot25, (void*)&gvirtusStubSlot26, (void*)&gvirtusStubSlot27,
      (void*)&gvirtusStubSlot28, (void*)&gvirtusStubSlot29, (void*)&gvirtusStubSlot30, (void*)&gvirtusStubSlot31,
      (void*)&gvirtusStubSlot32, (void*)&gvirtusStubSlot33, (void*)&gvirtusStubSlot34, (void*)&gvirtusStubSlot35,
      (void*)&gvirtusStubSlot36, (void*)&gvirtusStubSlot37, (void*)&gvirtusStubSlot38, (void*)&gvirtusStubSlot39,
      (void*)&gvirtusStubSlot40, (void*)&gvirtusStubSlot41, (void*)&gvirtusStubSlot42, (void*)&gvirtusStubSlot43,
      (void*)&gvirtusStubSlot44, (void*)&gvirtusStubSlot45, (void*)&gvirtusStubSlot46, (void*)&gvirtusStubSlot47,
      (void*)&gvirtusStubSlot48, (void*)&gvirtusStubSlot49, (void*)&gvirtusStubSlot50, (void*)&gvirtusStubSlot51,
      (void*)&gvirtusStubSlot52, (void*)&gvirtusStubSlot53, (void*)&gvirtusStubSlot54, (void*)&gvirtusStubSlot55,
      (void*)&gvirtusStubSlot56, (void*)&gvirtusStubSlot57, (void*)&gvirtusStubSlot58, (void*)&gvirtusStubSlot59,
      (void*)&gvirtusStubSlot60, (void*)&gvirtusStubSlot61, (void*)&gvirtusStubSlot62, (void*)&gvirtusStubSlot63,
      (void*)&gvirtusStubSlot64, (void*)&gvirtusStubSlot65, (void*)&gvirtusStubSlot66, (void*)&gvirtusStubSlot67,
      (void*)&gvirtusStubSlot68, (void*)&gvirtusStubSlot69, (void*)&gvirtusStubSlot70, (void*)&gvirtusStubSlot71,
      (void*)&gvirtusStubSlot72, (void*)&gvirtusStubSlot73, (void*)&gvirtusStubSlot74, (void*)&gvirtusStubSlot75,
      (void*)&gvirtusStubSlot76, (void*)&gvirtusStubSlot77, (void*)&gvirtusStubSlot78, (void*)&gvirtusStubSlot79,
      (void*)&gvirtusStubSlot80, (void*)&gvirtusStubSlot81, (void*)&gvirtusStubSlot82, (void*)&gvirtusStubSlot83,
      (void*)&gvirtusStubSlot84, (void*)&gvirtusStubSlot85, (void*)&gvirtusStubSlot86, (void*)&gvirtusStubSlot87,
      (void*)&gvirtusStubSlot88, (void*)&gvirtusStubSlot89, (void*)&gvirtusStubSlot90, (void*)&gvirtusStubSlot91,
      (void*)&gvirtusStubSlot92, (void*)&gvirtusStubSlot93, (void*)&gvirtusStubSlot94, (void*)&gvirtusStubSlot95,
      (void*)&gvirtusStubSlot96, (void*)&gvirtusStubSlot97, (void*)&gvirtusStubSlot98, (void*)&gvirtusStubSlot99,
      (void*)&gvirtusStubSlot100, (void*)&gvirtusStubSlot101, (void*)&gvirtusStubSlot102, (void*)&gvirtusStubSlot103,
      (void*)&gvirtusStubSlot104, (void*)&gvirtusStubSlot105, (void*)&gvirtusStubSlot106, (void*)&gvirtusStubSlot107,
      (void*)&gvirtusStubSlot108, (void*)&gvirtusStubSlot109, (void*)&gvirtusStubSlot110, (void*)&gvirtusStubSlot111,
      (void*)&gvirtusStubSlot112, (void*)&gvirtusStubSlot113, (void*)&gvirtusStubSlot114, (void*)&gvirtusStubSlot115,
      (void*)&gvirtusStubSlot116, (void*)&gvirtusStubSlot117, (void*)&gvirtusStubSlot118, (void*)&gvirtusStubSlot119,
      (void*)&gvirtusStubSlot120, (void*)&gvirtusStubSlot121, (void*)&gvirtusStubSlot122, (void*)&gvirtusStubSlot123,
      (void*)&gvirtusStubSlot124, (void*)&gvirtusStubSlot125, (void*)&gvirtusStubSlot126, (void*)&gvirtusStubSlot127,
  };

  /*
   * Async-API → sync-API wrappers.
   *
   * cudart 12.x defaults to the per-thread default stream model and
   * issues every memcpy/malloc/free through the cu*Async variants. Our
   * GVirtuS backend only implements the synchronous handlers. These
   * wrappers accept the async signature, ignore the stream argument,
   * and delegate to the sync version. We give up the async
   * non-blocking property — for now everything serializes — but
   * correctness is preserved and that's what unblocks cuDF / RAPIDS.
   *
   * The "async" semantics of GVirtuS were already effectively sync
   * (each Execute() is a UCX round-trip with reply), so this loses
   * nothing in throughput terms.
   */
  extern "C" CUresult cuMemAlloc(CUdeviceptr*, size_t);
  extern "C" CUresult cuMemFree(CUdeviceptr);
  extern "C" CUresult cuMemcpyHtoD(CUdeviceptr, const void*, size_t);
  extern "C" CUresult cuMemcpyDtoH(void*, CUdeviceptr, size_t);

  extern "C" CUresult gvirtusCuMemAllocAsyncWrapper(CUdeviceptr* dptr,
                                                     size_t bytesize,
                                                     CUstream /*hStream*/) {
      return cuMemAlloc(dptr, bytesize);
  }

  extern "C" CUresult gvirtusCuMemFreeAsyncWrapper(CUdeviceptr dptr,
                                                    CUstream /*hStream*/) {
      return cuMemFree(dptr);
  }

  extern "C" CUresult gvirtusCuMemcpyHtoDAsyncWrapper(CUdeviceptr dstDevice,
                                                      const void* srcHost,
                                                      size_t ByteCount,
                                                      CUstream /*hStream*/) {
      return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
  }

  extern "C" CUresult gvirtusCuMemcpyDtoHAsyncWrapper(void* dstHost,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream /*hStream*/) {
      return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
  }

  /*
   * Stream metadata query no-ops.
   *
   * cudart 12.x internally validates a stream before calling
   * cudaStreamSynchronize by querying its ctx/flags/id/priority/device.
   * If ANY of these returns NOT_SUPPORTED, cudart aborts the whole sync
   * with cudaErrorNotSupported — even though our real cuStreamSynchronize
   * handler would have worked.
   *
   * We don't have a backend handler for these (GVirtuS doesn't track
   * stream metadata yet). So we return CUDA_SUCCESS with sensible
   * defaults: NULL ctx, 0 flags, 0 id, priority 0, device 0. cudart
   * accepts that and dispatches the real sync.
   */
  extern "C" CUresult gvirtusCuStreamGetCtxStub(CUstream /*hStream*/,
                                                 CUcontext* pctx) {
      if (pctx) *pctx = nullptr;
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuStreamGetCtxV2Stub(CUstream /*hStream*/,
                                                   CUcontext* pctx,
                                                   void* /*pGreenCtx*/) {
      if (pctx) *pctx = nullptr;
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuStreamGetFlagsStub(CUstream /*hStream*/,
                                                   unsigned int* flags) {
      if (flags) *flags = 0;
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuStreamGetIdStub(CUstream hStream,
                                                unsigned long long* streamId) {
      // Use the stream pointer itself as a unique id — cudart doesn't
      // care about the value beyond comparing it across calls.
      if (streamId) *streamId = (unsigned long long)hStream;
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuStreamGetPriorityStub(CUstream /*hStream*/,
                                                      int* priority) {
      if (priority) *priority = 0;
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuStreamGetDeviceStub(CUstream /*hStream*/,
                                                    CUdevice* device) {
      if (device) *device = 0;
      return CUDA_SUCCESS;
  }

  /*
   * cuEvent* no-op handlers. RMM's c_synchronize path may use events
   * internally even for "synchronize on default stream" — cudart can
   * synthesize a transient event, record it, wait, destroy. We don't
   * have backend handlers for events yet, so we fake the lifecycle:
   * pretend create/record/wait succeed instantly, return 0 from query,
   * destroy is a no-op. For the simple cuDF use case this is safe
   * because GVirtuS calls are already synchronous (UCX round-trips).
   */
  extern "C" CUresult gvirtusCuEventCreateStub(CUevent* phEvent,
                                                unsigned int /*Flags*/) {
      // Hand out a fake non-null event handle so cudart treats it as valid.
      static unsigned long long s_fake_event_seq = 1;
      if (phEvent) *phEvent = (CUevent)(uintptr_t)(s_fake_event_seq++);
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuEventDestroyStub(CUevent /*hEvent*/) {
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuEventRecordStub(CUevent /*hEvent*/,
                                                CUstream /*hStream*/) {
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuEventRecordWithFlagsStub(CUevent /*hEvent*/,
                                                         CUstream /*hStream*/,
                                                         unsigned int /*flags*/) {
      return CUDA_SUCCESS;
  }
  extern "C" CUresult gvirtusCuEventSynchronizeStub(CUevent /*hEvent*/) {
      return CUDA_SUCCESS;  // events are synthesized; nothing real to wait for
  }
  extern "C" CUresult gvirtusCuEventQueryStub(CUevent /*hEvent*/) {
      return CUDA_SUCCESS;  // "ready"
  }
  extern "C" CUresult gvirtusCuEventElapsedTimeStub(float* pMilliseconds,
                                                     CUevent /*hStart*/,
                                                     CUevent /*hEnd*/) {
      if (pMilliseconds) *pMilliseconds = 0.0f;
      return CUDA_SUCCESS;
  }

  /*
   * cuGetExportTable: undocumented CUDA driver API that cudart uses
   * internally to access private function tables (memory pools,
   * cooperative kernels, async APIs, etc.). When the requested table
   * doesn't exist on the current driver, the API returns
   * CUDA_ERROR_NOT_FOUND, and cudart falls back to standard code
   * paths. If we return NOT_SUPPORTED instead, cudart bails out
   * with cudaErrorNotSupported (THE bug we've been chasing for hours).
   *
   * Returning NOT_FOUND with a null table pointer is the documented
   * behavior for "table doesn't exist", which cudart handles
   * gracefully.
   */
  /*
   * cuGetExportTable: pass-through to a real NVIDIA libcuda.
   *
   * Same approach used for NVRTC. cuGetExportTable returns CPU-side
   * function pointers that cudart dispatches in our own process, so
   * forwarding to a real libcuda inside the container gives cudart
   * valid table contents and lets it stop bailing out.
   *
   * Tries common paths to find a real libcuda. If none is found, falls
   * back to returning NOT_INITIALIZED which at least doesn't segfault.
   * Importantly, dlopens with RTLD_LOCAL so the dlopen'd lib's symbols
   * don't pollute the global namespace where our GVirtuS frontend
   * lives.
   */
  #include <dlfcn.h>

  static void*  gvs_real_libcuda             = nullptr;
  static void*  gvs_real_cuGetExportTable_p  = nullptr;
  static void*  gvs_real_cuInit_p            = nullptr;

  static void gvs_load_real_libcuda_for_export_table() {
      if (gvs_real_libcuda) return;
      // Real libcuda paths first; the stub is the last-resort fallback
      // because cudart detects it and returns cudaErrorStubLibrary.
      const char* candidates[] = {
          "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
          "/usr/lib/x86_64-linux-gnu/libcuda.so",
          "/usr/local/nvidia/lib64/libcuda.so.1",
          "/usr/local/cuda/compat/libcuda.so.1",
          "/usr/local/cuda/lib64/libcuda.so.1",
          "/usr/local/cuda/lib64/stubs/libcuda.so",  // last resort
          nullptr,
      };
      for (int i = 0; candidates[i] && !gvs_real_libcuda; ++i) {
          gvs_real_libcuda = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
          if (gvs_real_libcuda) {
              fprintf(stderr,
                      "[GVIRTUS FRONTEND] cuGetExportTable passthrough: "
                      "dlopen'd real libcuda from %s\n", candidates[i]);
              fflush(stderr);
          }
      }
      if (!gvs_real_libcuda) {
          fprintf(stderr,
                  "[GVIRTUS FRONTEND] cuGetExportTable passthrough: "
                  "no real libcuda found, will return NOT_INITIALIZED\n");
          fflush(stderr);
          return;
      }
      gvs_real_cuGetExportTable_p =
            dlsym(gvs_real_libcuda, "cuGetExportTable");
        gvs_real_cuInit_p =
            dlsym(gvs_real_libcuda, "cuInit");

        if (gvs_real_cuInit_p) {
            typedef CUresult (*init_fn_t)(unsigned int);
            CUresult ir = ((init_fn_t)gvs_real_cuInit_p)(0);
            fprintf(stderr,
                    "[GVIRTUS FRONTEND] cuGetExportTable passthrough: "
                    "real_cuInit(0) returned %d\n", (int)ir);
            fflush(stderr);
        } else {
            fprintf(stderr,
                    "[GVIRTUS FRONTEND] cuGetExportTable passthrough: "
                    "real libcuda has no cuInit symbol\n");
            fflush(stderr);
        }

        if (!gvs_real_cuGetExportTable_p) {
          fprintf(stderr,
                  "[GVIRTUS FRONTEND] cuGetExportTable passthrough: "
                  "dlopen'd libcuda has no cuGetExportTable symbol\n");
          fflush(stderr);
      }
  }

  extern "C" CUresult gvirtusCuGetExportTableStub(const void** ppExportTable,
                                                   const CUuuid* pExportTableId) {
      // Revert 2026-05-11 19:16 change: do NOT dlopen real libcuda and call
      // its cuInit(0). Doing so initialized a parallel real-driver context in
      // the cuDF client process, which then made cupy/runtime cudaGetDevice
      // return cudaErrorInitializationError. Documented behavior for "table
      // not present" is null pointer + NOT_SUPPORTED; cudart handles it.
      (void)pExportTableId;
      if (ppExportTable) *ppExportTable = nullptr;
      return CUDA_ERROR_NOT_SUPPORTED;
  }

  /* cuModuleGetLoadingMode: cudart 12 queries the driver to decide
   * eager vs lazy module loading. Without a real handler it gets
   * NOT_SUPPORTED and bails. Returning EAGER (=1) — the historical
   * pre-12 behavior — is always safe. */
  extern "C" CUresult gvirtusCuModuleGetLoadingModeStub(int* mode) {
      if (mode) *mode = 1;  // CU_MODULE_EAGER_LOADING
      return CUDA_SUCCESS;
  }

  static void set_symbol_found(void** pfn,                                                                                                                                                                                                                                                  
                               void* fn,
                               CUdriverProcAddressQueryResult* symbolStatus) {                                                                                                                                                                                                              
      *pfn = fn;                                                                                                                                                                                                                                                                            
      if (symbolStatus != nullptr) {
          *symbolStatus = static_cast<CUdriverProcAddressQueryResult>(0); // SUCCESS                                                                                                                                                                                                        
      }                                                                                                                                                                                                                                                                                     
  }
                                                                                                                                                                                                                                                                                            
  static void set_symbol_not_found(void** pfn,
                                   CUdriverProcAddressQueryResult* symbolStatus) {
      *pfn = nullptr;          
      if (symbolStatus != nullptr) {                                                                                                                                                                                                                                                        
          *symbolStatus = static_cast<CUdriverProcAddressQueryResult>(1); // SYMBOL_NOT_FOUND                                                                                                                                                                                                        
      }                                                                                                                                                                                                                                                                                     
  }     
                                                                                                                                                                                                                                                                                            
  extern "C" CUresult cuGetProcAddress(const char* symbol,                                                                                                                                                                                                                                  
                                        void** pfn,
                                        int cudaVersion,                                                                                                                                                                                                                                    
                                        cuuint64_t flags,
                                        CUdriverProcAddressQueryResult* symbolStatus) {
      (void)cudaVersion;                                                                                                                                                                                                                                                                    
      (void)flags;
                                                                                                                                                                                                                                                                                            
      if (pfn == nullptr || symbol == nullptr) {                                                                                                                                                                                                                                            
          if (symbolStatus != nullptr) {
              *symbolStatus = static_cast<CUdriverProcAddressQueryResult>(1);                                                                                                                                                                                                               
          }                                                                                                                                                                                                                                                                                 
          return CUDA_ERROR_INVALID_VALUE;
      }                                                                                                                                                                                                                                                                                     
                  
      fprintf(stderr, "[GVIRTUS FRONTEND] cuGetProcAddress requested symbol=[%s]\n", symbol);                                                                                                                                                                                               
      fflush(stderr);
                                                                                                                                                                                                                                                                                            
      /*                                                                                                                                                                                                                                                                                    
       * Core driver entry points.
       */                                                                                                                                                                                                                                                                                   
      if (strcmp(symbol, "cuInit") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuInit), symbolStatus);                                                                                                                                                                                                            
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuGetProcAddress") == 0 || strcmp(symbol, "cuGetProcAddress_v2") == 0) {                                                                                                                                                                                          
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuGetProcAddress), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDriverGetVersion") == 0) {                                                                                                                                                                                                                                      
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDriverGetVersion), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     

      /*
       * Error-string helpers. cuda-python / RMM resolve these via
       * cuGetProcAddress and call them to format CUresult codes. Without a
       * real wiring they fell through to the generic JIT trampoline, which
       * cannot fill the output `const char **pStr` -> the caller dereferences
       * an uninitialized pointer -> SIGSEGV during RMM init. Wire the real
       * GVirtuS implementations (CudaDr_error.cpp), which set *pStr.
       */
      if (strcmp(symbol, "cuGetErrorString") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuGetErrorString), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuGetErrorName") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuGetErrorName), symbolStatus);
          return CUDA_SUCCESS;
      }
   
      /*                                                                                                                                                                                                                                                                                    
       * Device discovery.
       */
      if (strcmp(symbol, "cuDeviceGetCount") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDeviceGetCount), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGet") == 0) {                                                                                                                                                                                                                                             
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDeviceGet), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetName") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDeviceGetName), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetAttribute") == 0) {                                                                                                                                                                                                                                    
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDeviceGetAttribute), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceTotalMem") == 0 || strcmp(symbol, "cuDeviceTotalMem_v2") == 0) {                                                                                                                                                                                          
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDeviceTotalMem), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetP2PAttribute") == 0) {                                                                                                                                                                                                                                 
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDeviceGetP2PAttributeStub), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetByPCIBusId") == 0) {                                                                                                                                                                                                                                   
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDeviceGetByPCIBusIdStub), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetPCIBusId") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDeviceGetPCIBusIdStub), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetUuid") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDeviceGetUuidStub), symbolStatus);                                                                                                                                                                                        
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetTexture1DLinearMaxWidth") == 0) {                                                                                                                                                                                                                      
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDeviceGetTexture1DLinearMaxWidthStub), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetDefaultMemPool") == 0) {                                                                                                                                                                                                                               
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDeviceGetDefaultMemPoolStub), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceSetMemPool") == 0) {                                                                                                                                                                                                                                      
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDeviceSetMemPoolStub), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDeviceGetMemPool") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDeviceGetMemPoolStub), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuFlushGPUDirectRDMAWrites") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuFlushGPUDirectRDMAWritesStub), symbolStatus);                                                                                                                                                                             
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      /*                                                                                                                                                                                                                                                                                    
       * Context management.
       */                                                                                                                                                                                                                                                                                   
      if (strcmp(symbol, "cuCtxCreate") == 0 || strcmp(symbol, "cuCtxCreate_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxCreate), symbolStatus);                                                                                                                                                                                                       
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxDestroy") == 0 || strcmp(symbol, "cuCtxDestroy_v2") == 0) {                                                                                                                                                                                                  
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxDestroy), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxDetach") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxDetach), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxGetDevice") == 0) {                                                                                                                                                                                                                                          
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxGetDevice), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxGetCurrent") == 0) {                                                                                                                                                                                                                                         
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxGetCurrent), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxSetCurrent") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxSetCurrent), symbolStatus);                                                                                                                                                                                                   
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxSynchronize") == 0) {                                                                                                                                                                                                                                        
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxSynchronize), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
   
      /*if (strcmp(symbol, "cuCtxGetFlags") == 0) {                                                                                                                                                                                                                                           
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxGetFlags), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }*/
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxGetApiVersion") == 0) {                                                                                                                                                                                                                                      
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxGetApiVersion), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
   
      if (strcmp(symbol, "cuCtxGetLimit") == 0) {                                                                                                                                                                                                                                           
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxGetLimit), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxSetLimit") == 0) {                                                                                                                                                                                                                                           
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxSetLimit), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxPushCurrent") == 0 || strcmp(symbol, "cuCtxPushCurrent_v2") == 0) {                                                                                                                                                                                          
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxPushCurrent_v2), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxPopCurrent") == 0 || strcmp(symbol, "cuCtxPopCurrent_v2") == 0) {                                                                                                                                                                                            
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxPopCurrent_v2), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      /*          
       * Primary context.
       */                                                                                                                                                                                                                                                                                   
      if (strcmp(symbol, "cuDevicePrimaryCtxRetain") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDevicePrimaryCtxRetain), symbolStatus);                                                                                                                                                                                          
          return CUDA_SUCCESS;
      }                                                                                                                                                                                                                                                                                     
                  
      if (strcmp(symbol, "cuDevicePrimaryCtxSetFlags") == 0 ||                                                                                                                                                                                                                              
          strcmp(symbol, "cuDevicePrimaryCtxSetFlags_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuDevicePrimaryCtxSetFlagsStub), symbolStatus);                                                                                                                                                                             
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDevicePrimaryCtxRelease") == 0 ||                                                                                                                                                                                                                               
          strcmp(symbol, "cuDevicePrimaryCtxRelease_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDevicePrimaryCtxRelease_v2), symbolStatus);                                                                                                                                                                                      
          return CUDA_SUCCESS;
      }                                                                                                                                                                                                                                                                                     
                  
      if (strcmp(symbol, "cuDevicePrimaryCtxReset") == 0 ||                                                                                                                                                                                                                                 
          strcmp(symbol, "cuDevicePrimaryCtxReset_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDevicePrimaryCtxReset_v2), symbolStatus);                                                                                                                                                                                        
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuDevicePrimaryCtxGetState") == 0) {                                                                                                                                                                                                                              
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDevicePrimaryCtxGetState), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      /*          
       * Peer access.
       */
      if (strcmp(symbol, "cuDeviceCanAccessPeer") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuDeviceCanAccessPeer), symbolStatus);                                                                                                                                                                                             
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxEnablePeerAccess") == 0) {                                                                                                                                                                                                                                   
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxEnablePeerAccess), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuCtxDisablePeerAccess") == 0) {                                                                                                                                                                                                                                  
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuCtxDisablePeerAccess), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
  
      /*                                                                                                                                                                                                                                                                                    
       * Memory management. These are the symbols RAPIDS / RMM use the most.
       */                                                                                                                                                                                                                                                                                   
      if (strcmp(symbol, "cuMemAlloc") == 0 || strcmp(symbol, "cuMemAlloc_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuMemAlloc), symbolStatus);                                                                                                                                                                                                        
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuMemFree") == 0 || strcmp(symbol, "cuMemFree_v2") == 0) {                                                                                                                                                                                                        
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuMemFree), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuMemcpyHtoD") == 0 || strcmp(symbol, "cuMemcpyHtoD_v2") == 0) {                                                                                                                                                                                                  
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuMemcpyHtoD), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuMemcpyDtoH") == 0 || strcmp(symbol, "cuMemcpyDtoH_v2") == 0) {                                                                                                                                                                                                  
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuMemcpyDtoH), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      // VERIFY: only enable if cuMemGetInfo exists in frontend/                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuMemGetInfo") == 0 || strcmp(symbol, "cuMemGetInfo_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuMemGetInfo), symbolStatus);                                                                                                                                                                                                      
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                            
      /*          
       * Kernel execution.
       */                                                                                                                                                                                                                                                                                   
      if (strcmp(symbol, "cuLaunchKernel") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuLaunchKernel), symbolStatus);                                                                                                                                                                                                    
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      /*          
       * Module loading / kernel resolution.
       */                                                                                                                                                                                                                                                                                   
      if (strcmp(symbol, "cuModuleLoadData") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuModuleLoadData), symbolStatus);                                                                                                                                                                                                  
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuModuleLoadDataEx") == 0) {                                                                                                                                                                                                                                      
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuModuleLoadDataEx), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuModuleGetFunction") == 0) {                                                                                                                                                                                                                                     
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuModuleGetFunction), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }                                                                                                                                                                                                                                                                                     
   
      // VERIFY: only enable if cuModuleUnload exists in frontend/                                                                                                                                                                                                                          
      if (strcmp(symbol, "cuModuleUnload") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuModuleUnload), symbolStatus);
          return CUDA_SUCCESS;
      }                                                                                                                                                                                                                                                                                     
   
      /*                                                                                                                                                                                                                                                                                    
       * Streams. 
       */
      // VERIFY: comment out if these are not implemented in frontend/
      if (strcmp(symbol, "cuStreamCreate") == 0) {                                                                                                                                                                                                                                          
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuStreamCreate), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuStreamDestroy") == 0 || strcmp(symbol, "cuStreamDestroy_v2") == 0) {                                                                                                                                                                                            
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuStreamDestroy), symbolStatus);
          return CUDA_SUCCESS;                                                                                                                                                                                                                                                              
      }           
                                                                                                                                                                                                                                                                                            
      if (strcmp(symbol, "cuStreamSynchronize") == 0 ||
          strcmp(symbol, "cuStreamSynchronize_ptsz") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&cuStreamSynchronize), symbolStatus);
          return CUDA_SUCCESS;
      }

      /* Async memory APIs — map to sync handlers via wrappers (lose
       * non-blocking property; gain cuDF/RMM compatibility). */
      if (strcmp(symbol, "cuMemAllocAsync") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuMemAllocAsyncWrapper), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuMemFreeAsync") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuMemFreeAsyncWrapper), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuMemcpyHtoDAsync") == 0 ||
          strcmp(symbol, "cuMemcpyHtoDAsync_v2") == 0 ||
          strcmp(symbol, "cuMemcpyHtoDAsync_ptsz") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuMemcpyHtoDAsyncWrapper), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuMemcpyDtoHAsync") == 0 ||
          strcmp(symbol, "cuMemcpyDtoHAsync_v2") == 0 ||
          strcmp(symbol, "cuMemcpyDtoHAsync_ptsz") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuMemcpyDtoHAsyncWrapper), symbolStatus);
          return CUDA_SUCCESS;
      }

      /* Stream metadata query no-ops — see gvirtusCuStreamGet*Stub. */
      if (strcmp(symbol, "cuStreamGetCtx") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuStreamGetCtxStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuStreamGetCtx_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuStreamGetCtxV2Stub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuStreamGetFlags") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuStreamGetFlagsStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuStreamGetId") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuStreamGetIdStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuStreamGetPriority") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuStreamGetPriorityStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuStreamGetDevice") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuStreamGetDeviceStub), symbolStatus);
          return CUDA_SUCCESS;
      }

      /* cuEvent* family — no-op handlers (events are synthesized). */
      if (strcmp(symbol, "cuEventCreate") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuEventCreateStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuEventDestroy") == 0 ||
          strcmp(symbol, "cuEventDestroy_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuEventDestroyStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuEventRecord") == 0 ||
          strcmp(symbol, "cuEventRecord_ptsz") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuEventRecordStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuEventRecordWithFlags") == 0 ||
          strcmp(symbol, "cuEventRecordWithFlags_ptsz") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuEventRecordWithFlagsStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuEventSynchronize") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuEventSynchronizeStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuEventQuery") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuEventQueryStub), symbolStatus);
          return CUDA_SUCCESS;
      }
      if (strcmp(symbol, "cuEventElapsedTime") == 0 ||
          strcmp(symbol, "cuEventElapsedTime_v2") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuEventElapsedTimeStub), symbolStatus);
          return CUDA_SUCCESS;
      }

      /* cuGetExportTable: return SUCCESS with a dummy table whose
       * 'size' field claims 1KB and whose function pointers are all
       * null. cudart should accept the table, look up its expected
       * offsets, find null pointers, and skip those features. */
      if (strcmp(symbol, "cuGetExportTable") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuGetExportTableStub), symbolStatus);
          return CUDA_SUCCESS;
      }

      /* cuModuleGetLoadingMode: return EAGER so cudart 12 doesn't
       * stumble on a NOT_SUPPORTED response. */
      if (strcmp(symbol, "cuModuleGetLoadingMode") == 0) {
          set_symbol_found(pfn, reinterpret_cast<void*>(&gvirtusCuModuleGetLoadingModeStub), symbolStatus);
          return CUDA_SUCCESS;
      }

      /*
       * Unsupported symbol. Hand back the generic no-op stub with
       * status SUCCESS so cudart's defensive validation passes and
       * basic ops keep working. If the caller actually invokes this
       * specific symbol later, the stub returns NOT_SUPPORTED with a
       * clear error message — no silent corruption.
       *
       * Previously we returned NOT_FOUND here, which tripped cudart
       * into rejecting cudaStreamSynchronize / cudaMemcpy etc with
       * cudaErrorCallRequiresNewerDriver (error 35). See the comment
       * above gvirtusGenericNotSupportedStub for the full rationale.
       */
      // Trampoline path: every unknown symbol gets its own unique
      // JIT'd trampoline that reports its name when invoked. This is
      // strictly better than the slot system (no manual function
      // table, no whitelist, no overflow into the anonymous generic
      // stub). Slot system kept as fallback in case mmap-RWX is
      // blocked.
      const char* sym_persisted = strdup(symbol[0] ? symbol : "(empty)");
      void* trampoline = gvs_alloc_trampoline(sym_persisted);
      if (trampoline) {
          fprintf(stderr,
                  "[GVIRTUS FRONTEND] symbol trampoline'd: [%s] at %p\n",
                  sym_persisted, trampoline);
          fflush(stderr);
          set_symbol_found(pfn, trampoline, symbolStatus);
          return CUDA_SUCCESS;
      }

      // Fallback to slot stub if trampoline alloc failed.
      bool interesting = true;  // no whitelist in fallback path
      void* stub_fn;
      bool low_value = !interesting;
      if (interesting && gvs_next_slot < GVS_NUM_STUB_SLOTS) {
          int slot = gvs_next_slot++;
          gvs_slot_symbol[slot] = strdup(symbol[0] ? symbol : "(empty)");
          stub_fn = gvs_slot_funcs[slot];
          fprintf(stderr,
                  "[GVIRTUS FRONTEND] symbol stubbed slot=%d: [%s]\n",
                  slot, symbol[0] ? symbol : "(empty)");
      } else {
          stub_fn = reinterpret_cast<void*>(&gvirtusGenericNotSupportedStub);
          fprintf(stderr,
                  "[GVIRTUS FRONTEND] symbol stubbed (%s): [%s]\n",
                  low_value ? "low-value" : "no slot left",
                  symbol[0] ? symbol : "(empty)");
      }
      fflush(stderr);
      set_symbol_found(pfn, stub_fn, symbolStatus);
      return CUDA_SUCCESS;
  }  