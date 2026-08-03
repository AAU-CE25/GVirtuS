// Auto-generated stubs for cudart symbols not implemented by GVirtuS frontend.
// Each stub returns cudaErrorNotSupported (71). Purpose: complete cudart ABI surface.

#include <cstdio>

typedef int cudaError_t_local;
#define CUDART_STUB_NOT_SUPPORTED 71

#ifdef GVIRTUS_LOG_STUB_CALLS
  #define STUB_LOG(n) do { fprintf(stderr, "[CUDART_STUB] %s -> NOT_SUPPORTED\n", n); fflush(stderr); } while(0)
#else
  #define STUB_LOG(n) do {} while(0)
#endif


// Forward decl for legacy impl that the _ptsz wrapper forwards to.
extern "C" int cudaStreamSynchronize(void* stream);

extern "C" {

__attribute__((visibility("default"))) cudaError_t_local cudaArrayGetInfo() { STUB_LOG("cudaArrayGetInfo"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaArrayGetMemoryRequirements() { STUB_LOG("cudaArrayGetMemoryRequirements"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaArrayGetPlane() { STUB_LOG("cudaArrayGetPlane"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaArrayGetSparseProperties() { STUB_LOG("cudaArrayGetSparseProperties"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaCreateSurfaceObject() { STUB_LOG("cudaCreateSurfaceObject"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaCtxResetPersistingL2Cache() { STUB_LOG("cudaCtxResetPersistingL2Cache"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDestroyExternalMemory() { STUB_LOG("cudaDestroyExternalMemory"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDestroyExternalSemaphore() { STUB_LOG("cudaDestroyExternalSemaphore"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDestroySurfaceObject() { STUB_LOG("cudaDestroySurfaceObject"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDestroyTextureObject() { STUB_LOG("cudaDestroyTextureObject"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceFlushGPUDirectRDMAWrites() { STUB_LOG("cudaDeviceFlushGPUDirectRDMAWrites"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetByPCIBusId() { STUB_LOG("cudaDeviceGetByPCIBusId"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetCacheConfig() { STUB_LOG("cudaDeviceGetCacheConfig"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetGraphMemAttribute() { STUB_LOG("cudaDeviceGetGraphMemAttribute"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetLimit() { STUB_LOG("cudaDeviceGetLimit"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetMemPool() { STUB_LOG("cudaDeviceGetMemPool"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetNvSciSyncAttributes() { STUB_LOG("cudaDeviceGetNvSciSyncAttributes"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetP2PAttribute() { STUB_LOG("cudaDeviceGetP2PAttribute"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetSharedMemConfig() { STUB_LOG("cudaDeviceGetSharedMemConfig"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGetTexture1DLinearMaxWidth() { STUB_LOG("cudaDeviceGetTexture1DLinearMaxWidth"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceGraphMemTrim() { STUB_LOG("cudaDeviceGraphMemTrim"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceRegisterAsyncNotification() { STUB_LOG("cudaDeviceRegisterAsyncNotification"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceSetGraphMemAttribute() { STUB_LOG("cudaDeviceSetGraphMemAttribute"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceSetMemPool() { STUB_LOG("cudaDeviceSetMemPool"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceSetSharedMemConfig() { STUB_LOG("cudaDeviceSetSharedMemConfig"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaDeviceUnregisterAsyncNotification() { STUB_LOG("cudaDeviceUnregisterAsyncNotification"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamConsumerAcquireFrame() { STUB_LOG("cudaEGLStreamConsumerAcquireFrame"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamConsumerConnect() { STUB_LOG("cudaEGLStreamConsumerConnect"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamConsumerConnectWithFlags() { STUB_LOG("cudaEGLStreamConsumerConnectWithFlags"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamConsumerDisconnect() { STUB_LOG("cudaEGLStreamConsumerDisconnect"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamConsumerReleaseFrame() { STUB_LOG("cudaEGLStreamConsumerReleaseFrame"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamProducerConnect() { STUB_LOG("cudaEGLStreamProducerConnect"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamProducerDisconnect() { STUB_LOG("cudaEGLStreamProducerDisconnect"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamProducerPresentFrame() { STUB_LOG("cudaEGLStreamProducerPresentFrame"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEGLStreamProducerReturnFrame() { STUB_LOG("cudaEGLStreamProducerReturnFrame"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaEventCreateFromEGLSync() { STUB_LOG("cudaEventCreateFromEGLSync"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaExternalMemoryGetMappedBuffer() { STUB_LOG("cudaExternalMemoryGetMappedBuffer"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaExternalMemoryGetMappedMipmappedArray() { STUB_LOG("cudaExternalMemoryGetMappedMipmappedArray"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaFreeMipmappedArray() { STUB_LOG("cudaFreeMipmappedArray"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaFuncGetName() { STUB_LOG("cudaFuncGetName"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaFuncGetParamInfo() { STUB_LOG("cudaFuncGetParamInfo"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaFuncSetSharedMemConfig() { STUB_LOG("cudaFuncSetSharedMemConfig"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGLGetDevices() { STUB_LOG("cudaGLGetDevices"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetDeviceFlags() { STUB_LOG("cudaGetDeviceFlags"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetDeviceProperties() { STUB_LOG("cudaGetDeviceProperties"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetDriverEntryPointByVersion_ptsz() { STUB_LOG("cudaGetDriverEntryPointByVersion_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetDriverEntryPoint_ptsz() { STUB_LOG("cudaGetDriverEntryPoint_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetExportTable() { STUB_LOG("cudaGetExportTable"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetFuncBySymbol() { STUB_LOG("cudaGetFuncBySymbol"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetKernel() { STUB_LOG("cudaGetKernel"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetMipmappedArrayLevel() { STUB_LOG("cudaGetMipmappedArrayLevel"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetSurfaceObjectResourceDesc() { STUB_LOG("cudaGetSurfaceObjectResourceDesc"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetTextureObjectResourceDesc() { STUB_LOG("cudaGetTextureObjectResourceDesc"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetTextureObjectResourceViewDesc() { STUB_LOG("cudaGetTextureObjectResourceViewDesc"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGetTextureObjectTextureDesc() { STUB_LOG("cudaGetTextureObjectTextureDesc"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddChildGraphNode() { STUB_LOG("cudaGraphAddChildGraphNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddDependencies() { STUB_LOG("cudaGraphAddDependencies"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddDependencies_v2() { STUB_LOG("cudaGraphAddDependencies_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddEmptyNode() { STUB_LOG("cudaGraphAddEmptyNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddEventRecordNode() { STUB_LOG("cudaGraphAddEventRecordNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddEventWaitNode() { STUB_LOG("cudaGraphAddEventWaitNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddExternalSemaphoresSignalNode() { STUB_LOG("cudaGraphAddExternalSemaphoresSignalNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddExternalSemaphoresWaitNode() { STUB_LOG("cudaGraphAddExternalSemaphoresWaitNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddHostNode() { STUB_LOG("cudaGraphAddHostNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddKernelNode() { STUB_LOG("cudaGraphAddKernelNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddMemAllocNode() { STUB_LOG("cudaGraphAddMemAllocNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddMemFreeNode() { STUB_LOG("cudaGraphAddMemFreeNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddMemcpyNode() { STUB_LOG("cudaGraphAddMemcpyNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddMemcpyNode1D() { STUB_LOG("cudaGraphAddMemcpyNode1D"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddMemcpyNodeFromSymbol() { STUB_LOG("cudaGraphAddMemcpyNodeFromSymbol"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddMemcpyNodeToSymbol() { STUB_LOG("cudaGraphAddMemcpyNodeToSymbol"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddMemsetNode() { STUB_LOG("cudaGraphAddMemsetNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddNode() { STUB_LOG("cudaGraphAddNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphAddNode_v2() { STUB_LOG("cudaGraphAddNode_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphChildGraphNodeGetGraph() { STUB_LOG("cudaGraphChildGraphNodeGetGraph"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphClone() { STUB_LOG("cudaGraphClone"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphConditionalHandleCreate() { STUB_LOG("cudaGraphConditionalHandleCreate"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphDestroyNode() { STUB_LOG("cudaGraphDestroyNode"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphEventRecordNodeGetEvent() { STUB_LOG("cudaGraphEventRecordNodeGetEvent"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphEventRecordNodeSetEvent() { STUB_LOG("cudaGraphEventRecordNodeSetEvent"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphEventWaitNodeGetEvent() { STUB_LOG("cudaGraphEventWaitNodeGetEvent"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphEventWaitNodeSetEvent() { STUB_LOG("cudaGraphEventWaitNodeSetEvent"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecChildGraphNodeSetParams() { STUB_LOG("cudaGraphExecChildGraphNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecEventRecordNodeSetEvent() { STUB_LOG("cudaGraphExecEventRecordNodeSetEvent"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecEventWaitNodeSetEvent() { STUB_LOG("cudaGraphExecEventWaitNodeSetEvent"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecExternalSemaphoresSignalNodeSetParams() { STUB_LOG("cudaGraphExecExternalSemaphoresSignalNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecExternalSemaphoresWaitNodeSetParams() { STUB_LOG("cudaGraphExecExternalSemaphoresWaitNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecGetFlags() { STUB_LOG("cudaGraphExecGetFlags"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecHostNodeSetParams() { STUB_LOG("cudaGraphExecHostNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecKernelNodeSetParams() { STUB_LOG("cudaGraphExecKernelNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecMemcpyNodeSetParams() { STUB_LOG("cudaGraphExecMemcpyNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecMemcpyNodeSetParams1D() { STUB_LOG("cudaGraphExecMemcpyNodeSetParams1D"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecMemcpyNodeSetParamsFromSymbol() { STUB_LOG("cudaGraphExecMemcpyNodeSetParamsFromSymbol"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecMemcpyNodeSetParamsToSymbol() { STUB_LOG("cudaGraphExecMemcpyNodeSetParamsToSymbol"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecMemsetNodeSetParams() { STUB_LOG("cudaGraphExecMemsetNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExecNodeSetParams() { STUB_LOG("cudaGraphExecNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExternalSemaphoresSignalNodeGetParams() { STUB_LOG("cudaGraphExternalSemaphoresSignalNodeGetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExternalSemaphoresSignalNodeSetParams() { STUB_LOG("cudaGraphExternalSemaphoresSignalNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExternalSemaphoresWaitNodeGetParams() { STUB_LOG("cudaGraphExternalSemaphoresWaitNodeGetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphExternalSemaphoresWaitNodeSetParams() { STUB_LOG("cudaGraphExternalSemaphoresWaitNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphGetEdges() { STUB_LOG("cudaGraphGetEdges"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphGetEdges_v2() { STUB_LOG("cudaGraphGetEdges_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphGetRootNodes() { STUB_LOG("cudaGraphGetRootNodes"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphHostNodeGetParams() { STUB_LOG("cudaGraphHostNodeGetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphHostNodeSetParams() { STUB_LOG("cudaGraphHostNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphInstantiateWithParams() { STUB_LOG("cudaGraphInstantiateWithParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphInstantiateWithParams_ptsz() { STUB_LOG("cudaGraphInstantiateWithParams_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphKernelNodeCopyAttributes() { STUB_LOG("cudaGraphKernelNodeCopyAttributes"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphKernelNodeGetAttribute() { STUB_LOG("cudaGraphKernelNodeGetAttribute"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphKernelNodeGetParams() { STUB_LOG("cudaGraphKernelNodeGetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphKernelNodeSetAttribute() { STUB_LOG("cudaGraphKernelNodeSetAttribute"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphKernelNodeSetParams() { STUB_LOG("cudaGraphKernelNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemAllocNodeGetParams() { STUB_LOG("cudaGraphMemAllocNodeGetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemFreeNodeGetParams() { STUB_LOG("cudaGraphMemFreeNodeGetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemcpyNodeGetParams() { STUB_LOG("cudaGraphMemcpyNodeGetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemcpyNodeSetParams() { STUB_LOG("cudaGraphMemcpyNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemcpyNodeSetParams1D() { STUB_LOG("cudaGraphMemcpyNodeSetParams1D"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemcpyNodeSetParamsFromSymbol() { STUB_LOG("cudaGraphMemcpyNodeSetParamsFromSymbol"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemcpyNodeSetParamsToSymbol() { STUB_LOG("cudaGraphMemcpyNodeSetParamsToSymbol"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemsetNodeGetParams() { STUB_LOG("cudaGraphMemsetNodeGetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphMemsetNodeSetParams() { STUB_LOG("cudaGraphMemsetNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeFindInClone() { STUB_LOG("cudaGraphNodeFindInClone"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeGetDependencies() { STUB_LOG("cudaGraphNodeGetDependencies"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeGetDependencies_v2() { STUB_LOG("cudaGraphNodeGetDependencies_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeGetDependentNodes() { STUB_LOG("cudaGraphNodeGetDependentNodes"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeGetDependentNodes_v2() { STUB_LOG("cudaGraphNodeGetDependentNodes_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeGetEnabled() { STUB_LOG("cudaGraphNodeGetEnabled"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeGetType() { STUB_LOG("cudaGraphNodeGetType"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeSetEnabled() { STUB_LOG("cudaGraphNodeSetEnabled"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphNodeSetParams() { STUB_LOG("cudaGraphNodeSetParams"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphReleaseUserObject() { STUB_LOG("cudaGraphReleaseUserObject"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphRemoveDependencies() { STUB_LOG("cudaGraphRemoveDependencies"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphRemoveDependencies_v2() { STUB_LOG("cudaGraphRemoveDependencies_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphRetainUserObject() { STUB_LOG("cudaGraphRetainUserObject"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphicsEGLRegisterImage() { STUB_LOG("cudaGraphicsEGLRegisterImage"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphicsGLRegisterImage() { STUB_LOG("cudaGraphicsGLRegisterImage"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphicsResourceGetMappedEglFrame() { STUB_LOG("cudaGraphicsResourceGetMappedEglFrame"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphicsResourceGetMappedMipmappedArray() { STUB_LOG("cudaGraphicsResourceGetMappedMipmappedArray"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphicsSubResourceGetMappedArray() { STUB_LOG("cudaGraphicsSubResourceGetMappedArray"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphicsVDPAURegisterOutputSurface() { STUB_LOG("cudaGraphicsVDPAURegisterOutputSurface"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaGraphicsVDPAURegisterVideoSurface() { STUB_LOG("cudaGraphicsVDPAURegisterVideoSurface"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaImportExternalMemory() { STUB_LOG("cudaImportExternalMemory"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaImportExternalSemaphore() { STUB_LOG("cudaImportExternalSemaphore"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaInitDevice() { STUB_LOG("cudaInitDevice"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaLaunchCooperativeKernel() { STUB_LOG("cudaLaunchCooperativeKernel"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaLaunchCooperativeKernelMultiDevice() { STUB_LOG("cudaLaunchCooperativeKernelMultiDevice"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaLaunchCooperativeKernel_ptsz() { STUB_LOG("cudaLaunchCooperativeKernel_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaLaunchHostFunc_ptsz() { STUB_LOG("cudaLaunchHostFunc_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaLaunchKernelExC_ptsz() { STUB_LOG("cudaLaunchKernelExC_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMallocFromPoolAsync() { STUB_LOG("cudaMallocFromPoolAsync"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMallocFromPoolAsync_ptsz() { STUB_LOG("cudaMallocFromPoolAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMallocMipmappedArray() { STUB_LOG("cudaMallocMipmappedArray"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemAdvise() { STUB_LOG("cudaMemAdvise"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemAdvise_v2() { STUB_LOG("cudaMemAdvise_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPoolExportPointer() { STUB_LOG("cudaMemPoolExportPointer"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPoolExportToShareableHandle() { STUB_LOG("cudaMemPoolExportToShareableHandle"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPoolGetAccess() { STUB_LOG("cudaMemPoolGetAccess"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPoolImportFromShareableHandle() { STUB_LOG("cudaMemPoolImportFromShareableHandle"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPoolImportPointer() { STUB_LOG("cudaMemPoolImportPointer"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPrefetchAsync() { STUB_LOG("cudaMemPrefetchAsync"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPrefetchAsync_ptsz() { STUB_LOG("cudaMemPrefetchAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPrefetchAsync_v2() { STUB_LOG("cudaMemPrefetchAsync_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemPrefetchAsync_v2_ptsz() { STUB_LOG("cudaMemPrefetchAsync_v2_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemRangeGetAttribute() { STUB_LOG("cudaMemRangeGetAttribute"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemRangeGetAttributes() { STUB_LOG("cudaMemRangeGetAttributes"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy2DArrayToArray_ptds() { STUB_LOG("cudaMemcpy2DArrayToArray_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy2DFromArrayAsync_ptsz() { STUB_LOG("cudaMemcpy2DFromArrayAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy2DFromArray_ptds() { STUB_LOG("cudaMemcpy2DFromArray_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy2DToArrayAsync_ptsz() { STUB_LOG("cudaMemcpy2DToArrayAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy2DToArray_ptds() { STUB_LOG("cudaMemcpy2DToArray_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
// RETIRADO 2026-08-03: implementado de verdad en CudaRt_ptsz.cpp. Lo
// encontro tests/semantic/semantic_conformance.cu: el brazo ptsz fallaba las
// cuatro propiedades de memoria SINCRONA con cudaErrorNotSupported (71)
// mientras nativo y el brazo handle pasaban 7/7.
// __attribute__((visibility("default"))) cudaError_t_local cudaMemcpy2D_ptds() { STUB_LOG("cudaMemcpy2D_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy3DPeer() { STUB_LOG("cudaMemcpy3DPeer"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy3DPeerAsync() { STUB_LOG("cudaMemcpy3DPeerAsync"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy3DPeerAsync_ptsz() { STUB_LOG("cudaMemcpy3DPeerAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpy3DPeer_ptds() { STUB_LOG("cudaMemcpy3DPeer_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
// RETIRADO 2026-08-03: implementado de verdad en CudaRt_ptsz.cpp. Lo
// encontro tests/semantic/semantic_conformance.cu: el brazo ptsz fallaba las
// cuatro propiedades de memoria SINCRONA con cudaErrorNotSupported (71)
// mientras nativo y el brazo handle pasaban 7/7.
// __attribute__((visibility("default"))) cudaError_t_local cudaMemcpy3D_ptds() { STUB_LOG("cudaMemcpy3D_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpyArrayToArray_ptds() { STUB_LOG("cudaMemcpyArrayToArray_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpyFromArrayAsync_ptsz() { STUB_LOG("cudaMemcpyFromArrayAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpyFromArray_ptds() { STUB_LOG("cudaMemcpyFromArray_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
// RETIRADO 2026-08-03: implementado de verdad en CudaRt_ptsz.cpp. Lo
// encontro tests/semantic/semantic_conformance.cu: el brazo ptsz fallaba las
// cuatro propiedades de memoria SINCRONA con cudaErrorNotSupported (71)
// mientras nativo y el brazo handle pasaban 7/7.
// __attribute__((visibility("default"))) cudaError_t_local cudaMemcpyFromSymbol_ptds() { STUB_LOG("cudaMemcpyFromSymbol_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpyPeer() { STUB_LOG("cudaMemcpyPeer"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpyToArrayAsync_ptsz() { STUB_LOG("cudaMemcpyToArrayAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemcpyToArray_ptds() { STUB_LOG("cudaMemcpyToArray_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
// RETIRADO 2026-08-03: implementado de verdad en CudaRt_ptsz.cpp. Lo
// encontro tests/semantic/semantic_conformance.cu: el brazo ptsz fallaba las
// cuatro propiedades de memoria SINCRONA con cudaErrorNotSupported (71)
// mientras nativo y el brazo handle pasaban 7/7.
// __attribute__((visibility("default"))) cudaError_t_local cudaMemcpyToSymbol_ptds() { STUB_LOG("cudaMemcpyToSymbol_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
// RETIRADO 2026-08-03: implementado de verdad en CudaRt_ptsz.cpp. Lo
// encontro tests/semantic/semantic_conformance.cu: el brazo ptsz fallaba las
// cuatro propiedades de memoria SINCRONA con cudaErrorNotSupported (71)
// mientras nativo y el brazo handle pasaban 7/7.
// __attribute__((visibility("default"))) cudaError_t_local cudaMemcpy_ptds() { STUB_LOG("cudaMemcpy_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
// RETIRADO 2026-08-03: implementado de verdad en CudaRt_ptsz.cpp. Lo
// encontro tests/semantic/semantic_conformance.cu: el brazo ptsz fallaba las
// cuatro propiedades de memoria SINCRONA con cudaErrorNotSupported (71)
// mientras nativo y el brazo handle pasaban 7/7.
// __attribute__((visibility("default"))) cudaError_t_local cudaMemset2D_ptds() { STUB_LOG("cudaMemset2D_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemset3DAsync() { STUB_LOG("cudaMemset3DAsync"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMemset3DAsync_ptsz() { STUB_LOG("cudaMemset3DAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
// RETIRADO 2026-08-03: implementado de verdad en CudaRt_ptsz.cpp. Lo
// encontro tests/semantic/semantic_conformance.cu: el brazo ptsz fallaba las
// cuatro propiedades de memoria SINCRONA con cudaErrorNotSupported (71)
// mientras nativo y el brazo handle pasaban 7/7.
// __attribute__((visibility("default"))) cudaError_t_local cudaMemset3D_ptds() { STUB_LOG("cudaMemset3D_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
// RETIRADO 2026-08-03: implementado de verdad en CudaRt_ptsz.cpp. Lo
// encontro tests/semantic/semantic_conformance.cu: el brazo ptsz fallaba las
// cuatro propiedades de memoria SINCRONA con cudaErrorNotSupported (71)
// mientras nativo y el brazo handle pasaban 7/7.
// __attribute__((visibility("default"))) cudaError_t_local cudaMemset_ptds() { STUB_LOG("cudaMemset_ptds"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMipmappedArrayGetMemoryRequirements() { STUB_LOG("cudaMipmappedArrayGetMemoryRequirements"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaMipmappedArrayGetSparseProperties() { STUB_LOG("cudaMipmappedArrayGetSparseProperties"); return CUDART_STUB_NOT_SUPPORTED; }
// cudaOccupancyAvailableDynamicSMemPerBlock: real impl in CudaRt_occupancy.cpp
__attribute__((visibility("default"))) cudaError_t_local cudaOccupancyMaxActiveClusters() { STUB_LOG("cudaOccupancyMaxActiveClusters"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaOccupancyMaxPotentialClusterSize() { STUB_LOG("cudaOccupancyMaxPotentialClusterSize"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaSignalExternalSemaphoresAsync() { STUB_LOG("cudaSignalExternalSemaphoresAsync"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaSignalExternalSemaphoresAsync_ptsz() { STUB_LOG("cudaSignalExternalSemaphoresAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaSignalExternalSemaphoresAsync_v2() { STUB_LOG("cudaSignalExternalSemaphoresAsync_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaSignalExternalSemaphoresAsync_v2_ptsz() { STUB_LOG("cudaSignalExternalSemaphoresAsync_v2_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamAttachMemAsync() { STUB_LOG("cudaStreamAttachMemAsync"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamAttachMemAsync_ptsz() { STUB_LOG("cudaStreamAttachMemAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamBeginCaptureToGraph() { STUB_LOG("cudaStreamBeginCaptureToGraph"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamBeginCaptureToGraph_ptsz() { STUB_LOG("cudaStreamBeginCaptureToGraph_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamCopyAttributes() { STUB_LOG("cudaStreamCopyAttributes"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamCopyAttributes_ptsz() { STUB_LOG("cudaStreamCopyAttributes_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetAttribute() { STUB_LOG("cudaStreamGetAttribute"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetAttribute_ptsz() { STUB_LOG("cudaStreamGetAttribute_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetCaptureInfo() { STUB_LOG("cudaStreamGetCaptureInfo"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetCaptureInfo_ptsz() { STUB_LOG("cudaStreamGetCaptureInfo_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetCaptureInfo_v2_ptsz() { STUB_LOG("cudaStreamGetCaptureInfo_v2_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetCaptureInfo_v3() { STUB_LOG("cudaStreamGetCaptureInfo_v3"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetCaptureInfo_v3_ptsz() { STUB_LOG("cudaStreamGetCaptureInfo_v3_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetFlags() { STUB_LOG("cudaStreamGetFlags"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetFlags_ptsz() { STUB_LOG("cudaStreamGetFlags_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetId() { STUB_LOG("cudaStreamGetId"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamGetId_ptsz() { STUB_LOG("cudaStreamGetId_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamSetAttribute() { STUB_LOG("cudaStreamSetAttribute"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamSetAttribute_ptsz() { STUB_LOG("cudaStreamSetAttribute_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamUpdateCaptureDependencies() { STUB_LOG("cudaStreamUpdateCaptureDependencies"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamUpdateCaptureDependencies_ptsz() { STUB_LOG("cudaStreamUpdateCaptureDependencies_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamUpdateCaptureDependencies_v2() { STUB_LOG("cudaStreamUpdateCaptureDependencies_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaStreamUpdateCaptureDependencies_v2_ptsz() { STUB_LOG("cudaStreamUpdateCaptureDependencies_v2_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaThreadGetCacheConfig() { STUB_LOG("cudaThreadGetCacheConfig"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaThreadGetLimit() { STUB_LOG("cudaThreadGetLimit"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaThreadSetCacheConfig() { STUB_LOG("cudaThreadSetCacheConfig"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaThreadSetLimit() { STUB_LOG("cudaThreadSetLimit"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaUserObjectCreate() { STUB_LOG("cudaUserObjectCreate"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaUserObjectRelease() { STUB_LOG("cudaUserObjectRelease"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaUserObjectRetain() { STUB_LOG("cudaUserObjectRetain"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaVDPAUGetDevice() { STUB_LOG("cudaVDPAUGetDevice"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaVDPAUSetVDPAUDevice() { STUB_LOG("cudaVDPAUSetVDPAUDevice"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaWaitExternalSemaphoresAsync() { STUB_LOG("cudaWaitExternalSemaphoresAsync"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaWaitExternalSemaphoresAsync_ptsz() { STUB_LOG("cudaWaitExternalSemaphoresAsync_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaWaitExternalSemaphoresAsync_v2() { STUB_LOG("cudaWaitExternalSemaphoresAsync_v2"); return CUDART_STUB_NOT_SUPPORTED; }
__attribute__((visibility("default"))) cudaError_t_local cudaWaitExternalSemaphoresAsync_v2_ptsz() { STUB_LOG("cudaWaitExternalSemaphoresAsync_v2_ptsz"); return CUDART_STUB_NOT_SUPPORTED; }

}
