#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(GraphicsUnregisterResource) {
    return std::make_shared<Result>(cudaErrorNotSupported);
}
CUDA_ROUTINE_HANDLER(GraphicsMapResources) {
    return std::make_shared<Result>(cudaErrorNotSupported);
}
CUDA_ROUTINE_HANDLER(GraphicsUnmapResources) {
    return std::make_shared<Result>(cudaErrorNotSupported);
}
CUDA_ROUTINE_HANDLER(GraphicsResourceGetMappedPointer) {
    return std::make_shared<Result>(cudaErrorNotSupported);
}
CUDA_ROUTINE_HANDLER(GraphicsSubResourceGetMappedArray) {
    return std::make_shared<Result>(cudaErrorNotSupported);
}
CUDA_ROUTINE_HANDLER(GLSetGLDevice) { return std::make_shared<Result>(cudaErrorNotSupported); }
CUDA_ROUTINE_HANDLER(GraphicsGLRegisterBuffer) { return std::make_shared<Result>(cudaErrorNotSupported); }
CUDA_ROUTINE_HANDLER(GraphicsResourceSetMapFlags) { return std::make_shared<Result>(cudaErrorNotSupported); }
