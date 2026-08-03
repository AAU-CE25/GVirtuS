#include <cstdio>
#include <cuda.h>
int main(){
    printf("ATTR_GPU_DIRECT_RDMA_SUPPORTED        = %d\n", (int)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED);
    printf("ATTR_GPU_DIRECT_RDMA_FLUSH_OPTIONS    = %d\n", (int)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS);
    printf("ATTR_GPU_DIRECT_RDMA_WRITES_ORDERING  = %d\n", (int)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING);
    printf("FLUSH_OPTION_HOST                     = %d\n", (int)CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST);
    printf("FLUSH_OPTION_MEMOPS                   = %d\n", (int)CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS);
    printf("ORDERING_NONE                         = %d\n", (int)CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE);
    printf("ORDERING_OWNER                        = %d\n", (int)CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER);
    printf("ORDERING_ALL_DEVICES                  = %d\n", (int)CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES);
    printf("FLUSH_TARGET_CURRENT_CTX              = %d\n", (int)CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX);
    printf("FLUSH_TO_OWNER                        = %d\n", (int)CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
    printf("FLUSH_TO_ALL_DEVICES                  = %d\n", (int)CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES);
    return 0;
}
