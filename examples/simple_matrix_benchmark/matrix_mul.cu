#include <iostream>
#include <cublas_v2.h>

int main() {
    // Matrix A (2x3)
    float A[6] = {1, 2, 3, 4, 5, 6};
    // Matrix B (3x2)
    float B[6] = {7, 8, 9, 10, 11, 12};
    // Matrix C (2x2), the result
    float C[4] = {0};

    float *d_A, *d_B, *d_C;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate device memory
    cudaMalloc((void **)&d_A, 6 * sizeof(float));
    cudaMalloc((void **)&d_B, 6 * sizeof(float));
    cudaMalloc((void **)&d_C, 4 * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 6 * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // A: 2x3, B: 3x2, C: 2x2
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 3, &alpha, d_A, 2, d_B, 3, &beta, d_C, 2);

    // Copy result back to host
    cudaMemcpy(C, d_C, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Matrix C: ";
    for (int i = 0; i < 4; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free resources
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
