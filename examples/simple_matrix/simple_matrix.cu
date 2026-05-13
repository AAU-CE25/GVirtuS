#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    const int N = 2048;
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f);
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    cublasHandle_t handle;
    cublasCreate(&handle);

    auto t0 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    auto t1 = std::chrono::high_resolution_clock::now();

    volatile long long waste = 0;
    for (long long i = 0; i < 500000000LL; ++i) {
        waste += i;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A, N,
                d_B, N,
                &beta,
                d_C, N);

    auto t3 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "C[0] = " << C[0] << std::endl;

    auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };

    std::cout << "Timing us - h2d: " << us(t0, t1)
              << ", cpu: " << us(t1, t2)
              << ", gemm: " << us(t2, t3)
              << ", d2h: " << us(t3, t4)
              << ", total: " << us(t0, t4)
              << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}