#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <cublas_v2.h>

using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void run_benchmark(cublasHandle_t handle, int N, bool quiet = false) {
    size_t elems = (size_t)N * N;
    size_t bytes = elems * sizeof(float);

    // Allocate and initialize host matrices
    float *h_A = new float[elems];
    float *h_B = new float[elems];
    float *h_C = new float[elems];
    for (size_t i = 0; i < elems; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    memset(h_C, 0, bytes);

    float *d_A, *d_B, *d_C;

    // Allocate device memory
    auto t_alloc_start = clk::now();
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);
    auto t_alloc_end = clk::now();

    // H2D transfer
    auto t_h2d_start = clk::now();
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    auto t_h2d_end = clk::now();

    // SGEMM: C = A * B
    const float alpha = 1.0f, beta = 0.0f;
    auto t_gemm_start = clk::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha, d_A, N, d_B, N,
                &beta, d_C, N);
    auto t_gemm_end = clk::now();

    // D2H transfer
    auto t_d2h_start = clk::now();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    auto t_d2h_end = clk::now();

    // Cleanup device
    auto t_free_start = clk::now();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    auto t_free_end = clk::now();

    double alloc_ms = ms(t_alloc_start, t_alloc_end);
    double h2d_ms = ms(t_h2d_start, t_h2d_end);
    double gemm_ms = ms(t_gemm_start, t_gemm_end);
    double d2h_ms = ms(t_d2h_start, t_d2h_end);
    double free_ms = ms(t_free_start, t_free_end);
    double total_ms = alloc_ms + h2d_ms + gemm_ms + d2h_ms + free_ms;

    if (!quiet) {
        std::cout << N
                  << "," << alloc_ms
                  << "," << h2d_ms
                  << "," << gemm_ms
                  << "," << d2h_ms
                  << "," << free_ms
                  << "," << total_ms
                  << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    std::vector<int> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};

    // Create cuBLAS handle once
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warmup run
    std::cerr << "Warmup..." << std::flush;
    run_benchmark(handle, 32, true);
    std::cerr << " done" << std::endl;

    // Print CSV header
    std::cout << "N,alloc_ms,h2d_ms,gemm_ms,d2h_ms,free_ms,total_ms" << std::endl;

    for (int N : sizes) {
        run_benchmark(handle, N);
    }

    cublasDestroy(handle);
    return 0;
}
