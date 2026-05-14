/**
 * Test 2: Matrix Multiplication Benchmark (cuBLAS via GVirtuS)
 *
 * Performs C = A * B where A is NxN and B is NxN using cublasSgemm.
 * Measures end-to-end time including memory transfers and compute.
 *
 * Usage: ./matrix_mul_bench [N] [num_runs]
 *   N        — matrix dimension (NxN), default: 512
 *   num_runs — number of repetitions, default: 10
 *
 * Output: prints CSV lines to stdout
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static void fill_random(std::vector<float> &v) {
    for (auto &x : v) {
        x = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char *argv[]) {
    int N = 512;
    int num_runs = 10;

    if (argc > 1) N = std::stoi(argv[1]);
    if (argc > 2) num_runs = std::stoi(argv[2]);

    size_t matrix_size = static_cast<size_t>(N) * N;
    size_t bytes = matrix_size * sizeof(float);

    // Host allocations
    std::vector<float> h_A(matrix_size);
    std::vector<float> h_B(matrix_size);
    std::vector<float> h_C(matrix_size, 0.0f);

    fill_random(h_A);
    fill_random(h_B);

    // Device allocations
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warmup
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // CSV header
    std::cout << "run,n,matrix_bytes,h2d_us,compute_us,d2h_us,total_us" << std::endl;

    for (int run = 0; run < num_runs; run++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Host to Device
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Compute: C = A * B
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta,
                    d_C, N);
        auto t2 = std::chrono::high_resolution_clock::now();

        // Device to Host
        cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
        auto t3 = std::chrono::high_resolution_clock::now();

        double h2d_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        double compute_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        double d2h_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        double total_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0).count();

        std::cout << run << "," << N << "," << bytes << "," << h2d_us << "," << compute_us << ","
                  << d2h_us << "," << total_us << std::endl;
    }

    // Quick sanity check: C[0][0] should be non-zero
    if (h_C[0] == 0.0f && N > 1) {
        std::cerr << "WARNING: C[0][0] is zero, computation may have failed" << std::endl;
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
