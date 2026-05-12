#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cublas_v2.h>

using clk = std::chrono::steady_clock;

static long ms(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

int main()
{
    const char *env_n = std::getenv("MATRIX_N");
    int N = env_n ? std::atoi(env_n) : 512;
    if (N <= 0)
        N = 512;

    std::cerr << "BENCHMARK_MATRIX_N=" << N << std::endl;

    size_t elems = (size_t)N * N;
    size_t bytes = elems * sizeof(float);

    // ── malloc ────────────────────────────────────────────────────────────────
    auto t0 = clk::now();
    float *h_A = new float[elems];
    float *h_B = new float[elems];
    float *h_C = new float[elems];
    for (size_t i = 0; i < elems; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }
    auto t1 = clk::now();
    std::cerr << "STAGE_MALLOC_MS=" << ms(t0, t1) << std::endl;

    // ── cudaMalloc ────────────────────────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    auto tca = clk::now();
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);
    auto tcb = clk::now();
    std::cerr << "STAGE_CUDAMALLOC_MS=" << ms(tca, tcb) << std::endl;

    // ── H2D ──────────────────────────────────────────────────────────────────
    auto t2 = clk::now();
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t3 = clk::now();
    std::cerr << "STAGE_H2D_MS=" << ms(t2, t3) << std::endl;

    // ── cublasCreate ─────────────────────────────────────────────────────────
    cublasHandle_t handle;
    auto t4 = clk::now();
    cublasCreate(&handle);
    auto t5 = clk::now();
    std::cerr << "STAGE_CUBLAS_CREATE_MS=" << ms(t4, t5) << std::endl;

    // ── SGEMM: C = A * B ─────────────────────────────────────────────────────
    const float alpha = 1.0f, beta = 0.0f;
    auto t6 = clk::now();
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha, d_A, N, d_B, N,
                &beta, d_C, N);
    cudaDeviceSynchronize();
    auto t7 = clk::now();
    std::cerr << "STAGE_GEMM_MS=" << ms(t6, t7) << std::endl;

    // ── D2H ──────────────────────────────────────────────────────────────────
    auto t8 = clk::now();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto t9 = clk::now();
    std::cerr << "STAGE_D2H_MS=" << ms(t8, t9) << std::endl;

    // ── result check ─────────────────────────────────────────────────────────
    // h_C[0] should equal N (sum of N ones). Must be BEFORE cleanup/delete[].
    float expected = (float)N;
    bool pass = std::abs(h_C[0] - expected) < 1.0f;
    std::cerr << "RESULT_CHECK=" << (pass ? "PASS" : "FAIL")
              << " (got=" << h_C[0] << " expected=" << expected << ")" << std::endl;

    // ── cleanup ───────────────────────────────────────────────────────────────
    auto t10 = clk::now();
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    auto t11 = clk::now();
    std::cerr << "STAGE_CLEANUP_MS=" << ms(t10, t11) << std::endl;

    // ── total ─────────────────────────────────────────────────────────────────
    long total = ms(t0, t11);
    std::cerr << "BENCHMARK_RESULT_MS=" << total << std::endl;

    std::cout << "OK N=" << N << " total_ms=" << total << std::endl;
    return 0;
}