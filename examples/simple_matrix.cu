#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

static int getenv_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    return std::atoi(value);
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err == cudaSuccess) return;
    std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(1);
}

static void check_cublas(cublasStatus_t status, const char* msg) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    std::cerr << "cuBLAS error: " << msg << " (status " << static_cast<int>(status) << ")" << std::endl;
    std::exit(1);
}

static double elapsed_us(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
               end - start
           ).count();
}

static double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
               end - start
           ).count();
}

int main(int argc, char** argv) {
    int n = getenv_int("MATRIX_SIZE", 256);
    int iters = getenv_int("ITERATIONS", 10);
    int warmup = getenv_int("WARMUP", 1);

    const char* alloc_mode_env = std::getenv("ALLOCATION_MODE");
    const std::string alloc_mode = alloc_mode_env != nullptr ? alloc_mode_env : "reuse";
    const bool per_iter_alloc = (alloc_mode == "per_iter" ||
                                 alloc_mode == "per-iter" ||
                                 alloc_mode == "alloc");

    const bool print_per_iter = getenv_int("PRINT_MEM_PER_ITER", 1) != 0;

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) iters = std::atoi(argv[2]);
    if (argc > 3) warmup = std::atoi(argv[3]);

    if (n <= 0 || iters <= 0 || warmup < 0) {
        std::cerr << "Usage: " << argv[0] << " [N] [ITERS] [WARMUP]" << std::endl;
        return 1;
    }

    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = elems * sizeof(float);
    const double bytes_gb = static_cast<double>(bytes) / 1e9;
    const double h2d_bytes_gb = static_cast<double>(2 * bytes) / 1e9;
    const double d2h_bytes_gb = static_cast<double>(bytes) / 1e9;

    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto alloc_device = [&]() {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes), "cudaMalloc d_A");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes), "cudaMalloc d_B");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes), "cudaMalloc d_C");
    };

    auto free_device = [&]() {
        if (d_A != nullptr) check_cuda(cudaFree(d_A), "cudaFree d_A");
        if (d_B != nullptr) check_cuda(cudaFree(d_B), "cudaFree d_B");
        if (d_C != nullptr) check_cuda(cudaFree(d_C), "cudaFree d_C");
        d_A = nullptr;
        d_B = nullptr;
        d_C = nullptr;
    };

    check_cuda(cudaMallocHost(reinterpret_cast<void **>(&h_A), bytes), "cudaMallocHost h_A");
    check_cuda(cudaMallocHost(reinterpret_cast<void **>(&h_B), bytes), "cudaMallocHost h_B");
    check_cuda(cudaMallocHost(reinterpret_cast<void **>(&h_C), bytes), "cudaMallocHost h_C");

    for (size_t i = 0; i < elems; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    if (!per_iter_alloc) {
        alloc_device();
    }

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");

    for (int i = 0; i < warmup; ++i) {
        if (per_iter_alloc) {
            alloc_device();
        }

        check_cuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy A warmup");
        check_cuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy B warmup");

        check_cublas(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                        d_A, n, d_B, n, &beta, d_C, n),
            "cublasSgemm warmup"
        );

        check_cuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C warmup");

        if (per_iter_alloc) {
            free_device();
        }
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    float total_sgemm_ms = 0.0f;
    double host_total_ms = 0.0;
    double h2d_total_us = 0.0;
    double d2h_total_us = 0.0;

    for (int i = 0; i < iters; ++i) {
        const auto host_start = std::chrono::steady_clock::now();

        if (per_iter_alloc) {
            alloc_device();
        }

        const auto h2d_start = std::chrono::steady_clock::now();

        check_cuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
        check_cuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

        const auto h2d_end = std::chrono::steady_clock::now();

        check_cuda(cudaEventRecord(start), "cudaEventRecord start");

        check_cublas(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                        d_A, n, d_B, n, &beta, d_C, n),
            "cublasSgemm"
        );

        check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        const auto d2h_start = std::chrono::steady_clock::now();

        check_cuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");

        const auto d2h_end = std::chrono::steady_clock::now();

        if (per_iter_alloc) {
            free_device();
        }

        const auto host_end = std::chrono::steady_clock::now();

        float sgemm_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&sgemm_ms, start, stop), "cudaEventElapsedTime");

        const double h2d_us = elapsed_us(h2d_start, h2d_end);
        const double d2h_us = elapsed_us(d2h_start, d2h_end);
        const double host_ms = elapsed_ms(host_start, host_end);

        total_sgemm_ms += sgemm_ms;
        host_total_ms += host_ms;
        h2d_total_us += h2d_us;
        d2h_total_us += d2h_us;

        const double h2d_gbps = h2d_bytes_gb / (h2d_us / 1e6);
        const double d2h_gbps = d2h_bytes_gb / (d2h_us / 1e6);

        if (print_per_iter) {
            std::cout << "CSV_MEM_ITER,"
                      << n << ","
                      << i << ","
                      << bytes << ","
                      << h2d_us << ","
                      << d2h_us << ","
                      << h2d_gbps << ","
                      << d2h_gbps << ","
                      << sgemm_ms << ","
                      << host_ms
                      << std::endl;
        }
    }

    const float expected = static_cast<float>(n);
    double max_err = 0.0;

    for (size_t i = 0; i < elems; ++i) {
        const double err = std::fabs(static_cast<double>(h_C[i]) - expected);
        if (err > max_err) max_err = err;
    }

    const double tol = 1e-2 * static_cast<double>(n);
    const bool ok = max_err <= tol;

    const double avg_sgemm_ms = static_cast<double>(total_sgemm_ms) / static_cast<double>(iters);
    const double avg_host_ms = host_total_ms / static_cast<double>(iters);
    const double avg_h2d_us = h2d_total_us / static_cast<double>(iters);
    const double avg_d2h_us = d2h_total_us / static_cast<double>(iters);

    const double avg_h2d_ms = avg_h2d_us / 1000.0;
    const double avg_d2h_ms = avg_d2h_us / 1000.0;

    const double avg_h2d_gbps = h2d_bytes_gb / (avg_h2d_us / 1e6);
    const double avg_d2h_gbps = d2h_bytes_gb / (avg_d2h_us / 1e6);

    std::cout << "size=" << n
              << " iters=" << iters
              << " avg_sgemm_ms=" << avg_sgemm_ms
              << " avg_host_ms=" << avg_host_ms
              << " avg_h2d_us=" << avg_h2d_us
              << " avg_h2d_ms=" << avg_h2d_ms
              << " avg_d2h_us=" << avg_d2h_us
              << " avg_d2h_ms=" << avg_d2h_ms
              << " avg_h2d_GBps=" << avg_h2d_gbps
              << " avg_d2h_GBps=" << avg_d2h_gbps
              << " check=" << (ok ? "pass" : "fail")
              << " max_abs_err=" << max_err
              << " alloc_mode=" << alloc_mode
              << std::endl;

    std::cout << "CSV,"
              << n << ","
              << iters << ","
              << avg_sgemm_ms << ","
              << avg_host_ms << ","
              << avg_h2d_us << ","
              << avg_d2h_us << ","
              << avg_h2d_ms << ","
              << avg_d2h_ms << ","
              << avg_h2d_gbps << ","
              << avg_d2h_gbps
              << std::endl;

    std::cout << "CSV_HEADER,size,iters,avg_sgemm_ms,avg_host_ms,avg_h2d_us,avg_d2h_us,avg_h2d_ms,avg_d2h_ms,avg_h2d_GBps,avg_d2h_GBps"
              << std::endl;

    int exit_code = 0;

    if (!ok) {
        std::cerr << "Matrix check failed: expected=" << expected
                  << " max_abs_err=" << max_err
                  << " tol=" << tol << std::endl;
        exit_code = 2;
    }

    if (n <= 4) {
        std::cout << "Matrix C: ";
        for (int i = 0; i < n * n; ++i) std::cout << h_C[i] << " ";
        std::cout << std::endl;
    }

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check_cublas(cublasDestroy(handle), "cublasDestroy");

    if (!per_iter_alloc) {
        free_device();
    }

    check_cuda(cudaFreeHost(h_A), "cudaFreeHost h_A");
    check_cuda(cudaFreeHost(h_B), "cudaFreeHost h_B");
    check_cuda(cudaFreeHost(h_C), "cudaFreeHost h_C");

    return exit_code;
}
