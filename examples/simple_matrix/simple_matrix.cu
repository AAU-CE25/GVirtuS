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

int main(int argc, char** argv) {
    int n = getenv_int("MATRIX_SIZE", 256);
    int iters = getenv_int("ITERATIONS", 10);
    int warmup = getenv_int("WARMUP", 1);
    const char* alloc_mode_env = std::getenv("ALLOCATION_MODE");
    const std::string alloc_mode = alloc_mode_env != nullptr ? alloc_mode_env : "reuse";
    const bool per_iter_alloc = (alloc_mode == "per_iter" || alloc_mode == "per-iter" ||
                                 alloc_mode == "alloc");

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) iters = std::atoi(argv[2]);
    if (argc > 3) warmup = std::atoi(argv[3]);

    if (n <= 0 || iters <= 0 || warmup < 0) {
        std::cerr << "Usage: " << argv[0] << " [N] [ITERS] [WARMUP]" << std::endl;
        return 1;
    }

    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = elems * sizeof(float);

    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
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
            "cublasSgemm warmup");
        check_cuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C warmup");
        if (per_iter_alloc) {
            free_device();
        }
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    float total_ms = 0.0f;
    double host_total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto host_start = std::chrono::steady_clock::now();
        if (per_iter_alloc) {
            alloc_device();
        }
        check_cuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
        check_cuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");
        check_cuda(cudaEventRecord(start), "cudaEventRecord start");
        check_cublas(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                        d_A, n, d_B, n, &beta, d_C, n),
            "cublasSgemm");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        check_cuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");
        if (per_iter_alloc) {
            free_device();
        }
        const auto host_end = std::chrono::steady_clock::now();

        float ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        total_ms += ms;
        host_total_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                             host_end - host_start)
                             .count();
    }

    const float expected = static_cast<float>(n);
    double max_err = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        const double err = std::fabs(static_cast<double>(h_C[i]) - expected);
        if (err > max_err) max_err = err;
    }
    const double tol = 1e-2 * static_cast<double>(n);
    const bool ok = max_err <= tol;

    const double avg_ms = static_cast<double>(total_ms) / static_cast<double>(iters);
    const double host_avg_ms = host_total_ms / static_cast<double>(iters);
    std::cout << "size=" << n << " iters=" << iters
              << " avg_sgemm_ms=" << avg_ms
              << " avg_host_ms=" << host_avg_ms
              << " check=" << (ok ? "pass" : "fail")
              << " max_abs_err=" << max_err
              << " alloc_mode=" << alloc_mode << std::endl;
    std::cout << "CSV," << n << "," << iters << "," << avg_ms << "," << host_avg_ms
              << std::endl;

    int exit_code = 0;
    if (!ok) {
        std::cerr << "Matrix check failed: expected=" << expected
                  << " max_abs_err=" << max_err << " tol=" << tol << std::endl;
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
