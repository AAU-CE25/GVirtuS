#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::string payload_label(int N) {
    double bytes = double(N) * double(N) * sizeof(float);
    const char* units[] = {"B", "KB", "MB", "GB"};
    int u = 0;

    while (bytes >= 1024.0 && u < 3) {
        bytes /= 1024.0;
        ++u;
    }

    std::ostringstream oss;
    if (std::fabs(bytes - std::round(bytes)) < 1e-9) {
        oss << static_cast<long long>(std::round(bytes)) << units[u];
    } else {
        oss << std::fixed << std::setprecision(1) << bytes << units[u];
    }
    return oss.str();
}

struct Result {
    std::string status = "FAILED";
    double wall_ms = 0.0;
    double malloc_ms = 0.0;
    double cudamalloc_ms = 0.0;
    double h2d_ms = 0.0;
    double cublas_create_ms = 0.0;
    double gemm_ms = 0.0;
    double d2h_ms = 0.0;
    double cleanup_ms = 0.0;
    float got = 0.0f;
    float expected = 0.0f;
};

static Result run_once(int N) {
    Result r;
    r.expected = static_cast<float>(N);

    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t bytes = elems * sizeof(float);

    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cublasHandle_t handle = nullptr;

    auto cleanup = [&]() {
        auto t0 = clk::now();
        if (handle) cublasDestroy(handle);
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (h_A) std::free(h_A);
        if (h_B) std::free(h_B);
        if (h_C) std::free(h_C);
        auto t1 = clk::now();
        r.cleanup_ms = ms(t0, t1);
    };

    auto wall0 = clk::now();

    auto t0 = clk::now();
    h_A = static_cast<float*>(std::malloc(bytes));
    h_B = static_cast<float*>(std::malloc(bytes));
    h_C = static_cast<float*>(std::malloc(bytes));
    auto t1 = clk::now();
    r.malloc_ms = ms(t0, t1);

    if (!h_A || !h_B || !h_C) {
        cleanup();
        r.wall_ms = ms(wall0, clk::now());
        return r;
    }

    for (size_t i = 0; i < elems; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    t0 = clk::now();
    if (cudaMalloc(&d_A, bytes) != cudaSuccess ||
        cudaMalloc(&d_B, bytes) != cudaSuccess ||
        cudaMalloc(&d_C, bytes) != cudaSuccess) {
        cleanup();
        r.wall_ms = ms(wall0, clk::now());
        return r;
    }
    t1 = clk::now();
    r.cudamalloc_ms = ms(t0, t1);

    t0 = clk::now();
    if (cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        r.wall_ms = ms(wall0, clk::now());
        return r;
    }
    t1 = clk::now();
    r.h2d_ms = ms(t0, t1);

    t0 = clk::now();
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        r.wall_ms = ms(wall0, clk::now());
        return r;
    }
    t1 = clk::now();
    r.cublas_create_ms = ms(t0, t1);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    t0 = clk::now();
    if (cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N) != CUBLAS_STATUS_SUCCESS ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        r.wall_ms = ms(wall0, clk::now());
        return r;
    }
    t1 = clk::now();
    r.gemm_ms = ms(t0, t1);

    t0 = clk::now();
    if (cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        r.wall_ms = ms(wall0, clk::now());
        return r;
    }
    t1 = clk::now();
    r.d2h_ms = ms(t0, t1);

    r.got = h_C[0];
    if (std::fabs(r.got - r.expected) < 1e-3f) {
        r.status = "OK";
    }

    cleanup();
    r.wall_ms = ms(wall0, clk::now());
    return r;
}

static std::vector<int> parse_sizes() {
    const char* env = std::getenv("SIZES");
    std::string s = env ? env : "256 512 1024 2048 4096 8192 16384";

    std::istringstream iss(s);
    std::vector<int> sizes;
    int n = 0;
    while (iss >> n) {
        sizes.push_back(n);
    }
    return sizes;
}

int main() {
    int runs = std::getenv("RUNS") ? std::atoi(std::getenv("RUNS")) : 50;
    int warmups = std::getenv("WARMUPS") ? std::atoi(std::getenv("WARMUPS")) : 5;

    std::string csv_path = std::getenv("STEADY_CSV")
        ? std::getenv("STEADY_CSV")
        : "benchmark_results/SimpleMatrix_Steady_Baremetal/results.csv";

    auto sizes = parse_sizes();

    std::ofstream csv(csv_path);
    csv << "configuration,phase,run,matrix_n,payload_label,status,"
        << "wall_s,result_ms,malloc_ms,cudamalloc_ms,h2d_ms,cublas_create_ms,"
        << "gemm_ms,d2h_ms,cleanup_ms,got,expected\n";

    auto process0 = clk::now();

    for (int N : sizes) {
        std::cerr << "STEADY_MATRIX_N=" << N
                  << " payload=" << payload_label(N)
                  << " warmups=" << warmups
                  << " runs=" << runs << std::endl;

        for (int i = 1; i <= warmups; ++i) {
            Result r = run_once(N);

            csv << "baremetal_steady,warmup," << i << ","
                << N << "," << payload_label(N) << ","
                << r.status << ","
                << std::fixed << std::setprecision(9) << r.wall_ms / 1000.0 << ","
                << std::setprecision(3) << r.wall_ms << ","
                << r.malloc_ms << "," << r.cudamalloc_ms << "," << r.h2d_ms << ","
                << r.cublas_create_ms << "," << r.gemm_ms << "," << r.d2h_ms << ","
                << r.cleanup_ms << "," << r.got << "," << r.expected << "\n";

            std::cout << "[warmup] N=" << N
                      << " run=" << i
                      << " status=" << r.status
                      << " wall_ms=" << std::fixed << std::setprecision(3) << r.wall_ms
                      << " gemm_ms=" << r.gemm_ms
                      << " d2h_ms=" << r.d2h_ms
                      << std::endl;
        }

        for (int i = 1; i <= runs; ++i) {
            Result r = run_once(N);

            csv << "baremetal_steady,measure," << i << ","
                << N << "," << payload_label(N) << ","
                << r.status << ","
                << std::fixed << std::setprecision(9) << r.wall_ms / 1000.0 << ","
                << std::setprecision(3) << r.wall_ms << ","
                << r.malloc_ms << "," << r.cudamalloc_ms << "," << r.h2d_ms << ","
                << r.cublas_create_ms << "," << r.gemm_ms << "," << r.d2h_ms << ","
                << r.cleanup_ms << "," << r.got << "," << r.expected << "\n";

            std::cout << "[measure] N=" << N
                      << " run=" << i
                      << " status=" << r.status
                      << " wall_ms=" << std::fixed << std::setprecision(3) << r.wall_ms
                      << " gemm_ms=" << r.gemm_ms
                      << " d2h_ms=" << r.d2h_ms
                      << std::endl;
        }
    }

    auto process1 = clk::now();

    std::ofstream wall("benchmark_results/SimpleMatrix_Steady_Baremetal/process_wall.txt");
    wall << "process_wall_s=" << std::fixed << std::setprecision(6)
         << ms(process0, process1) / 1000.0 << "\n";

    return 0;
}
