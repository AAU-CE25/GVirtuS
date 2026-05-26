// Per-call timing bench: shows individual H2D and D2H latencies
// (since [GVS PROFILE] in Frontend.cpp only fires for big REQUEST payloads,
// which misses D2H — small request, big response).
//
// Usage:
//   THREADS=1 ITERS=4 ./percall_bench
//
// Output prints one line per call: kind, iter, elapsed_us, GB/s.

#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

static constexpr size_t MB = 1024ull * 1024ull;
static constexpr size_t BUF_SIZE = 64ull * MB;

struct ThreadArgs {
    int tid;
    int iterations;
};

static double now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               steady_clock::now().time_since_epoch())
        .count();
}

void *worker(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;

    float *h_buf = (float *)malloc(BUF_SIZE);
    if (!h_buf) {
        fprintf(stderr, "[tid=%d] host malloc failed\n", a->tid);
        return nullptr;
    }
    for (size_t i = 0; i < BUF_SIZE / sizeof(float); ++i) {
        h_buf[i] = (float)(a->tid + 1);
    }

    float *d_buf = nullptr;
    cudaError_t err = cudaMalloc(&d_buf, BUF_SIZE);
    if (err != cudaSuccess) {
        fprintf(stderr, "[tid=%d] cudaMalloc failed: %s\n", a->tid,
                cudaGetErrorString(err));
        free(h_buf);
        return nullptr;
    }

    fprintf(stderr,
            "tid=%d kind=cudaMalloc ok (initial setup, not timed below)\n",
            a->tid);

    for (int i = 0; i < a->iterations; ++i) {
        // ----- H2D -----
        double t0 = now_us();
        cudaMemcpy(d_buf, h_buf, BUF_SIZE, cudaMemcpyHostToDevice);
        double t1 = now_us();
        double h2d_us = t1 - t0;
        double h2d_gbps = (double)BUF_SIZE / (h2d_us / 1e6) / 1e9;
        fprintf(stderr,
                "tid=%d iter=%d kind=H2D %.0f us  (%.3f GB/s)\n",
                a->tid, i, h2d_us, h2d_gbps);

        // ----- D2H -----
        double t2 = now_us();
        cudaMemcpy(h_buf, d_buf, BUF_SIZE, cudaMemcpyDeviceToHost);
        double t3 = now_us();
        double d2h_us = t3 - t2;
        double d2h_gbps = (double)BUF_SIZE / (d2h_us / 1e6) / 1e9;
        fprintf(stderr,
                "tid=%d iter=%d kind=D2H %.0f us  (%.3f GB/s)\n",
                a->tid, i, d2h_us, d2h_gbps);
    }

    cudaFree(d_buf);
    free(h_buf);
    return nullptr;
}

int main(int argc, char **argv) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : 1;
    int iterations = (argc > 2) ? atoi(argv[2]) : 4;
    if (num_threads < 1) num_threads = 1;
    if (iterations < 1) iterations = 1;

    fprintf(stderr,
            "percall_bench: threads=%d iterations=%d buf=%zuMB\n",
            num_threads, iterations, BUF_SIZE / MB);

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArgs> args(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        args[i] = {i, iterations};
        if (pthread_create(&threads[i], nullptr, worker, &args[i]) != 0) {
            fprintf(stderr, "pthread_create failed for tid=%d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    fprintf(stderr, "percall_bench: done\n");
    return 0;
}
