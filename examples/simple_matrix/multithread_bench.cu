// Multi-thread cudaMemcpy bench: N pthreads, each doing M iterations of
// cudaMemcpy 64MB H2D+D2H against the same logical GPU. With GVirtuS's
// per-pthread Frontend model, each thread should open its own UCX
// connection to the backend, so the total wall-clock should stay roughly
// flat as N grows — if it scales as O(N), there is contention somewhere
// (likely the listener's shared UCX worker mutex or a backend lock).

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
    double *elapsed_ms;  // wall time spent inside the worker loop
};

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

    // H2D + D2H bidirectional. With the bidirectional-RMA protocol
    // (client advertises its rx_pool rkeys to the server at Connect time)
    // BOTH directions go through ucp_put_nbx, not the AM-stream path.
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < a->iterations; ++i) {
        cudaMemcpy(d_buf, h_buf, BUF_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(h_buf, d_buf, BUF_SIZE, cudaMemcpyDeviceToHost);
    }
    auto t1 = std::chrono::steady_clock::now();

    *a->elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaFree(d_buf);
    free(h_buf);
    return nullptr;
}

int main(int argc, char **argv) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : 1;
    int iterations = (argc > 2) ? atoi(argv[2]) : 2;
    if (num_threads < 1) num_threads = 1;
    if (iterations < 1) iterations = 1;

    printf("multithread_bench: threads=%d iterations=%d buf=%zuMB\n",
           num_threads, iterations, BUF_SIZE / MB);

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArgs> args(num_threads);
    std::vector<double> elapsed(num_threads, 0.0);

    auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        args[i] = {i, iterations, &elapsed[i]};
        if (pthread_create(&threads[i], nullptr, worker, &args[i]) != 0) {
            fprintf(stderr, "pthread_create failed for tid=%d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    auto t_end = std::chrono::steady_clock::now();

    double wall_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    double per_thread_mean = 0;
    double per_thread_max = 0;
    for (double e : elapsed) {
        per_thread_mean += e;
        if (e > per_thread_max) per_thread_max = e;
    }
    per_thread_mean /= num_threads;

    size_t bytes_per_thread =
        (size_t)iterations * 2 * BUF_SIZE;  // H2D + D2H per iteration
    size_t total_bytes = bytes_per_thread * (size_t)num_threads;
    double agg_gbps = (double)total_bytes / (wall_ms / 1000.0) / 1e9;

    printf("\n=== RESULTS ===\n");
    printf("wall_clock_total     %.2f ms\n", wall_ms);
    printf("per_thread_mean      %.2f ms\n", per_thread_mean);
    printf("per_thread_max       %.2f ms\n", per_thread_max);
    printf("total_bytes          %.0f MB\n", total_bytes / 1.0 / MB);
    printf("aggregate_BW         %.3f GB/s\n", agg_gbps);
    printf("scaling_efficiency   %.0f%% (wall vs serial estimate)\n",
           100.0 * per_thread_mean / wall_ms);

    return 0;
}
