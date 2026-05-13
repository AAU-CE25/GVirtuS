// test_async_overlap.cu
// Measures pipeline overlap efficiency in GVirtuS vs native CUDA
// Tests whether cudaMemcpyAsync + kernel can overlap across two streams
//
// Three scenarios:
//   1. SEQUENTIAL  — sync after every op, no overlap possible
//   2. PIPELINED   — two streams, interleaved H2D + kernel + D2H (ideal overlap)
//   3. SYNC_HEAVY  — pipelined but with cudaDeviceSynchronize after every call
//
// On native CUDA:  PIPELINED should be ~50% faster than SEQUENTIAL
// On GVirtuS:      PIPELINED ≈ SEQUENTIAL (no overlap due to blocking AM calls)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d — %s\n",                         \
              __FILE__, __LINE__, cudaGetErrorString(err));                  \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// Simple compute kernel: element-wise saxpy, enough work to be measurable
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Returns wall-clock milliseconds
static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// N: number of floats per buffer — large enough to make transfers visible
#define N        (1 << 22)   // 4M floats = 16 MB per buffer
#define REPEATS  5           // warm-up + timed runs
#define BLOCKS   ((N + 255) / 256)
#define THREADS  256

int main(int argc, char **argv) {
    int device = 0;
    if (argc > 1) device = atoi(argv[1]);
    CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("N = %d floats (%.1f MB per buffer)\n\n", N, N * 4.0 / 1e6);

    // --- Allocate pinned host and device memory for 2 streams ---
    float *h_x[2], *h_y[2], *h_out[2];
    float *d_x[2], *d_y[2];
    cudaStream_t stream[2];

    for (int s = 0; s < 2; s++) {
        CHECK(cudaMallocHost(&h_x[s],   N * sizeof(float)));
        CHECK(cudaMallocHost(&h_y[s],   N * sizeof(float)));
        CHECK(cudaMallocHost(&h_out[s], N * sizeof(float)));
        CHECK(cudaMalloc(&d_x[s], N * sizeof(float)));
        CHECK(cudaMalloc(&d_y[s], N * sizeof(float)));
        CHECK(cudaStreamCreate(&stream[s]));

        // Init host data
        for (int i = 0; i < N; i++) {
            h_x[s][i] = (float)(i + s * N);
            h_y[s][i] = 1.0f;
        }
    }

    // ---------------------------------------------------------------
    // SCENARIO 1: SEQUENTIAL — full sync after every operation
    // ---------------------------------------------------------------
    printf("=== SCENARIO 1: Sequential (sync after every op) ===\n");
    double seq_times[REPEATS];
    for (int r = 0; r < REPEATS; r++) {
        double t0 = now_ms();

        for (int s = 0; s < 2; s++) {
            // H2D
            CHECK(cudaMemcpy(d_x[s], h_x[s], N * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_y[s], h_y[s], N * sizeof(float), cudaMemcpyHostToDevice));
            // Kernel
            saxpy<<<BLOCKS, THREADS>>>(N, 2.0f, d_x[s], d_y[s]);
            CHECK(cudaDeviceSynchronize());
            // D2H
            CHECK(cudaMemcpy(h_out[s], d_y[s], N * sizeof(float), cudaMemcpyDeviceToHost));
        }

        seq_times[r] = now_ms() - t0;
        printf("  run %d: %.2f ms\n", r + 1, seq_times[r]);
    }

    // ---------------------------------------------------------------
    // SCENARIO 2: PIPELINED — two streams, overlapped
    // ---------------------------------------------------------------
    printf("\n=== SCENARIO 2: Pipelined (two streams, async) ===\n");
    double pipe_times[REPEATS];
    for (int r = 0; r < REPEATS; r++) {
        double t0 = now_ms();

        // Interleave stream 0 and stream 1 ops to maximise overlap
        CHECK(cudaMemcpyAsync(d_x[0], h_x[0], N * sizeof(float), cudaMemcpyHostToDevice, stream[0]));
        CHECK(cudaMemcpyAsync(d_x[1], h_x[1], N * sizeof(float), cudaMemcpyHostToDevice, stream[1]));

        CHECK(cudaMemcpyAsync(d_y[0], h_y[0], N * sizeof(float), cudaMemcpyHostToDevice, stream[0]));
        CHECK(cudaMemcpyAsync(d_y[1], h_y[1], N * sizeof(float), cudaMemcpyHostToDevice, stream[1]));

        saxpy<<<BLOCKS, THREADS, 0, stream[0]>>>(N, 2.0f, d_x[0], d_y[0]);
        saxpy<<<BLOCKS, THREADS, 0, stream[1]>>>(N, 2.0f, d_x[1], d_y[1]);

        CHECK(cudaMemcpyAsync(h_out[0], d_y[0], N * sizeof(float), cudaMemcpyDeviceToHost, stream[0]));
        CHECK(cudaMemcpyAsync(h_out[1], d_y[1], N * sizeof(float), cudaMemcpyDeviceToHost, stream[1]));

        // Single sync point at the end
        CHECK(cudaStreamSynchronize(stream[0]));
        CHECK(cudaStreamSynchronize(stream[1]));

        pipe_times[r] = now_ms() - t0;
        printf("  run %d: %.2f ms\n", r + 1, pipe_times[r]);
    }

    // ---------------------------------------------------------------
    // SCENARIO 3: SYNC_HEAVY — pipelined API calls but sync after each
    // ---------------------------------------------------------------
    printf("\n=== SCENARIO 3: Sync-heavy (async API + cudaDeviceSynchronize each call) ===\n");
    double syncheavy_times[REPEATS];
    for (int r = 0; r < REPEATS; r++) {
        double t0 = now_ms();

        for (int s = 0; s < 2; s++) {
            CHECK(cudaMemcpyAsync(d_x[s], h_x[s], N * sizeof(float), cudaMemcpyHostToDevice, stream[s]));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpyAsync(d_y[s], h_y[s], N * sizeof(float), cudaMemcpyHostToDevice, stream[s]));
            CHECK(cudaDeviceSynchronize());
            saxpy<<<BLOCKS, THREADS, 0, stream[s]>>>(N, 2.0f, d_x[s], d_y[s]);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpyAsync(h_out[s], d_y[s], N * sizeof(float), cudaMemcpyDeviceToHost, stream[s]));
            CHECK(cudaDeviceSynchronize());
        }

        syncheavy_times[r] = now_ms() - t0;
        printf("  run %d: %.2f ms\n", r + 1, syncheavy_times[r]);
    }

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    // Use last 3 runs as stable measurement (skip warm-up)
    double seq_avg = 0, pipe_avg = 0, synch_avg = 0;
    for (int r = REPEATS - 3; r < REPEATS; r++) {
        seq_avg   += seq_times[r];
        pipe_avg  += pipe_times[r];
        synch_avg += syncheavy_times[r];
    }
    seq_avg /= 3; pipe_avg /= 3; synch_avg /= 3;

    double overlap_pct = (seq_avg - pipe_avg) / seq_avg * 100.0;
    if (overlap_pct < 0) overlap_pct = 0;

    printf("\n=== RESULTS SUMMARY ===\n");
    printf("  Sequential avg (last 3):    %.2f ms\n", seq_avg);
    printf("  Pipelined avg (last 3):     %.2f ms\n", pipe_avg);
    printf("  Sync-heavy avg (last 3):    %.2f ms\n", synch_avg);
    printf("\n");
    printf("  Pipeline speedup:           %.2fx\n", seq_avg / pipe_avg);
    printf("  Overlap efficiency:         %.1f%%\n", overlap_pct);
    printf("\n");
    printf("  Interpretation:\n");
    if (overlap_pct >= 30.0) {
        printf("  -> OVERLAP ACTIVE: async streams are genuinely concurrent.\n");
        printf("     (Expected on native CUDA with pinned memory and 2 copy engines)\n");
    } else if (overlap_pct >= 5.0) {
        printf("  -> PARTIAL OVERLAP: some concurrency, but serialization present.\n");
    } else {
        printf("  -> NO OVERLAP: async calls are serialized (GVirtuS AM barrier).\n");
        printf("     cudaMemcpyAsync is effectively synchronous over the network.\n");
    }

    // CSV output for benchmark.sh ingestion
    printf("\nCSV:\nscenario,avg_ms,speedup_vs_seq,overlap_pct\n");
    printf("sequential,%.2f,1.00,0.0\n", seq_avg);
    printf("pipelined,%.2f,%.2f,%.1f\n", pipe_avg, seq_avg / pipe_avg, overlap_pct);
    printf("sync_heavy,%.2f,%.2f,0.0\n", synch_avg, seq_avg / synch_avg);

    // Cleanup
    for (int s = 0; s < 2; s++) {
        cudaFreeHost(h_x[s]); cudaFreeHost(h_y[s]); cudaFreeHost(h_out[s]);
        cudaFree(d_x[s]); cudaFree(d_y[s]);
        cudaStreamDestroy(stream[s]);
    }

    return 0;
}