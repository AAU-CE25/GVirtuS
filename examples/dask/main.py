"""
Dask + CuPy matrix multiplication via GVirtuS.

CuPy issues CUDA driver calls that are intercepted by GVirtuS libcuda.so
and forwarded over TCP to the backend GPU.

Usage:
    python main.py [--size N] [--chunks C]

    --size N    Matrix dimension (NxN). Default: 4096
    --chunks C  Dask chunk size along each axis. Default: 1024
"""

import argparse
import time

import numpy as np
import cupy as cp
import dask
import dask.array as da


def make_random_matrix(n: int, chunk: int, dtype=np.float32) -> da.Array:
    """Create a random NxN Dask array backed by CuPy chunks."""
    return da.random.random((n, n), chunks=(chunk, chunk)).map_blocks(
        cp.asarray, dtype=dtype
    )


def run(size: int, chunks: int) -> None:
    print(f"Matrix size : {size}x{size}  (chunks={chunks}x{chunks})")
    print(f"Dask version: {dask.__version__}  |  CuPy version: {cp.__version__}")

    # ── Build matrices (materialize immediately so random state is fixed) ────
    print("Generating matrices ...")
    A = make_random_matrix(size, chunks).persist()
    B = make_random_matrix(size, chunks).persist()

    # ── Multiply ──────────────────────────────────────────────────────────────
    print("Computing A @ B ...")
    t0 = time.perf_counter()
    C = da.matmul(A, B)
    result = C.compute()          # triggers GPU kernels via GVirtuS
    elapsed = time.perf_counter() - t0

    # ── Verify ────────────────────────────────────────────────────────────────
    # Run a small standalone matmul and compare against numpy.
    # We do NOT slice the large result — a tile of A@B is not the same as
    # (tile of A) @ (tile of B) because the full matmul sums over all columns.
    tile_size = min(256, size)
    chunk = min(chunks, tile_size)
    A_small = make_random_matrix(tile_size, chunk).persist()
    B_small = make_random_matrix(tile_size, chunk).persist()
    C_small = da.matmul(A_small, B_small).compute()
    a_cpu = cp.asnumpy(A_small.compute())
    b_cpu = cp.asnumpy(B_small.compute())
    expected = a_cpu @ b_cpu
    actual   = cp.asnumpy(C_small)
    max_err  = float(np.max(np.abs(actual - expected)))

    print(f"Elapsed     : {elapsed:.3f}s")
    print(f"Result shape: {result.shape}  dtype={result.dtype}")
    print(f"Max abs err (tile {tile_size}x{tile_size} vs numpy): {max_err:.6f}")
    assert max_err < 1e-2, f"Result looks wrong: max_err={max_err}"
    print("OK — matrix multiplication verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask+CuPy matmul via GVirtuS")
    parser.add_argument("--size",   type=int, default=4096, help="Matrix dimension N (NxN)")
    parser.add_argument("--chunks", type=int, default=1024, help="Chunk size per axis")
    args = parser.parse_args()

    run(args.size, args.chunks)
