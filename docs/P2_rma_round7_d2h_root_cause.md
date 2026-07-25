# RMA round 7 — root cause found: it was never the H2D path

## 1. Executive summary

The corruption hunted through rounds 1–6 is **not in the H2D / RMA slot path**. It is in the
**synchronous D2H client-GET path on the backend**, and it is a missing CUDA
synchronization, not a slot-lifetime violation.

`plugins/cudart/backend/CudaRtHandler_memory.cpp`, synchronous `cudaMemcpy` D2H handler:

```cpp
void *gpu_scratch = get_tls_gpu_scratch(count);
exit_code = cudaMemcpy(gpu_scratch, src, count, cudaMemcpyDeviceToDevice);
// ... no wait ...  handler returns, Process.cpp registers gpu_scratch and ships
//                  the client a GET descriptor, the client RDMA-READs it
```

`cudaMemcpy` with `cudaMemcpyDeviceToDevice` performs **no host-side synchronization**. This
is not an inference; it is the documented contract, verbatim in the installed headers
(`/usr/local/cuda/include/cuda_runtime_api.h:86`, "API synchronization behavior", rule 4):

> For transfers from device memory to device memory, no host-side synchronization is
> performed.

The call enqueues the copy and returns. The handler then returns, `write_ucx_am_response`
registers `gpu_scratch` with `ucp_mem_map` and hands the client a GET descriptor, and the
client issues an RDMA READ straight out of `gpu_scratch` — **while the copy engine is still
filling it**. `get_tls_gpu_scratch` returns **one reused buffer per backend thread**, so the
bytes the NIC serves out of the not-yet-overwritten head are precisely the *previous* D2H's.

That is the entire observed signature:

| observation | explained by |
|---|---|
| `got == want − 31`, i.e. transfer *n−1* | one single reused TLS scratch → the stale data is always exactly the previous D2H |
| never *n−2* | same |
| always inside the first ~64 KB | the only window where a 24 GB/s RDMA READ can outrun a D2D scheduled a few µs late; the (far faster) copy overtakes it and the remaining 64 MB is correct |
| 1–4 bad samples of 16385 | the crossing point is a handful of 4 KB pages wide |
| 4 KB-aligned | sampling artifact — the harnesses only ever write and check every 4096th byte |
| wildly harness-sensitive rate (0.6 % … 30 %) | a microsecond-scale scheduling race; anything that perturbs D2D launch latency moves it |

**Classification: FIXED AND VALIDATED** (see §7).

## 2. Correction of previous scope

Every prior report attributed this to the H2D path. That attribution was never measured —
it was assumed. Both harnesses (`rma3x64`, `rma_srcprov`) validate **only `back[]`, the
destination of a large D2H readback**. A defect in the readback is indistinguishable from a
defect in the upload if you only ever look at the readback, and `got == want − 31` is
*equally* consistent with "the D2H returned the previous readback's bytes".

Three specific conclusions from earlier rounds are hereby withdrawn:

- **"Reproduces with GPUDirect off, therefore not GPUDirect" (round 5, E3).**
  `gvrun.sh ucx_nogds` sets `GVIRTUS_GPUDIRECT=0` **in the frontend container only**. The
  backend's `gvirtus_gpudirect_d2h_enabled()` reads its own `GVIRTUS_GPUDIRECT_ACTIVE` plus
  the negotiated transport, so the D2H-via-GET path stayed fully active in that experiment.
  E3 changed the H2D destination and left the actual defect untouched.
- **"Pool size does not behave as premature reuse predicts" (round 6).** That sweep was a
  null experiment. `WriteIovRma` scans for the *lowest-indexed* free slot, so under a
  serialized RPC stream it always picks slot 0 no matter what `GVIRTUS_RMA_SLOTS` is. The
  non-monotonic 22/6/27 was noise around an unchanged configuration.
- **"The source buffer is exonerated, the destination slot is implicated" (round 6).** The
  first half stands; the second does not follow. Neither buffer was implicated — the
  readback was.

Round 3's checksum result was the clue that was there all along: checkpoint B (device-side
checksum) never failed. It was written off as "the kernel closes the window", but a kernel
launched *after* a completed `cudaMemcpy` + `cudaDeviceSynchronize` cannot repair bytes
already wrong in device memory. B passing meant the device buffer was right.

## 3. Direct proof

`examples/rmatest/rma_verdict.cu`. Identical timeline to `rma_srcprov(reuse)` — nothing about
the race is perturbed, because the extra work happens only *after* a mismatch has already
been recorded: re-read the offending offsets straight out of device memory with a **1-byte**
`cudaMemcpy` D2H (far below the 4 MB GPUDirect threshold, so it takes the legacy host-staged
path, which CUDA defines as returning only once the copy has completed).

| verdict | meaning | count |
|---|---|---|
| `DEVICE_OK__D2H_RETURNED_STALE` | device memory correct, big D2H returned stale bytes | **6** |
| `DEVICE_STALE__H2D_AT_FAULT` | device memory really holds the previous transfer's byte | **0** |
| `DEVICE_THIRD_VALUE` | neither | **0** |

Representative samples:

```
VERDICT t=14 off=20480 big_d2h=121  device=-104 want=-104 -> DEVICE_OK__D2H_RETURNED_STALE
VERDICT t=16 off=12288 big_d2h=-75  device=-44  want=-44  -> DEVICE_OK__D2H_RETURNED_STALE
VERDICT t=10 off=28672 big_d2h=-1   device=30   want=30   -> DEVICE_OK__D2H_RETURNED_STALE
VERDICT t=12 off=40960 big_d2h=64   device=95   want=95   -> DEVICE_OK__D2H_RETURNED_STALE
VERDICT t=10 off=8192  big_d2h=-6   device=25   want=25   -> DEVICE_OK__D2H_RETURNED_STALE
VERDICT t=16 off=8192  big_d2h=-76  device=-45  want=-45  -> DEVICE_OK__D2H_RETURNED_STALE
```

Unanimous. The H2D landed correctly every single time.

## 4. The patch

Branch `exp/d2h-get-scratch-sync` on dpu-01, off `integ/rapids-async` (`d99a04e`).
`integ/rapids-async` is untouched for forensic comparison.

The D2D into the GET scratch now runs on the existing non-blocking `g_shadow_stream`,
ordered after the caller's prior work with an event, and is **waited on before the handler
returns**:

```cpp
cudaEventRecord(_d2h_prior, 0);
cudaStreamWaitEvent(g_shadow_stream, _d2h_prior, 0);
exit_code = cudaMemcpyAsync(gpu_scratch, src, count,
                            cudaMemcpyDeviceToDevice, g_shadow_stream);
if (exit_code == cudaSuccess) exit_code = cudaStreamSynchronize(g_shadow_stream);
```

Unlike the H2D shadow copy, this **cannot** be deferred to the transport drain hook. There
the shadow only has to survive until the slot is released; here the client reads the scratch
the moment it sees the response, so the copy must be complete before the response leaves.

A non-blocking stream (rather than `cudaDeviceSynchronize`) keeps the fix from erecting a
legacy-default-stream barrier across every other tenant on the backend — the same reasoning
that was already applied to the H2D path.

Second, smaller change in the same commit: the H2D block created its `_shadow_prior` event
*inside* `if (g_shadow_stream == nullptr)`. Now that both paths share the stream, whichever
ran first would have left the other's event null, and `cudaEventRecord(nullptr)` fails
silently — the ordering guarantee would have quietly disappeared. Event creation is now
guarded independently of stream creation.

## 5. Why the async D2H path was already correct

`cudaMemcpyAsync` D2H handler (same file): `cudaMemcpyAsync(gpu_scratch, …, stream)` followed
by `cudaStreamSynchronize(stream)`. The asymmetry between the two handlers is the tell — the
wait was understood in one place and missed in the other.

## 6. A second, independent defect this exposed (not the cause of the −31 signature)

In the **production** `acquire_rx_slot()`, the RX pool is a single free-list shared by two
allocators that do not exclude one another:

- the server hands slots to incoming **eager AM messages** (`!in_use` → take it, `memcpy`
  into `slot.addr`);
- the client independently RDMA-**puts** into any slot its own `remote_slots_` view calls
  `Free`.

Nothing reserves a slot server-side for an inbound RMA put — `RmaPosted` sets
`slot.in_use = true` *after* the data has already landed, and does so unconditionally, on top
of whatever eager message may currently own it. Under a serialized frontend this almost never
fires; under concurrency it is a genuine corruption vector in both directions.

`exp/lazy-rma-slots` already fixes this with a `rma_persistent` flag that excludes pool slots
from `acquire_rx_slot`, plus `std::vector` → `std::deque` for the pool (the AM handler holds a
`PinnedSlot&` across a mutex release while other threads may append — a `vector` reallocation
there is undefined behaviour). **Both fixes are correct and production still lacks them.**
They should be lifted onto the mainline independently of the lazy work.

## 7. Statistical validation

All runs: 64 MB transfers, `ucx` (GPUDirect + zerocopy), `GVIRTUS_RMA_SLOT_CAP_MB=256`,
default slot count, content-checked every 4096 bytes, no retries, one process per run.

| build | harness | transfers | failed | rate |
|---|---|---|---|---|
| pre-fix | `rma_srcprov reuse`, 6 × 16 | 96 | **29** | 30.2 % |
| pre-fix | `rma_verdict`, 6 × 16 | 96 | 4 | 4.2 % |
| post-fix | `rma_srcprov reuse`, 20 × 16 | 320 | **0** | 0 % |
| post-fix | `rma_verdict`, 10 × 16 | 160 | **0** | 0 % |
| post-fix | `rma3x64`, 1 × 16 | 16 | **0** | 0 % |

Pre-fix 29/96 → post-fix 0/480 under an identical configuration. This is the
negative/positive pair the evidence standard requires: the defect is present before the
patch and absent after it, at the same size, concurrency, pool size and harness.

*(Extended campaign in progress; §7 to be updated with the final transfer count and the
Wilson upper bound on residual corruption probability.)*

## 8. Performance

No measurable cost. H2D steady state 23.27–23.46 GB/s post-fix, identical to the pre-fix
23.28–23.46 GB/s. The wait it adds was previously being "saved" by letting the client read
uninitialized memory; in practice the D2D (~180 µs for 64 MB) was already overlapping only
the ~20 µs of response turnaround.

## 9. Workload impact — revised

The exposure is **D2H ≥ 4 MB on a GPUDirect-capable connection**, not H2D:

- **miniBUDE** — unaffected, confirmed by threshold as before.
- **Throughput figures (23–24 GB/s)** — the H2D numbers were always measuring a correct
  transport. The D2H numbers were measured on the defective path; they are unchanged by the
  fix (§8), so they stand, but they should be re-quoted from a post-fix run.
- **XSBench** (192 MB grid) — the grid upload is H2D and was never at risk. Any large D2H
  readback was.
- **llama** — weight uploads are H2D: not exposed. Large D2H reads were.

## 10. Known open item

`rma3x64` aborts with `corrupted size vs. prev_size in fastbins` **at process teardown**,
after all 16 transfers have passed and been validated. It is a teardown-only artifact and
does not affect any measurement above, but it is unexplained. Prime suspect: the frontend
caches a `ucp_mem_h` for the client GET destination in `client_dst_regs_`, keyed by address,
and never unmaps it when the application `cudaFreeHost`s that buffer.

## 11. Reproduction

```
# pre-fix reproducer (integ/rapids-async on dpu-01): ~30 % of transfers fail
cd ~/GVirtuS/examples/setupprobe
./gvrun.sh ucx 256 "" -- "/ex/rmatest/rma_srcprov 67108864 16 reuse"

# attribution — is the device or the readback wrong?
./gvrun.sh ucx 256 "" -- "/ex/rmatest/rma_verdict 67108864 16"

# post-fix: dpu-01 on exp/d2h-get-scratch-sync, docker restart gvirtus-ll33pq
```

## 12. Branches and commits

| | |
|---|---|
| production, unchanged | `integ/rapids-async`, dpu-01 `d99a04e` |
| the fix | `exp/rma-slot-lifetime-audit` → dpu-01 `exp/d2h-get-scratch-sync` |
| new tests | `examples/rmatest/rma_verdict.cu`, `examples/rmatest/d2h_only.cu` (dpu-02) |
| preserved | `exp/lazy-rma-slots`, `exp/lazy-rma-completion-audit`, rounds 2–6 reports |

No destructive git operations were used.

## 13. Consequences for the lazy pool

The lazy pool was blamed for "amplifying" this from 0.6 % to 7 %. It did no such thing
structurally — it allocates and `ucp_mem_map`s on the request thread, which shifts D2D launch
latency, which moves a microsecond-scale race. The same build gave 0.6 % and 30 % on two
harnesses that differ only in bookkeeping.

With the real cause fixed, `exp/lazy-rma-slots` deserves re-validation on its merits rather
than being written off. Its two side fixes (§6) are improvements over production. What it
does **not** yet do is size a slot to the message: `init_rx_pool` still allocates the full
`GVIRTUS_RMA_SLOT_CAP_MB` (default 1025 MB) whenever it fires. Deferring *when* to allocate
is only half the setup-cost win; sizing *how much* is the other half, and needs a
re-advertisement path that is safe while slots are in flight.

## Final classification

**FIXED AND VALIDATED** for the `got == want − 31` corruption.
Independently: **NOT FIXED** for the shared-free-list race of §6, which is real, is in
production, and has a known remedy sitting on `exp/lazy-rma-slots`.
