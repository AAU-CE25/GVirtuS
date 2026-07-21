import re, statistics, collections
from datetime import datetime
pat = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z .*AM routine '([^']+)' returned .*req_id=(\d+)")
def ts(s): return datetime.strptime(s[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
rows=[]
for line in open("/tmp/llama_prof_backend.txt", errors="ignore"):
    m=pat.match(line)
    if m: rows.append((ts(m.group(1)), m.group(2), int(m.group(3))))
t0=rows[0][0]; tN=rows[-1][0]
last_reg=max(i for i,(t,n,r) in enumerate(rows) if 'Register' in n)
reg_end=rows[last_reg][0]
print(f"REGISTRATION: {last_reg+1} calls in {reg_end-t0:.2f}s")
print(f"GENERATION:   {len(rows)-last_reg-1} calls in {tN-reg_end:.2f}s")

# gaps in generation phase only
gaps=collections.defaultdict(list)
prev=rows[last_reg][0]
for t,n,r in rows[last_reg+1:]:
    gaps[n].append((t-prev)*1e6); prev=t   # us
print(f"\n=== GENERATION phase: gap-before-dispatch, MEDIAN us (robust) ===")
print(f"{'count':>7} {'median_us':>10} {'mean_us':>9} {'total_s':>8}  routine")
tot=collections.Counter({n:sum(v)/1e6 for n,v in gaps.items()})
for n,_ in tot.most_common(12):
    v=gaps[n]
    print(f"{len(v):>7} {statistics.median(v):>10.0f} {statistics.mean(v):>9.0f} {sum(v)/1e6:>8.1f}  {n}")

# per-token: cudaLaunchKernel count as proxy; tokens ~ launches / launches_per_token
launches=len(gaps.get('cudaLaunchKernel',[]))
print(f"\ntotal cudaLaunchKernel in gen: {launches}")
# count decode steps: cudaMemcpyAsync often 1-2 per token for logits; use cudaStreamSynchronize
syncs=len(gaps.get('cudaStreamSynchronize',[]))
print(f"cudaStreamSynchronize in gen: {syncs}")
# big-gap analysis: how much total time is in gaps > 1ms
allg=[(t-rows[i+last_reg][0]) for i,(t,n,r) in enumerate(rows[last_reg+1:])]
big=[g for g in allg if g>0.001]
print(f"\ngaps>1ms: {len(big)} of {len(allg)}, summing {sum(big):.1f}s of {sum(allg):.1f}s total")
print(f"gaps>10ms: {len([g for g in allg if g>0.01])}, summing {sum(g for g in allg if g>0.01):.1f}s")
