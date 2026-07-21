import sys
def step(msg):
    print(f"[cudf_smoke] {msg}", flush=True)

step("step1: import rmm")
import rmm
step("step2: rmm.reinitialize(pool_allocator=False)")
rmm.reinitialize(pool_allocator=False, managed_memory=False)
step("step3: import cudf")
import cudf
step(f"cudf {cudf.__version__}")
step("step4: build DataFrame")
df = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
step("step5: print DataFrame")
print(df, flush=True)
step("step6: reduction sum(a)")
s = int(df["a"].sum())
step(f"sum(a) = {s} (expect 10)")
step("step7: filter a>2")
df2 = df[df["a"] > 2]
print(df2, flush=True)
step("ALL DONE")
