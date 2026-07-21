import faulthandler; faulthandler.enable()
import pylibcugraph as plc, cupy as cp, numpy as np
print("start", flush=True)
s=[0,1,1,2,2,3,3,0,0,2]; d=[1,0,2,1,3,2,0,3,2,0]
src=cp.asarray(s+d,dtype=cp.int32); dst=cp.asarray(d+s,dtype=cp.int32)
w=cp.asarray([1.0]*len(src),dtype=cp.float32)
h=plc.ResourceHandle(); props=plc.GraphProperties(is_symmetric=True,is_multigraph=False)
G=plc.SGGraph(h,props,src,dst,weight_array=w,store_transposed=False,renumber=True,do_expensive_check=False)
print("graph built", flush=True)
def test(name, fn):
    try: fn(); print(f"OK   {name}", flush=True)
    except TypeError as e: print(f"SIG  {name}: {str(e)[:100]}", flush=True)
    except Exception as e: print(f"ERR  {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
# community with random_state (2nd arg) - correct signatures now
test("leiden", lambda: plc.leiden(h, 42, G, 100, 1.0, 1.0, False))
test("ecg",    lambda: plc.ecg(h, 42, G, 0.05, 16, 100, 1e-6, 1.0, False))
print("DONE", flush=True)
