import faulthandler; faulthandler.enable()
import pylibcugraph as plc, cupy as cp, numpy as np, inspect
print("pylibcugraph", plc.__version__, flush=True)
s=[0,1,1,2,2,3,3,0,0,2]; d=[1,0,2,1,3,2,0,3,2,0]
src=cp.asarray(s+d,dtype=cp.int32); dst=cp.asarray(d+s,dtype=cp.int32)
w=cp.asarray([1.0]*len(src),dtype=cp.float32)
h=plc.ResourceHandle(); props=plc.GraphProperties(is_symmetric=True,is_multigraph=False)
G=plc.SGGraph(h,props,src,dst,weight_array=w,store_transposed=False,renumber=True,do_expensive_check=False)
print("graph built", flush=True)
v2=cp.asarray([0,1],dtype=cp.int32); v2b=cp.asarray([2,3],dtype=cp.int32)
def sig(n):
    try: return str(inspect.signature(getattr(plc,n)))[:160]
    except: return "?"
def test(name, fn):
    print(f"[{name}] sig={sig(name)}", flush=True)
    try: fn(); print(f"OK   {name}", flush=True)
    except TypeError as e: print(f"SIG  {name}: {str(e)[:100]}", flush=True)
    except Exception as e: print(f"ERR  {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
# deterministic similarity (safe)
test("jaccard_coefficients",  lambda: plc.jaccard_coefficients(h,G,v2,v2b,False,False))
test("sorensen_coefficients", lambda: plc.sorensen_coefficients(h,G,v2,v2b,False,False))
test("overlap_coefficients",  lambda: plc.overlap_coefficients(h,G,v2,v2b,False,False))
test("cosine_coefficients",   lambda: plc.cosine_coefficients(h,G,v2,v2b,False,False))
test("k_core",                lambda: plc.k_core(h,G,1,"bidirectional",None,False))
# community with randomness (might segfault like sampling) - LAST
test("leiden", lambda: plc.leiden(h,G,100,1.0,1.0,False))
test("ecg",    lambda: plc.ecg(h,G,0.05,16,100,1e-6,1.0,None,False))
print("DONE", flush=True)
