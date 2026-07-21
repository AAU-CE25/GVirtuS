import faulthandler; faulthandler.enable()
import pylibcugraph as plc, cupy as cp, numpy as np, inspect
print("pylibcugraph", plc.__version__, flush=True)
s=[0,1,1,2,2,3,3,0,0,2]; d=[1,0,2,1,3,2,0,3,2,0]
src=cp.asarray(s+d,dtype=cp.int32); dst=cp.asarray(d+s,dtype=cp.int32)
w=cp.asarray([1.0]*len(src),dtype=cp.float32)
h=plc.ResourceHandle(); props=plc.GraphProperties(is_symmetric=True,is_multigraph=False)
G =plc.SGGraph(h,props,src,dst,weight_array=w,store_transposed=False,renumber=True,do_expensive_check=False)
Gt=plc.SGGraph(h,props,src,dst,weight_array=w,store_transposed=True, renumber=True,do_expensive_check=False)
print("graphs built", flush=True)
v2=cp.asarray([0,1],dtype=cp.int32); v2b=cp.asarray([2,3],dtype=cp.int32)
def sig(n):
    f=getattr(plc,n,None)
    try: return str(inspect.signature(f))[:180]
    except: return "?"
def test(name, fn):
    print(f"[{name}] sig={sig(name)}", flush=True)
    try: fn(); print(f"OK   {name}", flush=True)
    except TypeError as e: print(f"SIG  {name}: {str(e)[:100]}", flush=True)
    except Exception as e: print(f"ERR  {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
test("eigenvector_centrality", lambda: plc.eigenvector_centrality(h,Gt,1e-6,100,False))
test("katz_centrality",        lambda: plc.katz_centrality(h,Gt,None,0.01,1.0,1e-6,100,False))
test("k_core",                 lambda: plc.k_core(h,G,2,None,None,False))
test("two_hop_neighbors",      lambda: plc.two_hop_neighbors(h,G,v2,False))
test("degrees",                lambda: plc.degrees(h,G,v2,False))
test("jaccard_coefficients",   lambda: plc.jaccard_coefficients(h,G,None,v2,v2b,False,False))
test("sorensen_coefficients",  lambda: plc.sorensen_coefficients(h,G,None,v2,v2b,False,False))
test("overlap_coefficients",   lambda: plc.overlap_coefficients(h,G,None,v2,v2b,False,False))
test("betweenness_centrality", lambda: plc.betweenness_centrality(h,G,None,None,False,False,False))
print("DONE", flush=True)
