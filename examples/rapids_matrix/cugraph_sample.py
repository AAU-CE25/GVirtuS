import faulthandler; faulthandler.enable()
import pylibcugraph as plc, cupy as cp, numpy as np, inspect
print("pylibcugraph", plc.__version__, "has_uns:", hasattr(plc,"uniform_neighbor_sample"), flush=True)
if hasattr(plc,"uniform_neighbor_sample"):
    try: print("SIG:", str(inspect.signature(plc.uniform_neighbor_sample))[:420], flush=True)
    except Exception as e: print("sig?:", e, flush=True)
srcs=cp.asarray([0,0,1,1,2,2,3,3,4,4,0,2,1,3],dtype=cp.int32)
dsts=cp.asarray([1,2,2,3,3,4,4,0,0,1,3,0,4,1],dtype=cp.int32)
w=cp.asarray([1.0]*len(srcs),dtype=cp.float32)
h=plc.ResourceHandle(); props=plc.GraphProperties(is_symmetric=False,is_multigraph=False)
G=plc.SGGraph(h,props,srcs,dsts,weight_array=w,store_transposed=False,renumber=True,do_expensive_check=False)
print("graph built (5 nodes, 14 edges)", flush=True)
start=cp.asarray([0,1],dtype=cp.int32); fanout=np.asarray([2,2],dtype=np.int32)
attempts=[
 ("pos6", lambda: plc.uniform_neighbor_sample(h,G,start,fanout,False,False)),
 ("kw", lambda: plc.uniform_neighbor_sample(h,G,start_list=start,h_fan_out=fanout,with_replacement=False,do_expensive_check=False)),
]
done=False
for name,fn in attempts:
    if done: break
    try:
        res=fn(); k=len(res) if isinstance(res,(tuple,list)) else 1
        print(f"SAMPLE-OK ({name}) returned {k} arrays", flush=True); done=True
    except TypeError as e:
        print(f"SIG ({name}): {str(e)[:130]}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"ERR ({name}): {type(e).__name__}: {str(e)[:150]}", flush=True); done=True
print("DONE", flush=True)
