import faulthandler; faulthandler.enable()
print("importing pylibcugraph...", flush=True)
import pylibcugraph as plc
print("PYLIBCUGRAPH-IMPORT-OK", getattr(plc,"__version__","?"), flush=True)
import cupy as cp
print("building small graph (6 edges, 4 nodes)...", flush=True)
srcs = cp.asarray([0,1,1,2,2,3], dtype=cp.int32)
dsts = cp.asarray([1,0,2,1,3,2], dtype=cp.int32)
wgts = cp.asarray([1.0]*6, dtype=cp.float32)
handle = plc.ResourceHandle()
props  = plc.GraphProperties(is_symmetric=False, is_multigraph=False)
G = plc.SGGraph(handle, props, srcs, dsts, weight_array=wgts,
                store_transposed=True, renumber=True, do_expensive_check=False)
print("SGGraph-BUILT-OK", flush=True)
try:
    res = plc.pagerank(handle, G, None, None, None, None,
                       alpha=0.85, epsilon=1e-6, max_iterations=100,
                       do_expensive_check=False)
    verts, prs = res[0], res[1]
    print("PAGERANK-OK verts=", cp.asnumpy(verts).tolist(),
          "pr=", [round(float(x),4) for x in cp.asnumpy(prs).tolist()], flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print("PAGERANK-ERR", type(e).__name__, str(e)[:200], flush=True)
print("DONE", flush=True)
