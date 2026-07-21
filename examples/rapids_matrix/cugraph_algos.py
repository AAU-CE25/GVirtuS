import faulthandler; faulthandler.enable()
import pylibcugraph as plc, cupy as cp
print("pylibcugraph", plc.__version__, flush=True)
# small SYMMETRIC graph (both directions) so undirected algos work: 0-1-2-3-0 ring + chords
s=[0,1,1,2,2,3,3,0]; d=[1,0,2,1,3,2,0,3]
src=cp.asarray(s+d,dtype=cp.int32); dst=cp.asarray(d+s,dtype=cp.int32)
w=cp.asarray([1.0]*len(src),dtype=cp.float32)
h=plc.ResourceHandle(); props=plc.GraphProperties(is_symmetric=True,is_multigraph=False)
G=plc.SGGraph(h,props,src,dst,weight_array=w,store_transposed=False,renumber=True,do_expensive_check=False)
print("graph built", flush=True)
import inspect
def test(name, fn):
    try:
        fn(); print(f"OK   {name}", flush=True)
    except TypeError as e:
        print(f"SIG  {name}: {str(e)[:90]}", flush=True)   # my signature wrong, not a GVirtuS gap
    except Exception as e:
        print(f"ERR  {name}: {type(e).__name__}: {str(e)[:110]}", flush=True)
test("bfs",            lambda: plc.bfs(h,G,cp.asarray([0],dtype=cp.int32),False,10,True,False))
test("sssp",           lambda: plc.sssp(h,G,0,1e30,True,False))
test("hits",           lambda: plc.hits(h,G,1e-5,100,None,None,False,False))
test("core_number",    lambda: plc.core_number(h,G,"bidirectional",False))
test("triangle_count", lambda: plc.triangle_count(h,G,None,False))
test("louvain",        lambda: plc.louvain(h,G,100,1e-7,1.0,False))
test("katz",           lambda: plc.katz_centrality(h,G,None,0.01,1.0,1e-6,100,False,False))
test("wcc",            lambda: plc.weakly_connected_components(h,G,None,None,None,None,False))
print("DONE", flush=True)
