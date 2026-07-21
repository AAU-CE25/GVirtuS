import ctypes
def sym(lib, s):
    try:
        h=ctypes.CDLL(lib); return ctypes.cast(getattr(h,s), ctypes.c_void_p).value
    except Exception as e: return None
import rmm
rmm.reinitialize(pool_allocator=False, managed_memory=False)
b=rmm.DeviceBuffer(size=64)
for s in ('cudaMemcpyAsync','cudaMalloc'):
    a_gvs=sym('/opt/GVirtuS/lib/frontend/libcudart.so.12', s)
    a_glob=sym(None, s)
    print(f'ADDR {s}: gvs_libcudart={hex(a_gvs) if a_gvs else a_gvs}  RTLD_DEFAULT={hex(a_glob) if a_glob else a_glob}  SAME={a_gvs==a_glob}', flush=True)
    if a_glob:
        with open('/proc/self/maps') as f:
            for line in f:
                p=line.split(); lo,hi=[int(x,16) for x in p[0].split('-')]
                if lo<=a_glob<hi: print(f'   RTLD_DEFAULT {s} resolves into: {p[-1]}', flush=True); break
print('DONE', flush=True)
