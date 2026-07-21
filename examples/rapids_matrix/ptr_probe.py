import ctypes, numpy as np, rmm
def P(n,v): print(f'PROBE {n}: {v}', flush=True)
rmm.reinitialize(pool_allocator=False, managed_memory=False)
b = rmm.DeviceBuffer(size=64)
P('b.ptr', hex(b.ptr)); P('b.size', b.size)
lib = ctypes.CDLL('libcudart.so.12'); vp=ctypes.c_void_p; sz=ctypes.c_size_t
lib.cudaMemcpy.argtypes=[vp,vp,sz,ctypes.c_int]
lib.cudaMemcpyAsync.argtypes=[vp,vp,sz,ctypes.c_int,vp]
lib.cudaStreamSynchronize.argtypes=[vp]
H2D=1; D2H=2
src=(ctypes.c_ubyte*64)(*range(64))
P('H2D to b.ptr', lib.cudaMemcpy(vp(b.ptr), ctypes.cast(src,vp), 64, H2D))
dst=(ctypes.c_ubyte*64)()
P('D2H async stream0 on b.ptr', lib.cudaMemcpyAsync(ctypes.cast(dst,vp), vp(b.ptr), 64, D2H, None))
P('sync stream0', lib.cudaStreamSynchronize(None)); P('  first8', list(dst[:8]))
# now the real RMM path
try:
    h = np.frombuffer(b.copy_to_host(), dtype=np.uint8)
    P('rmm copy_to_host OK first8', h[:8].tolist())
except Exception as e:
    P('rmm copy_to_host EXC', repr(e)[:140])
P('DONE','')
