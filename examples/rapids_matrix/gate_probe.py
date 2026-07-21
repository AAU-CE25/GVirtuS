import ctypes, numpy as np, rmm
def P(n,v): print(f'GATE {n}: {v}', flush=True)
rmm.reinitialize(pool_allocator=False, managed_memory=False)
b = rmm.DeviceBuffer(size=64)
lib = ctypes.CDLL('libcudart.so.12'); vp=ctypes.c_void_p; sz=ctypes.c_size_t
lib.cudaMemcpy.argtypes=[vp,vp,sz,ctypes.c_int]
lib.cudaMemcpyAsync.argtypes=[vp,vp,sz,ctypes.c_int,vp]
lib.cudaStreamSynchronize.argtypes=[vp]
H2D=1; D2H=2
src=(ctypes.c_ubyte*64)(*range(64)); lib.cudaMemcpy(vp(b.ptr), ctypes.cast(src,vp),64,H2D)
P('b.ptr', hex(b.ptr)); P('b.ptr hi32', hex(b.ptr>>32))
# ctypes dst (low heap)
dctypes=(ctypes.c_ubyte*64)()
addr_c=ctypes.addressof(dctypes)
P('ctypes-dst addr', hex(addr_c)); P('ctypes-dst hi32', hex(addr_c>>32))
P('D2H ctypes-dst', lib.cudaMemcpyAsync(vp(addr_c), vp(b.ptr),64,D2H,None)); lib.cudaStreamSynchronize(None)
# numpy dst (mmap 0x7f..)
dnp=np.empty(64,np.uint8); addr_n=dnp.ctypes.data
P('numpy-dst addr', hex(addr_n)); P('numpy-dst hi32', hex(addr_n>>32))
P('hi32 match src?', (addr_n>>32)==(b.ptr>>32))
P('D2H numpy-dst', lib.cudaMemcpyAsync(vp(addr_n), vp(b.ptr),64,D2H,None)); lib.cudaStreamSynchronize(None)
P('numpy first8', dnp[:8].tolist())
P('DONE','')
