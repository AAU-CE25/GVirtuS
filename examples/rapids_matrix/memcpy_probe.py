import ctypes
lib = ctypes.CDLL('libcudart.so.12')   # GVirtuS (LD_PRELOAD'd)
vp=ctypes.c_void_p; sz=ctypes.c_size_t
lib.cudaMalloc.argtypes=[ctypes.POINTER(vp),sz]
lib.cudaMemcpy.argtypes=[vp,vp,sz,ctypes.c_int]
lib.cudaMemcpyAsync.argtypes=[vp,vp,sz,ctypes.c_int,vp]
lib.cudaStreamCreate.argtypes=[ctypes.POINTER(vp)]
lib.cudaStreamSynchronize.argtypes=[vp]
H2D=1; D2H=2
def P(n,v): print(f'[probe] {n}: {v}', flush=True)
d=vp()
P('cudaMalloc(64)', lib.cudaMalloc(ctypes.byref(d),64)); P('  dptr', hex(d.value or 0))
src=(ctypes.c_ubyte*64)(*range(64))
P('H2D sync', lib.cudaMemcpy(d, ctypes.cast(src,vp), 64, H2D))
dst=(ctypes.c_ubyte*64)()
P('D2H SYNC', lib.cudaMemcpy(ctypes.cast(dst,vp), d, 64, D2H)); P('  first8', list(dst[:8]))
d2=(ctypes.c_ubyte*64)()
P('D2H ASYNC stream0', lib.cudaMemcpyAsync(ctypes.cast(d2,vp), d, 64, D2H, None))
P('sync stream0', lib.cudaStreamSynchronize(None)); P('  first8', list(d2[:8]))
st=vp()
P('cudaStreamCreate', lib.cudaStreamCreate(ctypes.byref(st))); P('  stream', hex(st.value or 0))
d3=(ctypes.c_ubyte*64)()
P('D2H ASYNC createdStream', lib.cudaMemcpyAsync(ctypes.cast(d3,vp), d, 64, D2H, st))
P('sync createdStream', lib.cudaStreamSynchronize(st)); P('  first8', list(d3[:8]))
P('DONE','')
