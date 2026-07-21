import ctypes
lib = ctypes.CDLL('libcudart.so.12')
vp=ctypes.c_void_p; sz=ctypes.c_size_t
lib.cudaMalloc.argtypes=[ctypes.POINTER(vp),sz]
lib.cudaMemcpy.argtypes=[vp,vp,sz,ctypes.c_int]
lib.cudaMemcpyAsync.argtypes=[vp,vp,sz,ctypes.c_int,vp]
lib.cudaStreamSynchronize.argtypes=[vp]
H2D=1; D2H=2
def P(n,v): print(f'[sprobe] {n}: {v}', flush=True)
d=vp(); lib.cudaMalloc(ctypes.byref(d),64)
src=(ctypes.c_ubyte*64)(*range(64)); lib.cudaMemcpy(d, ctypes.cast(src,vp),64,H2D)
for name,sval in [('legacy(0x1)',1),('PTDS(0x2)',2)]:
    dd=(ctypes.c_ubyte*64)()
    e=lib.cudaMemcpyAsync(ctypes.cast(dd,vp), d, 64, D2H, vp(sval))
    s=lib.cudaStreamSynchronize(vp(sval))
    P(f'D2H ASYNC {name} memcpy', e); P(f'D2H ASYNC {name} sync', s); P('  first8', list(dd[:8]))
# what stream does RMM actually use?
try:
    import rmm
    rmm.reinitialize(pool_allocator=False, managed_memory=False)
    from rmm.pylibrmm.stream import DEFAULT_STREAM
    b=rmm.DeviceBuffer(size=64)
    P('rmm DEFAULT_STREAM repr', repr(DEFAULT_STREAM))
    for attr in ('value','_stream','c_stream'):
        if hasattr(DEFAULT_STREAM, attr): P(f'  DEFAULT_STREAM.{attr}', getattr(DEFAULT_STREAM,attr))
except Exception as ex:
    P('rmm introspect EXC', repr(ex)[:120])
P('DONE','')
