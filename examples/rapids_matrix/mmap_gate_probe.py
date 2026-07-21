import ctypes, rmm
def P(n,v): print(f'MG {n}: {v}', flush=True)
rmm.reinitialize(pool_allocator=False, managed_memory=False)
b = rmm.DeviceBuffer(size=64)
lib = ctypes.CDLL('libcudart.so.12'); vp=ctypes.c_void_p; sz=ctypes.c_size_t
lib.cudaMemcpy.argtypes=[vp,vp,sz,ctypes.c_int]
lib.cudaMemcpyAsync.argtypes=[vp,vp,sz,ctypes.c_int,vp]
lib.cudaStreamSynchronize.argtypes=[vp]
H2D=1; D2H=2
srcbuf=(ctypes.c_ubyte*64)(*range(64)); lib.cudaMemcpy(vp(b.ptr), ctypes.cast(srcbuf,vp),64,H2D)
P('b.ptr', hex(b.ptr)); P('b.ptr_hi32', hex(b.ptr>>32))
libc = ctypes.CDLL('libc.so.6'); libc.mmap.restype=ctypes.c_void_p
libc.mmap.argtypes=[vp,sz,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
base = b.ptr & ~0xffffffff; p=0
for off in (0x100000, 0x2000000, 0x10000000, 0x40000000):
    cand = libc.mmap(vp(base+off), 4096, 3, 0x22, -1, 0)
    if cand and (cand>>32)==(b.ptr>>32): p=cand; break
P('mmap_addr', hex(p)); P('mmap_hi32', hex(p>>32)); P('hi32_match', (p>>32)==(b.ptr>>32))
e=lib.cudaMemcpyAsync(vp(p), vp(b.ptr), 64, D2H, None); lib.cudaStreamSynchronize(None)
P('D2H_high_matching_dst', e)
low=(ctypes.c_ubyte*64)()
e2=lib.cudaMemcpyAsync(ctypes.cast(low,vp), vp(b.ptr), 64, D2H, None); lib.cudaStreamSynchronize(None)
P('D2H_far_low_dst', e2)
P('DONE','')
