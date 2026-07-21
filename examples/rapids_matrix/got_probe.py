import ctypes, rmm
def P(m): print(f'[got] {m}', flush=True)
rmm.reinitialize(pool_allocator=False, managed_memory=False)
b=rmm.DeviceBuffer(size=64)
gvs=ctypes.CDLL('/opt/GVirtuS/lib/frontend/libcudart.so.12')
myaddr=ctypes.cast(gvs.cudaMemcpyAsync, ctypes.c_void_p).value
P(f'GVirtuS cudaMemcpyAsync addr = {hex(myaddr)}')
import numpy as np
try:
    b.copy_to_host()
except Exception as e:
    P(f'EXC {e!r}')
# find device_buffer.so base (lowest mapping)
base=None; dbpath=None
with open('/proc/self/maps') as f:
    for line in f:
        if 'device_buffer.cpython' in line:
            st=int(line.split('-')[0],16)
            if base is None: base=st; dbpath=line.split()[-1]
P(f'device_buffer base={hex(base)}')
got=base+0x9f5e8
val=ctypes.cast(got, ctypes.POINTER(ctypes.c_uint64)).contents.value
P(f'GOT[cudaMemcpyAsync] resolved = {hex(val)}  == GVirtuS? {val==myaddr}')
with open('/proc/self/maps') as f:
    for line in f:
        p=line.split(); lo,hi=[int(x,16) for x in p[0].split('-')]
        if lo<=val<hi: P(f'GOT target lib: {p[-1]}'); break
P('DONE')
