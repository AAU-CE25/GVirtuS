import faulthandler; faulthandler.enable()
import cudf, cupy
from rmm import pylibrmm
from rmm.pylibrmm.stream import Stream
st = Stream(obj=cupy.cuda.get_current_stream())
buf = pylibrmm.device_buffer.DeviceBuffer(size=24, stream=st)
print("A DeviceBuffer ok ptr=", hex(buf.ptr), flush=True)
# Hypothesis: device_id=-1 triggers cudaPointerGetAttributes -> crash. Try explicit 0.
print("B UnownedMemory device_id=0 (explicit, skips pointerGetAttributes)...", flush=True)
try:
    mem = cupy.cuda.UnownedMemory(ptr=buf.ptr, size=buf.size, owner=buf, device_id=0)
    print("   OK device_id=0 works -> pointerGetAttributes is the culprit", flush=True)
except Exception as e:
    print("   ERR", type(e).__name__, str(e)[:80], flush=True)
print("C direct cudaPointerGetAttributes on rmm ptr...", flush=True)
import cupy_backends.cuda.api.runtime as rt
try:
    a = rt.pointerGetAttributes(buf.ptr)
    print("   pointerGetAttributes OK device=", a.device, "type=", a.type, flush=True)
except Exception as e:
    print("   pointerGetAttributes ERR", type(e).__name__, str(e)[:80], flush=True)
print("ALL DONE", flush=True)
