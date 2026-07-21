print('start', flush=True)
import cudf
print('cudf imported', flush=True)
import cupy
print('cupy imported', flush=True)
import cupy_backends.cuda.api.runtime as rt
try:
    print('getDeviceCount =', rt.getDeviceCount(), flush=True)
except Exception as e:
    print('getDeviceCount ERR:', type(e).__name__, str(e)[:140], flush=True)
try:
    a = cupy.array([1,2,3]); print('cupy.array sum =', int(a.sum()), flush=True)
except Exception as e:
    print('cupy.array ERR:', type(e).__name__, str(e)[:140], flush=True)
# which libcudart does cupy see?
import ctypes.util, os
print('which libcudart on maps:', flush=True)
os.system("grep -aoE '/[^ ]*libcudart[^ ]*' /proc/self/maps | sort -u")
print('DONE', flush=True)
