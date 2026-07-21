import cupy
print('cupy', cupy.__version__, flush=True)
try:
    a = cupy.asarray([1,2,3,4])
    print('asarray+sum =', int(a.sum()), flush=True)
    b = (a * 2).get()   # elementwise + D2H
    print('elementwise+get =', b.tolist(), flush=True)
    print('CUPY STANDALONE OK', flush=True)
except Exception as e:
    print('ERR:', type(e).__name__, str(e)[:130], flush=True)
