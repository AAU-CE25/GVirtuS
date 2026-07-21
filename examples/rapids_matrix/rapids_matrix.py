import numpy as np, cupy as cp

print("CuPy", cp.__version__)

# Matmul GRANDE: cada matriz 2048x2048x4 = 16 MB (> umbral 4 MB de GPUDirect)
# -> las subidas H2D (cp.asarray) y la bajada D2H (cp.asnumpy) deberian disparar B3 gpu-split
n = 2048
mb = n * n * 4 // (1024 * 1024)
ha = np.ones((n, n), np.float32)
hb = np.ones((n, n), np.float32)

da = cp.asarray(ha)   # H2D 16 MB
db = cp.asarray(hb)   # H2D 16 MB
dc = da @ db          # cublasGemmEx en la L40S remota
hc = cp.asnumpy(dc)   # D2H 16 MB

err = float(np.abs(hc - (ha @ hb)).max())
print("matmul %dx%d (%dMB/matrix) max_abs_diff=%.2e" % (n, n, mb, err),
      "PASS" if err < 1.0 else "FAIL")
print("ALL DONE")
