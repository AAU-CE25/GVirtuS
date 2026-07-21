import sys
def p(m): print(f"[numba_probe] {m}", flush=True)

p("import numba")
import numba
p(f"numba {numba.__version__}")
p("import numba.cuda")
from numba import cuda
p("cuda.is_available() ...")
try:
    p(f"  is_available -> {cuda.is_available()}")
except Exception as e:
    p(f"  is_available EXC {e!r}")
p("cuda.detect() ...")
try:
    p(f"  detect -> {cuda.detect()}")
except Exception as e:
    p(f"  detect EXC {e!r}")
p("cuda.current_context() ...")
try:
    ctx = cuda.current_context()
    p(f"  current_context -> {ctx}")
except Exception as e:
    p(f"  current_context EXC {e!r}")
p("DONE")
