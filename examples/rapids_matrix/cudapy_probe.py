import sys
def p(msg): print(f"[cudapy_probe] {msg}", flush=True)

p("import cuda.bindings runtime+driver")
from cuda.bindings import runtime, driver

# DRIVER API layer (should reach GVS libcuda shim -> RPC for device enum)
p("driver.cuInit(0) ...")
try: p(f"  cuInit -> {driver.cuInit(0)}")
except Exception as e: p(f"  cuInit EXC {e!r}")
p("driver.cuDeviceGetCount() ...")
try: p(f"  cuDeviceGetCount -> {driver.cuDeviceGetCount()}")
except Exception as e: p(f"  cuDeviceGetCount EXC {e!r}")
p("driver.cuDeviceGet(0) ...")
try: p(f"  cuDeviceGet(0) -> {driver.cuDeviceGet(0)}")
except Exception as e: p(f"  cuDeviceGet EXC {e!r}")

# RUNTIME API layer (RMM uses this) - does it reach GVS libcudart?
p("runtime.cudaGetDeviceCount() ...")
try: p(f"  cudaGetDeviceCount -> {runtime.cudaGetDeviceCount()}")
except Exception as e: p(f"  cudaGetDeviceCount EXC {e!r}")
p("runtime.cudaGetDevice() ...")
try: p(f"  cudaGetDevice -> {runtime.cudaGetDevice()}")
except Exception as e: p(f"  cudaGetDevice EXC {e!r}")
p("runtime.cudaSetDevice(0) ...")
try: p(f"  cudaSetDevice(0) -> {runtime.cudaSetDevice(0)}")
except Exception as e: p(f"  cudaSetDevice EXC {e!r}")

p("DONE")
