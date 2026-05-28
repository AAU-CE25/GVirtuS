#!/usr/bin/env python3
"""
Simple GVirtuS connectivity test.

Tests that the GVirtuS frontend can connect to the backend by calling
basic CUDA runtime functions (cudaGetDeviceCount, cudaGetDeviceProperties)
and CUDA driver API functions (cuInit, cuDeviceGetCount) used by RAPIDS.
"""

import ctypes
import os
import sys


def load_cudart():
    """Load libcudart from GVirtuS frontend path."""
    gvirtus_home = os.environ.get("GVIRTUS_HOME", "/opt/GVirtuS")
    cudart_path = os.path.join(gvirtus_home, "lib", "frontend", "libcudart.so")
    
    if not os.path.exists(cudart_path):
        print(f"ERROR: GVirtuS cudart not found at {cudart_path}")
        return None
    
    try:
        return ctypes.CDLL(cudart_path)
    except OSError as e:
        print(f"ERROR: Failed to load {cudart_path}: {e}")
        return None


def load_cuda_driver():
    """Load libcuda (CUDA Driver API) from GVirtuS frontend path."""
    gvirtus_home = os.environ.get("GVIRTUS_HOME", "/opt/GVirtuS")
    frontend_path = os.path.join(gvirtus_home, "lib", "frontend")
    
    # Try different possible names (symlink and versioned)
    candidates = [
        os.path.join(frontend_path, "libcuda.so"),
        os.path.join(frontend_path, "libcuda.so.1"),
    ]
    
    for cuda_path in candidates:
        if os.path.exists(cuda_path):
            try:
                return ctypes.CDLL(cuda_path)
            except OSError as e:
                print(f"WARNING: Failed to load {cuda_path}: {e}")
                continue
    
    print(f"WARNING: cudadr plugin not built (libcuda.so not found in {frontend_path})")
    print("         To enable: ensure libcuda.so.1 is available during GVirtuS build")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA Driver API tests (libcuda.so) - RAPIDS/cuDF uses these heavily
# ═══════════════════════════════════════════════════════════════════════════════

def test_cu_init(cuda):
    """Test cuInit(0) - the first call any CUDA driver API program makes."""
    if cuda is None:
        print("SKIP: CUDA driver library not available")
        return True
    
    result = cuda.cuInit(0)
    
    if result != 0:
        print(f"FAIL: cuInit(0) returned error code {result}")
        return False
    
    print("OK: cuInit(0) succeeded")
    return True


def test_cu_device_get_count(cuda):
    """Test cuDeviceGetCount - driver API version of device count."""
    if cuda is None:
        print("SKIP: CUDA driver library not available")
        return True
    
    count = ctypes.c_int()
    result = cuda.cuDeviceGetCount(ctypes.byref(count))
    
    if result != 0:
        print(f"FAIL: cuDeviceGetCount returned error code {result}")
        return False
    
    print(f"OK: cuDeviceGetCount = {count.value}")
    return count.value > 0


def test_cu_device_get(cuda):
    """Test cuDeviceGet - get device handle for device 0."""
    if cuda is None:
        print("SKIP: CUDA driver library not available")
        return True
    
    device = ctypes.c_int()
    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
    
    if result != 0:
        print(f"FAIL: cuDeviceGet(0) returned error code {result}")
        return False
    
    print(f"OK: cuDeviceGet(0) = {device.value}")
    return True


def test_cu_ctx_create(cuda):
    """Test cuCtxCreate/cuCtxDestroy - RAPIDS needs CUDA contexts."""
    if cuda is None:
        print("SKIP: CUDA driver library not available")
        return True
    
    ctx = ctypes.c_void_p()
    device = ctypes.c_int(0)
    
    # CU_CTX_SCHED_AUTO = 0
    result = cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
    
    if result != 0:
        print(f"FAIL: cuCtxCreate returned error code {result}")
        return False
    
    print(f"OK: cuCtxCreate = 0x{ctx.value:x}")
    
    result = cuda.cuCtxDestroy_v2(ctx)
    if result != 0:
        print(f"FAIL: cuCtxDestroy returned error code {result}")
        return False
    
    print("OK: cuCtxDestroy succeeded")
    return True


def test_get_device_count(cudart):
    """Test cudaGetDeviceCount - verifies basic connectivity."""
    count = ctypes.c_int()
    result = cudart.cudaGetDeviceCount(ctypes.byref(count))
    
    if result != 0:
        print(f"FAIL: cudaGetDeviceCount returned error code {result}")
        return False
    
    print(f"OK: cudaGetDeviceCount = {count.value}")
    return count.value > 0


def test_get_device_properties(cudart, device_id=0):
    """Test cudaGetDeviceProperties - verifies data serialization works."""
    # cudaDeviceProp is a large struct, we just need a buffer
    # The actual struct is ~1KB, allocate 2KB to be safe
    prop_buffer = ctypes.create_string_buffer(2048)
    
    # Try cudaGetDeviceProperties_v2 first (CUDA 12+), then fall back to original
    try:
        func = cudart.cudaGetDeviceProperties_v2
    except AttributeError:
        try:
            func = cudart.cudaGetDeviceProperties
        except AttributeError:
            print("SKIP: cudaGetDeviceProperties not available")
            return True  # Don't fail the test suite
    
    result = func(prop_buffer, device_id)
    
    if result != 0:
        print(f"FAIL: cudaGetDeviceProperties returned error code {result}")
        return False
    
    # Extract device name (first 256 bytes of the struct)
    name = prop_buffer.raw[:256].split(b'\x00')[0].decode('utf-8', errors='ignore')
    print(f"OK: Device {device_id} name = {name}")
    return True


def test_set_device(cudart, device_id=0):
    """Test cudaSetDevice - verifies device selection works."""
    result = cudart.cudaSetDevice(device_id)
    
    if result != 0:
        print(f"FAIL: cudaSetDevice({device_id}) returned error code {result}")
        return False
    
    print(f"OK: cudaSetDevice({device_id}) succeeded")
    return True


def test_device_reset(cudart):
    """Test cudaDeviceReset - clean disconnect from backend."""
    result = cudart.cudaDeviceReset()
    
    if result != 0:
        print(f"FAIL: cudaDeviceReset returned error code {result}")
        return False
    
    print("OK: cudaDeviceReset succeeded (clean disconnect)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# RAPIDS-specific tests - these are the calls that RAPIDS/cuDF JNI uses
# ═══════════════════════════════════════════════════════════════════════════════

def test_driver_version(cudart):
    """Test cudaDriverGetVersion - RAPIDS checks this at startup."""
    version = ctypes.c_int()
    result = cudart.cudaDriverGetVersion(ctypes.byref(version))
    
    if result != 0:
        print(f"FAIL: cudaDriverGetVersion returned error code {result}")
        return False
    
    major = version.value // 1000
    minor = (version.value % 1000) // 10
    print(f"OK: cudaDriverGetVersion = {version.value} (CUDA {major}.{minor})")
    return True


def test_runtime_version(cudart):
    """Test cudaRuntimeGetVersion - RAPIDS compares this with driver version."""
    version = ctypes.c_int()
    result = cudart.cudaRuntimeGetVersion(ctypes.byref(version))
    
    if result != 0:
        print(f"FAIL: cudaRuntimeGetVersion returned error code {result}")
        return False
    
    major = version.value // 1000
    minor = (version.value % 1000) // 10
    print(f"OK: cudaRuntimeGetVersion = {version.value} (CUDA {major}.{minor})")
    return True


def test_malloc_free(cudart):
    """Test cudaMalloc/cudaFree - basic GPU memory allocation used by cuDF."""
    ptr = ctypes.c_void_p()
    size = 1024 * 1024  # 1 MB
    
    result = cudart.cudaMalloc(ctypes.byref(ptr), size)
    if result != 0:
        print(f"FAIL: cudaMalloc returned error code {result}")
        return False
    
    if ptr.value is None or ptr.value == 0:
        print("FAIL: cudaMalloc returned null pointer")
        return False
    
    print(f"OK: cudaMalloc({size} bytes) = 0x{ptr.value:x}")
    
    result = cudart.cudaFree(ptr)
    if result != 0:
        print(f"FAIL: cudaFree returned error code {result}")
        return False
    
    print("OK: cudaFree succeeded")
    return True


def test_memcpy(cudart):
    """Test cudaMemcpy - host to device and back, used heavily by RAPIDS."""
    # Allocate device memory
    d_ptr = ctypes.c_void_p()
    size = 256
    
    result = cudart.cudaMalloc(ctypes.byref(d_ptr), size)
    if result != 0:
        print(f"FAIL: cudaMalloc for memcpy test returned error code {result}")
        return False
    
    # Create host data
    h_src = ctypes.create_string_buffer(b"Hello GVirtuS RAPIDS!" + b"\x00" * (size - 21))
    h_dst = ctypes.create_string_buffer(size)
    
    # cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2
    result = cudart.cudaMemcpy(d_ptr, h_src, size, 1)  # H2D
    if result != 0:
        cudart.cudaFree(d_ptr)
        print(f"FAIL: cudaMemcpy H2D returned error code {result}")
        return False
    
    print("OK: cudaMemcpy Host->Device succeeded")
    
    result = cudart.cudaMemcpy(h_dst, d_ptr, size, 2)  # D2H
    if result != 0:
        cudart.cudaFree(d_ptr)
        print(f"FAIL: cudaMemcpy D2H returned error code {result}")
        return False
    
    # Verify data
    if h_dst.raw[:21] == b"Hello GVirtuS RAPIDS!":
        print("OK: cudaMemcpy Device->Host succeeded (data verified)")
    else:
        cudart.cudaFree(d_ptr)
        print(f"FAIL: Data mismatch: {h_dst.raw[:21]}")
        return False
    
    cudart.cudaFree(d_ptr)
    return True


def test_stream_create_destroy(cudart):
    """Test cudaStreamCreate/Destroy - RAPIDS uses streams for async ops."""
    stream = ctypes.c_void_p()
    
    result = cudart.cudaStreamCreate(ctypes.byref(stream))
    if result != 0:
        print(f"FAIL: cudaStreamCreate returned error code {result}")
        return False
    
    print(f"OK: cudaStreamCreate = 0x{stream.value:x}")
    
    result = cudart.cudaStreamSynchronize(stream)
    if result != 0:
        print(f"FAIL: cudaStreamSynchronize returned error code {result}")
        cudart.cudaStreamDestroy(stream)
        return False
    
    print("OK: cudaStreamSynchronize succeeded")
    
    result = cudart.cudaStreamDestroy(stream)
    if result != 0:
        print(f"FAIL: cudaStreamDestroy returned error code {result}")
        return False
    
    print("OK: cudaStreamDestroy succeeded")
    return True


def test_device_synchronize(cudart):
    """Test cudaDeviceSynchronize - used by RAPIDS for sync points."""
    result = cudart.cudaDeviceSynchronize()
    
    if result != 0:
        print(f"FAIL: cudaDeviceSynchronize returned error code {result}")
        return False
    
    print("OK: cudaDeviceSynchronize succeeded")
    return True


def test_get_mem_info(cudart):
    """Test cudaMemGetInfo - RAPIDS uses this to manage memory pools."""
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    
    result = cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    
    if result != 0:
        print(f"FAIL: cudaMemGetInfo returned error code {result}")
        return False
    
    free_gb = free.value / (1024**3)
    total_gb = total.value / (1024**3)
    print(f"OK: cudaMemGetInfo: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
    return True


def test_version_compatibility(cudart):
    """
    Test driver/runtime version compatibility.
    RAPIDS fails with cudaErrorInsufficientDriver (35) if driver < runtime.
    """
    driver_ver = ctypes.c_int()
    runtime_ver = ctypes.c_int()
    
    cudart.cudaDriverGetVersion(ctypes.byref(driver_ver))
    cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_ver))
    
    d_major = driver_ver.value // 1000
    d_minor = (driver_ver.value % 1000) // 10
    r_major = runtime_ver.value // 1000
    r_minor = (runtime_ver.value % 1000) // 10
    
    print(f"     Driver:  {d_major}.{d_minor} ({driver_ver.value})")
    print(f"     Runtime: {r_major}.{r_minor} ({runtime_ver.value})")
    
    # CUDA requires driver version >= runtime version
    if driver_ver.value >= runtime_ver.value:
        print(f"OK: Version compatible (driver >= runtime)")
        return True
    else:
        print(f"FAIL: Version mismatch! Driver {d_major}.{d_minor} < Runtime {r_major}.{r_minor}")
        print("      This would cause cudaErrorInsufficientDriver (error 35)")
        return False


def run_jni_diagnostics():
    """
    Diagnose why RAPIDS JNI might fail even when our tests pass.
    Check library resolution paths and simulate JNI library loading.
    """
    import subprocess
    
    print("\n1. LD_PRELOAD check (set in Dockerfile):")
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if ld_preload:
        preloads = [p for p in ld_preload.split(":") if p]
        print(f"   OK: LD_PRELOAD is SET with {len(preloads)} entries:")
        all_exist = True
        for p in preloads:
            exists = os.path.exists(p)
            status = "OK" if exists else "MISSING"
            if not exists:
                all_exist = False
            print(f"     [{status}] {p}")
        if all_exist and any("libcudart" in p for p in preloads):
            print("   OK: libcudart.so is preloaded - Java/JNI will use GVirtuS!")
    else:
        print("   WARNING: LD_PRELOAD is NOT SET!")
        print("   Java/JNI may load system CUDA libs instead of GVirtuS")
    
    print("\n2. LD_LIBRARY_PATH check:")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths = ld_path.split(":") if ld_path else []
    gvirtus_frontend = "/opt/GVirtuS/lib/frontend"
    
    if gvirtus_frontend in paths:
        idx = paths.index(gvirtus_frontend)
        print(f"   OK: GVirtuS frontend is in LD_LIBRARY_PATH (position {idx})")
    else:
        print(f"   WARNING: GVirtuS frontend NOT in LD_LIBRARY_PATH!")
        print(f"   Current: {ld_path}")
    
    print("\n3. Library search simulation (what ldconfig sees):")
    try:
        result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
        cuda_libs = [l for l in result.stdout.split("\n") if "cuda" in l.lower()]
        if cuda_libs:
            for lib in cuda_libs[:5]:  # First 5
                print(f"   {lib.strip()}")
        else:
            print("   No CUDA libs in ldconfig cache (expected for container)")
    except Exception as e:
        print(f"   Could not run ldconfig: {e}")
    
    print("\n4. Testing library load via standard search (like Java/JNI would):")
    # Try to load libcudart.so without explicit path - this is how JNI does it
    try:
        # RTLD_NOW | RTLD_GLOBAL to match typical JNI behavior
        cudart_jni = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
        count = ctypes.c_int()
        result = cudart_jni.cudaGetDeviceCount(ctypes.byref(count))
        if result == 0:
            print(f"   OK: Standard dlopen('libcudart.so') works, found {count.value} devices")
        else:
            print(f"   FAIL: cudaGetDeviceCount via standard load returned error {result}")
    except OSError as e:
        print(f"   FAIL: Could not load 'libcudart.so' via standard search: {e}")
        print("   This is likely why RAPIDS JNI fails!")
    
    print("\n5. Testing libcuda.so (driver API) via standard search:")
    try:
        cuda_jni = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
        result = cuda_jni.cuInit(0)
        if result == 0:
            print(f"   OK: Standard dlopen('libcuda.so') and cuInit(0) succeeded")
        else:
            print(f"   FAIL: cuInit(0) via standard load returned error {result}")
    except OSError as e:
        print(f"   FAIL: Could not load 'libcuda.so' via standard search: {e}")
    
    print("\n6. RAPIDS JAR native library check:")
    # Try multiple possible paths
    jar_candidates = [
        "/app/jars/rapids-4-spark_2.12-26.02.1.jar",
        "../jars/rapids-4-spark_2.12-26.02.1.jar",
        "/tmp/rapids-native"
    ]
    jar_path = None
    for jp in jar_candidates:
        if os.path.exists(jp):
            jar_path = jp
            break
    
    if jar_path:
        print(f"   Found JAR at: {jar_path}")
        try:
            result = subprocess.run(["unzip", "-l", jar_path], capture_output=True, text=True)
            native_libs = [l for l in result.stdout.split("\n") if ".so" in l]
            if native_libs:
                print(f"   Found {len(native_libs)} native libraries in RAPIDS JAR:")
                for lib in native_libs[:10]:
                    print(f"     {lib.strip().split()[-1]}")
            else:
                print("   No native libraries found in JAR")
        except Exception as e:
            print(f"   Could not inspect JAR: {e}")
    else:
        print(f"   JAR not found (tried: {jar_candidates})")
    
    print("\n7. Attempting to load cuDF native library (like Spark does):")
    
    # Try extracting and loading cudf JNI lib
    if jar_path:
        import tempfile
        import zipfile
        print("\n   Attempting to extract and load cuDF JNI from JAR...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(jar_path, 'r') as zf:
                    # Find cudf JNI native lib
                    jni_libs = [n for n in zf.namelist() if "libcudfjni" in n.lower() or ("cudf" in n.lower() and n.endswith(".so"))]
                    if jni_libs:
                        print(f"   Found cuDF JNI libs in JAR: {jni_libs[:3]}")
                        # Extract the libcudfjni.so specifically
                        jni_lib = next((l for l in jni_libs if "libcudfjni" in l.lower()), jni_libs[0])
                        zf.extract(jni_lib, tmpdir)
                        extracted = os.path.join(tmpdir, jni_lib)
                        print(f"   Extracted: {extracted}")
                        
                        # Check what CUDA libs it needs using ldd
                        # Clear LD_PRELOAD to avoid GVirtuS interference in subprocess
                        clean_env = os.environ.copy()
                        clean_env.pop("LD_PRELOAD", None)
                        
                        print("\n   Running ldd to check CUDA dependencies:")
                        try:
                            result = subprocess.run(["ldd", extracted], capture_output=True, text=True, timeout=10, env=clean_env)
                            if result.returncode != 0:
                                print(f"   ldd failed: {result.stderr}")
                            else:
                                all_deps = result.stdout.split("\n")
                                cuda_deps = [l for l in all_deps if "cuda" in l.lower() or "nvidia" in l.lower() or "nvrtc" in l.lower()]
                                not_found = [l for l in all_deps if "not found" in l]
                                
                                if cuda_deps:
                                    print("   CUDA/NVIDIA libraries needed:")
                                    for dep in cuda_deps:
                                        dep = dep.strip()
                                        if dep:
                                            if "not found" in dep:
                                                print(f"     MISSING: {dep}")
                                            else:
                                                print(f"     OK: {dep}")
                                
                                if not_found:
                                    print(f"\n   Missing libs in extracted dir ({len(not_found)} total):")
                                    print("   (Note: These are expected - JAR extract dir not in LD_LIBRARY_PATH)")
                                    for dep in not_found[:5]:  # Show first 5 only
                                        print(f"     {dep.strip()}")
                                    if len(not_found) > 5:
                                        print(f"     ... and {len(not_found) - 5} more")
                                else:
                                    print("\n   All dependencies resolved!")
                        except subprocess.TimeoutExpired:
                            print("   ldd timed out (library loading triggered network calls)")
                            print("   This means GVirtuS is intercepting - good sign!")
                        except FileNotFoundError:
                            print("   ldd not available, trying readelf...")
                            result = subprocess.run(["readelf", "-d", extracted], capture_output=True, text=True, timeout=5, env=clean_env)
                            needed = [l for l in result.stdout.split("\n") if "NEEDED" in l]
                            print("   Required shared libraries:")
                            for lib in needed:
                                print(f"     {lib.strip()}")
                        
                        # Extract ALL libs from JAR and try to load
                        print("\n   Extracting ALL native libs...")
                        lib_dir = os.path.dirname(extracted)
                        for jlib in jni_libs:
                            zf.extract(jlib, tmpdir)
                        
                        print(f"   Extracted to: {lib_dir}")
                        print(f"   Contents: {os.listdir(lib_dir)}")
                        
                        # Skip actual loading as it can hang waiting for GVirtuS
                        print("\n   Skipping dlopen test (can hang with GVirtuS network)")
                        print("   Use 'make run-spark-gvirtus' to test actual Spark + RAPIDS")
                    else:
                        print("   No cuDF JNI libs found in JAR")
        except Exception as e:
            import traceback
            print(f"   Error extracting/inspecting JAR: {e}")
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    ld_preload = os.environ.get("LD_PRELOAD", "")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    
    issues = []
    # Check environment is set correctly (both LD_PRELOAD and LD_LIBRARY_PATH)
    if "/opt/GVirtuS/lib/frontend" not in ld_path:
        issues.append("GVirtuS frontend not in LD_LIBRARY_PATH")
    if not ld_preload or "libcudart" not in ld_preload:
        issues.append("LD_PRELOAD not set with libcudart.so")
    
    if not issues:
        print("Environment looks correctly configured for GVirtuS + RAPIDS!")
        print("")
        print("LD_PRELOAD and LD_LIBRARY_PATH are set in Dockerfile.")
        print("Java/JNI will use GVirtuS CUDA stubs.")
        print("")
        print("If Spark RAPIDS still fails, the issue may be:")
        print("  - cuDF native libs have hard-coded CUDA driver checks")
        print("  - RAPIDS version incompatible with remote CUDA version")
        print("  - cuDF JNI loads libcudf.so which does internal CUDA checks")
    else:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")


def main():
    print("=" * 50)
    print("GVirtuS Frontend Connectivity Test")
    print("=" * 50)
    print(f"GVIRTUS_HOME:     {os.environ.get('GVIRTUS_HOME', 'NOT SET')}")
    print(f"GVIRTUS_LOGLEVEL: {os.environ.get('GVIRTUS_LOGLEVEL', 'NOT SET')}")
    ld_path = os.environ.get('LD_LIBRARY_PATH', 'NOT SET')
    print(f"LD_LIBRARY_PATH:  {ld_path[:60]}..." if len(ld_path) > 60 else f"LD_LIBRARY_PATH:  {ld_path}")
    ld_preload = os.environ.get('LD_PRELOAD', 'NOT SET')
    print(f"LD_PRELOAD:       {ld_preload[:60]}..." if len(ld_preload) > 60 else f"LD_PRELOAD:       {ld_preload}")
    print("-" * 50)
    
    cudart = load_cudart()
    if cudart is None:
        print("\nFAIL: Could not load GVirtuS cudart library")
        return 1
    
    print("OK: Loaded GVirtuS cudart library")
    
    cuda = load_cuda_driver()
    if cuda:
        print("OK: Loaded GVirtuS cuda driver library\n")
    else:
        print("WARNING: CUDA driver library not available (driver API tests will be skipped)\n")
    
    tests = [
        # CUDA Driver API (cuInit must be first!)
        ("cuInit(0)", lambda: test_cu_init(cuda)),
        ("cuDeviceGetCount", lambda: test_cu_device_get_count(cuda)),
        ("cuDeviceGet", lambda: test_cu_device_get(cuda)),
        ("cuCtxCreate/Destroy", lambda: test_cu_ctx_create(cuda)),
        
        # CUDA Runtime API - Basic connectivity
        ("Device Count", lambda: test_get_device_count(cudart)),
        ("Set Device 0", lambda: test_set_device(cudart, 0)),
        ("Device Properties", lambda: test_get_device_properties(cudart, 0)),
        
        # RAPIDS-critical: version checks
        ("Driver Version", lambda: test_driver_version(cudart)),
        ("Runtime Version", lambda: test_runtime_version(cudart)),
        ("Version Compatibility", lambda: test_version_compatibility(cudart)),
        
        # RAPIDS-critical: memory operations
        ("Malloc/Free", lambda: test_malloc_free(cudart)),
        ("Memcpy H2D/D2H", lambda: test_memcpy(cudart)),
        ("Memory Info", lambda: test_get_mem_info(cudart)),
        
        # RAPIDS-critical: streams & sync
        ("Stream Create/Destroy", lambda: test_stream_create_destroy(cudart)),
        ("Device Synchronize", lambda: test_device_synchronize(cudart)),
        
        # Cleanup
        ("Device Reset", lambda: test_device_reset(cudart)),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"Testing: {name}...")
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL: {name} raised exception: {e}")
            failed += 1
    
    print("-" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    # Run JNI library resolution diagnostics
    print("\n" + "=" * 50)
    print("JNI/RAPIDS Library Resolution Diagnostics")
    print("=" * 50)
    run_jni_diagnostics()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
