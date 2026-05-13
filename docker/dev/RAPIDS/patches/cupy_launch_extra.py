#!/usr/bin/env python3
"""
Patch CuPy v13.3.0 source to use CUDA extra-parameter mechanism in cuLaunchKernel.

Replaces the default kernelParams (void**) approach with the extra parameter
mechanism (CU_LAUNCH_PARAM_BUFFER_POINTER + SIZE), which sends args as a single
contiguous buffer with known size. This is required for GVirtuS to marshal
kernel arguments correctly across the network.

Usage: python3 cupy_launch_extra.py /path/to/cupy/source/root
"""
import sys, os, re

cupy_root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cupy"

# ---- 1) function.pxd: add `size` field to CPointer base ---------------
path = os.path.join(cupy_root, "cupy/cuda/function.pxd")
with open(path) as f: c = f.read()
old = "cdef class CPointer:\n    cdef void* ptr\n"
new = "cdef class CPointer:\n    cdef void* ptr\n    cdef int gvirtus_size  # bytes for kernel-arg buffer packing\n"
if old in c and new not in c:
    c = c.replace(old, new)
    with open(path, "w") as f: f.write(c)
    print(f"[1] {path}: added size field")
elif new in c:
    print(f"[1] {path}: size field already present")
else:
    print(f"[1] FAIL: anchor not found in {path}"); sys.exit(1)

# ---- 2) function.pyx: set size in each CPointer subclass __init__ ----
path = os.path.join(cupy_root, "cupy/cuda/function.pyx")
with open(path) as f: c = f.read()

sizes_to_set = [
    ("def __init__(self, p=0):\n        self.ptr = <void*>p",
     "def __init__(self, p=0):\n        self.ptr = <void*>p\n        self.gvirtus_size = sizeof(void*)"),
    ("def __init__(self, int8_t v):\n        self.val = v\n        self.ptr = <void*>&self.val",
     "def __init__(self, int8_t v):\n        self.val = v\n        self.ptr = <void*>&self.val\n        self.gvirtus_size = 1"),
    ("def __init__(self, int16_t v):\n        self.val = v\n        self.ptr = <void*>&self.val",
     "def __init__(self, int16_t v):\n        self.val = v\n        self.ptr = <void*>&self.val\n        self.gvirtus_size = 2"),
    ("def __init__(self, int32_t v):\n        self.val = v\n        self.ptr = <void*>&self.val",
     "def __init__(self, int32_t v):\n        self.val = v\n        self.ptr = <void*>&self.val\n        self.gvirtus_size = 4"),
    ("def __init__(self, int64_t v):\n        self.val = v\n        self.ptr = <void*>&self.val",
     "def __init__(self, int64_t v):\n        self.val = v\n        self.ptr = <void*>&self.val\n        self.gvirtus_size = 8"),
    ("def __init__(self, double complex v):\n        self.val = v\n        self.ptr = <void*>&self.val",
     "def __init__(self, double complex v):\n        self.val = v\n        self.ptr = <void*>&self.val\n        self.gvirtus_size = 16"),
    ("def __init__(self, uintmax_t v):\n        self.val = v\n        self.ptr = <void*>&self.val",
     "def __init__(self, uintmax_t v):\n        self.val = v\n        self.ptr = <void*>&self.val\n        self.gvirtus_size = sizeof(uintmax_t)"),
    ("def __init__(self, intptr_t v):\n        self.val = v\n        self.ptr = <void*>&self.val",
     "def __init__(self, intptr_t v):\n        self.val = v\n        self.ptr = <void*>&self.val\n        self.gvirtus_size = sizeof(intptr_t)"),
    ("self.ptr = <void*><size_t>v.__array_interface__[\"data\"][0]",
     "self.ptr = <void*><size_t>v.__array_interface__[\"data\"][0]\n        self.gvirtus_size = sizeof(void*)"),
]
for old, new in sizes_to_set:
    if old in c and new not in c:
        c = c.replace(old, new)
print(f"[2] function.pyx: set size in CPointer subclasses")

# ---- 3) function.pyx: replace _launch with extra-mechanism version ----
old_launch = """    cdef list pargs = []
    cdef vector.vector[void*] kargs
    cdef CPointer cp
    kargs.reserve(len(args))
    for a in args:
        cp = _pointer(a)
        pargs.append(cp)  # keep the CPointer objects alive
        kargs.push_back(cp.ptr)

    runtime._ensure_context()"""

new_launch = """    cdef list pargs = []
    cdef vector.vector[void*] kargs
    cdef vector.vector[int] ksizes
    cdef CPointer cp
    kargs.reserve(len(args))
    ksizes.reserve(len(args))
    for a in args:
        cp = _pointer(a)
        pargs.append(cp)  # keep the CPointer objects alive
        kargs.push_back(cp.ptr)
        ksizes.push_back(cp.gvirtus_size if cp.gvirtus_size > 0 else <int>sizeof(void*))

    # GVirtuS interop: pack args into contiguous buffer for CUDA extra-param
    # mechanism. CUDA needs natural alignment per arg (here we conservatively
    # align to 8 bytes which is >= max scalar alignment; CArray fields align 8).
    cdef int _total = 0
    cdef int _i, _sz, _pad, _align
    for _i in range(<int>ksizes.size()):
        _sz = ksizes[_i]
        _align = _sz if _sz > 0 and _sz < 8 else 8
        _pad = (_align - (_total % _align)) % _align
        _total += _pad + _sz
    cdef vector.vector[char] _packed
    _packed.resize(_total if _total > 0 else 1)
    cdef char* _packed_ptr = _packed.data()
    cdef int _off = 0
    cdef const char* _src
    for _i in range(<int>ksizes.size()):
        _sz = ksizes[_i]
        _align = _sz if _sz > 0 and _sz < 8 else 8
        _pad = (_align - (_off % _align)) % _align
        if _pad:
            memset(_packed_ptr + _off, 0, _pad)
            _off += _pad
        _src = <const char*>kargs[_i]
        memcpy(_packed_ptr + _off, _src, _sz)
        _off += _sz
    cdef size_t _packed_total_size = <size_t>_total
    cdef void* _extra[5]
    _extra[0] = <void*>1   # CU_LAUNCH_PARAM_BUFFER_POINTER
    _extra[1] = <void*>_packed_ptr
    _extra[2] = <void*>2   # CU_LAUNCH_PARAM_BUFFER_SIZE
    _extra[3] = <void*>&_packed_total_size
    _extra[4] = <void*>0   # CU_LAUNCH_PARAM_END

    runtime._ensure_context()"""

if old_launch in c:
    c = c.replace(old_launch, new_launch)
    print("[3a] _launch: packed-buffer + extra prep added")
else:
    print("[3a] old _launch not found, maybe already patched"); sys.exit(1)

# Replace the actual launchKernel call to use extra
old_call = """        driver.launchKernel(
            func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
            <int>shared_mem, stream, <intptr_t>kargs.data(), <intptr_t>0)"""
new_call = """        driver.launchKernel(
            func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
            <int>shared_mem, stream, <intptr_t>0, <intptr_t>_extra)"""
if old_call in c:
    c = c.replace(old_call, new_call)
    print("[3b] launchKernel: now using extra param")
else:
    print("[3b] launchKernel call not found exactly; check manually")

# Need memcpy/memset includes at the top of function.pyx
if "from libc.string cimport memcpy, memset" not in c:
    c = c.replace("from libc.stdint cimport", "from libc.string cimport memcpy, memset\nfrom libc.stdint cimport", 1)
    print("[3c] added memcpy/memset cimport")

with open(path, "w") as f: f.write(c)

# ---- 4) _carray.pyx: set size for CArray and CIndexer based on ndim --
path = os.path.join(cupy_root, "cupy/_core/_carray.pyx")
with open(path) as f: c = f.read()

# In CArray.init, ndim = shape.size(). CArray<T, NDIM> in C++ is:
#   T* data (8) + Py_ssize_t size (8) + shape[NDIM] (8*NDIM) + strides[NDIM] (8*NDIM)
# = 16 + 16 * NDIM
old = """        self.val.data = data_ptr
        self.val.size = data_size
        for i in range(ndim):
            shape_and_strides[i] = shape[i]
            shape_and_strides[i + ndim] = strides[i]
        self.ptr = <void*>&self.val"""
new = """        self.val.data = data_ptr
        self.val.size = data_size
        for i in range(ndim):
            shape_and_strides[i] = shape[i]
            shape_and_strides[i + ndim] = strides[i]
        self.ptr = <void*>&self.val
        # GVirtuS interop: actual CUDA-side struct size for CArray<T, NDIM>
        self.gvirtus_size = <int>(16 + 16 * ndim)"""
if old in c:
    c = c.replace(old, new); print("[4a] CArray.size set")
else:
    print("[4a] CArray.init body not exactly found")

# CIndexer.init: kernel struct CIndexer<NDIM> is size (8) + shape[NDIM] (8*NDIM)
old_idx = """        self.val.size = size
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t>shape.size()):
            self.val.shape_and_index[i] = shape[i]
        self.ptr = <void*>&self.val"""
new_idx = """        self.val.size = size
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t>shape.size()):
            self.val.shape_and_index[i] = shape[i]
        self.ptr = <void*>&self.val
        # GVirtuS interop: actual CUDA-side struct size for CIndexer<NDIM>
        self.gvirtus_size = <int>(8 + 8 * ndim)"""
if old_idx in c:
    c = c.replace(old_idx, new_idx); print("[4b] CIndexer.size set")
else:
    print("[4b] CIndexer.init body not exactly found")

with open(path, "w") as f: f.write(c)
print("\nDONE")

# ---- 5) _scalar.pyx: mirror size -> gvirtus_size in all CScalar size setters ----
import re as _re
path = os.path.join(cupy_root, "cupy/_core/_scalar.pyx")
with open(path) as f: c = f.read()
# Pattern: lines like "    self.size = X" or "    ret.size = Y" — duplicate as gvirtus_size
# Process line by line
lines = c.split("\n")
out_lines = []
prev_was_gvirtus = False
for line in lines:
    out_lines.append(line)
    m = _re.match(r"^(\s+)(\w+)\.size = (.+)$", line)
    if m:
        indent, lvalue, expr = m.group(1), m.group(2), m.group(3)
        # Skip if next line is already setting gvirtus_size (idempotent)
        new_line = f"{indent}{lvalue}.gvirtus_size = {expr}"
        out_lines.append(new_line)
c_new = "\n".join(out_lines)
if c_new != c:
    with open(path, "w") as f: f.write(c_new)
    print("[5] _scalar.pyx: mirrored .size = X assignments as .gvirtus_size = X")
else:
    print("[5] _scalar.pyx: no changes needed")
