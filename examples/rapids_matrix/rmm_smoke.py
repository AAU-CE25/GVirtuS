import sys
def step(m): print(f'[rmm_smoke] {m}', flush=True)
def dumpmaps(tag):
    print(f'[rmm_smoke] MAPS({tag}):', flush=True)
    seen=set()
    with open('/proc/self/maps') as f:
        for line in f:
            if 'libcudart' in line or 'libcuda.so' in line:
                p=line.split()[-1]
                if p not in seen: seen.add(p); print('  MAP', p, flush=True)
step('step1: import rmm'); import rmm
step('step2: reinitialize'); rmm.reinitialize(pool_allocator=False, managed_memory=False)
step('step3: DeviceBuffer(64)'); b = rmm.DeviceBuffer(size=64); step(f'OK size={b.size}')
dumpmaps('before-copy')
step('step4: copy_to_host')
import numpy as np
try:
    host = np.frombuffer(b.copy_to_host(), dtype=np.uint8)
    step(f'OK host len={len(host)}'); step('ALL DONE')
except Exception as e:
    step(f'EXC {e!r}')
    dumpmaps('after-fail')
