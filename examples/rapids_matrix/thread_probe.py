import threading, sys
def p(m): print(f"[thread_probe] {m}", flush=True)

from cuda.bindings import runtime

p("MAIN thread: cudaGetDevice (opens connection #1)")
p(f"  main -> {runtime.cudaGetDevice()}")

result = {}
def worker():
    p("WORKER thread: cudaGetDevice (should open connection #2)")
    try:
        result['r'] = runtime.cudaGetDevice()
        p(f"  worker -> {result['r']}")
    except Exception as e:
        p(f"  worker EXC {e!r}")
    p("WORKER thread: done")

t = threading.Thread(target=worker)
p("starting worker thread")
t.start()
t.join(timeout=45)
p(f"join returned; worker still alive? {t.is_alive()}")
p("MAIN DONE")
