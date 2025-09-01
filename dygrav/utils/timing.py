import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    dt = (time.time() - t0) * 1000.0
    print(f"[TIMER] {name}: {dt:.2f} ms")
