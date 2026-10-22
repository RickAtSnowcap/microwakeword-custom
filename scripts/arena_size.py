"""Find the minimum TFLite-Micro tensor arena that lets a model allocate and run one inference
(binary search with the tflite_micro Python interpreter)."""
import sys
import numpy as np
from tflite_micro.python.tflite_micro import runtime

def ok(path, size):
    try:
        it = runtime.Interpreter.from_file(path, arena_size=size)
        d = it.get_input_details(0)
        it.set_input(np.zeros(d["shape"], dtype=d["dtype"]), 0)
        it.invoke()
        return True
    except Exception:
        return False

for path in sys.argv[1:]:
    lo, hi = 1024, 1 << 20
    assert ok(path, hi)
    while hi - lo > 16:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(path, mid) else (mid, hi)
    print(f"{path}: min arena {hi} bytes")
