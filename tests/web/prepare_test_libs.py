# Prepare test library for js.
import tvm
from tvm.contrib import emscripten
import os

def prepare_test_libs(base_path):
    target = "llvm -target=asmjs-unknown-emscripten -system-lib"
    if not tvm.module.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    fadd1 = tvm.build(s, [A, B], target, name="add_one")
    obj_path = os.path.join(base_path, "test_add_one.bc")
    fadd1.save(obj_path)
    emscripten.create_js(os.path.join(base_path, "test_module.js"), obj_path)

if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_test_libs(os.path.join(curr_path, "../../lib"))
