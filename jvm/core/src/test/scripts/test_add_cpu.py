import os

import tvm
from tvm.contrib import cc, util

def test_add(target_dir):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    fadd = tvm.build(s, [A, B, C], "llvm", target_host="llvm", name="myadd")

    fadd.save(os.path.join(target_dir, "add_cpu.o"))
    cc.create_shared(os.path.join(target_dir, "add_cpu.so"),
            [os.path.join(target_dir, "add_cpu.o")])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(-1)
    test_add(sys.argv[1])
