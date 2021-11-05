import tvm
import numpy as np
from tvm import te
from tvm import relay
import os


def prepare_relay_libs():
    x = relay.var("x", shape=(10,), dtype="float32")
    y = relay.add(x, x)
    func = relay.Function([x], y)
    mod = tvm.IRModule.from_expr(func)

    with tvm.transform.PassContext(opt_level=1):
        mod = relay.build(mod, target="llvm  --system-lib --runtime=c", params=None)

    mod.lib.export_library("/tmp/libdouble.tar")


def prepare_te_libs():
    A = te.placeholder((10,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) * 2.0, name="B")
    s = te.create_schedule(B.op)

    fadd_syslib = tvm.build(
        s, [A, B], "llvm --system-lib --runtime=c", name="tvmgen_default_fused_add"
    )
    fadd_syslib.export_library("/tmp/libdouble.tar")


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_relay_libs()
    # prepare_te_libs()
