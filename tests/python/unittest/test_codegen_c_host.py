import tvm
import numpy as np
from tvm import relay
from tvm.contrib import util

def test_add():
    shape = (1024,)
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A")
    B = tvm.placeholder(tvm_shape, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.create_schedule(C.op)

    mod_host = tvm.build(s, [A, B, C], "c", name="fadd")
    temp = util.tempdir()
    path_dso = temp.relpath("temp.so")
    mod_host.export_library(path_dso)
    mod = tvm.module.load(path_dso)
    fadd = mod["fadd"]
    ctx = tvm.cpu(0)
    # launch the kernel.
    a = tvm.nd.array(np.random.uniform(size=shape).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=shape).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(shape, dtype=C.dtype), ctx)
    fadd(a, b, c)
    tvm.testing.assert_allclose(
        c.asnumpy(), a.asnumpy() + b.asnumpy())


def test_add_pipeline():
    shape = (1024,)
    tvm_shape = tvm.convert(shape)
    A = tvm.placeholder(tvm_shape, name="A")
    B = tvm.placeholder(tvm_shape, name="B")
    AA = tvm.compute(tvm_shape, lambda *i: A(*i), name="A")
    BB = tvm.compute(tvm_shape, lambda *i: B(*i), name="B")
    T = tvm.compute(A.shape, lambda *i: AA(*i) + BB(*i), name="T")
    C = tvm.compute(A.shape, lambda *i: T(*i), name="C")
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
    s[C].vectorize(xi)

    # TODO: Make this `with` clause more fine-grained.
    if not tvm.module.enabled("llvm"):
        return
    # Specifically allow offset to test codepath when offset is available
    Ab = tvm.decl_buffer(
        A.shape, A.dtype,
        elem_offset=tvm.var("Aoffset"),
        offset_factor=8,
        name="A")
    binds = {A : Ab}
    # BUILD and invoke the kernel.
    with tvm.build_config(offset_factor=4):
        f1 = tvm.lower(s, [A,B,C], name="fadd_pipeline")
        fsplits = [x for x in tvm.ir_pass.SplitHostDevice(f1)]
        fsplits[0] = tvm.ir_pass.LowerTVMBuiltin(fsplits[0])
        mod_host = tvm.codegen.build_module(fsplits[0], "c")
    temp = util.tempdir()
    path_dso = temp.relpath("temp.so")
    mod_host.export_library(path_dso)
    mod = tvm.module.load(path_dso)
    fadd = mod["fadd_pipeline"]
    ctx = tvm.cpu(0)
    # launch the kernel.
    a = tvm.nd.array(np.random.uniform(size=shape).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=shape).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(shape, dtype=C.dtype), ctx)
    fadd(a, b, c)
    tvm.testing.assert_allclose(
        c.asnumpy(), a.asnumpy() + b.asnumpy())


if __name__ == "__main__":
    test_add()
    test_add_pipeline()
