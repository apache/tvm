import tvm
import numpy as np
import os

os.environ["XCL_EMULATION_MODE"] = "1"
os.environ["CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA"] = "1"

@tvm.register_func
def tvm_callback_vhls_postproc(code):
    """Hook to inspect the Vivado HLS code before actually run it"""
    print(code)
    return code

def test_exp():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: tvm.exp(A(*i)), name='B')
    s = tvm.create_schedule(B.op)
    # create iter var and assign them tags.
    px, x = s[B].split(B.op.axis[0], nparts=1)
    s[B].bind(px, tvm.thread_axis("pipeline"))

    # one line to build the function.
    def check_device(device, host="llvm"):
        if not tvm.module.enabled(host):
            return
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            return
        fexp = tvm.build(s, [A, B],
                         device, host,
                         name="myexp")
        ctx = tvm.context(device, 0)
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
        fexp(a, b)
        np.testing.assert_allclose(
            b.asnumpy(), np.exp(a.asnumpy()), rtol=1e-5)

    check_device("sdaccel")
    if "AWS_PLATFORM" in os.environ:
        check_device("sdaccel -device=" + os.environ.get("AWS_PLATFORM"))

    check_device("aocl_sw_emu")

def test_multi_kernel():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    D = tvm.compute(A.shape, lambda *i: A(*i) + C(*i), name='D')
    s = tvm.create_schedule(D.op)
    # create iter var and assign them tags.
    px, x = s[C].split(C.op.axis[0], nparts=1)
    s[C].bind(px, tvm.thread_axis("pipeline"))
    px, x = s[D].split(D.op.axis[0], nparts=1)
    s[D].bind(px, tvm.thread_axis("pipeline"))

    # one line to build the function.
    def check_device(device, host="llvm"):
        if not tvm.module.enabled(host):
            return
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            return
        fadd = tvm.build(s, [A, B, C, D],
                         device, host,
                         name="myadd")
        ctx = tvm.context(device, 0)
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.random.uniform(size=n).astype(C.dtype), ctx)
        d = tvm.nd.array(np.random.uniform(size=n).astype(D.dtype), ctx)
        fadd(a, b, c, d)
        np.testing.assert_allclose(
            d.asnumpy(), a.asnumpy() * 2 + b.asnumpy(), rtol=1e-5)

    check_device("sdaccel")
    check_device("aocl_sw_emu")


if __name__ == "__main__":
    test_exp()
    test_multi_kernel()
