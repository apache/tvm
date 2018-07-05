import tvm
import numpy as np
import os

os.environ["XCL_EMULATION_MODE"] = "1"

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
    def check_device(device, host="stackvm"):
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


if __name__ == "__main__":
    test_exp()
