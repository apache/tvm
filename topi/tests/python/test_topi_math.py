import numpy as np
import tvm
import topi
import topi.testing
from topi import util


def test_util():
    x = tvm.const(100)
    assert util.get_const_int(x) == 100
    assert util.get_const_tuple((x, x)) == (100, 100)


def test_ewise():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')

    shape = (20, 3)

    def test_apply(func, name, f_numpy):
        B = func(A)
        assert tuple(B.shape) == tuple(A.shape)
        assert B.op.body[0].name == name
        a_np = np.random.uniform(low=1e-5, size=shape).astype(A.dtype)
        a_np = np.abs(a_np)
        b_np = f_numpy(a_np)

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            with tvm.target.create(device):
                s = topi.generic.schedule_injective(B)
            a = tvm.nd.array(a_np, ctx)
            b = tvm.nd.array(np.zeros_like(b_np), ctx)
            foo = tvm.build(s, [A, B], device, name=name)
            foo(a, b)
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, atol=1e-5)

        for device in ['cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'llvm']:
            check_device(device)

    test_apply(topi.exp, "exp", np.exp)
    test_apply(topi.tanh, "tanh", np.tanh)
    test_apply(topi.sigmoid, "sigmoid", lambda x:1/(1+np.exp(-x)))
    test_apply(topi.log, "log", np.log)
    test_apply(topi.sqrt, "sqrt", np.sqrt)

if __name__ == "__main__":
    test_util()
    test_ewise()
