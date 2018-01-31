import numpy as np
import tvm
import topi
from topi import util


def test_util():
    x = tvm.const(100)
    assert util.get_const_int(x) == 100
    assert util.get_const_tuple((x, x)) == (100, 100)


def test_ewise():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')

    def test_apply(func, name):
        B = func(A)
        assert tuple(B.shape) == tuple(A.shape)
        assert B.op.body[0].name == name

    test_apply(topi.exp, "exp")
    test_apply(topi.tanh, "tanh")
    test_apply(topi.sigmoid, "sigmoid")
    test_apply(topi.log, "log")
    test_apply(topi.sqrt, "sqrt")

def verify_matmul(l_shape, r_shape, transpose_a, transpose_b):
    l = tvm.placeholder(l_shape, name="l")
    r = tvm.placeholder(r_shape, name="r")

    ret = topi.matmul(l, r, transpose_a, transpose_b)
    s = tvm.create_schedule([ret.op])

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        ctx = tvm.context(device, 0)
        f = tvm.build(s, [l, r, ret], device, name="matmul")

        np_l = np.random.uniform(size=l_shape).astype(l.dtype)
        np_r = np.random.uniform(size=r_shape).astype(l.dtype)
        tvm_l = tvm.nd.array(np_l, ctx)
        tvm_r = tvm.nd.array(np_r, ctx)

        if len(l_shape) == 1:
            np_l = np.array([np_l])
        if len(r_shape) == 1:
            np_r = np.array([np_r])
        if transpose_a:
            np_l = np.transpose(np_l)
        if transpose_b:
            np_r = np.transpose(np_r)
        if len(np_r.shape) > 2:
            reorder = list(range(len(np_r.shape)))
            np_r = np.transpose(np_r, [reorder[-2]] + reorder[:-2] + [reorder[-1]])
        np_ret = np.dot(np_l, np_r)
        out = tvm.nd.array(np.zeros(np_ret.shape).astype(ret.dtype), ctx)
        f(tvm_l, tvm_r, out)
        np.testing.assert_allclose(out.asnumpy(), np_ret, rtol=1e-5)

    for device in ["llvm"]:
        check_device(device)

def test_matmul():
    verify_matmul((3, 4, 5), (5, 6), False, False)
    verify_matmul((5, 4, 3), (6, 5), True, True)
    verify_matmul((3, 4, 5), (6, 5), False, True)
    verify_matmul((2, 2), (2, 1), True, False)
    verify_matmul((5, 4, 3), (5, 6), True, False)
    verify_matmul((3, 5), (5, 6, 8), False, False)
    verify_matmul((5, 3), (8, 6, 5), True, True)
    verify_matmul((3, 5), (8, 6, 5), False, True)
    verify_matmul((5, 3), (5, 6, 8), True, False)
    verify_matmul((5,), (5, 1), False, False)
    verify_matmul((5,), (5, ), False, True)
    verify_matmul((5,), (5, 6, 7), False, False)
    verify_matmul((5,), (5,), True, False)

if __name__ == "__main__":
    test_util()
    test_ewise()
    test_matmul()
