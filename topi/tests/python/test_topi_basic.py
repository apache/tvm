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


if __name__ == "__main__":
    test_util()
    test_ewise()
