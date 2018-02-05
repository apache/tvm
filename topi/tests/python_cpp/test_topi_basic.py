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

    test_apply(topi.cpp.exp, "exp")
    test_apply(topi.cpp.tanh, "tanh")
    test_apply(topi.cpp.sigmoid, "sigmoid")
    test_apply(topi.cpp.log, "log")
    test_apply(topi.cpp.sqrt, "sqrt")

def test_flatten_tag():
    A = tvm.placeholder((3, 4), name='A')
    B = topi.cpp.nn.flatten(A)
    assert B.op.tag == topi.tag.INJECTIVE

if __name__ == "__main__":
    test_util()
    test_ewise()
    test_flatten_tag()
