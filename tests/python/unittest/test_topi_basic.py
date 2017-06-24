import tvm
import topi

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


if __name__ == "__main__":
    test_ewise()
