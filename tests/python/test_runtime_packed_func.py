import tvm
import numpy as np

def test_function():
    ctx = tvm.cpu(0)
    x = np.random.randint(0, 10, size=(3, 4))
    x = np.array(x)
    y = tvm.nd.array(x, ctx=ctx)

    f = tvm.codegen.DummyHelloFunction()
    f(y, 10)


def test_get_global():
    targs = (10, 10.0, "hello")
    # register into global function table
    @tvm.register_func
    def my_packed_func(*args):
        assert(tuple(args) == targs)
    # get it out from global function table
    f = tvm.get_global_func("my_packed_func")
    assert isinstance(f, tvm.nd.Function)
    f(*targs)


def test_convert():
    # convert a function to tvm function
    targs = (10, 10.0, "hello", 10)
    def myfunc(*args):
        assert(tuple(args) == targs)

    f = tvm.convert(myfunc)
    assert isinstance(f, tvm.nd.Function)
    f(*targs)


if __name__ == "__main__":
    test_function()
    test_convert()
    test_get_global()
