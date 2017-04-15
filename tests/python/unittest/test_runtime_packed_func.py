import tvm
import numpy as np

def test_get_global():
    targs = (10, 10.0, "hello")
    # register into global function table
    @tvm.register_func
    def my_packed_func(*args):
        assert(tuple(args) == targs)
        return 10
    # get it out from global function table
    f = tvm.get_global_func("my_packed_func")
    assert isinstance(f, tvm.Function)
    y = f(*targs)
    assert y == 10


def test_return_func():
    def addy(y):
        def add(x):
            return tvm.convert(x + y)
        return add
    myf = tvm.convert(addy)
    f = myf(10)
    assert f(11).value == 21


def test_convert():
    # convert a function to tvm function
    targs = (10, 10.0, "hello", 10)
    def myfunc(*args):
        assert(tuple(args) == targs)

    f = tvm.convert(myfunc)
    assert isinstance(f, tvm.Function)
    f(*targs)

def test_byte_array():
    s = "hello"
    a = bytearray(s, encoding="ascii")

    def myfunc(ss):
        assert ss == a
    f = tvm.convert(myfunc)
    f(a)

if __name__ == "__main__":
    test_convert()
    test_get_global()
    test_return_func()
    test_byte_array()
