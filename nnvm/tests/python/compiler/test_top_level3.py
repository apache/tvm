import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm.testing.check_computation import check_function

def check_map(symfunc, np_func, np_backward=None, dtype="float32", rnd_min=-1, rnd_max=1):
    x = sym.Variable("x")
    y = symfunc(x)
    shape = {'x': (1, 3, 32, 32)}
    check_function(y, lambda x: np_func(x), np_backward,
                   dtype=dtype, shape=shape, in_range=(rnd_min, rnd_max))


def test_floor():
    check_map(sym.floor, np.floor)

def test_ceil():
    check_map(sym.ceil, np.ceil)

def test_trunc():
    check_map(sym.trunc, np.trunc)

def test_round():
    check_map(sym.round, np.round)

def test_abs():
    check_map(sym.abs, np.abs)
    check_map(sym.abs, np.abs, dtype = "int32")
    check_map(sym.abs, np.abs, dtype = "int8")

def test_shift():
    n = 3
    for dtype in ["int32", "int8"]:
        check_map(lambda x : x >> n, lambda x: x >> n, dtype=dtype, rnd_min=-100, rnd_max=100)
        check_map(lambda x : x << n, lambda x: x << n, dtype=dtype, rnd_min=-100, rnd_max=100)

if __name__ == "__main__":
    test_shift()
    test_floor()
    test_ceil()
    test_round()
    test_abs()
    test_trunc()
