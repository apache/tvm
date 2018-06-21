import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from test_top_level1 import helper

def check_map(symfunc, np_func, np_backward=None):
    x = sym.Variable("x")
    y = symfunc(x)
    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [('x', dshape, x)]
    helper(y, inputs, dtype, lambda x: np_func(x), np_backward)


def test_floor():
    check_map(sym.floor, np.floor)

def test_ceil():
    check_map(sym.ceil, np.ceil)

def test_trunc():
    check_map(sym.trunc, np.trunc)

def test_round():
    check_map(sym.round, np.round)


if __name__ == "__main__":
    test_floor()
    test_ceil()
    test_round()
    test_trunc()
