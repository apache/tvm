import nnvm.symbol as sym
from nnvm import NNVMError

def test_dense():
    x = sym.Variable('x')
    y = sym.dense(x, units=3, name="dense")
    assert y.list_input_names() == ['x', 'dense_weight', 'dense_bias']

if __name__ == "__main__":
    test_dense()
