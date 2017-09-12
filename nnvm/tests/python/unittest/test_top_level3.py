import nnvm.symbol as sym

def test_reshape():
    x = sym.Variable("x")
    y = sym.reshape(x, shape=(10, 20), name="y")
    assert(y.list_input_names() == ["x"])


def test_scalar_op():
    x = sym.Variable("x")
    y = (1 / (x * 2) - 1) ** 2
    assert(y.list_input_names() == ["x"])

def test_leaky_relu():
    x = sym.Variable("x")
    y = sym.leaky_relu(x, alpha=0.1)
    assert(y.list_input_names() == ["x"])


if __name__ == "__main__":
    test_scalar_op()
    test_reshape()
    test_leaky_relu()
