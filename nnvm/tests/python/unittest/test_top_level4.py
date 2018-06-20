import nnvm.symbol as sym

def test_binary_broadcast():
    x = sym.Variable('x')
    y = sym.Variable('y')
    z = x + y
    z = x * y
    z = x - y
    z = x / y


def test_broadcast_to():
    x = sym.Variable('x')
    y = sym.broadcast_to(x, shape=(3, 3))
    assert y.list_input_names() == ["x"]


if __name__ == "__main__":
    test_binary_broadcast()
    test_broadcast_to()
