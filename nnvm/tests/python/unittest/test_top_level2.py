import nnvm.symbol as sym

def test_conv2d():
    x = sym.Variable('x')
    y = sym.conv2d(x, channels=3, kernel_size=(3, 3),
                   name="y", use_bias=False)
    assert y.list_input_names() == ["x", "y_weight"]


def test_max_pool2d():
    x = sym.Variable('x')
    y = sym.max_pool2d(x, pool_size=(3, 3), name="y")
    y = sym.global_max_pool2d(y)
    assert y.list_input_names() == ["x"]


if __name__ == "__main__":
    test_conv2d()
    test_max_pool2d()
