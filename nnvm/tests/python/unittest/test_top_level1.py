import nnvm.symbol as sym
import nnvm.graph as graph

def test_dense():
    x = sym.Variable('x')
    x1 = sym.dense(x, units=3, name="dense")
    x2 = sym.flatten(x1)
    x3 = sym.softmax(x2)
    assert x3.list_input_names() == ['x', 'dense_weight', 'dense_bias']


def test_concatenate_split():
    x = sym.Variable('x')
    y = sym.Variable('y')
    y = sym.concatenate(x, y)
    assert y.list_input_names() == ['x', 'y']
    z = sym.split(y, indices_or_sections=10)
    assert len(z.list_output_names()) == 10
    z = sym.split(y, indices_or_sections=[10, 20])
    assert len(z.list_output_names()) == 3

def test_expand_dims():
    x = sym.Variable('x')
    y = sym.expand_dims(x, axis=1, num_newaxis=2)
    assert y.list_input_names() == ['x']


def test_unary():
    x = sym.Variable('x')
    x = sym.exp(x)
    x = sym.log(x)
    x = sym.sigmoid(x)
    x = sym.tanh(x)
    x = sym.relu(x)
    assert x.list_input_names() == ['x']


def test_batchnorm():
    x = sym.Variable('x')
    x = sym.batch_norm(x, name="bn")
    assert x.list_input_names() == [
        "x", "bn_gamma", "bn_beta", "bn_moving_mean", "bn_moving_var"]


if __name__ == "__main__":
    test_concatenate_split()
    test_expand_dims()
    test_dense()
    test_unary()
    test_batchnorm()
