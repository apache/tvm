import nnvm.symbol as sym
from nnvm import NNVMError

def test_dense():
    x = sym.Variable('x')
    y = sym.dense(x, units=30, name="fc")
    assert y.list_input_names() == ["x", "fc_weight", "fc_bias"]

def test_batch_norm():
    x = sym.Variable('x')
    y = sym.dense(x, units=30, name="fc")
    z = sym.batch_norm(x, name='bn')
    assert z.list_input_names('aux_state') == ['bn_moving_mean', 'bn_moving_var']
    assert z.list_input_names('read_only') == ['x', 'bn_gamma', 'bn_beta']

def test_compose():
    x = sym.Variable('x')
    z = sym.Variable('z')
    y = sym.exp(sym.elemwise_add(x, x, name='add', gpu=2),
                name='exp', gpu=1, attr={"kk": "1"})

    assert y.list_input_names() == ['x']
    assert y.list_output_names() == ["exp_output"]
    assert y.list_attr()['gpu'] == '1'
    z = y.get_internals()
    assert z['add_output'].list_output_names() == ['add_output']
    assert y.list_attr(recursive=True)['add$gpu'] == '2'

def test_default_input():
    x = sym.Variable('x')
    y = sym.dense(data=x, units=30, name='fc', use_bias=False)
    assert y.list_input_names() == ['x', 'fc_weight']
    tname = [z.list_output_names()[0] for z in y.list_input_variables()]
    assert tname == y.list_input_names()
    try:
        z = sym.elemwise_add(x)
        assert False
    except NNVMError:
        pass

def test_copy():
    x = sym.Variable('x')
    z = sym.Variable('z')
    y = sym.exp(sym.elemwise_add(x, x, name='add', gpu=2),
                name='exp', gpu=1, attr={"kk": "1"})
    assert y.__copy__().debug_str() == y.debug_str()


def test_op_name():
    x = sym.Variable('x')
    y = sym.exp(x)
    op_name = y.attr("op_name")
    op_func = sym.__dict__[op_name]
    z = op_func(x)

if __name__ == "__main__":
    test_op_name()
    test_copy()
    test_default_input()
    test_compose()
    test_batch_norm()
