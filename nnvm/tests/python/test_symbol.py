import nnvm.symbol as sym
from nnvm import NNVMError

def test_compose():
    x = sym.Variable('x')
    z = sym.Variable('z')
    y = sym.exp(sym.add(x, x, name='add', gpu=2),
                name='exp', gpu=1, attr={"kk": "1"})

    assert y.list_input_names() == ['x']
    assert y.list_output_names() == ["exp_output"]
    assert y.list_attr()['gpu'] == '1'
    z = y.get_internals()
    assert z['add_output'].list_output_names() == ['add_output']
    assert y.list_attr(recursive=True)['add$gpu'] == '2'

def test_default_input():
    x = sym.Variable('x')
    y = sym.conv2d(data=x, name='conv')
    assert y.list_input_names() == ['x', 'conv_weight']
    tname = [z.list_output_names()[0] for z in y.list_input_variables()]
    assert tname == y.list_input_names()
    try:
        z = sym.add(x)
        assert False
    except NNVMError:
        pass

def test_mutate_input():
    x = sym.Variable('x')
    y = sym.conv2d(data=x, name='conv')
    z = sym.assign(x, y)
    t = sym.add(z, x)

    try:
        z = sym.assign(z, z)
        assert False
    except NNVMError:
        pass

def test_copy():
    x = sym.Variable('x')
    z = sym.Variable('z')
    y = sym.exp(sym.add(x, x, name='add', gpu=2),
                name='exp', gpu=1, attr={"kk": "1"})
    assert y.__copy__().debug_str() == y.debug_str()

def test_control_dep():
    x = sym.Variable('x')
    y = sym.conv2d(data=x, name='conv')
    z = sym.assign(x, y)
    t = sym.add(x, x)
    t._add_control_deps([z, y])

if __name__ == "__main__":
    test_copy()
    test_default_input()
    test_compose()
    test_mutate_input()
    test_control_dep()
