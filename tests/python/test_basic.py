import tvm
from tvm import expr

def test_bind():
    x = tvm.Var('x')
    y = x + 1
    z = tvm.bind(y, {x: tvm.const(10) + 9})
    assert tvm.format_str(z) == '((10 + 9) + 1)'


def test_basic():
    a= tvm.Var('a')
    b = tvm.Var('b')
    c =  a + b
    assert tvm.format_str(c) == '(%s + %s)' % (a.name, b.name)


if __name__ == "__main__":
    test_basic()
    test_bind()
