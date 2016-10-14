import tvm

def test_bind():
    x = tvm.Var('x')
    y = x + 1
    z = tvm.bind(y, {x: tvm.const(10) + 9})
    assert tvm.format_str(z) == '((10 + 9) + 1)'


def test_basic():
    a = tvm.Var('a')
    b = tvm.Var('b')
    c =  a + b
    assert tvm.format_str(c) == '(%s + %s)' % (a.name, b.name)

def test_simplify():
    a = tvm.Var('a')
    b = tvm.Var('b')
    e1 = a * (2 + 1) + b * 1
    e2 = a * (2 + 1) - b * 1
    e3 = tvm.max(a * 3.3 + 5, 3 + 3.3 * a)
    e4 = a - a
    assert tvm.format_str(tvm.simplify(e1)) == '((%s * 3) + %s)' % (a.name, b.name)
    assert tvm.format_str(tvm.simplify(e2)) == '((%s * 3) + (%s * -1))' % (a.name, b.name)
    assert tvm.format_str(tvm.simplify(e3)) == '((%s * 3.3) + 5)' % (a.name)
    assert tvm.format_str(tvm.simplify(e4)) == '0'

if __name__ == "__main__":
    test_basic()
    test_bind()
    test_simplify()
