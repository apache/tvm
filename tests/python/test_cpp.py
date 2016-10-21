from tvm import cpp as tvm


def test_basic():
    a = tvm.Var('a')
    b = tvm.Var('b')
    c =  a + b
    assert a == c.lhs
    assert c.dtype == tvm.int32
    assert tvm.format_str(c) == '(%s + %s)' % (a.name, b.name)


def test_array():
    a = tvm.Var('a')
    x = tvm.function._symbol([1,2,a])


def assert_equal(x, y):
    z = tvm.simplify(x - y)
    assert isinstance(z, tvm.expr.IntExpr)
    assert z.value == 0


def test_simplify():
    a = tvm.Var('a')
    b = tvm.Var('b')
    e1 = a * (2 + 1) + b * 1
    e2 = a * (2 + 1) - b * 1
    e3 = tvm.max(a * 3 + 5, 3 + 3 * a)
    e4 = a - a

    assert_equal(e1, a * 3 + b)
    assert_equal(e2, a * 3 - b)
    assert_equal(e3, a * 3 + 5)
    assert_equal(e4, 0)


if __name__ == "__main__":
    test_basic()
    test_array()
    test_simplify()
