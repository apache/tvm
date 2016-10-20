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
    print type(x)
    print len(x)
    print x[4]


if __name__ == "__main__":
    test_basic()
    test_array()
