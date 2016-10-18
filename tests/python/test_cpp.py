from tvm import cpp as tvm

def test_basic():
    a = tvm.Var('a')
    b = tvm.Var('b')
    z = tvm.max(a, b)
    assert tvm.format_str(z) == 'max(%s, %s)' % (a.name, b.name)

if __name__ == "__main__":
    test_basic()
