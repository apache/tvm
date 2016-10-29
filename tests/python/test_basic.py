import tvm

def test_const():
    x = tvm.const(1)
    assert x.dtype == 'int32'
    assert isinstance(x, tvm.expr.IntImm)

def test_make():
    x = tvm.const(1)
    y = tvm.make.IntImm('int32', 1)
    z = x + y
    print(tvm.format_str(z))

def test_ir():
    x = tvm.const(1)
    y = tvm.make.IntImm('int32', 1)
    z = x + y
    stmt = tvm.make.Evaluate(z)
    assert isinstance(stmt, tvm.stmt.Evaluate)

def test_basic():
    a = tvm.Var('a')
    b = tvm.Var('b')
    c =  a + b
    assert tvm.format_str(c) == '(%s + %s)' % (a.name, b.name)

def test_array():
    a = tvm.convert([1,2,3])

def test_stmt():
    tvm.make.For(tvm.Var('i'), 0, 1,
                 tvm.stmt.For.Serial, 0,
                 tvm.make.Evaluate(0))



if __name__ == "__main__":
    test_const()
    test_make()
    test_ir()
    test_basic()
    test_stmt()
