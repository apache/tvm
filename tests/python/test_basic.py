import tvm

def test_const():
    x = tvm.const(1)
    assert x.dtype == 'int32'
    assert isinstance(x, tvm.expr.IntImm)

def test_make():
    x = tvm.const(1)
    y = tvm.make.IntImm('int32', 1)
    z = x + y
    print tvm.format_str(z)

def test_ir():
    x = tvm.const(1)
    y = tvm.make.IntImm('int32', 1)
    z = x + y
    stmt = tvm.make.Evaluate(z)
    assert isinstance(stmt, tvm.stmt.Evaluate)
    print tvm.format_str(stmt)

if __name__ == "__main__":
    test_const()
    test_make()
    test_ir()
