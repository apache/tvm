import tvm

def test_const():
    x = tvm.const(1)
    print(x.dtype)
    assert x.dtype == tvm.int32
    assert isinstance(x, tvm.expr.IntImm)

def test_const_saveload_json():
    # save load json
    x = tvm.const(1)
    y = tvm.const(10)
    z = x + y
    z = z + z
    json_str = tvm.save_json(z)
    zz = tvm.load_json(json_str)
    assert tvm.save_json(zz) == tvm.save_json(z)

def test_make():
    x = tvm.const(1)
    y = tvm.make.IntImm('int32', 1)
    z = x + y
    print(z)

def test_ir():
    x = tvm.const(1)
    y = tvm.make.IntImm('int32', 1)
    z = x + y
    stmt = tvm.make.Evaluate(z)
    assert isinstance(stmt, tvm.stmt.Evaluate)

def test_ir2():
    x = tvm.var("n")
    a = tvm.var("array", tvm.handle)
    st = tvm.make.Store(a, x + 1, 1)
    assert isinstance(st, tvm.stmt.Store)
    assert(st.buffer_var == a)

def test_let():
    x = tvm.var('x')
    y = tvm.var('y')
    stmt = tvm.make.LetStmt(
        x, 10, tvm.make.Evaluate(x + 1));

def test_attr():
    x = tvm.var('x')
    y = tvm.var('y')
    stmt = tvm.make.AttrStmt(
        y, "stride", 10, tvm.make.Evaluate(x + 1));
    assert stmt.node == y

    a = tvm.convert(1)
    assert a.value == 1
    try:
        a.no_field
        assert False
    except AttributeError:
        pass


def test_basic():
    a = tvm.var('a')
    b = tvm.var('b')
    c =  a + b
    assert str(c) == '(%s + %s)' % (a.name, b.name)


def test_stmt():
    x = tvm.make.Evaluate(0)
    tvm.make.For(tvm.var('i'), 0, 1,
                 tvm.stmt.For.Serial, 0,
                 x)

def test_dir():
    x = tvm.var('x')
    dir(x)

def test_dtype():
    x = tvm.var('x')
    assert x.dtype == 'int32'
    y = tvm.var('y')
    assert (x > y).dtype == 'uint1'

if __name__ == "__main__":
    test_attr()
    test_const()
    test_const_saveload_json()
    test_make()
    test_ir()
    test_basic()
    test_stmt()
    test_let()
    test_dir()
    test_dtype()
