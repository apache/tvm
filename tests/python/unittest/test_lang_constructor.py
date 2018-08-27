import tvm

def test_expr_constructor():
    x = tvm.expr.Var("xx", "float32")
    assert isinstance(x, tvm.expr.Var)
    assert x.name == "xx"

    x = tvm.expr.Reduce(None, [1],
                        [tvm.api._IterVar((0, 1), "x", 2)],
                        None, 0)
    assert isinstance(x, tvm.expr.Reduce)
    assert x.combiner == None
    assert x.value_index == 0

    x = tvm.expr.FloatImm("float32", 1.0)
    assert isinstance(x, tvm.expr.FloatImm)
    assert x.value == 1.0
    assert x.dtype == "float32"

    x = tvm.expr.IntImm("int64", 2)
    assert isinstance(x, tvm.expr.IntImm)
    assert x.value == 2
    assert x.dtype == "int64"

    x = tvm.expr.UIntImm("uint16", 2)
    assert isinstance(x, tvm.expr.UIntImm)
    assert x.value == 2
    assert x.dtype == "uint16"

    x = tvm.expr.StringImm("xyza")
    assert isinstance(x, tvm.expr.StringImm)
    assert x.value == "xyza"

    x = tvm.expr.Cast("float32", tvm.expr.IntImm("int32", 1))
    assert isinstance(x, tvm.expr.Cast)
    assert x.dtype == "float32"
    assert x.value.value == 1

    a = tvm.const(1.0, dtype="float32")
    b = tvm.var("x", dtype="float32")

    for cls in [tvm.expr.Add,
                tvm.expr.Sub,
                tvm.expr.Mul,
                tvm.expr.Div,
                tvm.expr.Mod,
                tvm.expr.Min,
                tvm.expr.Max,
                tvm.expr.LT,
                tvm.expr.LE,
                tvm.expr.GT,
                tvm.expr.GE]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)


    a = tvm.convert(tvm.var("x") > 1)
    b = tvm.convert(tvm.var("x") == 1)

    for cls in [tvm.expr.And,
                tvm.expr.Or]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    x = tvm.expr.Not(a)
    assert isinstance(x, tvm.expr.Not)
    assert x.a == a

    x = tvm.expr.Select(a, a, b)
    assert isinstance(x, tvm.expr.Select)
    assert x.true_value == a
    assert x.false_value == b
    assert x.condition == a

    buffer_var = tvm.var("x", dtype="handle")
    x = tvm.expr.Load("float32", buffer_var, 1, a)
    assert isinstance(x, tvm.expr.Load)
    assert x.dtype == "float32"
    assert x.buffer_var == buffer_var
    assert x.index.value == 1
    assert x.predicate == a

    x = tvm.expr.Ramp(1, 2, 10)
    assert isinstance(x, tvm.expr.Ramp)
    assert x.base.value == 1
    assert x.stride.value == 2
    assert x.lanes == 10

    x = tvm.expr.Broadcast(a, 10)
    assert isinstance(x, tvm.expr.Broadcast)
    assert x.value == a
    assert x.lanes == 10

    x = tvm.expr.Shuffle([a], [0])
    assert isinstance(x, tvm.expr.Shuffle)
    assert x.vectors[0] == a
    assert x.indices[0].value == 0

    x = tvm.expr.Call("float32", "xyz", [a], tvm.expr.Call.Extern, None, 0)
    assert isinstance(x, tvm.expr.Call)
    assert x.dtype == "float32"
    assert x.name == "xyz"
    assert x.args[0] == a
    assert x.call_type == tvm.expr.Call.Extern
    assert x.func == None
    assert x.value_index == 0

    v = tvm.var("aa")
    x = tvm.expr.Let(v, 1, v)
    assert x.var == v
    assert x.value.value == 1
    assert x.body == v


def test_stmt_constructor():
    v = tvm.var("aa")
    buffer_var = tvm.var("buf", dtype="handle")
    nop = tvm.stmt.Evaluate(1)
    x = tvm.stmt.LetStmt(v, 1, tvm.stmt.Evaluate(1))
    assert isinstance(x, tvm.stmt.LetStmt)
    assert x.var == v
    assert x.value.value == 1
    assert isinstance(x.body, tvm.stmt.Evaluate)

    x = tvm.stmt.AttrStmt(v == 1, "xx", 1, tvm.stmt.Evaluate(1))
    assert isinstance(x, tvm.stmt.AttrStmt)
    assert x.value.value == 1

    x = tvm.stmt.Block(tvm.stmt.Evaluate(11),
                       nop)
    assert isinstance(x, tvm.stmt.Block)
    assert x.first.value.value == 11
    assert x.rest == nop

    x = tvm.stmt.AssertStmt(tvm.const(1, "uint1"),
                            tvm.convert("hellow"),
                            nop)
    assert isinstance(x, tvm.stmt.AssertStmt)
    assert x.body == nop

    x = tvm.stmt.ProducerConsumer(None, True, nop)
    assert isinstance(x, tvm.stmt.ProducerConsumer)
    assert x.body == nop

    x = tvm.stmt.For(tvm.var("x"), 0, 10, 0, 0, nop)
    assert isinstance(x, tvm.stmt.For)
    assert x.min.value == 0
    assert x.extent.value == 10
    assert x.body == nop

    x = tvm.stmt.Store(buffer_var, 1, 10, tvm.const(1, "uint1"))
    assert isinstance(x, tvm.stmt.Store)
    assert x.buffer_var == buffer_var
    assert x.index.value == 10
    assert x.value.value == 1

    tensor = tvm.placeholder((), dtype="float32")
    x = tvm.stmt.Provide(tensor.op, 0, 10, [])
    assert isinstance(x, tvm.stmt.Provide)
    assert x.value_index == 0
    assert x.value.value == 10

    x = tvm.stmt.Allocate(buffer_var, "float32", [10],
                          tvm.const(1, "uint1"), nop)
    assert isinstance(x, tvm.stmt.Allocate)
    assert x.dtype == "float32"
    assert x.buffer_var == buffer_var
    assert x.body == nop

    x = tvm.stmt.AttrStmt(buffer_var, "xyz", 1, nop)
    assert isinstance(x, tvm.stmt.AttrStmt)
    assert x.node == buffer_var
    assert x.attr_key == "xyz"
    assert x.body == nop

    x = tvm.stmt.Free(buffer_var)
    assert isinstance(x, tvm.stmt.Free)
    assert x.buffer_var == buffer_var

    x = tvm.stmt.Realize(None, 0, "float", [], tvm.const(1, "uint1"), nop)
    assert isinstance(x, tvm.stmt.Realize)
    assert x.body == nop

    x = tvm.stmt.IfThenElse(tvm.const(1, "uint1"),
                            tvm.stmt.Evaluate(11),
                            nop)
    assert isinstance(x, tvm.stmt.IfThenElse)
    assert x.then_case.value.value == 11
    assert x.else_case == nop

    x = tvm.stmt.Prefetch(None, 1, "float32", [])
    assert isinstance(x, tvm.stmt.Prefetch)
    assert x.value_index == 1


if __name__ == "__main__":
    test_expr_constructor()
    test_stmt_constructor()
