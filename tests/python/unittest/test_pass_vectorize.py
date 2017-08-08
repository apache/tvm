import tvm

def test_vectorize_loop():
    dtype = 'int64'
    n = tvm.var('n')
    ib = tvm.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, n) as i:
        with ib.for_range(0, 4, for_type="vectorize") as j:
            A[j] = tvm.const(1, A.dtype)
    stmt = ib.get()

    assert isinstance(stmt.body, tvm.stmt.For)
    stmt = tvm.ir_pass.VectorizeLoop(stmt)
    assert isinstance(stmt, tvm.stmt.For)
    assert not isinstance(stmt.body, tvm.stmt.For)
    assert isinstance(stmt.body.index, tvm.expr.Ramp)
    assert isinstance(stmt.body.value, tvm.expr.Broadcast)

def test_vectorize_vector():
    dtype = 'int64'
    n = tvm.var('n')
    ib = tvm.ir_builder.create()
    A = ib.pointer("float32x4", name="A")
    with ib.for_range(0, n) as i:
        with ib.for_range(0, 4, for_type="vectorize") as j:
            A[j] = tvm.const(1, A.dtype)
    stmt = ib.get()
    assert isinstance(stmt.body, tvm.stmt.For)
    stmt = tvm.ir_pass.VectorizeLoop(stmt)
    assert isinstance(stmt, tvm.stmt.For)
    assert not isinstance(stmt.body, tvm.stmt.For)
    assert isinstance(stmt.body.index, tvm.expr.Ramp)
    assert isinstance(stmt.body.value, tvm.expr.Broadcast)


def test_vectorize_with_if():
    n = tvm.var('n')
    x = tvm.var('x')
    ib = tvm.ir_builder.create()
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 4, for_type="vectorize") as i:
        with ib.if_scope(x < n):
            A[i] = A[i] + 1
        with ib.else_scope():
            with ib.if_scope(i < n):
                A[i] = 2.0
    stmt = ib.get()
    stmt = tvm.ir_pass.VectorizeLoop(stmt)
    assert isinstance(stmt, tvm.stmt.IfThenElse)
    assert isinstance(stmt.then_case.index, tvm.expr.Ramp)
    assert isinstance(stmt.then_case.value, tvm.expr.Add)
    assert stmt.then_case.value.dtype == "float32x4"
    assert isinstance(stmt.else_case, tvm.stmt.For)

if __name__ == "__main__":
    test_vectorize_vector()
    test_vectorize_with_if()
    test_vectorize_loop()
