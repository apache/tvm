import tvm

def test_inline():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute((m,), lambda i,: A[i] + 10, name='T')
    stmt = tvm.make.Evaluate(T[10] + 11 * T[100])
    stmt = tvm.ir_pass.Inline(
        stmt, T.op, [x.var for x in T.op.axis], T.op.body[0])
    print(stmt)
    assert(tvm.ir_pass.VerifySSA(stmt))

    try:
        # pass in int array(wrong argument type)
        # must raise an error
        stmt = tvm.ir_pass.Inline(
            T.op, [1,2,3], T.op.body, stmt)
        assert False
    except tvm.TVMError:
        pass

def test_inline2():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute((m,), lambda i,: A[i] + 10, name='T')
    stmt = tvm.make.Evaluate(tvm.exp(T[10]) + 11 * T[100])
    stmt = tvm.ir_pass.Inline(
        stmt, T.op, [x.var for x in T.op.axis], T.op.body[0])
    def check(op):
        if isinstance(op, tvm.expr.Call):
            assert op.func != T.op
    tvm.ir_pass.PostOrderVisit(stmt, check)


if __name__ == "__main__":
    test_inline2()
    test_inline()
