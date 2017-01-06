import tvm

def test_inline():
    m = tvm.Var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute((m,), lambda i,: A[i] + 10, name='T')
    stmt = tvm.make.Evaluate(T[10] + 11 * T[100])
    stmt = tvm.ir_pass.Inline(
        T, [x.var for x in T.op.axis], T.op.body, stmt)
    print(stmt)
    assert(tvm.ir_pass.VerifySSA(stmt))

    try:
        # pass in int array(wrong argument type)
        # must raise an error
        stmt = tvm.ir_pass.Inline(
            T, [1,2,3], T.op.body, stmt)
        assert False
    except tvm.TVMError:
        pass



if __name__ == "__main__":
    test_inline()
