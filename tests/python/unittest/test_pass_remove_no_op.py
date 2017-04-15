import tvm

def test_remove_no_op():
    i = tvm.var('i')
    j = tvm.var('j')
    k = tvm.var('k')
    m = tvm.var('m')
    n = tvm.var('n')
    dtype = 'int64'
    Ab = tvm.decl_buffer((n, ), dtype)
    stmt = tvm.make.For(
        i, 0, 4, 0, 0,
        tvm.make.For(
            j, 0, n, 0, 0,
            tvm.make.For(
                k, 0, m, 0, 0,
                tvm.make.IfThenElse(
                    (i*m+j+k < n), tvm.make.Evaluate(m), tvm.make.Evaluate(n)))))
    ret = tvm.ir_pass.RemoveNoOp(stmt)
    assert(isinstance(ret, tvm.stmt.Evaluate))
    store = tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 1,
                           i + 1)
    stmt2 = tvm.make.Block(stmt, store)
    assert(tvm.ir_pass.RemoveNoOp(stmt2) == store)


if __name__ == "__main__":
    test_remove_no_op()
