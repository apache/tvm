import tvm

def test_vectorize_loop():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    j = tvm.var('j')
    VECTORIZE = 2
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, n, 2, 0, 0,
        tvm.make.For(j, 0, 4, VECTORIZE, 0,
                     tvm.make.Store(Ab.data,
                                    tvm.make.Load(dtype, Ab.data, i) + 1,
                                    j + 1)))
    assert isinstance(stmt.body, tvm.stmt.For)
    stmt = tvm.ir_pass.VectorizeLoop(stmt)
    assert isinstance(stmt, tvm.stmt.For)
    assert not isinstance(stmt.body, tvm.stmt.For)
    print(stmt)

if __name__ == "__main__":
    test_vectorize_loop()
