import tvm

def test_unroll_loop():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    j = tvm.var('j')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, n, 2, 0, 0,
        tvm.make.For(j, 0, 8, 3, 0,
                     tvm.make.Store(Ab.data,
                                    tvm.make.Load(dtype, Ab.data, i) + 1,
                                    j + 1)))
    assert isinstance(stmt, tvm.stmt.For)
    stmt = tvm.ir_pass.UnrollLoop(stmt, 4)
    assert not isinstance(stmt, tvm.stmt.For)
    print(stmt)

if __name__ == "__main__":
    test_unroll_loop()
