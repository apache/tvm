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
    ret = tvm.ir_pass.UnrollLoop(stmt, 2, 0, True)
    assert not isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 4, 0, False)
    assert isinstance(ret, tvm.stmt.For)
    assert ret.for_type == tvm.stmt.For.Unrolled

if __name__ == "__main__":
    test_unroll_loop()
