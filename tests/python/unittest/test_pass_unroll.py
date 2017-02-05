import tvm

def test_unroll_loop():
    dtype = 'int64'
    n = tvm.Var('n')
    Ab = tvm.Buffer((n, ), dtype)
    i = tvm.Var('i')
    j = tvm.Var('j')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, n, 2, 0, 0,
        tvm.make.For(j, 0, n, 0, 0,
                     tvm.make.Store(Ab.data,
                                    tvm.make.Load(dtype, Ab.data, i) + 1,
                                    j + 1)))
    stmt = tvm.ir_pass.UnrollLoop(stmt, 8)
    print(stmt)

if __name__ == "__main__":
    test_unroll_loop()
