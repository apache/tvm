import tvm
import numpy

def test_simplify():
    """Not yet working, mock design"""
    dtype = 'int64'
    n = tvm.Var('n')
    Ab = tvm.Buffer((n, ), dtype)
    i = tvm.Var('i')
    j = tvm.Var('j')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, 2, n, 0, 0,
        tvm.make.For(j, 0, n, 0, 0,
                     tvm.make.IfThenElse(
                         tvm.make.LT(i + 2, n),
                         tvm.make.Store(Ab.data,
                                        tvm.make.Load(dtype, Ab.data, i + 4) + 1,
                                        (j + 1) * 4 - 4 * j + i),
                         None)))
    print(stmt)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    print(stmt)

if __name__ == "__main__":
    test_simplify()
