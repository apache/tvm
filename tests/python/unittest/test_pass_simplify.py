import tvm
import numpy

def test_simplify():
    """Not yet working, mock design"""
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    j = tvm.var('j')
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
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)


def test_basic():
    m = tvm.var('m')
    ret = tvm.ir_pass.CanonicalSimplify(tvm.make.Evaluate(m-1))
    assert str(ret.value) == "(m - 1)"


def test_bound():
    m = tvm.var('m')
    vrange = tvm.convert({m: tvm.Range(tvm.const(0), tvm.const(10))})
    ret = tvm.ir_pass.Simplify(m % 10, vrange)
    assert ret == m

def test_canonical():
    x = tvm.var("x")
    z = tvm.const(3)
    ret = tvm.ir_pass.CanonicalSimplify(x / (z*z) - x / (z*z))
    assert(tvm.ir_pass.Equal(ret, 0))

    ret = tvm.ir_pass.CanonicalSimplify(x / (z+z) - x / (z+z))
    assert(tvm.ir_pass.Equal(ret, 0))

    #make sure terms are ordered based on their top operators (e.g., / always precedes %)
    ret1 = tvm.ir_pass.CanonicalSimplify(x % 3 + x / 3)
    ret2 = tvm.ir_pass.CanonicalSimplify(x / 3 + x % 3)
    assert(tvm.ir_pass.Equal(ret1, ret2))

    #when top operators match, compare string representation of terms
    ret1 = tvm.ir_pass.CanonicalSimplify(x % 4 + x % 3)
    ret2 = tvm.ir_pass.CanonicalSimplify(x % 3 + x % 4)
    assert (tvm.ir_pass.Equal(ret1, ret2))

if __name__ == "__main__":
    test_bound()
    test_basic()
    test_simplify()
    test_canonical()
