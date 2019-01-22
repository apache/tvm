import tvm
import numpy
from tvm import comm_reducer
from tvm.ir_pass import Simplify, CanonicalSimplify, Equal

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
    vrange = tvm.convert({m: tvm.Range(tvm.const(0, "int32"), tvm.const(10, "int32"))})
    ret = tvm.ir_pass.Simplify(m % 10, vrange)
    assert ret == m

def test_canonical():
    x = tvm.var("x")
    z = tvm.const(3, "int32")
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


def test_simplify_combiner():
    dummy = tvm.var('dummy')

    prod = comm_reducer(lambda x, y: x*y, lambda t0: tvm.const(1, t0))

    sum_or_prod = comm_reducer(lambda x, y: tvm.expr.Select(dummy < 0,
                                                            x + y, x*y),
                               lambda t0: tvm.expr.Select(dummy < 0,
                                                          tvm.const(0, t0), tvm.const(1, t0)))

    sum_and_prod = comm_reducer(lambda x, y: (x[0] + y[0],
                                              x[1]*y[1]),
                                lambda t0, t1: (tvm.const(0, t0),
                                                tvm.const(5, t0) - tvm.const(4, t0)))

    sum_and_prod2 = comm_reducer(lambda x, y: (x[0] + y[0],
                                               x[1]*y[1] + 0*x[0] + y[0] - y[0]),
                                 lambda t0, t1: (tvm.const(5, t0) - tvm.const(5, t0),
                                                 tvm.const(1, t1)))

    some_reducer1 = comm_reducer(lambda x, y: (x[0] + y[0],
                                               x[0] + y[0] + x[1] + y[1],
                                               x[0]*y[2] + y[0]*x[2],
                                               x[1] + y[2],
                                               4.0),
                                 lambda t0, t1, t2, t3, t4: (tvm.const(0, t0),
                                                             tvm.const(1, t1),
                                                             tvm.const(2, t2),
                                                             tvm.const(3, t3),
                                                             tvm.const(4, t4)))

    k = tvm.reduce_axis((0, 10), name="k")
    A = tvm.placeholder((10,), name='A')

    # Test that SimplifyCombiner makes use of vranges
    vrange = {dummy: tvm.Range(-10, -5)}
    assert Equal(Simplify(sum_or_prod(A[k], k), vrange), tvm.sum(A[k], k))
    vrange = {dummy: tvm.Range(5, 10)}
    assert Equal(Simplify(sum_or_prod(A[k], k), vrange), prod(A[k], k))

    assert Equal(Simplify(sum_and_prod((A[k], A[10-k]), k)[0]), tvm.sum(A[k], k))
    assert Equal(Simplify(sum_and_prod((A[k], A[10-k]), k)[1]), prod(A[10-k], k))

    assert Equal(Simplify(sum_and_prod2((A[k], A[10-k]), k)[0]), tvm.sum(A[k], k))
    assert Equal(Simplify(sum_and_prod2((A[k], A[10-k]), k)[1]), prod(A[10-k], k))

    reference_simplified_sources = [[A[0]],
                                    [A[0], A[1]],
                                    [A[0], A[2]],
                                    [A[0], A[1], A[2], A[3]],
                                    [A[4]]]
    for j in range(5):
        # Here we use the j-th component of the result, so only it and the components it
        # depends on are left.
        simplified = Simplify(some_reducer1((A[0], A[1], A[2], A[3], A[4]), k)[j])

        # Check that the remaining components are the expected ones.
        for lhs, rhs in zip(simplified.source, reference_simplified_sources[j]):
            assert Equal(lhs, rhs)

    # Test that components with side effects are not removed
    side_effect = lambda *xs: tvm.make.Call("int32", "dummy", xs, tvm.expr.Call.Intrinsic, None, 0)
    assert Equal(Simplify(sum_and_prod((A[k], side_effect(A[10-k])), k)[0]),
                 sum_and_prod((A[k], side_effect(A[10-k])), k)[0])
    assert Equal(Simplify(sum_and_prod((side_effect(A[k]), A[10-k]), k)[0]),
                 tvm.sum(side_effect(A[k]), k))


def test_simplify_reduce():
    k = tvm.reduce_axis((0, 10), name="k")
    j = tvm.reduce_axis((-5, 3), name="j")
    A = tvm.placeholder((10,), name='A')

    assert Equal(Simplify(tvm.sum(k/10, k)), tvm.sum(tvm.const(0, "int32"), k))
    assert Equal(Simplify(tvm.sum(A[3], [])), A[3])
    assert Equal(Simplify(tvm.sum(tvm.expr.Select(k + j < 12, k + j, 0), [k, j])),
                 tvm.sum(k + j, [k, j]))


if __name__ == "__main__":
    test_bound()
    test_basic()
    test_simplify()
    test_canonical()
    test_simplify_combiner()
    test_simplify_reduce()
