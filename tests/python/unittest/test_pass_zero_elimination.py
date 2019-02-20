import random
import sys
import numpy as np
import tvm
from tvm import comm_reducer
from tvm.testing import estimate_performance
from tvm.ir_pass import Simplify, Equal, LiftNonzeronessCondition, IsSumCombiner, \
    CanFactorZeroFromCombiner, InlineTailCall, InlineTensors, SolveSystemOfInequalities, \
    SimplifyDomain, SimplifyReductionDomain, ExtractAsTensorMaybe, ExtractReductions, \
    ExtractNonTopReductions, OptimizeAndLiftNonzeronessConditions

def get_shape(tensor):
    return [s.value for s in tensor.shape]

def check_eq(t1, t2, args):
    s1 = tvm.create_schedule(t1.op)
    m1 = tvm.build(s1, [t1] + args)

    s2 = tvm.create_schedule(t2.op)
    m2 = tvm.build(s2, [t2] + args)

    for _ in range(5):
        arg_vals = [tvm.ndarray.array(np.random.uniform(-10, 10, size=get_shape(a))
                                      .astype(a.dtype))
                    for a in [t1] + args]
        m1(*arg_vals)
        res1 = arg_vals[0].asnumpy()
        m2(*arg_vals)
        res2 = arg_vals[0].asnumpy()

        np.testing.assert_allclose(res1, res2, atol=1e-3, rtol=1e-2)

def check_symeq(expr1, expr2):
    expr1 = tvm.ir_pass.Simplify(tvm.ir_pass.CanonicalSimplify(expr1))
    expr2 = tvm.ir_pass.Simplify(tvm.ir_pass.CanonicalSimplify(expr2))

    if tvm.ir_pass.Equal(expr1, expr2):
        return

    diff = tvm.ir_pass.Simplify(tvm.ir_pass.CanonicalSimplify(expr1 - expr2))
    if not Equal(diff, tvm.const(0, expr1.dtype)):
        raise AssertionError("Expressions {} and {} are not equal, their diff is {}"
                             .format(expr1, expr2, diff))

def compute(shape, fcompute):
    """Like tvm.compute but automatically extracts reductions."""
    return tvm.compute(shape,
                       lambda *vs: ExtractNonTopReductions(
                           fcompute(*vs), vs, {v: tvm.Range(0, s) for v, s in zip(vs, shape)}))

def check_tensor_symeq(A, B):
    if not isinstance(B, tvm.tensor.Tensor):
        B = compute(A.shape, B)
    vmap = {a.var: b.var for a, b in zip(A.op.axis, B.op.axis)}
    expr_a = tvm.ir_pass.Substitute(A.op.body[A.value_index], vmap)
    expr_b = B.op.body[B.value_index]
    expr_a = tvm.ir_pass.CanonicalSimplify(InlineTensors(expr_a, [], True))
    expr_b = tvm.ir_pass.CanonicalSimplify(InlineTensors(expr_b, [], True))
    if not Equal(expr_a, expr_b):
        print(expr_a)
        print(expr_b)
        raise AssertionError("The expressions are not equal")

def check_eq_bruteforce(expr1, expr2, vranges):
    def _compute_body(*us):
        vmap = {v: u + r.min for (v, r), u in zip(vranges.items(), us)}
        return tvm.ir_pass.Substitute(expr1 == expr2, vmap)

    A = compute([r.extent.value for v, r in vranges.items()], _compute_body)
    args = [tvm.ndarray.empty(A.shape, A.dtype)]
    sch = tvm.create_schedule(A.op)
    mod = tvm.build(sch, [A])
    mod(*args)
    res = args[0].asnumpy()
    if not np.all(res):
        indices = list(np.argwhere(res == 0)[0])
        counterex = [(str(v), i + r.min) for (v, r), i in zip(vranges.items(), indices)]
        counterex = ", ".join([v + " = " + str(i) for v, i in sorted(counterex)])
        raise AssertionError("Expressions {}\nand {}\nare not equal on {}\n"
                             "Counterexample: {}"
                             .format(expr1, expr2, vranges, counterex))

prod_combiner = comm_reducer(lambda x, y: x*y, lambda t0: tvm.const(1, t0))
sum_combiner = comm_reducer(lambda x, y: x + y, lambda t0: tvm.const(0, t0))
sum2_combiner = comm_reducer(lambda x, y: y + x, lambda t0: tvm.const(0, t0))
sum_derivative_combiner = comm_reducer(lambda x, y: (x[0] + y[0], y[1] + x[1]),
                                       lambda t0, t1: (tvm.const(0, t0), tvm.const(0, t1)))
prod_derivative_combiner = comm_reducer(lambda x, y: (x[0]*y[0], x[0]*y[1] + x[1]*y[0]),
                                        lambda t0, t1: (tvm.const(1, t0), tvm.const(0, t1)))
sum_both_combiner = comm_reducer(lambda x, y: (x[0] + y[0], x[0] + y[0] + x[1] + y[1]),
                                        lambda t0, t1: (tvm.const(0, t0), tvm.const(0, t1)))
xor_combiner = comm_reducer(lambda x, y: x ^ y, lambda t0: tvm.const(0, t0))

def test_is_sum_combiner():
    k = tvm.reduce_axis((0, 10), name="k")
    i = tvm.const(0, "int32")
    f = tvm.const(0.0, "float32")
    assert IsSumCombiner(sum_combiner(i, k).combiner)
    assert IsSumCombiner(sum_combiner(f, k).combiner)
    assert IsSumCombiner(sum2_combiner(i, k).combiner)
    assert IsSumCombiner(sum2_combiner(f, k).combiner)
    assert not IsSumCombiner(sum_derivative_combiner((f, f), k)[0].combiner)
    assert not IsSumCombiner(prod_combiner(f, k).combiner)
    assert not IsSumCombiner(prod_derivative_combiner((f, f), k)[1].combiner)

def test_can_factor_zero_from_combiner():
    k = tvm.reduce_axis((0, 10), name="k")
    i = tvm.const(0, "int32")
    f = tvm.const(0.0, "float32")
    assert CanFactorZeroFromCombiner(sum_combiner(i, k).combiner, 0)
    assert CanFactorZeroFromCombiner(sum2_combiner(f, k).combiner, 0)
    assert CanFactorZeroFromCombiner(sum_derivative_combiner((f, f), k)[0].combiner, 0)
    assert CanFactorZeroFromCombiner(sum_derivative_combiner((f, f), k)[0].combiner, 1)
    assert not CanFactorZeroFromCombiner(prod_derivative_combiner((f, f), k)[0].combiner, 0)
    assert CanFactorZeroFromCombiner(prod_derivative_combiner((f, f), k)[0].combiner, 1)
    assert CanFactorZeroFromCombiner(sum_both_combiner((f, f), k)[0].combiner, 0)
    assert not CanFactorZeroFromCombiner(sum_both_combiner((f, f), k)[0].combiner, 1)

def test_lift_nonzeroness_condition():
    k = tvm.reduce_axis((0, 5), name="k")
    l = tvm.reduce_axis((0, 5), name="l")
    n = tvm.reduce_axis((0, 5), name="n")
    A = tvm.placeholder((10,), name='A')

    def _check(shape, fun, A=A):
        T1 = tvm.compute(shape, fun)
        T2 = tvm.compute(shape, lambda *args: LiftNonzeronessCondition(fun(*args)))
        check_eq(T1, T2, [A])
        assert isinstance(T2.op.body[0], tvm.expr.Select)

    _check((10,), lambda i: A[i])
    _check((10,), lambda i: A[i] + (i % 2 == 0))
    _check((10,), lambda i: A[i]*(i % 2 == 0) + (i % 2 == 0))
    _check((10,), lambda i: tvm.expr.Select((i % 2 == 0), A[i], 0.0))
    _check((10,), lambda i: tvm.expr.Select((i % 2 == 0), A[i], 0.0) + (i % 2 == 0))
    _check((10,), lambda i: tvm.expr.Select((i % 2 == 0), 0.0, A[i]) + (i % 2 == 0))
    def e1(i): return tvm.expr.Select((i % 2 == 1), 0.0, A[i])
    def e2(i): return tvm.expr.Select((i % 2 == 0), A[(i + 1) % 10], 0.0)
    def e3(i): return tvm.expr.Select((i % 2 == 1), A[i], 0.0)
    _check((10,), lambda i: e1(i) + e2(i) + e3(i) + e1(i)*e2(i))
    _check((10,), lambda i: e1(i)*e3(i))
    _check((10,), lambda i: e1(i)*e2(i))
    _check((10,10), lambda i, j: A[i]*(i == j) + A[j]*(i == 2*j) + A[j]*(j == i))
    _check((10,10), lambda i, j: tvm.min(A[i]*(i == j), A[j]*(i == 2*j)))
    _check((10,10), lambda i, j: tvm.max(A[i]*(i == j), A[j]*(i == 2*j)))
    _check((10,10), lambda i, j: A[i]*(i == j) - A[j]*(i == 2*j))
    _check((10,10), lambda i, j: A[i]*(i == j) / (1 + tvm.abs(A[j]*(i == 2*j))))
    _check((10,10), lambda i, j: i*(i < j) + j*(i > j))
    _check((10,10), lambda i, j: i*(i < j) % (1 + j*(i > j)))

    def _check_symeq(expr1, expr2):
        expr1 = LiftNonzeronessCondition(expr1)
        expr2 = LiftNonzeronessCondition(expr2)
        print(expr1)
        print(expr2)
        print()
        check_symeq(expr1, expr2)

    _check_symeq(tvm.expr.Select(tvm.expr.EQ(k, l), 0.0, tvm.expr.Cast('float32', (k < n))),
                 tvm.expr.Select(tvm.expr.And((k < n), tvm.expr.NE(k, l)), 1.0, 0.0))
    _check_symeq(tvm.min(tvm.expr.Cast('int32', k < n)*l, tvm.expr.Select(k >= n, 0, 1)),
                 tvm.expr.Select(k < n, tvm.min(l, 1), 0))

def test_inline_tail_call():
    A = tvm.compute((10, 10), lambda i, j: i + j*j)
    B = tvm.compute((5, 6), lambda k, l: A[k + l, k + 1])
    C = InlineTailCall(B)
    resbody = lambda k, l: k + l + (k + 1)*(k + 1)
    check_symeq(C.op.body[0], resbody(*[iv.var for iv in C.op.axis]))

def test_inline_tensors():
    A = tvm.compute((10, 10), lambda i, j: i + j)
    B = tvm.compute((10, 10), lambda i, j: i * j)
    C = tvm.compute((10, 10), lambda i, j: A[i, j] + B[i, j])
    k = tvm.reduce_axis((0, 5), name="k")
    D = tvm.compute((10, 10), lambda i, j: tvm.sum(A[i, k], k))
    E = tvm.compute((10, 10), lambda i, j: A[2, j] + C[i, 2] + D[i, j])

    R = InlineTensors(E)
    resbody = lambda i, j: 2 + j + i + 2 + i*2 + D[i, j]
    check_symeq(R.op.body[0], resbody(*[iv.var for iv in R.op.axis]))

    R = InlineTensors(E, [A])
    resbody = lambda i, j: 2 + j + C[i, 2] + D[i, j]
    check_symeq(R.op.body[0], resbody(*[iv.var for iv in R.op.axis]))

    R = InlineTensors(E, [A, C])
    resbody = lambda i, j: 2 + j + ((i + 2) + B[i, 2]) + D[i, j]
    check_symeq(R.op.body[0], resbody(*[iv.var for iv in R.op.axis]))

    R = InlineTensors(E, [B, C])
    resbody = lambda i, j: A[2, j] + (A[i, 2] + i*2) + D[i, j]
    check_symeq(R.op.body[0], resbody(*[iv.var for iv in R.op.axis]))

def test_solve_system_of_inequalities():
    seed = random.randrange(sys.maxsize)
    print("\nseed: {}\n".format(seed))
    random.seed(seed)

    def _check(variables, formulas, coef=(-5, 5), bounds=(-20, 20)):
        vs = [tvm.var("x" + str(i)) for i in range(variables)]

        fs = []
        for i in range(formulas):
            s1 = sum([v*random.randint(coef[0], coef[1]) for v in vs])
            s1 += random.randint(coef[0], coef[1])
            s2 = sum([v*random.randint(coef[0], coef[1]) for v in vs])
            s2 += random.randint(coef[0], coef[1])
            op = random.choice([tvm.expr.EQ, tvm.expr.LE, tvm.expr.LT, tvm.expr.GE, tvm.expr.GT])
            fs.append(op(s1, s2))

        vranges = {v: tvm.Range(bounds[0], bounds[1] + 1) for v in vs}

        before = tvm.all(*fs)
        print(before)
        after = tvm.all(*SolveSystemOfInequalities(fs, vs, vranges))
        print(after)
        print()

        check_eq_bruteforce(before, after, vranges)

    for i in range(3):
        _check(1, 1)
    for i in range(3):
        _check(1, 2)

    for i in range(3):
        _check(2, 1)
    for i in range(3):
        _check(2, 2)
    for i in range(3):
        _check(2, 3)

    # Somewhere here coefficients in the results become too large, leading to overflow,
    # so we use smaller initial coefficients

    for i in range(5):
        _check(3, 3, coef=(-2,2))
    for i in range(5):
        _check(3, 4, coef=(-2,2))

    for i in range(5):
        _check(4, 3, coef=(-1,1))

    for i in range(5):
        _check(10, 2, coef=(-1,1), bounds=(0, 4))
    for i in range(5):
        _check(10, 3, coef=(0,1), bounds=(0, 4))

def test_simplify_domain():
    # Note that here we test both SimplifyDomain and SimplifyReductionDomain.
    def _check(cond, axis, volume, vranges={}):
        vranges_with_axis = dict(vranges)
        vranges_with_axis.update({iv.var: iv.dom for iv in axis})
        variables = [iv.var for iv in axis]
        new_cond, new_axis, old_to_new, new_to_old = SimplifyDomain(cond, variables,
                                                                    vranges_with_axis)

        print("old", axis, cond)
        print("new", new_axis, new_cond)
        print("old_to_new", old_to_new)
        print("new_to_old", new_to_old)
        print()

        cond_subst = tvm.ir_pass.Substitute(cond, old_to_new)
        new_vranges = vranges.copy()
        new_vranges.update({v.var: v.dom for v in new_axis})
        # If new_cond is true in the new domain, then cond_subst must also be true in the new
        # domain, but the reverse is not necessarily true
        check_eq_bruteforce(tvm.all(new_cond, cond_subst), new_cond, new_vranges)

        new_cond_subst = tvm.ir_pass.Substitute(new_cond, new_to_old)
        old_vranges = vranges.copy()
        old_vranges.update({v.var: v.dom for v in axis})
        check_eq_bruteforce(cond, tvm.all(cond, new_cond_subst), old_vranges)

        # Also check SimplifyReductionDomain
        reduction = xor_combiner(sum([v*(i + 1) for i, v in enumerate(axis)]), axis)
        new_reduction = SimplifyReductionDomain(reduction, vranges)
        check_eq_bruteforce(reduction, new_reduction, vranges)

        vol = np.prod([iv.dom.extent.value for iv in new_axis])
        if vol != volume:
            raise AssertionError("New volume is {} != {}\n"
                                 "Old domain {} where {}\nNew domain {} where {}"
                                 .format(vol, volume, axis, cond, new_axis, new_cond))

    k = tvm.reduce_axis((0, 5), name="k")
    l = tvm.reduce_axis((0, 5), name="l")
    n = tvm.reduce_axis((0, 5), name="n")

    _check((k <= l), [k, l, n], 125)
    _check((k < l), [k, l, n], 80)
    _check(tvm.expr.EQ(k, l), [k, l, n], 25)
    _check(tvm.all(tvm.expr.EQ(k, l), (l < n)), [k, l, n], 16)
    _check(tvm.expr.EQ(2*l, k), [k, l, n], 15)
    # TODO: the result depends on the order of variables because we don't have a proper solver for
    # systems of linear equations yet
    _check(tvm.expr.EQ(2*l, k), [n, l, k], 25)
    _check(tvm.all(l - k < 2, 2*n == k), [k, l, n], 15)
    _check(tvm.all(l - k < 2, l >= k), [k, l, n], 50)

    some_var = tvm.var('some_var')
    _check(tvm.all(l - k < some_var, l >= k), [k, l, n], 50, {some_var: tvm.Range(0, 3)})
    _check(tvm.all(l - k < some_var, l >= k), [k, l, n], 25, {some_var: tvm.Range(0, 2)})


    k = tvm.reduce_axis((-3, 2), name="k")
    l = tvm.reduce_axis((-3, 2), name="l")
    n = tvm.reduce_axis((-3, 2), name="n")

    _check((k < l), [k, l, n], 80)
    _check(tvm.expr.EQ(k, l), [k, l, n], 25)
    _check(tvm.all(tvm.expr.EQ(k, l), (l < n)), [k, l, n], 16)
    # Now there are only two possible values for l: {l = -1, k = -2} and {l = 0, k = 0}
    _check(tvm.expr.EQ(2*l, k), [k, l, n], 10)
    # TODO: the result depends on the order of variables because we don't have a proper solver for
    # systems of linear equations
    _check(tvm.expr.EQ(2*l, k), [n, l, k], 25)
    _check(tvm.all(l - k < 2, 2*n == k), [k, l, n], 10)
    _check(tvm.all(l - k < 2, l >= k), [k, l, n], 50)

    some_var = tvm.var('some_var')
    _check(tvm.all(l - k < some_var, l >= k), [k, l, n], 50, {some_var: tvm.Range(0, 3)})
    _check(tvm.all(l - k < some_var, l >= k), [k, l, n], 25, {some_var: tvm.Range(0, 2)})


    k = tvm.reduce_axis((0, 6), name="k")
    l = tvm.reduce_axis((0, 5), name="l")
    n = tvm.reduce_axis((0, 30), name="n")

    _check(tvm.all(k + l*6 == n), [k, l, n], 30)
    _check(tvm.all(k + l*6 == n), [n, k, l], 30)
    _check(tvm.all(k + l*6 == n), [n, l, k], 30)

    _check(tvm.all(n / 5 == k, n % 5 == l), [l, k, n], 30)
    # TODO: Same thing with the order
    _check(tvm.all(n / 5 == k, n % 5 == l), [n, l, k], 30)

    k = tvm.reduce_axis((0, 10), name="k")
    l = tvm.reduce_axis((0, 10), name="l")
    # TODO: This is not fully optimized because we don't have a solver
    _check(tvm.all((l + k)%3 <= 1, (l + k)/3 <= 2), [l, k], 144)

def test_extract_as_tensor_maybe():
    def _check(shape, fcompute, volume=None, vranges={}):
        def fcompute_extracted(*variables):
            vranges_updated = dict(vranges)
            vranges_updated.update({v: tvm.Range(0, s) for v, s in zip(variables, shape)})
            expr = fcompute(*variables)
            if isinstance(expr, tvm.expr.Select):
                new_true_value = ExtractAsTensorMaybe(expr.true_value,
                                                      expr.condition,
                                                      variables,
                                                      vranges_updated)
                expr = tvm.expr.Select(expr.condition,
                                       new_true_value,
                                       expr.false_value)
                if volume is not None:
                    assert isinstance(new_true_value, tvm.expr.Call)
                    vol = np.prod([iv.dom.extent.value for iv in new_true_value.func.axis])
                    if vol != volume:
                        raise AssertionError("New volume is {} != {}"
                                             .format(vol, volume))
            return expr

        A = tvm.compute(shape, fcompute)
        B = tvm.compute(shape, fcompute_extracted)
        check_eq(A, B, [])

    _check((10, 10), lambda i, j: tvm.expr.Select(i < 3, i + j, 0), volume=30)
    _check((10, 10), lambda i, j: tvm.expr.Select(i < 3, j, 0), volume=10)
    _check((10, 10), lambda i, j: tvm.expr.Select(i < 3, i, 0), volume=3)
    _check((10, 10), lambda i, j: tvm.expr.Select(tvm.all(i < j, j < 5), i + j, 0), volume=16)
    # This one doesn't get extracted
    _check((10, 10), lambda i, j: tvm.expr.Select(i <= j, i + j, 0))

def test_extract_reductions():
    k = tvm.reduce_axis((0, 10), name="k")
    l = tvm.reduce_axis((0, 10), name="l")
    n = tvm.reduce_axis((0, 10), name="n")

    A = tvm.compute((10, 10),
                    lambda i, j:
                        ExtractReductions(sum_combiner(i + k + xor_combiner(j*k + l, l), k),
                                          [i, j],
                                          {i: tvm.Range(0, 10), j: tvm.Range(0, 10)}))
    B = tvm.compute((10, 10), lambda j, k: xor_combiner(j*k + l, l))
    C = tvm.compute((10, 10), lambda i, j: sum_combiner(i + k + B[j, k], k))
    check_eq(C, A, [])

    fcompute = lambda i, j: \
        ExtractReductions(sum_both_combiner((prod_derivative_combiner((i*n + 2*k, j + k), k)[1],
                                             xor_combiner(j*n + l, l)), n)[1],
                          [i, j],
                          {i: tvm.Range(0, 10), j: tvm.Range(0, 10)})
    A = tvm.compute((10, 10), fcompute)
    _, B = tvm.compute((10, 10, 10),
                       lambda i, j, n: prod_derivative_combiner((i*n + 2*k, j + k), k))
    C = tvm.compute((10, 10), lambda j, n: xor_combiner(j*n + l, l))
    _, D = tvm.compute((10, 10), lambda i, j: sum_both_combiner((B[i, j, n], C[j, n]), n))
    check_eq(A, D, [])

def test_optimize_and_lift_nonzeroness():
    k = tvm.reduce_axis((0, 10), name="k")
    l = tvm.reduce_axis((0, 10), name="l")
    n = tvm.reduce_axis((0, 10), name="n")
    A = tvm.placeholder((10, 10), name="A")

    zero = tvm.const(0, 'float32')

    B = compute((10, 10), lambda i, j: tvm.sum((i == j)*A[i, k] + A[k, j]*(i == j), k))
    B = OptimizeAndLiftNonzeronessConditions(B)
    R = lambda i, j: tvm.expr.Select(i == j,
                                     tvm.sum(A[j, k] + A[k, j], k),
                                     zero)
    check_tensor_symeq(B, R)

    # TODO: This test is unstable: sometimes the resulting condition looks like
    #       (i == j)*(j == i) instead of (i == j)
    #  B = compute((10, 10), lambda i, j: tvm.sum((i == j)*(i == k)*A[i, k] +
    #                                             (i == j)*A[k, j]*(i == k), k))
    #  B = OptimizeAndLiftNonzeronessConditions(B)
    #  R = lambda i, j: tvm.expr.Select(i == j, A[j, j]*2.0, zero)
    #  check_tensor_symeq(B, R)

    B = compute((10, 10), lambda i, j: tvm.sum((i < j)*(j < k)*A[j, k], k))
    B = OptimizeAndLiftNonzeronessConditions(B)
    k1 = tvm.reduce_axis((2, 10), name="k1")
    R = compute((10, 10), lambda i, j:
                tvm.expr.Select(tvm.all(i < j, j < 10),
                                tvm.sum(tvm.expr.Select(j < k1, A[j, k1], zero), k1),
                                zero))
    check_eq(B, R, [A])
    assert estimate_performance(B) <= estimate_performance(R)

    # TODO: This one needs the equation solver
    #  B = compute((10, 10), lambda i, j: tvm.sum((i <= j)*(j <= k)*A[j, k], k, where=(i >= k)))
    #  B = OptimizeAndLiftNonzeronessConditions(B)
    #  R = compute((10, 10), lambda i, j: tvm.expr.Select((i == j), A[i, i], zero))
    #  check_eq(B, R, [A])
    #  assert estimate_performance(B) <= estimate_performance(R)

    B = compute((10, 10),
                lambda i, j: prod_derivative_combiner((A[j, k], (i <= j)*(j < k)*A[i, k]), k)[1])
    B = OptimizeAndLiftNonzeronessConditions(B)
    R = compute((10, 10), lambda i, j:
                tvm.expr.Select(tvm.all(i <= j, j < 10),
                                prod_derivative_combiner((A[j, k], (j < k)*A[i, k]), k)[1],
                                zero))
    check_eq(B, R, [A])
    assert estimate_performance(B) <= estimate_performance(R)

if __name__ == "__main__":
    test_is_sum_combiner()
    test_can_factor_zero_from_combiner()
    test_lift_nonzeroness_condition()
    test_inline_tail_call()
    test_inline_tensors()
    test_solve_system_of_inequalities()
    test_simplify_domain()
    test_extract_as_tensor_maybe()
    test_extract_reductions()
    test_optimize_and_lift_nonzeroness()
