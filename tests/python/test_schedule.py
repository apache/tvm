import tvm

def test_schedule_create():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    AA = tvm.compute((m, l), lambda i, j: A[i, j])
    T = tvm.compute((m, n, l), lambda i, j, k: A(i, k) * B(j, k))

    sch_T = tvm.Schedule(T.op, scope="shared")
    sch_A = tvm.Schedule(AA.op, scope="global")

    xo, xi = sch_T.split(T.op.axis[0], factor=10)
    xi1, xi2 = sch_T.split(xi, factor=2)

    sch_A.compute_at(sch_T, xi1)
    xo, xi = sch_A.split(AA.op.axis[0], factor=10)

    sch_T.reorder(xi2, xi1)
    assert T.op.axis[1] in sch_T.leaf_iter_vars

def test_reorder():
    m = tvm.Var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute(m, lambda i: A[i+1])

    sch_T = tvm.Schedule(T.op, scope="shared")
    xo, xi = sch_T.split(T.op.axis[0], factor=10)
    xi1, xi2 = sch_T.split(xi, factor=2)
    order = (xi2, xi1, xo)
    assert tuple(sch_T.leaf_iter_vars) != order
    sch_T.reorder(*order)
    assert tuple(sch_T.leaf_iter_vars) == order

def test_split():
    m = tvm.Var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute((m,), lambda i: A[i])

    sT = tvm.Schedule(T.op)
    xo, xi = sT.split(T.op.axis[0], factor=10)
    assert tuple(sT.leaf_iter_vars) == (xo, xi)


def test_tile():
    m = tvm.Var('m')
    n = tvm.Var('n')
    A = tvm.placeholder((m, n), name='A')
    T = tvm.compute((m, n), lambda i, j: A[i, j])

    sch_T = tvm.Schedule(T.op, scope="shared")
    xo, yo, xi, yi = sch_T.tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
    assert tuple(sch_T.leaf_iter_vars) == (xo, yo, xi, yi)

if __name__ == "__main__":
    test_schedule_create()
    test_reorder()
    test_tile()
    test_split()
