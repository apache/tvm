import tvm
import pickle as pkl

def test_schedule_create():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    AA = tvm.compute((m, l), lambda i, j: A[i, j])
    T = tvm.compute((m, n, l), lambda i, j, k: AA(i, k) * B(j, k))
    s = tvm.Schedule(T.op)
    s[AA].set_scope("shared")
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    xi1, xi2 = s[T].split(xi, factor=2)
    s[AA].compute_at(s[T], xi1)
    xo, xi = s[AA].split(AA.op.axis[0], factor=10)
    s[T].reorder(xi2, xi1)
    assert T.op.axis[1] in s[T].leaf_iter_vars

    # save load json
    json_str = tvm.save_json(s)
    s_loaded = tvm.load_json(json_str)
    assert isinstance(s_loaded, tvm.schedule.Schedule)
    assert(str(s_loaded.roots[0].body) == str(s.roots[0].body))

    # pickle unpickle
    dump = pkl.dumps(s)
    s_loaded = pkl.loads(dump)
    assert isinstance(s_loaded, tvm.schedule.Schedule)
    assert(str(s_loaded.roots[0].body) == str(s.roots[0].body))

def test_reorder():
    m = tvm.Var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute(m, lambda i: A[i+1])

    s = tvm.Schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    xi1, xi2 = s[T].split(xi, factor=2)
    order = (xi2, xi1, xo)
    assert tuple(s[T].leaf_iter_vars) != order
    s[T].reorder(*order)
    assert tuple(s[T].leaf_iter_vars) == order

def test_split():
    m = tvm.Var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute((m,), lambda i: A[i])

    s = tvm.Schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    assert tuple(s[T].leaf_iter_vars) == (xo, xi)


def test_tile():
    m = tvm.Var('m')
    n = tvm.Var('n')
    A = tvm.placeholder((m, n), name='A')
    T = tvm.compute((m, n), lambda i, j: A[i, j])

    s = tvm.Schedule(T.op)
    xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
    assert tuple(s[T].leaf_iter_vars) == (xo, yo, xi, yi)

if __name__ == "__main__":
    test_schedule_create()
    test_reorder()
    test_tile()
    test_split()
