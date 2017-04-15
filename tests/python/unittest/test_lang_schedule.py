import tvm
import pickle as pkl

def test_schedule_create():
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    AA = tvm.compute((m, l), lambda i, j: A[i, j])
    T = tvm.compute((m, n, l), lambda i, j, k: AA(i, k) * B(j, k))
    s = tvm.create_schedule(T.op)
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
    assert(str(s_loaded.outputs[0].body) == str(s.outputs[0].body))

    # pickle unpickle
    dump = pkl.dumps(s)
    s_loaded = pkl.loads(dump)
    assert isinstance(s_loaded, tvm.schedule.Schedule)
    assert(str(s_loaded.outputs[0].body) == str(s.outputs[0].body))

def test_reorder():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute(m, lambda i: A[i+1])

    s = tvm.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    xi1, xi2 = s[T].split(xi, factor=2)
    order = (xi2, xi1, xo)
    assert tuple(s[T].leaf_iter_vars) != order
    s[T].reorder(*order)
    assert tuple(s[T].leaf_iter_vars) == order

def test_split():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')
    T = tvm.compute((m,), lambda i: A[i])

    s = tvm.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=10)
    assert tuple(s[T].leaf_iter_vars) == (xo, xi)


def test_tile():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    T = tvm.compute((m, n), lambda i, j: A[i, j])

    s = tvm.create_schedule(T.op)
    xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
    assert tuple(s[T].leaf_iter_vars) == (xo, yo, xi, yi)


def test_fuse():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    T = tvm.compute((m, n), lambda i, j: A[i, j])

    s = tvm.create_schedule(T.op)
    xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
    fused = s[T].fuse(yo, xo)
    assert any(isinstance(x, tvm.schedule.Fuse) for x in s[T].relations)
    assert tuple(s[T].leaf_iter_vars) == (fused, xi, yi)

def test_vectorize():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    T = tvm.compute((m, n), lambda i, j: A[i, j])

    s = tvm.create_schedule(T.op)
    xo, yo, xi, yi = s[T].tile(T.op.axis[0], T.op.axis[1], x_factor=10, y_factor=5)
    s[T].vectorize(yi)
    s[T].unroll(xi)
    UNROLL = tvm.collections.IterVar.Unrolled
    VECTORIZE = tvm.collections.IterVar.Vectorized
    assert s[T].iter_var_attrs[xi].iter_type == UNROLL
    assert s[T].iter_var_attrs[yi].iter_type == VECTORIZE

def test_rfactor():
    n = tvm.var('n')
    k1 = tvm.reduce_axis((0, n), name="k1")
    k2 = tvm.reduce_axis((0, n), name="k2")
    A = tvm.placeholder((n, n, n), name='A')
    B = tvm.compute((n, ), lambda i: tvm.sum(A[i, k1, k2], axis=[k1, k2]))
    # normal schedule
    s = tvm.create_schedule(B.op)
    BF = s.rfactor(B, k1)
    assert(tuple(BF.shape) == (n, n))
    assert(set(BF.op.body.axis) == set([k2]))
    assert(s[B].op.body.axis[0].dom.extent == n)
    assert(len(s[B].all_iter_vars) == 2)
    # schedule with splot
    s = tvm.create_schedule(B.op)
    ko, ki = s[B].split(k1, factor=4)
    xo, xi = s[B].split(B.op.axis[0], factor=8)
    BF = s.rfactor(B, ki)
    assert(BF.shape[0].value == 4)
    assert(BF.shape[1] == n)
    assert(BF.op.body.axis[0] ==  k2)
    assert(BF.op.body.axis[1].var ==  ko.var)
    assert(s[B].op.body.axis[0].dom.extent.value == 4)


if __name__ == "__main__":
    test_rfactor()
    test_schedule_create()
    test_reorder()
    test_tile()
    test_split()
    test_fuse()
    test_vectorize()
