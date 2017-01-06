import tvm

def test_bound_inference():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
    sA1 = tvm.Schedule(A1.op)
    sA2 = tvm.Schedule(A2.op)
    xo, xi = sA2.split(A2.op.dim_var[0], 8)
    sA1.compute_at(sA2, xo)
    bounds = tvm.schedule.InferBound(sA2)
    assert isinstance(bounds, tvm.collections.Map)
    print(bounds[A1.op.dim_var[0]])
    print(bounds[A1.op.dim_var[1]])


def test_create_read_graph():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j])
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3)
    g = tvm.schedule.CreateReadGraph(A2.op)
    assert g[A2.op][0] == A1
    assert g[A1.op][0] == A
    post_order = tvm.schedule.PostDFSOrder(A2.op, g)
    assert(post_order[0] == A1.op)
    assert(post_order[1] == A2.op)


if __name__ == "__main__":
    test_bound_inference()
    test_create_read_graph()
