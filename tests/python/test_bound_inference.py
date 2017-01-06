import tvm

def test_bound1():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
    sA1 = tvm.Schedule(A1.op)
    sA2 = tvm.Schedule(A2.op)
    xo, xi = sA2.split(A2.op.axis[0], 8)
    sA1.compute_at(sA2, xo)
    bounds = tvm.schedule.InferBound(sA2)
    assert isinstance(bounds, tvm.collections.Map)
    assert(bounds[A1.op.axis[0]].extent.value == 8)

def test_bound2():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
    sA1 = tvm.Schedule(A1.op)
    sA2 = tvm.Schedule(A2.op)
    xo, yo, xi, yi = sA2.tile(A2.op.axis[0], A2.op.axis[1], 8, 8)
    sA1.compute_at(sA2, yo)
    bounds = tvm.schedule.InferBound(sA2)
    assert isinstance(bounds, tvm.collections.Map)
    assert(bounds[A1.op.axis[0]].extent.value == 8)
    assert(bounds[A1.op.axis[1]].extent.value == 8)

def test_bound3():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
    sA1 = tvm.Schedule(A1.op, scope="shared")
    sA2 = tvm.Schedule(A2.op)
    thread_x = tvm.IterVar((0, 16), thread_tag="threadIdx.x")
    xo, xi = sA2.split(A2.op.axis[0], 32)
    xi0, xi1 = sA2.split(xi, outer=thread_x)
    yo, yi = sA2.split(A2.op.axis[1], 16)
    sA2.reorder(xo, xi0, yo, xi1, yi)
    sA1.compute_at(sA2, yo)

    bounds = tvm.schedule.InferBound(sA2)
    assert isinstance(bounds, tvm.collections.Map)
    assert(bounds[A1.op.axis[0]].extent.value==32)
    assert(bounds[A1.op.axis[1]].extent.value==16)


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
    test_bound3()
    test_bound1()
    test_bound2()
    test_create_read_graph()
