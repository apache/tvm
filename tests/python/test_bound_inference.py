import tvm

def test_bound_inference():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j])
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3)
    sA1 = tvm.Schedule(A1.op)
    sA2 = tvm.Schedule(A2.op)
    xo, xi = sA1.split(A1.op.dim_var[0], factor=8)
    sA2.compute_at(sA1, xi)

    bounds = tvm.schedule.InferBound(sA1)
    assert isinstance(bounds, tvm.collections.Map)
    print(bounds)

if __name__ == "__main__":
    test_bound_inference()
