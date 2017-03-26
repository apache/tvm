import tvm

def lower(s, args):
    binds = {}
    arg_list = []

    for x in args:
        assert isinstance(x, tvm.tensor.Tensor)
        buf = tvm.Buffer(x.shape, dtype=x.dtype, name=x.op.name)
        binds[x] = buf
        arg_list.append(buf)
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.StorageFlatten(stmt, binds)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    return stmt

def test_basic_pipeline():
    n = tvm.convert(128)
    A = tvm.placeholder((n,), name='A')
    stages = []
    num_stage = 3

    B = A
    for k in range(num_stage):
        stages.append(B)
        B = tvm.compute((n,), lambda i: B[i] + k, name="A%s" % k)

    s = tvm.Schedule(B.op)
    px = tvm.thread_axis((0, 1), "pipeline")
    xo, xi = s[B].split(B.op.axis[0], outer=px)
    xo, xi = s[B].split(xi, factor=4)
    for S in stages:
        s[S].compute_at(s[B], xo)

    stmt = lower(s, [A, B])
    stmt = tvm.ir_pass.SplitPipeline(stmt, False)
    print(stmt)
    stmt = tvm.ir_pass.NarrowChannelAccess(stmt)
    print(stmt)
    assert(tvm.ir_pass.VerifySSA(stmt))

def test_conv1d():
    n = tvm.Var('n')
    A = tvm.compute((n+2), lambda i: 1,  name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = tvm.compute(n, computeB, name='B')
    s = tvm.Schedule(B.op)
    px = tvm.thread_axis((0, 1), "pipeline")
    xo, xi = s[B].split(B.op.axis[0], outer=px)
    s[A].compute_at(s[B], px)
    stmt = lower(s, [B])
    stmt = tvm.ir_pass.SplitPipeline(stmt, False)
    print(stmt)
    stmt = tvm.ir_pass.NarrowChannelAccess(stmt)
    print(stmt)


if __name__ == "__main__":
    test_basic_pipeline()
    test_conv1d()
