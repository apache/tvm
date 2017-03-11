import tvm

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
    xo, xi = s[B].split(B.op.axis[0], factor=4)
    for S in stages:
        s[S].compute_at(s[B], xo)

    # Lowering
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.Buffer(A.shape, A.dtype, name='A')
    Bb = tvm.Buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B:Bb})
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.SplitPipeline(stmt)
    print(stmt)
    assert(tvm.ir_pass.VerifySSA(stmt))

if __name__ == "__main__":
    test_basic_pipeline()
