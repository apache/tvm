import tvm

def test_storage_share():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    num_stage = 5
    B = A
    for t in range(num_stage):
        B = tvm.compute((m, l), lambda i, j: B[i, j] + (t+1), name='A%d' % t)

    s = tvm.create_schedule(B.op)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb})
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    # verify only have two allocations.
    # verify that the data is folded.
    num_alloc = [0]
    def verify(n):
        if isinstance(n, tvm.stmt.Allocate):
            num_alloc[0] += 1
        elif isinstance(n, tvm.stmt.Store):
            assert n.buffer_var != n.value.a.buffer_var
    tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert num_alloc[0] == 2

def test_storage_share_gpu():
    m = tvm.var('m')
    A = [tvm.placeholder((m), name='A')]
    num_stage = 5
    for t in range(num_stage):
        A.append(tvm.compute((m,), lambda i: A[-1][i] + (t+1), name='A%d_s' % t))
        A.append(tvm.compute((m,), lambda i: A[-1][i], name='A%d' % t))
    s = tvm.create_schedule(A[-1].op)
    for t in range(num_stage):
        x = A[2*t+2].op.axis[0]
        bx, tx = s[A[2*t+2]].split(x, factor=32)
        s[A[2*t+2]].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[A[2*t+2]].bind(tx, tvm.thread_axis("threadIdx.x"))
        s[A[2*t+1]].compute_at(s[A[2*t+2]], tx)
        s[A[2*t+1]].set_scope("shared")

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A[0].shape, A[0].dtype, name='A')
    Bb = tvm.decl_buffer(A[0].shape, A[0].dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A[0]: Ab, A[-1]: Bb})
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.StorageRewrite(stmt)
    alloc_stats = {"global": 0, "shared": 0}

    def verify(n):
        if isinstance(n, tvm.stmt.AttrStmt):
            if n.attr_key == "storage_scope":
                alloc_stats[n.value.value] += 1
    tvm.ir_pass.PostOrderVisit(stmt, verify)
    assert alloc_stats["global"] == 2
    assert alloc_stats["shared"] == num_stage


if __name__ == "__main__":
    test_storage_share_gpu()
    test_storage_share()
