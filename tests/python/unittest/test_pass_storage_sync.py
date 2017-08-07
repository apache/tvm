import tvm

def test_storage_sync():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')

    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], factor=8)
    s[A2].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[A1].compute_at(s[A2], xo)
    s[A1].set_scope("shared")

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)
    f = tvm.ir_pass.MakeAPI(stmt, "test", [Ab, A2b], 0, True)
    flist = tvm.ir_pass.SplitHostDevice(f)
    f = flist[1]
    f = tvm.ir_pass.ThreadSync(f, "shared")
    body_list = tvm.make.stmt_list(f.body.body.body.body)
    assert(body_list[1].value.name == "tvm_storage_sync")


def test_coproc_sync():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    cp = tvm.thread_axis((0, 1), "cop")
    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        A[i] = A[i] + 1
        with ib.for_range(0, 10, name="j") as j:
            ib.scope_attr(cp, "coproc_scope", 1)
            A[j] = A[j] + 2
    body = ib.get()
    body = tvm.ir_pass.CoProcSync(body)
    body = body.body.body.body
    assert(tvm.make.stmt_list(body)[-1].value.name == "cop.coproc_sync")


if __name__ == "__main__":
    test_coproc_sync()
    test_storage_sync()
