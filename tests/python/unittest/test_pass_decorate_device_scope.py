import tvm

def test_decorate_device():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')

    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], factor=8)
    s[A1].compute_at(s[A2], xo)
    s[A1].set_scope("shared")

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt1 = tvm.ir_pass.Simplify(stmt)
    stmt2 = tvm.ir_pass.DecorateDeviceScope(stmt1)
    assert isinstance(stmt2, tvm.stmt.AttrStmt)
    assert stmt2.attr_key == "device_scope"
    assert stmt1 == stmt2.body
    
if __name__ == "__main__":
    test_decorate_device()

