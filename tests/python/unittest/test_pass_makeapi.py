import tvm
import numpy

def test_makeapi():
    """Not yet working, mock design"""
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule(C.op)

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    Cb = tvm.decl_buffer(C.shape, C.dtype, name='C')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B:Bb, C:Cb}, 64)

    num_unpacked_args = 2
    f = tvm.ir_pass.MakeAPI(
        stmt, "myadd", [n, Ab, Bb, Cb], num_unpacked_args, True)
    assert(f.handle_data_type[Ab.data].dtype == Ab.dtype)
    assert(len(f.args) == 5)
    output_ssa = False


if __name__ == "__main__":
    test_makeapi()
