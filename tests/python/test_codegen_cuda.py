import tvm
import numpy

def mock_test_add():
    """Not yet working, mock design"""
    n = tvm.Var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.Schedule(C.op)

    # GPU schedule have to split by gridIdx and threadIdx
    num_thread = 256
    grid_x = tvm.IterVar(thread_tag="gridIdx.x")
    thread_x = tvm.IterVar((0, num_thread), thread_tag="threadIdx.x")
    _, x = s[C].split(C.op.axis[0], factor=num_thread, outer=grid_x)
    _, x = s[C].split(x, outer=thread_x)
    # compile to IR
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.ir_pass.ScheduleOps(s, bounds)


    Ab = tvm.Buffer(A.shape, A.dtype, name='A')
    Bb = tvm.Buffer(B.shape, B.dtype, name='B')
    Cb = tvm.Buffer(C.shape, C.dtype, name='C')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B:Bb, C:Cb})
    print(stmt)
    output_ssa = False
    code = tvm.codegen.CompileToC(stmt, "myadd",
                                  [Ab.ptr, Bb.ptr, Cb.ptr, n],
                                  output_ssa)

    print(code)
    def codegen():
        # generate host/device code
        host_code, device_code = tvm.codegen.GenCUDA(
            s,
            inputs={A: Ab, B:Bb},
            outputs={C: Cb},
            args=[A, B, C])
        # generate a function based on the code
        f = tvm.cuda.build_function(host_code, device_code)
        # create arrays
        a = tvm.nd.array(np.ones(10), ctx=tvm.gpu(0))
        b = tvm.nd.array(np.ones(10), ctx=tvm.gpu(0))
        c = tvm.nd.array(np.zeros(10), ctx=tvm.gpu(0))
        # calll the generated code
        f(a, b, c)
        # sync the result
        np.testing.assert_equal(c.asnumpy(), np.ones(10) * 2)


if __name__ == "__main__":
    mock_test_add()
