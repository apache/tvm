import tvm
import numpy as np

def test_add_pipeline():
    """Not yet working, mock design"""
    n = tvm.Var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.Schedule(C.op)

    # GPU schedule have to split by gridIdx and threadIdx
    num_thread = 256
    grid_x = tvm.IterVar(thread_tag="blockIdx.x")
    thread_x = tvm.IterVar((0, num_thread), thread_tag="threadIdx.x")
    _, x = s[C].split(C.op.axis[0], factor=num_thread, outer=grid_x)
    _, x = s[C].split(x, outer=thread_x)

    # compile to IR
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.Buffer(A.shape, A.dtype, name='A')
    Bb = tvm.Buffer(B.shape, B.dtype, name='B')
    Cb = tvm.Buffer(C.shape, C.dtype, name='C')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B:Bb, C:Cb})
    stmt = tvm.ir_pass.Simplify(stmt)
    fapi = tvm.ir_pass.MakeAPI(stmt, "myadd", [Ab, Bb, Cb], 3)
    fsplits = tvm.ir_pass.SplitHostDevice(fapi)

    def check_cuda():
        output_ssa = False
        for f in fsplits[1:]:
            print(tvm.codegen.CompileToC(f, output_ssa, "cuda"))

        # build and invoke the kernel.
        fcuda = tvm.codegen.BuildNVRTC(fsplits, "stackvm")
        num_device = 1
        for i in range(num_device):
            ctx = tvm.gpu(i)
            if not ctx.enabled:
                continue
            # launch the kernel.
            n = 1027
            a = tvm.nd.array(np.random.uniform(size=n).astype(Ab.dtype), ctx)
            b = tvm.nd.array(np.random.uniform(size=n).astype(Bb.dtype), ctx)
            c = tvm.nd.array(np.zeros(n, dtype=Cb.dtype), ctx)
            fcuda(a, b, c)
            np.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy())

    def check_opencl():
        output_ssa = False
        for f in fsplits[1:]:
            print(tvm.codegen.CompileToC(f, output_ssa, "opencl"))

        # build and invoke the kernel.
        fcl = tvm.codegen.BuildOpenCL(fsplits, "stackvm")
        # Disable OpenCL runtime test for now,
        # since the local worksize on CPU might be too large.
        num_device = 0
        for i in range(num_device):
            ctx = tvm.cl(i)
            if not ctx.enabled:
                continue
            # launch the kernel.
            n = 1027
            a = tvm.nd.array(np.random.uniform(size=n).astype(Ab.dtype), ctx)
            b = tvm.nd.array(np.random.uniform(size=n).astype(Bb.dtype), ctx)
            c = tvm.nd.array(np.zeros(n, dtype=Cb.dtype), ctx)
            fcl(a, b, c)
            np.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy())

    tvm.init_opencl()
    if tvm.cl(0).enabled:
        check_opencl()

    if tvm.gpu(0).enabled:
        check_cuda()

if __name__ == "__main__":
    test_add_pipeline()
