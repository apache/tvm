import tvm
import numpy as np

def test_add_pipeline():
    nn = 64
    max_threads = 4
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')

    def extern_generator(ins, outs):
        """Manually write the IR for the extern function, add pipeline"""
        ib = tvm.ir_builder.create()
        with ib.for_range(0, (n+1) // 2) as i:
            ib.emit(outs[0].vstore(i*2, ins[0].vload(i*2, "float32x2") + tvm.const(1, "float32x2")))
        return ib.get()

    def extern_generator_gpu(ins, outs):
        """Manually write the IR for the extern function, add pipeline"""
        ib = tvm.ir_builder.create()
        bx = tvm.thread_axis("blockIdx.x")
        tx = tvm.thread_axis("threadIdx.x")
        ib.scope_attr(bx, "thread_extent", (nn+max_threads-1) // max_threads)
        ib.scope_attr(tx, "thread_extent", max_threads)
        idx = bx.var * max_threads + tx.var
        with ib.if_scope(ib.likely(idx < n)):
            ib.emit(outs[0].vstore(idx*2, ins[0].vload(idx*2, "float32x2") + tvm.const(1, "float32x2")))
        return ib.get()

    C_cpu = tvm.extern(A.shape, [A], extern_generator, name='C')
    C_gpu = tvm.extern(A.shape, [A], extern_generator_gpu, name='C')
    s_cpu = tvm.create_schedule(C_cpu.op)
    s_gpu = tvm.create_schedule(C_gpu.op)
    print(tvm.lower(s_cpu, [A, C_cpu], simple_mode=True))
    print(tvm.lower(s_gpu, [A, C_gpu], simple_mode=True))

    def check_target(target):
        if not tvm.module.enabled(target):
            return
        s = s_gpu if target in ['opencl', 'cuda'] else s_cpu
        C = C_gpu if target in ['opencl', 'cuda'] else C_cpu
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)
        ctx = tvm.context(target, 0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)
        
    check_target("llvm")
    check_target("opencl")
    check_target("cuda")

def test_pack_buffer_simple():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    def extern_generator(ins, outs):
        """Manually write the IR for the extern function, add pipeline."""
        return tvm.call_packed("my_extern_array_func1", ins[0], outs[0])

    C = tvm.extern(A.shape, [A], extern_generator, name='C')
    s = tvm.create_schedule(C.op)

    @tvm.register_func
    def my_extern_array_func1(aa, bb):
        aa.copyto(bb)


    def check_target(target):
        if not tvm.module.enabled(target):
            return
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)

        f(a, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy())
    check_target("stackvm")
    check_target("llvm")


def test_pack_buffer_intermediate():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute((n,), lambda i: A[i] + 1, name="B")
    def extern_generator(ins, outs):
        """Manually write the IR for the extern function, add pipeline."""
        return tvm.call_packed("my_extern_array_func2", ins[0], outs[0])

    C = tvm.extern(B.shape, [B], extern_generator, name='C')
    s = tvm.create_schedule(C.op)

    def check_target(target):
        if not tvm.module.enabled(target):
            return
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)

        @tvm.register_func
        def my_extern_array_func2(aa, bb):
            assert aa.shape == a.shape
            np.testing.assert_allclose(
                aa.asnumpy(), a.asnumpy() + 1)
            aa.copyto(bb)

        f(a, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + 1)

    check_target("llvm")


if __name__ == "__main__":
    test_pack_buffer_simple()
    test_pack_buffer_intermediate()
    test_add_pipeline()
