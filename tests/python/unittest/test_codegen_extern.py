import tvm
import numpy as np

def test_add_pipeline():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')

    def extern_generator(ins, outs):
        """Manually write the IR for the extern function, add pipeline"""
        ib = tvm.ir_builder.create()
        dout = ib.buffer_ptr(outs[0])
        din = ib.buffer_ptr(ins[0])
        with ib.for_range(0, n) as i:
            dout[i] = din[i] + 1
        return ib.get()

    C = tvm.extern(A.shape, [A], extern_generator, name='C')
    s = tvm.create_schedule(C.op)

    def check_llvm():
        if not tvm.codegen.enabled("llvm"):
            return
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], "llvm")
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        f(a, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + 1)
    check_llvm()


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
        if not tvm.codegen.enabled(target):
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
        if not tvm.codegen.enabled(target):
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
