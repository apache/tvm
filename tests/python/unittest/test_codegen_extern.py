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


if __name__ == "__main__":
    test_add_pipeline()
