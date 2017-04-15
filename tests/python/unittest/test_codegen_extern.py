import tvm
import numpy as np

def test_add_pipeline():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    def extern_generator(ins, outs):
        """Manually write the IR for the extern function, add pipeline"""
        i = tvm.var('i')
        stmt = tvm.make.For(
            i, 0, n, 0, 0,
            tvm.make.Store(outs[0].data,
                           tvm.make.Load(A.dtype, ins[0].data, i) +
                           1, i))
        return stmt
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
