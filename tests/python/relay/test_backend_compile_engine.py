import tvm
import tvm.testing
import numpy as np
from tvm import relay


def test_compile_engine():
    engine = relay.backend.compile_engine.get()
    def get_func(shape):
        x = relay.var("x", shape=shape)
        y = relay.add(x, x)
        z = relay.add(y, x)
        f = relay.ir_pass.infer_type(relay.Function([x], z))
        return f
    z1 = engine.lower(get_func((10,)), "llvm")
    z2 = engine.lower(get_func((10,)), "llvm")
    z3 = engine.lower(get_func(()), "llvm")
    assert z1.same_as(z2)
    assert not z3.same_as(z1)
    if tvm.context("cuda").exist:
        z4 = engine.lower(get_func(()), "cuda")
        assert not z3.same_as(z4)

    # Test JIT target
    for target in ["llvm"]:
        ctx = tvm.context(target)
        if ctx.exist:
            f = engine.jit(get_func((10,)), target)
            x = tvm.nd.array(np.ones(10).astype("float32"), ctx=ctx)
            y = tvm.nd.empty((10,), ctx=ctx)
            f(x, y)
            tvm.testing.assert_allclose(
                y.asnumpy(), x.asnumpy() * 3)
    engine.dump()


if __name__ == "__main__":
    test_compile_engine()
