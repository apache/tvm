import tvm
import numpy as np

def enabled_ctx_list():
    ctx_list = [tvm.cpu(0), tvm.gpu(0), tvm.opencl(0)]
    ctx_list = [ctx for ctx in ctx_list if ctx.enabled]
    return ctx_list

ENABLED_CTX_LIST = enabled_ctx_list()
print("Testing using contexts:", ENABLED_CTX_LIST)


def test_nd_create():
    for ctx in ENABLED_CTX_LIST:
        for dtype in ["float32", "int8", "uint16"]:
            x = np.random.randint(0, 10, size=(3, 4))
            x = np.array(x, dtype=dtype)
            y = tvm.nd.array(x, ctx=ctx)
            z = y.copyto(ctx)
            assert y.dtype == x.dtype
            assert y.shape == x.shape
            assert isinstance(y, tvm.nd.NDArray)
            np.testing.assert_equal(x, y.asnumpy())
            np.testing.assert_equal(x, z.asnumpy())
        # no need here, just to test usablity
        tvm.nd.sync(ctx)

if __name__ == "__main__":
    test_nd_create()
