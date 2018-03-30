import tvm
import numpy as np

def enabled_ctx_list():
    ctx_list = [('cpu', tvm.cpu(0)),
                ('gpu', tvm.gpu(0)),
                ('cl', tvm.opencl(0)),
                ('metal', tvm.metal(0)),
                ('rocm', tvm.rocm(0)),
                ('vulkan', tvm.vulkan(0)),
                ('vpi', tvm.vpi(0))]
    for k, v  in ctx_list:
        assert tvm.context(k, 0) == v
    ctx_list = [x[1] for x in ctx_list if x[1].exist]
    return ctx_list

ENABLED_CTX_LIST = enabled_ctx_list()
print("Testing using contexts:", ENABLED_CTX_LIST)


def test_nd_create():
    for ctx in ENABLED_CTX_LIST:
        for dtype in ["uint8", "int8", "uint16", "int16", "uint32", "int32",
                      "float32"]:
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
        ctx.sync()


if __name__ == "__main__":
    test_nd_create()
