import numpy as np
import tvm
import topi
import nnvm.symbol as sym
import nnvm.compiler
import nnvm.runtime
from nnvm.testing.config import test_ctx_list

def verify_transpose(dshape, axes):
    x = sym.Variable("x")
    if axes:
        y = sym.transpose(x, axes=axes)
    else:
        y = sym.transpose(x)
    y = y + 1
    dtype = "float32"
    for target, ctx in test_ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = nnvm.runtime.create(graph, lib, ctx)
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.run(x=data)
        out_np = np.transpose(data.asnumpy(), axes=axes) + 1
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


def verify_reduce(dshape, fnp, fsym, **kwargs):
    x = sym.Variable("x")
    y = fsym(x + 1, **kwargs)
    dtype = "float32"
    for target, ctx in test_ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = nnvm.runtime.create(graph, lib, ctx)
        # set input
        data = np.random.uniform(size=dshape).astype(dtype)
        out_np = fnp(data + 1, **kwargs)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


def test_tranpose():
    verify_transpose((2, 3, 4), (0, 2, 1))
    verify_transpose((2, 3, 4), None)


def test_reduce():
    verify_reduce((2, 3, 4), np.max, sym.max, axis=1, keepdims=True)
    verify_reduce((4, 4, 3), np.min, sym.min, keepdims=True)
    verify_reduce((4, 4, 3), np.sum, sym.sum, axis=(0, 2))


if __name__ == "__main__":
    test_reduce()
    test_tranpose()
