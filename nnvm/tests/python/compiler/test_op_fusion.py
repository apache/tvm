import nnvm
import numpy as np
import tvm
import topi.testing
from tvm.contrib import graph_runtime
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr
from nnvm.testing import ctx_list

def test_ewise_injective():
    x = sym.Variable("x")
    y = x * 2
    y = sym.flatten(y) + 1
    dshape = (10, 2, 3)
    shape_dict = {"x": dshape}
    dtype = "float32"
    target = "llvm"
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        assert graph.index.num_nodes == 2
        m = graph_runtime.create(graph, lib, ctx)
        x_np = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=x_np)
        out = m.get_output(0, tvm.nd.empty((10, 6)))
        np.testing.assert_allclose(
            out.asnumpy(),  x_np.reshape(out.shape) * 2 + 1,
            atol=1e-5, rtol=1e-5)


def test_conv_ewise_injective():
    x = sym.Variable("x")
    y = sym.conv2d(x, channels=32, kernel_size=(3, 3), groups=32,
                   name="y", padding=(1,1))
    y = sym.flatten(y + 1) + 1
    dtype = "float32"
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    oshape = (1, 32* 18 * 18)
    shape_dict = {"x": dshape}

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        # print(graph.ir(join_entry_attrs=["shape"]))
        assert graph.index.num_nodes == 5
        # set input
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
        bias = tvm.nd.array(np.random.uniform(size=kshape[0]).astype(dtype))
        m.run(x=data, y_weight=kernel, y_bias=bias)
        # get output
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        c_np = topi.testing.depthwise_conv2d_python_nchw(
            data.asnumpy(), kernel.asnumpy(), (1,1), 'SAME')
        c_np = c_np + bias.asnumpy().reshape(kshape[0], 1, 1) + 1
        c_np = c_np.reshape(c_np.shape[0], np.prod(c_np.shape[1:])) + 1
        np.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


def test_injective_reduce_injective():
    x = sym.Variable("x")
    x = sym.flatten(x) + 1
    y = sym.sum(x, axis=1)
    dtype = "float32"
    dshape = (32, 1, 18, 18)
    shape_dict = {"x": dshape}

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        assert graph.index.num_nodes == 2
        data = np.random.uniform(size=dshape).astype(dtype)
        m.run(x=data)
        c_np = np.sum(data.reshape(32, 18 * 18) + 1, axis=1)
        # get output
        out = m.get_output(0, tvm.nd.empty(c_np.shape, dtype))
        np.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


if __name__ == "__main__":
    test_injective_reduce_injective()
    test_ewise_injective()
    test_conv_ewise_injective()
