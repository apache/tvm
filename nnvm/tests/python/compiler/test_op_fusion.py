import nnvm
import numpy as np
import tvm
import topi.testing
from tvm.contrib import graph_runtime
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr
from nnvm.testing import ctx_list, utils

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


def test_injective_conv2d():
    channels = 16
    data = sym.Variable(name="data")
    pool = sym.global_avg_pool2d(data=data)
    weight = sym.reshape(pool, shape=[1, channels, 1, 1])
    residual = sym.conv2d(data=data, kernel_size=(3,3), channels=channels, padding=(1, 1),
                          layout="NCHW", kernel_layout="OIHW", use_bias=False, name="conv")
    net = weight * data + residual
    size = 56
    dtype="float32"
    dshape = (1, channels, size, size)
    kshape = (channels, channels, 3, 3)
    oshape = dshape
    shape_dict = {"data": dshape}

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(net, target, shape_dict)
        # data, global_avg_pool, conv weight, conv op, fused elemwise add
        assert graph.index.num_nodes == 5

        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
        m = graph_runtime.create(graph, lib, ctx)
        m.run(data=data, conv_weight=kernel)
        # get output
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        residual = topi.testing.conv2d_nchw_python(
            data.asnumpy(), kernel.asnumpy(), (1,1), 'SAME')
        weight = np.mean(data.asnumpy(), axis=(2, 3))
        c_np = weight[:, :, np.newaxis, np.newaxis] * data.asnumpy() + residual
        np.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


def build_and_run(sym, params, data, out_shape, target, ctx, opt_level=2):
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(sym, target, shape={"data":data.shape}, params=params)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out =  module.get_output(0, tvm.nd.empty(out_shape))
    return out.asnumpy(), graph


def test_fuse_conv2d_elu():
    def elu(data):
        return -0.5 * sym.relu(1 - sym.exp(data)) + sym.relu(data)

    def get_sym(out_channel):
        data = sym.Variable(name="data")
        data = sym.conv2d(data=data, kernel_size=(3,3), channels=out_channel, padding=(1, 1),
                          layout="NCHW", kernel_layout="OIHW", use_bias=True)
        data = sym.batch_norm(data)
        data = elu(data)
        return data

    in_channel = 8
    out_channel = 16
    size = 64
    dshape = (1, in_channel, size, size)
    oshape = (1, out_channel, size, size)
    data = np.random.uniform(-1, 1, dshape).astype(np.float32)

    for target, ctx in ctx_list():
        sym1 = get_sym(out_channel)
        sym2 = get_sym(out_channel)
        _, params1 = utils.create_workload(sym1, 1, dshape[1:], seed=0)
        _, params2 = utils.create_workload(sym2, 1, dshape[1:], seed=0)
        output1, g1 = build_and_run(sym1, params1, data, oshape, target, ctx, opt_level=2)
        output2, g2 = build_and_run(sym2, params2, data, oshape, target, ctx, opt_level=0)
        np.testing.assert_allclose(output1, output2, rtol=1e-5, atol=1e-5)
        # data, conv weight, bias, batch norm gamma, batch norm beta, conv op
        assert g1.index.num_nodes == 6

if __name__ == "__main__":
    test_injective_reduce_injective()
    test_ewise_injective()
    test_conv_ewise_injective()
    test_fuse_conv2d_elu()
    test_injective_conv2d()
