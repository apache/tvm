import numpy as np

import tvm
from tvm.contrib import graph_runtime
import topi
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list


def test_conv2d():
    x = sym.Variable("x")
    y = sym.conv2d(x, channels=10, kernel_size=(3,3),
                   name="y", padding=(1,1))
    dtype = "float32"
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    oshape = (1, 10, 18, 18)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
        bias = tvm.nd.array(np.random.uniform(size=kshape[0]).astype(dtype))
        m.run(x=data, y_weight=kernel, y_bias=bias)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        c_np = topi.testing.conv2d_nchw_python(
            data.asnumpy(), kernel.asnumpy(), 1, 1)
        c_np = c_np + bias.asnumpy().reshape(kshape[0], 1, 1)
        np.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


def test_grouped_conv2d():
    x = sym.Variable("x")
    y = sym.conv2d(x, channels=32, kernel_size=(3,3), groups=32,
                   name="y", padding=(1,1))
    dtype = "float32"
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    oshape = (1, 32, 18, 18)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
        bias = tvm.nd.array(np.random.uniform(size=kshape[0]).astype(dtype))
        m.run(x=data, y_weight=kernel, y_bias=bias)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        c_np = topi.testing.depthwise_conv2d_python_nchw(
            data.asnumpy(), kernel.asnumpy(), (1,1), 'SAME')
        c_np = c_np + bias.asnumpy().reshape(kshape[0], 1, 1)
        np.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


def test_conv2d_transpose():
    x = sym.Variable("x")
    y = sym.conv2d_transpose(x, channels=10, kernel_size=(3,3), strides=(2,2),
                             name="y", padding=(1,1), output_padding=(2,2))
    dtype = "float32"
    dshape = (1, 3, 18, 18)
    kshape = (3, 10, 3, 3)
    oshape = (1, 10, 37, 37)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
        bias = tvm.nd.array(np.random.uniform(size=kshape[1]).astype(dtype))
        m.run(x=data, y_weight=kernel, y_bias=bias)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        c_np = topi.testing.conv2d_transpose_nchw_python(
            data.asnumpy(), kernel.asnumpy(), 2, 1)
        c_np = c_np + bias.asnumpy().reshape(kshape[1], 1, 1)
        d_np = np.zeros(shape=oshape)
        d_np[:,:,0:c_np.shape[2],0:c_np.shape[3]] = c_np
        np.testing.assert_allclose(out.asnumpy(), d_np, rtol=1e-5)


def test_max_pool2d():
    x = sym.Variable("x")
    y = sym.max_pool2d(x, pool_size=(2,2), strides=(2,2),
                       padding=(0,0), name="y", ceil_mode=True)
    dtype = "float32"
    dshape = (1, 3, 28, 28)
    oshape = (1, 3, 14, 14)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        b_np = np.max(data.asnumpy().reshape(1,3,14,2,14,2), axis=(3,5))
        np.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


def test_avg_pool2d():
    x = sym.Variable("x")
    y = sym.avg_pool2d(x, pool_size=(2,2), strides=(2,2), padding=(0,0), name="y")
    dtype = "float32"
    dshape = (1, 3, 28, 28)
    oshape = (1, 3, 14, 14)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        b_np = np.mean(data.asnumpy().reshape(1,3,14,2,14,2), axis=(3,5))
        np.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


def test_global_max_pool2d():
    x = sym.Variable("x")
    y = sym.global_max_pool2d(x, name="y")
    dtype = "float32"
    dshape = (1, 1024, 7, 7)
    oshape = (1, 1024, 1, 1)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        b_np = np.max(data.asnumpy(), axis=(2,3), keepdims=True)
        np.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


def test_global_avg_pool2d():
    x = sym.Variable("x")
    y = sym.global_avg_pool2d(x, name="y")
    dtype = "float32"
    dshape = (1, 1024, 7, 7)
    oshape = (1, 1024, 1, 1)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        b_np = np.mean(data.asnumpy(), axis=(2,3), keepdims=True)
        np.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


def test_upsampling():
    x = sym.Variable("x")
    scale = 2
    y = sym.upsampling(x, scale=scale, name="y")
    dtype = "float32"
    dshape = (1, 16, 32, 32)
    oshape = (1, 16, 32*scale, 32*scale)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        a_np = np.random.uniform(size=dshape).astype(dtype)
        data = tvm.nd.array(a_np)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        b_np = topi.testing.upsampling_python(a_np, scale)
        np.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


if __name__ == "__main__":
    test_conv2d()
    test_grouped_conv2d()
    test_conv2d_transpose()
    test_max_pool2d()
    test_avg_pool2d()
    test_global_max_pool2d()
    test_global_avg_pool2d()
    test_upsampling()
