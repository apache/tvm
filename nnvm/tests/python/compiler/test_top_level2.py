# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np

import tvm
from tvm.contrib import graph_runtime
import topi
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list


def test_conv2d():
    def run_test_conv2d(sym, dtype, dshape, kshape, oshape, shape_dict, padding):
        for target, ctx in ctx_list():
            graph, lib, _ = nnvm.compiler.build(sym, target, shape_dict)
            m = graph_runtime.create(graph, lib, ctx)
            data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
            kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
            bias = tvm.nd.array(np.random.uniform(size=kshape[0]).astype(dtype))
            m.run(x=data, y_weight=kernel, y_bias=bias)
            out = m.get_output(0, tvm.nd.empty(oshape, dtype))
            c_np = topi.testing.conv2d_nchw_python(
                data.asnumpy(), kernel.asnumpy(), 1, padding)
            c_np = c_np + bias.asnumpy().reshape(kshape[0], 1, 1)
            tvm.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)

    x = sym.Variable("x")
    y = sym.conv2d(x, channels=10, kernel_size=(3,3),
                   name="y", padding=(1,1))
    dtype = "float32"
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    oshape = (1, 10, 18, 18)
    shape_dict = {"x": dshape}
    run_test_conv2d(y, dtype, dshape, kshape, oshape, shape_dict, (1,1))

    x = sym.Variable("x")
    y = sym.conv2d(x, channels=10, kernel_size=(1,3),
                   name="y", padding=(0,1))
    dtype = "float32"
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 1, 3)
    oshape = (1, 10, 224, 224)
    shape_dict = {"x": dshape}
    run_test_conv2d(y, dtype, dshape, kshape, oshape, shape_dict, (0,1))


def test_mixed_precision():
    x = sym.Variable("x")
    dtype = "int8"
    out_dtype="int32"
    y = sym.conv2d(x,
                   channels=10,
                   kernel_size=(3,3),
                   name="y",
                   padding=(1,1),
                   use_bias=False,
                   out_dtype="int32")
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    oshape = (1, 10, 18, 18)
    shape_dict = {"x": dshape}
    dtype_dict = {"x": dtype}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict, dtype_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(-127, 127, size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(-127, 127, size=kshape).astype(dtype))
        m.run(x=data, y_weight=kernel)
        out = m.get_output(0, tvm.nd.empty(oshape, out_dtype))
        c_np = topi.testing.conv2d_nchw_python(
            data.asnumpy().astype(out_dtype),
            kernel.asnumpy().astype(out_dtype), 1, 1)
        tvm.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


def test_dilated_conv2d():
    dilation = 3
    x = sym.Variable("x")
    y = sym.conv2d(x, channels=10, kernel_size=(3, 3), dilation=(dilation, dilation),
                   name="y", padding=(1, 1))
    dtype = "float32"
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    oshape = (1, 10, 14, 14)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        bias = tvm.nd.array(np.random.uniform(size=kshape[0]).astype(dtype))
        kernel_np = np.random.uniform(size=kshape).astype(dtype)
        kernel = tvm.nd.array(kernel_np)
        dkernel_np = topi.testing.dilate_python(kernel_np, (1, 1, dilation, dilation))
        m.run(x=data, y_weight=kernel, y_bias=bias)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        c_np = topi.testing.conv2d_nchw_python(
            data.asnumpy(), dkernel_np, 1, 1)
        c_np = c_np + bias.asnumpy().reshape(kshape[0], 1, 1)
        tvm.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


def test_grouped_conv2d_nchw():
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
        tvm.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)

def test_grouped_conv2d_nhwc():
    x = sym.Variable("x")
    y = sym.conv2d(x, channels=32, kernel_size=(3,3), groups=32,
                   name="y", padding=(1,1), layout="NHWC", kernel_layout ='HWOI')
    dtype = "float32"
    dshape = (1, 18, 18, 32)
    kshape = (3, 3, 32, 1)
    oshape = (1, 18, 18, 32)
    shape_dict = {"x": dshape}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        kernel = tvm.nd.array(np.random.uniform(size=kshape).astype(dtype))
        bias = tvm.nd.array(np.random.uniform(size=kshape[2]).astype(dtype))
        m.run(x=data, y_weight=kernel, y_bias=bias)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        c_np = topi.testing.depthwise_conv2d_python_nhwc(
            data.asnumpy(), kernel.asnumpy(), (1,1), 'SAME')
        c_np = c_np + bias.asnumpy().reshape(1, 1, kshape[2])
        tvm.testing.assert_allclose(out.asnumpy(), c_np, rtol=1e-5)


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
        tvm.testing.assert_allclose(out.asnumpy(), d_np, rtol=1e-5)


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
        tvm.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


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
        tvm.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


def test_avg_pool2d_no_count_pad():
    kh, kw = (4, 4)
    sh, sw = (2, 2)
    ph, pw = (2, 2)

    x = sym.Variable("x")
    y = sym.avg_pool2d(x, pool_size=(kh, kw), strides=(sw, sw), padding=(ph, pw),
                       name="y", count_include_pad=False)
    dtype = "float32"
    n = 1
    (ic, ih, iw) = (3, 28, 28)
    (oc, oh, ow) = (3, 15, 15)

    a_np = np.random.uniform(low=0.001, size=(n, ic, ih, iw)).astype(dtype)
    pad_np = np.zeros(shape=(n, ic, ih+2*ph, iw+2*pw)).astype(dtype)
    no_zero = (range(n), range(ic), (range(ph, ih+ph)), (range(pw, iw+pw)))
    pad_np[np.ix_(*no_zero)] = a_np
    b_np = np.zeros(shape=(n, oc, oh, ow)).astype(dtype)

    for i in range(oh):
        for j in range(ow):
            pad_count = np.sum(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] > 0, axis=(2,3))
            b_np[:,:,i,j] = np.sum(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw],
                                   axis=(2,3)) / np.maximum(pad_count, 1)
    b_np = np.maximum(b_np, 0.0)
    shape_dict = {"x": (n, ic, ih, iw)}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict)
        m = graph_runtime.create(graph, lib, ctx)
        data = tvm.nd.array(a_np)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty((n, oc, oh, ow), dtype))
        tvm.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


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
        tvm.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


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
        tvm.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)


def test_upsampling_nearest_neighbor():
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
        b_np = topi.testing.upsampling_python(a_np, (scale, scale), "NCHW")
        tvm.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5)

def test_upsampling_bilinear():
    x = sym.Variable("x")
    scale = 2
    y = sym.upsampling(x, scale=scale, method="BILINEAR", name="y", layout="NCHW")
    dtype = "float32"
    dshape = (1, 4, 32, 32)
    oshape = (1, 4, 32*scale, 32*scale)
    shape_dict = {"x": dshape}
    dtype_dict = {"x": dtype}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict, dtype_dict)
        m = graph_runtime.create(graph, lib, ctx)
        a_np = np.random.uniform(size=dshape).astype(dtype)
        data = tvm.nd.array(a_np)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        b_np = topi.testing.bilinear_resize_python(a_np, (32*scale, 32*scale), "NCHW")
        tvm.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5, atol=1e-5)

def test_resize_bilinear():
    x = sym.Variable("x")
    y = sym.resize(x, size=(60, 60), method="BILINEAR", name="y", layout="NHWC")
    dtype = "float32"
    dshape = (1, 32, 32, 4)
    oshape = (1, 60, 60, 4)
    shape_dict = {"x": dshape}
    dtype_dict = {"x": dtype}
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, shape_dict, dtype_dict)
        m = graph_runtime.create(graph, lib, ctx)
        a_np = np.random.uniform(size=dshape).astype(dtype)
        data = tvm.nd.array(a_np)
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape, dtype))
        b_np = topi.testing.bilinear_resize_python(a_np, (60, 60), "NHWC")
        tvm.testing.assert_allclose(out.asnumpy(), b_np, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_mixed_precision()
    test_conv2d()
    test_dilated_conv2d()
    test_grouped_conv2d_nchw()
    test_grouped_conv2d_nhwc()
    test_conv2d_transpose()
    test_max_pool2d()
    test_avg_pool2d()
    test_avg_pool2d_no_count_pad()
    test_global_max_pool2d()
    test_global_avg_pool2d()
    test_upsampling_nearest_neighbor()
    test_upsampling_bilinear()
    test_resize_bilinear()
