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
import attr
import numpy as np
import math
import torch
import torchvision
import topi
import topi.testing
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import onnx
from onnx import helper, TensorProto
import unittest
import scipy

def get_tvm_output(graph_def, input_data, target, ctx, output_shape=None, output_dtype='float32'):
    """ Generic function to execute and get tvm output"""
    target = 'llvm'
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        dtype_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_data[i].shape
            dtype_dict[input_names[i]] = input_data[i].dtype
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}
        dtype_dict = {input_names: input_data.dtype}

    mod, params = relay.frontend.from_onnx(graph_def, shape_dict)
    with relay.build_config(opt_level=1):
        graph, lib, params = relay.build(mod,
                                         target,
                                         params=params)

    ctx = tvm.cpu(0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_names):
            m.set_input(input_names[i], tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.asnumpy()

def get_caffe2_output(model, x, dtype='float32'):
    import caffe2.python.onnx.backend
    prepared_backend = caffe2.python.onnx.backend.prepare(model)
    W = {model.graph.input[0].name: x.astype(dtype)}
    c2_out = prepared_backend.run(W)[0]
    return c2_out


def verify_onnx_forward_impl(graph_file, data_shape, out_shape):
    dtype = 'float32'
    x = np.random.uniform(size=data_shape)
    model = onnx.load_model(graph_file)
    c2_out = get_caffe2_output(model, x, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, dtype)
        tvm.testing.assert_allclose(c2_out, tvm_out, rtol=1e-5, atol=1e-5)

def verify_super_resolution_example():
    verify_onnx_forward_impl(super_resolution, (1, 1, 224, 224), (1, 1, 672, 672))

def verify_squeezenet1_1():
    verify_onnx_forward_impl(squeezenet1_1, (1, 3, 224, 224), (1, 1000))

def verify_lenet():
    verify_onnx_forward_impl(lenet, (1, 1, 28, 28), (1, 10))

def verify_resnet18():
    verify_onnx_forward_impl(resnet18_1_0, (1, 3, 224, 224), (1, 1000))


def test_reshape():
    in_shape = (4, 3, 3, 4)
    ref_shape = (6, 2, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['ref_in'],
                                 value=onnx.helper.make_tensor(name = 'const_tensor',
                                                               data_type = onnx.TensorProto.INT32,
                                                               dims = ref_array.shape,
                                                               vals = ref_array.flatten().astype(int)))
    reshape_node = helper.make_node("Reshape", ["in", "ref_in"], ["out"])

    graph = helper.make_graph([ref_node, reshape_node],
                              "reshape_test",
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='reshape_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('int32')
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, 'float32')

    tvm.testing.assert_allclose(ref_shape, tvm_out.shape)

def test_shape():
    in_shape = (4, 3, 3, 4)
    ref_shape = (6, 2, 4, 3)

    ref_array = np.array(ref_shape)
    ref_node = onnx.helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['ref_in'],
                                 value=onnx.helper.make_tensor(name = 'const_tensor',
                                                               data_type = onnx.TensorProto.INT32,
                                                               dims = ref_array.shape,
                                                               vals = ref_array.flatten().astype(int)))
    reshape_node = helper.make_node("Reshape", ["in", "ref_in"], ["out"])

    shape_node = helper.make_node("Shape", ['out'], ['final_out'])

    graph = helper.make_graph([ref_node, reshape_node, shape_node],
                              "shape_test",
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("final_out",
                                            TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='shape_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('int32')
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, 'int32')

    tvm.testing.assert_allclose(ref_shape, tvm_out)

def _test_power_iteration(x_shape, y_shape):
    if isinstance(y_shape, int):
        y_shape = [y_shape]

    x = np.random.uniform(size=x_shape).astype(np.float32)
    y = np.random.uniform(size=y_shape).astype(np.float32)

    np_res = np.power(x, y).astype(np.float32)

    res = helper.make_node("Pow", ['x', 'y'], ['out'])

    graph = helper.make_graph([res],
                              'power_test',
                              inputs = [helper.make_tensor_value_info("x",
                                            TensorProto.FLOAT, list(x_shape)),
                                        helper.make_tensor_value_info("y",
                                            TensorProto.FLOAT, list(y_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(np_res.shape))])

    model = helper.make_model(graph, producer_name='power_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x, y], target, ctx, np_res.shape)
        tvm.testing.assert_allclose(np_res, tvm_out, rtol=1e-5, atol=1e-5)

def test_power():
    _test_power_iteration((1, 3), (1))
    _test_power_iteration((2, 3), (2, 3))
    _test_power_iteration((2, 3), (1, 3))

def test_squeeze():
    in_shape = (1, 3, 1, 3, 1, 1)
    out_shape = (3, 3)
    y = helper.make_node("Squeeze", ['in'], ['out'], axes=[0, 2, 4, 5])

    graph = helper.make_graph([y],
                              'squeeze_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='squeeze_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('float32')
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, 'float32')

    tvm.testing.assert_allclose(out_shape, tvm_out.shape)

def test_flatten():

    in_shape = (1, 3, 4, 4)
    axis = 1
    ref_shape = (1, 48)

    flatten_node = helper.make_node("Flatten", ["in"], ["out"], axis = axis)

    graph = helper.make_graph([flatten_node],
                              "flatten_test",
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(ref_shape))])

    model = helper.make_model(graph, producer_name='flatten_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('int32')
        tvm_out = get_tvm_output(model, x, target, ctx, ref_shape, 'float32')

    tvm.testing.assert_allclose(ref_shape, tvm_out.shape)

def test_unsqueeze():
    in_shape = (3, 3)
    axis = (0, 3, 4)
    out_shape = (1, 3, 3, 1, 1)
    y = helper.make_node("Unsqueeze", ['in'], ['out'], axes=list(axis))

    graph = helper.make_graph([y],
                              'squeeze_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='squeeze_test')

    for target, ctx in ctx_list():
        x = np.random.uniform(size=in_shape).astype('float32')
        tvm_out = get_tvm_output(model, x, target, ctx, out_shape, 'float32')

    tvm.testing.assert_allclose(out_shape, tvm_out.shape)

def verify_gather(in_shape, indices, axis, dtype):
    x = np.random.uniform(size=in_shape).astype(dtype)
    indices = np.array(indices, dtype="int32")
    out_np = np.take(x, indices, axis=axis)

    y = helper.make_node("Gather", ['in', 'indices'], ['out'], axis=axis)

    graph = helper.make_graph([y],
                              'gather_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(in_shape)),
                                        helper.make_tensor_value_info("indices",
                                            TensorProto.INT32, list(indices.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_np.shape))])
    model = helper.make_model(graph, producer_name='gather_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x, indices], target, ctx, out_np.shape)
        tvm.testing.assert_allclose(out_np, tvm_out)

def test_gather():
    verify_gather((4,), [1], 0, 'int32')
    verify_gather((1,4), [0], 0, 'int32')
    verify_gather((4,), [[[1,0],[0,1]]], 0, 'float32')
    verify_gather((2,2), [[[1,0],[0,1]]], 1, 'int32')
    verify_gather((3,3,3), [[[1,0]]], -1, 'int32')
    verify_gather((4,3,5,6), [[2,1,0,0]], 0, 'float32')

def _test_slice_iteration(indata, outdata, starts, ends, axes=None):
    if axes:
        y = helper.make_node("Slice", ['in'], ['out'], axes=axes, starts=starts, ends=ends)
    else:
        y = helper.make_node("Slice", ['in'], ['out'], starts=starts, ends=ends)

    graph = helper.make_graph([y],
                              'slice_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name='slice_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, 'float32')

    tvm.testing.assert_allclose(outdata, tvm_out)

def test_slice():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    _test_slice_iteration(x, x[0:3, 0:10], (0, 0), (3, 10), (0, 1))
    _test_slice_iteration(x, x[:, :, 3:4], (0, 0, 3), (20, 10, 4))
    _test_slice_iteration(x, x[:, 1:1000], (1), (1000), (1))
    _test_slice_iteration(x, x[:, 0:-1], (0), (-1), (1))

def _test_onnx_op_elementwise(inshape, outfunc, npargs, dtype, opname, kwargs):
    indata = np.random.uniform(-1, 1, size=inshape).astype(dtype)
    outdata = outfunc(indata, **npargs)

    y = helper.make_node(opname, ['in'], ['out'], **kwargs)

    graph = helper.make_graph([y],
                              opname+'_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name=opname+'_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, dtype)

    tvm.testing.assert_allclose(outdata, tvm_out)

def test_floor():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.floor, {}, 'float32', 'Floor', {})

def test_ceil():
    _test_onnx_op_elementwise((2, 4, 5, 6), np.ceil, {}, 'float32', 'Ceil', {})

def test_clip():
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              np.clip,
                              {'a_min': -1.0, 'a_max': 1.0},
                              'float32',
                              'Clip',
                              {'min': -1.0, 'max': 1.0})

def test_matmul():
    a_shape = (4, 3)
    b_shape = (3, 4)

    a_array = np.random.uniform(size=a_shape).astype('float32')
    b_array = np.random.uniform(size=b_shape).astype('float32')
    out_np = np.matmul(a_array, b_array)

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

    graph = helper.make_graph([mul_node],
                              "matmul_test",
                              inputs = [helper.make_tensor_value_info("a",
                                            TensorProto.FLOAT, list(a_shape)),
                                        helper.make_tensor_value_info("b",
                                            TensorProto.FLOAT, list(b_shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out_np.shape))])

    model = helper.make_model(graph, producer_name='matmul_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_array, b_array], target, ctx, out_np.shape)
        tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)

def verify_lrn(shape, nsize, dtype, alpha=None, beta=None, bias=None):
    in_array = np.random.uniform(size=shape).astype(dtype)

    if alpha == None and beta == None and bias==None:
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        node = onnx.helper.make_node('LRN', inputs=['in'], outputs=['out'], size=nsize)
    else:
        node = onnx.helper.make_node('LRN', inputs=['in'], outputs=['out'], alpha=alpha,
                                     beta=beta, bias=bias, size=nsize)

    graph = helper.make_graph([node],
                              "lrn_test",
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))])
    model = helper.make_model(graph, producer_name='lrn_test')

    def _get_python_lrn():
        square_sum = np.zeros(shape).astype(dtype)
        for n, c, h, w in np.ndindex(in_array.shape):
            square_sum[n, c, h, w] = sum(in_array[n,
                                         max(0, c - int(math.floor((nsize - 1) / 2))): \
                                             min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                         h,
                                         w] ** 2)
        py_out = in_array / ((bias + (alpha / nsize) * square_sum) ** beta)
        return py_out

    for target, ctx in ctx_list():
        input_name = model.graph.input[0].name
        py_out = _get_python_lrn()
        tvm_out = get_tvm_output(model, in_array, target, ctx, py_out.shape, 'float32')
        tvm.testing.assert_allclose(py_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_lrn():
    verify_lrn((5, 5, 5, 5), 3, 'float32')
    verify_lrn((5, 5, 5, 5), 3, 'float32', alpha=0.0002, beta=0.5, bias=2.0)


def verify_instance_norm(shape, axis=1):

    def _get_python_instance_norm(x, gamma, beta, epsilon=1e-5):
        dims_x = len(x.shape)
        axis = tuple(range(2, dims_x))
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        dim_ones = (1,) * (dims_x - 2)
        gamma = gamma.reshape(-1, *dim_ones)
        beta = beta.reshape(-1, *dim_ones)
        return gamma * (x - mean) / np.sqrt(var + epsilon) + beta

    x = np.random.randn(*shape).astype(np.float32)
    gamma = np.random.randn(shape[1]).astype(np.float32)
    beta = np.random.randn(shape[1]).astype(np.float32)
    epsilon = 1e-5
    y = _get_python_instance_norm(x, gamma, beta, epsilon).astype(np.float32)

    node = onnx.helper.make_node(
            'InstanceNormalization',
            inputs=['x', 'gamma', 'beta'],
            outputs=['y'],
            epsilon=epsilon,
        )
    graph = helper.make_graph([node],
                              "instance_norm_test",
                              inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape)),
                                      helper.make_tensor_value_info("gamma", TensorProto.FLOAT, (shape[1],)),
                                      helper.make_tensor_value_info("beta", TensorProto.FLOAT, (shape[1],))],
                              outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))])
    model = helper.make_model(graph, producer_name='instance_norm_test')
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x, gamma, beta], target, ctx, shape, 'float32')
        tvm.testing.assert_allclose(y, tvm_out, rtol=1e-5, atol=1e-5)


def test_instance_norm():
    verify_instance_norm((2, 3, 4, 5))
    verify_instance_norm((32, 64, 80, 64))
    verify_instance_norm((8, 6, 5))
    verify_instance_norm((8, 7, 6, 5, 4))


def _test_upsample_nearest():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3*scale, 3*scale)
    y = helper.make_node("Upsample", ['in'], ['out'], mode='nearest', scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = topi.testing.upsampling_python(in_array, (scale, scale), "NCHW")

    graph = helper.make_graph([y],
                              'upsample_nearest_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='upsample_nearest_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, 'float32')
        tvm.testing.assert_allclose(out_array, tvm_out)

def _test_upsample_bilinear():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3*scale, 3*scale)
    y = helper.make_node("Upsample", ['in'], ['out'], mode='linear', scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = topi.testing.bilinear_resize_python(in_array, (3*scale, 3*scale), "NCHW")

    graph = helper.make_graph([y],
                              'upsample_bilinear_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='upsample_bilinear_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, 'float32')
        tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)

def _test_upsample_bilinear_opset9():
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3*scale, 3*scale)
    y = helper.make_node("Upsample", ['in','scales'], ['out'], mode='linear')
    scales=[1.0, 1.0, 2.0, 2.0]
    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = topi.testing.bilinear_resize_python(in_array, (3*scale, 3*scale), "NCHW")

    ref_array = np.array(scales)
    ref_node = helper.make_node('Constant',
                                 inputs=[],
                                 outputs=['scales'],
                                 value=onnx.helper.make_tensor(name = 'const_tensor',
                                                               data_type = TensorProto.FLOAT,
                                                               dims = ref_array.shape,
                                                               vals = ref_array.flatten().astype(float)))

    graph = helper.make_graph([ref_node, y],
                              'upsample_bilinear_opset9_test',
                              inputs = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
                              outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))])

    model = helper.make_model(graph, producer_name='upsample_bilinear_opset9_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, in_array, target, ctx, out_shape, 'float32')
        tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)

def test_upsample():
    _test_upsample_nearest()
    _test_upsample_bilinear()
    _test_upsample_bilinear_opset9()

def _test_softmax(inshape, axis):
    opname = 'Softmax'
    indata = np.random.uniform(size=inshape).astype(np.float32)
    outshape = inshape
    outdata = topi.testing.softmax_python(indata)
    if isinstance(axis, int):
        y = helper.make_node(opname, ['in'], ['out'], axis = axis)
    elif axis is None:
        y = helper.make_node(opname, ['in'], ['out'])

    graph = helper.make_graph([y],
                              opname+'_test',
                              inputs = [helper.make_tensor_value_info("in",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name=opname+'_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outshape, 'float32')
        tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)

def test_softmax():
    _test_softmax((1, 10), None)
    _test_softmax((1, 10), 1)

def verify_min(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.min((a_np1, a_np2, a_np3), axis=0)

    min_node = helper.make_node("Min", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([min_node],
                              "Min_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Min_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_min():
    verify_min((1, 3, 20, 20))
    verify_min((20, 20))

def verify_max(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.max((a_np1, a_np2, a_np3), axis=0)

    max_node = helper.make_node("Max", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([max_node],
                              "Max_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Max_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_max():
    verify_max((1, 3, 20, 20))
    verify_max((20, 20))

def verify_mean(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.mean((a_np1, a_np2, a_np3), axis=0)

    mean_node = helper.make_node("Mean", ["a_np1", "a_np2", "a_np3"], ["out"])

    graph = helper.make_graph([mean_node],
                              "Mean_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np2",
                                            TensorProto.FLOAT, list(input_dim)),
                                        helper.make_tensor_value_info("a_np3",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='Mean_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1, a_np2, a_np3], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_mean():
    verify_mean((1, 3, 20, 20))
    verify_mean((20, 20))

def verify_hardsigmoid(input_dim, alpha, beta):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.clip(a_np1 * alpha + beta, 0, 1)

    hardsigmoid_node = helper.make_node("HardSigmoid", ["a_np1"], ["out"], alpha=alpha, beta=beta)

    graph = helper.make_graph([hardsigmoid_node],
                              "HardSigmoid_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.FLOAT, list(input_dim))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='HardSigmoid_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_hardsigmoid():
    verify_hardsigmoid((1, 3, 20, 20), 0.5, 0.6)
    verify_hardsigmoid((20, 20), 0.3, 0.4)

def verify_argmin(input_dim, axis=None, keepdims=None):
    def _argmin_numpy(data, axis=0, keepdims=True):
        result = np.argmin(data, axis=axis)
        if (keepdims == 1):
            result = np.expand_dims(result, axis)
        return result.astype(data.dtype)

    a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)
    if keepdims is None and axis is None:
        b_np = _argmin_numpy(a_np1)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'])
    elif axis is None:
        b_np = _argmin_numpy(a_np1, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     keepdims=keepdims)
    elif keepdims is None:
        b_np = _argmin_numpy(a_np1, axis=axis)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis)
    else:
        b_np = _argmin_numpy(a_np1, axis=axis, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMin',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis,
                                     keepdims=keepdims)
    graph = helper.make_graph([node],
                              "argmin_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.INT32, list(a_np1.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.INT32, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='argmin_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape, b_np.dtype)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def verify_argmax(input_dim, axis=None, keepdims=None):
    def _argmax_numpy(data, axis=0, keepdims=True):
        result = np.argmax(data, axis=axis)
        if (keepdims == 1):
            result = np.expand_dims(result, axis)
        return result.astype(data.dtype)

    a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)
    if keepdims is None and axis is None:
        b_np = _argmax_numpy(a_np1)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'])
    elif axis is None:
        b_np = _argmax_numpy(a_np1, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     keepdims=keepdims)
    elif keepdims is None:
        b_np = _argmax_numpy(a_np1, axis=axis)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis)
    else:
        b_np = _argmax_numpy(a_np1, axis=axis, keepdims=keepdims)
        node = onnx.helper.make_node('ArgMax',
                                     inputs=['a_np1'],
                                     outputs=['out'],
                                     axis=axis,
                                     keepdims=keepdims)

    graph = helper.make_graph([node],
                              "argmax_test",
                              inputs = [helper.make_tensor_value_info("a_np1",
                                            TensorProto.INT32, list(a_np1.shape))],
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.INT32, list(b_np.shape))])

    model = helper.make_model(graph, producer_name='argmax_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [a_np1], target, ctx, b_np.shape, b_np.dtype)
        tvm.testing.assert_allclose(b_np, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_arg_min_max():
    '''Verify argmin and argmax'''
    verify_argmin([3,4,4])
    verify_argmax([3,4,4])
    verify_argmin([3,4,4], axis=1)
    verify_argmax([3,4,4], axis=0)
    verify_argmin([3,4,4], keepdims=0)
    verify_argmax([3,4,4], keepdims=1)
    for axis in [None, 0,1,2]:
        for keepdims in [None, True,False]:
            verify_argmin([3,4,4], axis, keepdims)
            verify_argmax([3,4,4], axis, keepdims)

def verify_constantfill(is_shape, input_dim, out_dim, value, dtype, **kwargs):
    input_a = np.random.uniform(size=input_dim).astype(dtype)
    out = np.empty(shape=out_dim, dtype=dtype)
    out.fill(value)

    if is_shape == True:
        fill_node = helper.make_node("ConstantFill", [], ["out"], shape=input_dim, value=value, **kwargs)
    else:
        fill_node = helper.make_node("ConstantFill", ["input_a"], ["out"], value=value, dtype=dtype, **kwargs)

    if is_shape == True:
        inputs = []
    else:
        inputs = [helper.make_tensor_value_info("input_a",
                  TensorProto.FLOAT, list(input_dim))]

    graph = helper.make_graph([fill_node],
                              "fill_test",
                              inputs,
                              outputs = [helper.make_tensor_value_info("out",
                                            TensorProto.FLOAT, list(out.shape))])

    model = helper.make_model(graph, producer_name='fill_test')

    for target, ctx in ctx_list():
        if is_shape == True:
            tvm_out = get_tvm_output(model, [], target, ctx, out.shape)
        else:
            tvm_out = get_tvm_output(model, [input_a], target, ctx, out.shape)

        tvm.testing.assert_allclose(out, tvm_out, rtol=1e-5, atol=1e-5)

def test_constantfill():
    verify_constantfill(True, (2, 3, 4, 5), (2, 3, 4, 5), 10, 'float32')
    verify_constantfill(False, (2, 3, 4, 5), (2, 3, 4, 5), 10, 'float32')
    verify_constantfill(True, (2, 3, 4, 5), (2, 3, 4, 5, 4, 5, 6), 10, 'float32', extra_shape=(4, 5, 6))


def verify_pad(indata, pads, mode='constant', value=0.0):
    indata = np.array(indata).astype(np.float32)
    #  numpy expect result
    len_dim = len(pads) // 2
    np_pads = [(pads[i], pads[i+len_dim]) for i in range(len_dim)]
    #  onnx graph
    if mode in ['edge', 'reflect']:
        outdata = np.pad(indata, pad_width=np_pads, mode=mode)
        node = helper.make_node(
            'Pad',
            inputs=['input'],
            outputs=['output'],
            mode=mode,
            pads=pads,
        )
    else:
        outdata = np.pad(indata, pad_width=np_pads, mode='constant', constant_values=value)
        node = helper.make_node(
            'Pad',
            inputs=['input'],
            outputs=['output'],
            mode='constant',
            pads=pads,
            value=value
        )
    graph = helper.make_graph([node],
                              'pad_test',
                              inputs = [helper.make_tensor_value_info("input",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("output",
                                            TensorProto.FLOAT, list(outdata.shape))])
    model = helper.make_model(graph, producer_name='pad_test')
    #  tvm result
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, 'float32')
    tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)

def test_pad():
    verify_pad(np.random.randn(2, 2).astype(np.float32), [0, 1, 0, 0], 'constant', 0.0)
    verify_pad(np.random.randn(2, 3).astype(np.float32), [1, 0, 0, 1], 'constant', 0.0)
    verify_pad(np.random.randn(3, 2).astype(np.float32), [0, 0, 1, 0], 'constant', 5.0)
    verify_pad(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], 'edge')
    verify_pad(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], 'reflect')

def verify_reduce_x(name, indata, axis, keepdims):
    indata = np.array(indata).astype(np.float32)
    #  numpy expect result
    if name == 'ReduceMax':
        outdata = np.maximum.reduce(indata, axis=axis, keepdims=keepdims == 1)
    elif name == 'ReduceMin':
        outdata = np.minimum.reduce(indata, axis=axis, keepdims=keepdims == 1)
    elif name == 'ReduceSum':
        outdata = np.sum(indata, axis=axis, keepdims=keepdims == 1)
    elif name == 'ReduceMean':
        outdata = np.mean(indata, axis=axis, keepdims=keepdims == 1)
    else:
        raise Exception('unsupport op: {}'.format(name))
    if len(np.asarray(outdata).shape) == 0:
        outdata = np.asarray([outdata])
    #  onnx graph
    if axis is None:
        node = helper.make_node(name, inputs=['input'], outputs=['output'],
                                keepdims=keepdims)
    else:
        node = helper.make_node(name, inputs=['input'], outputs=['output'],
                                axes=axis, keepdims=keepdims)
    graph = helper.make_graph([node],
                              '{}_test'.format(name),
                              inputs = [helper.make_tensor_value_info("input",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("output",
                                            TensorProto.FLOAT, list(outdata.shape))])
    model = helper.make_model(graph, producer_name='{}_test'.format(name))
    #  tvm result
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, indata, target, ctx, outdata.shape, 'float32')
    tvm.testing.assert_allclose(outdata, tvm_out, rtol=1e-5, atol=1e-5)

def test_reduce_max():
    verify_reduce_x("ReduceMax",
                    np.random.randn(3, 2, 2).astype(np.float32),
                    axis=None, keepdims=1)
    verify_reduce_x("ReduceMax",
                    np.random.randn(3, 2, 3).astype(np.float32),
                    axis=None, keepdims=0)
    verify_reduce_x("ReduceMax",
                    np.random.randn(3, 3, 3).astype(np.float32),
                    axis=(1,), keepdims=1)

def test_reduce_min():
    verify_reduce_x("ReduceMin",
                    np.random.randn(3, 2, 2).astype(np.float32),
                    axis=None, keepdims=1)
    verify_reduce_x("ReduceMin",
                    np.random.randn(3, 2, 3).astype(np.float32),
                    axis=None, keepdims=0)
    verify_reduce_x("ReduceMin",
                    np.random.randn(3, 3, 3).astype(np.float32),
                    axis=(1,), keepdims=1)

def test_reduce_sum():
    verify_reduce_x("ReduceSum",
                    np.random.randn(3, 2, 2).astype(np.float32),
                    axis=None, keepdims=1)
    verify_reduce_x("ReduceSum",
                    np.random.randn(3, 2, 3).astype(np.float32),
                    axis=None, keepdims=0)
    verify_reduce_x("ReduceSum",
                    np.random.randn(3, 3, 3).astype(np.float32),
                    axis=(1,), keepdims=1)

def test_reduce_mean():
    verify_reduce_x("ReduceMean",
                    np.random.randn(3, 2, 2).astype(np.float32),
                    axis=None, keepdims=1)
    verify_reduce_x("ReduceMean",
                    np.random.randn(3, 2, 3).astype(np.float32),
                    axis=None, keepdims=0)
    verify_reduce_x("ReduceMean",
                    np.random.randn(3, 3, 3).astype(np.float32),
                    axis=(1,), keepdims=1)

def verify_split(indata, outdatas, split, axis=0):
    indata = np.array(indata).astype(np.float32)
    outdatas = [np.array(o).astype(np.float32) for o in outdatas]
    node = helper.make_node(
        'Split',
        inputs=['input'],
        outputs=['output_{}'.format(i) for i in range(len(split))],
        axis=axis,
        split=split
    )
    graph = helper.make_graph([node],
                              'split_test',
                              inputs = [helper.make_tensor_value_info("input",
                                            TensorProto.FLOAT, list(indata.shape))],
                              outputs = [helper.make_tensor_value_info("output_{}".format(i),
                                            TensorProto.FLOAT, list(outdatas[i].shape))
                                            for i in range(len(split))
                                         ])
    model = helper.make_model(graph, producer_name='split_test')

    for target, ctx in ctx_list():
        output_shape = [o.shape for o in outdatas]
        output_type = ['float32', 'float32', 'float32']
        tvm_out = get_tvm_output(model, indata, target, ctx, output_shape, output_type)
    for o, t in zip(outdatas, tvm_out):
        tvm.testing.assert_allclose(o, t)

def test_split():
    # 1D
    verify_split([1., 2., 3., 4., 5., 6.], [[1., 2.], [3., 4.], [5., 6.]], [2, 2, 2], 0)
    verify_split([1., 2., 3., 4., 5., 6.], [[1., 2.], [3.], [4., 5., 6.]], [2, 1, 3], 0)
    # 2D
    verify_split([[1., 2., 3., 4.], [7., 8., 9., 10.]],
                 [[[1., 2.], [7., 8.]], [[3., 4.], [9., 10.]]], [2, 2], 1)

def test_binary_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_binary_ops(op, x, y, out_np, broadcast=None):
        if broadcast is None:
            z = helper.make_node(op, ['in1', 'in2'], ['out'])
        else:
            z = helper.make_node(op, ['in1', 'in2'], ['out'], broadcast=1)
        graph = helper.make_graph([z],
                                   '_test',
                                  inputs = [helper.make_tensor_value_info("in1",
                                                TensorProto.FLOAT, list(in_shape)),
                                            helper.make_tensor_value_info("in2",
                                                TensorProto.FLOAT, list(in_shape))],
                                  outputs = [helper.make_tensor_value_info("out",
                                                TensorProto.FLOAT, list(out_shape))])
        model = helper.make_model(graph, producer_name='_test')
        for target, ctx in ctx_list():
            tvm_out = get_tvm_output(model, [x, y], target, ctx)
            tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)

    x = np.random.uniform(size=in_shape).astype(dtype)
    y = np.random.uniform(size=in_shape).astype(dtype)
    z = np.random.uniform(size=(3,)).astype(dtype)
    verify_binary_ops("Add",x, y, x + y, broadcast=None)
    verify_binary_ops("Add", x, z,  x + z, broadcast=True)
    verify_binary_ops("Sub", x, y, x - y, broadcast=None)
    verify_binary_ops("Sub", x, z, x - z, broadcast=True)
    verify_binary_ops("Mul",x, y, x * y, broadcast=None)
    verify_binary_ops("Mul", x, z,  x * z, broadcast=True)
    verify_binary_ops("Div", x, y, x / y, broadcast=None)
    verify_binary_ops("Div", x, z, x / z, broadcast=True)
    verify_binary_ops("Sum", x, y, x + y, broadcast=None)
    verify_binary_ops("Greater", x, y, x > y, broadcast=True)
    verify_binary_ops("Less", x, y, x < y, broadcast=True)
    verify_binary_ops("Equal", x, y, x == y, broadcast=True)

def test_single_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_single_ops(op, x, out_np, rtol=1e-5, atol=1e-5):
        z = helper.make_node(op, ['in1'], ['out'])
        graph = helper.make_graph([z],
                                   '_test',
                                  inputs = [helper.make_tensor_value_info("in1",
                                                TensorProto.FLOAT, list(in_shape)),],
                                  outputs = [helper.make_tensor_value_info("out",
                                                TensorProto.FLOAT, list(out_shape))])
        model = helper.make_model(graph, producer_name='_test')
        for target, ctx in ctx_list():
            tvm_out = get_tvm_output(model, [x], target, ctx)
            tvm.testing.assert_allclose(out_np, tvm_out, rtol=rtol, atol=atol)

    x = np.random.uniform(size=in_shape).astype(dtype)
    verify_single_ops("Neg",x, -x)
    verify_single_ops("Abs",x, np.abs(x))
    verify_single_ops("Reciprocal",x, 1/x)
    verify_single_ops("Sqrt",x, np.sqrt(x))
    verify_single_ops("Relu",x, np.maximum(x, 0))
    verify_single_ops("Exp",x, np.exp(x))
    verify_single_ops("Log",x, np.log(x))
    verify_single_ops("Log",x, np.log(x))
    verify_single_ops("Tanh",x, np.tanh(x))
    verify_single_ops("Sigmoid",x, 1 / (1 + np.exp(-x)))
    verify_single_ops("Softsign",x, x / (1 + np.abs(x)))
    verify_single_ops("SoftPlus",x, np.log(1 + np.exp(x)))

def test_leaky_relu():
    def leaky_relu_x(x, alpha):
        return np.where(x >= 0, x, x * alpha)
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              leaky_relu_x,
                              {'alpha': 0.25},
                              'float32',
                              'LeakyRelu',
                              {'alpha': 0.25})

def test_elu():
    def elu_x(x, alpha):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              elu_x,
                              {'alpha': 0.25},
                              'float32',
                              'Elu',
                              {'alpha': 0.25})

def test_selu():
    def selu_x(x, alpha, gamma):
        return gamma * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              selu_x,
                              {'alpha': 0.25, 'gamma': 0.3},
                              'float32',
                              'Selu',
                              {'alpha': 0.25, 'gamma': 0.3})

def test_ThresholdedRelu():
    def ThresholdedRelu_x(x, alpha):
        out_np = np.clip(x, alpha, np.inf)
        out_np[out_np == alpha] = 0
        return out_np
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              ThresholdedRelu_x,
                              {'alpha': 0.25},
                              'float32',
                              'ThresholdedRelu',
                              {'alpha': 0.25})

def test_ScaledTanh():
    def ScaledTanh_x(x, alpha, beta):
        return alpha * np.tanh(beta * x)
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              ScaledTanh_x,
                              {'alpha': 0.25, 'beta': 0.3},
                              'float32',
                              'ScaledTanh',
                              {'alpha': 0.25, 'beta': 0.3})

def test_ParametricSoftplus():
    def ParametricSoftplus_x(x, alpha, beta):
        return alpha * np.log(np.exp(beta * x) + 1)
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              ParametricSoftplus_x,
                              {'alpha': 0.25, 'beta': 0.3},
                              'float32',
                              'ParametricSoftplus',
                              {'alpha': 0.25, 'beta': 0.3})

def test_Scale():
    def Scale_x(x, scale):
        return scale * x
    _test_onnx_op_elementwise((2, 4, 5, 6),
                              Scale_x,
                              {'scale': 0.25},
                              'float32',
                              'Scale',
                              {'scale': 0.25})

def test_LogSoftmax():
    _test_onnx_op_elementwise((1, 4),
                              topi.testing.log_softmax_python,
                              {},
                              'float32',
                              'LogSoftmax',
                              {'axis': 1})


def check_torch_conversion(model, input_size):
    dummy_input = torch.randn(*input_size)
    file_name = '{}.onnx'.format(model.__name__)
    # Set verbose=True for more output
    torch.onnx.export(model(), dummy_input, file_name, export_params=True, verbose=False)
    onnx_model = onnx.load(file_name)
    for target, ctx in ctx_list():
        input_data = np.random.uniform(size=input_size).astype('int32')
        c2_out = get_caffe2_output(onnx_model, input_data)
        tvm_out = get_tvm_output(onnx_model, input_data, target, ctx)
        tvm.testing.assert_allclose(c2_out, tvm_out)

def test_resnet():
    check_torch_conversion(torchvision.models.resnet18, (1,3,224,224))
    # check_torch_conversion(torchvision.models.resnet101, (1,3,224,224))

# def test_alexnet():
    # Torch's ONNX export does not support the adaptive pooling used by AlexNet?
    # check_torch_conversion(torchvision.models.alexnet, (1,3,224,224))

# Torch's ONNX export does not support the adaptive pooling used by vgg16?
# def test_vgg16():
#     check_torch_conversion(torchvision.models.vgg16, (1,3,224,224))

# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_squeezenet():
#     # Torch's ONNX export does not support the max pooling used by Squezenet
#     check_torch_conversion(torchvision.models.squeezenet1_0, (1,3,224,224))

def test_densenet():
    check_torch_conversion(torchvision.models.densenet161, (1,3,224,224))

def test_inception():
    check_torch_conversion(torchvision.models.inception_v3, (1,3,224,224))

# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_googlenet():
#     check_torch_conversion(torchvision.models.googlenet, (1,3,224,224))

# TODO(@jroesch): Update Torch + ONNX to support this import.
# def test_shufflenetv2():
#     check_torch_conversion(torchvision.models.shufflenetv2, (1,3,224,224))

def test_sign():
    def Sign_x(x):
        return np.sign(x)
    _test_onnx_op_elementwise((3, 4, 5, 6),
                              Sign_x,
                              {},
                              'float32',
                              'Sign',
                              {})


def verify_not(indata, dtype):
    x = indata.astype(dtype)
    outdata = np.logical_not(x)

    node = helper.make_node('Not', inputs=['in'], outputs=['out'],)

    graph = helper.make_graph([node],
                              'not_test',
                              inputs=[helper.make_tensor_value_info("in", TensorProto.BOOL, list(x.shape))],
                              outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))])

    model = helper.make_model(graph, producer_name='not_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)


def test_not():
    # 2d
    verify_not(indata=(np.random.randn(3, 4) > 0), dtype=bool)
    # 3d
    verify_not(indata=(np.random.randn(3, 4, 5) > 0), dtype=bool)
    # 4d
    verify_not(indata=(np.random.randn(3, 4, 5, 6) > 0), dtype=bool)


def verify_and(indata, dtype):
    x = indata[0].astype(dtype)
    y = indata[1].astype(dtype)
    outdata = np.logical_and(x, y)

    node = helper.make_node('And', inputs=['in1', 'in2'], outputs=['out'], )

    graph = helper.make_graph([node],
                              'and_test',
                              inputs=[helper.make_tensor_value_info("in1", TensorProto.BOOL, list(x.shape)),
                                      helper.make_tensor_value_info("in2", TensorProto.BOOL, list(y.shape))],
                              outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))])

    model = helper.make_model(graph, producer_name='and_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [x, y], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)


def test_and():
    # 2d
    x = (np.random.randn(3, 4) > 0)
    y = (np.random.randn(3, 4) > 0)
    verify_and(indata=[x, y], dtype=bool)

    # 3d
    x = (np.random.randn(3, 4, 5) > 0)
    y = (np.random.randn(3, 4, 5) > 0)
    verify_and(indata=[x, y], dtype=bool)

    # 4d
    x = (np.random.randn(3, 4, 5, 6) > 0)
    y = (np.random.randn(3, 4, 5, 6) > 0)
    verify_and(indata=[x, y], dtype=bool)

    # 3d vs 1d
    x = (np.random.randn(3, 4, 5) > 0)
    y = (np.random.randn(5) > 0)
    verify_and(indata=[x, y], dtype=bool)

    # 3d vs 2d
    x = (np.random.randn(3, 4, 5) > 0)
    y = (np.random.randn(4, 5) > 0)
    verify_and(indata=[x, y], dtype=bool)


def verify_tile(indata, outdata, **kwargs):
    node = helper.make_node('Tile', inputs=['in'], outputs=['out'], **kwargs)
    graph = helper.make_graph([node],
                              'tile_test',
                              inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
                              outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))])

    model = helper.make_model(graph, producer_name='tile_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [indata], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)


def test_tile():
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z = np.tile(x, repeats)
    verify_tile(x, z, repeats=repeats)

def verify_erf(indata, outdata):
    node = helper.make_node('Erf', inputs=['in'], outputs=['out'])
    graph = helper.make_graph([node],
                              'erf_test',
                              inputs=[helper.make_tensor_value_info('in', TensorProto.FLOAT, list(indata.shape))],
                              outputs=[helper.make_tensor_value_info('out', TensorProto.FLOAT, list(outdata.shape))])
    model = helper.make_model(graph, producer_name='erf_test')

    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, [indata], target, ctx, outdata.shape)
        tvm.testing.assert_allclose(outdata, tvm_out)

def test_erf():
    x = np.random.rand(2, 3, 4, 6).astype(np.float32)
    z = scipy.special.erf(x)
    verify_erf(x, z)


if __name__ == '__main__':
    test_flatten()
    test_reshape()
    test_shape()
    test_power()
    test_squeeze()
    test_unsqueeze()
    test_slice()
    test_floor()
    test_ceil()
    test_clip()
    test_matmul()
    test_gather()
    test_lrn()
    test_instance_norm()
    test_upsample()
    test_forward_min()
    test_forward_max()
    test_forward_mean()
    test_forward_hardsigmoid()
    test_forward_arg_min_max()
    test_softmax()
    test_constantfill()
    test_reduce_max()
    test_reduce_min()
    test_reduce_sum()
    test_reduce_mean()
    test_pad()
    test_split()
    test_binary_ops()
    test_single_ops()
    test_leaky_relu()
    test_elu()
    test_selu()
    test_ThresholdedRelu()
    test_ScaledTanh()
    test_ParametricSoftplus()
    test_Scale()
    test_LogSoftmax()
    test_resnet()
    test_inception()
    test_densenet()
    test_sign()
    test_not()
    test_and()
    test_tile()
    test_erf()
