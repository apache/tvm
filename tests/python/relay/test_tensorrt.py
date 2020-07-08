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
import time

import tvm
from tvm import relay
import tvm.relay.testing
import tvm.relay.tensorrt
import pytest
from tvm.contrib import graph_runtime

def should_skip():
    if not tvm.runtime.enabled("cuda") or not tvm.gpu(0).exist:
        print("skip because cuda is not enabled.")
        return True
    if not relay.tensorrt.IsTrtRuntimeAvailable():
        print("skip because tensorrt runtime is not available")
        return True
    return False

def test_tensorrt_simple():
    if should_skip():
        return
    dtype = 'float32'
    xshape = (1, 3, 2, 2)
    yshape = (1, 3,  1,  1)
    zshape = (1,  1,  1,  1)
    x = relay.var('x', shape=(xshape), dtype=dtype)
    y = relay.var('y', shape=(yshape), dtype=dtype)
    z = relay.var('z', shape=(zshape), dtype=dtype)
    w = z * (x + y)
    out = relay.nn.relu(w)
    f = relay.Function([x, y, z], out)

    mod = tvm.IRModule()
    mod['main'] = f
    mod = relay.tensorrt.EnableTrt(mod)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "cuda")
    mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)
    y_data = np.random.uniform(-1, 1, yshape).astype(dtype)
    z_data = np.random.uniform(-1, 1, zshape).astype(dtype)
    mod.run(x=x_data, y=y_data, z=z_data)
    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]

def test_tensorrt_not_compatible():
    if should_skip():
        return
    dtype = 'float32'
    xshape = (1, 32, 14, 14)
    x = relay.var('x', shape=(xshape), dtype=dtype)
    y = relay.add(x, x)
    z = relay.erf(y)
    out = relay.nn.relu(z)
    f = relay.Function([x], out)
    mod = tvm.IRModule()
    mod['main'] = f
    mod = relay.tensorrt.EnableTrt(mod)
    assert not mod['main'].attrs

def test_tensorrt_ops():
    if should_skip():
        return
    def run_and_verify(config):
        f, input_shapes, is_param = config
        params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(np.float32) for x in is_param}
        input_dict = {k: np.random.uniform(-1, 1, v).astype(np.float32) for k, v in input_shapes.items() if k not in is_param}

        # Run TRT 
        mod = tvm.IRModule()
        mod['main'] = f
        mod = relay.tensorrt.EnableTrt(mod, params)
        with relay.build_config(opt_level=3):
            graph, lib, graph_params = relay.build(mod, "cuda", params=params)
        mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
        mod.set_input(**graph_params)
        mod.run(**input_dict)
        results = [mod.get_output(i) for i in range(mod.get_num_outputs())]

        # Run reference
        mod = tvm.IRModule()
        mod['main'] = f
        with relay.build_config(opt_level=3):
            graph, lib, graph_params = relay.build(mod, "cuda", params=params)
        mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
        mod.set_input(**graph_params)
        mod.run(**input_dict)
        ref_results = [mod.get_output(i) for i in range(mod.get_num_outputs())]
        
        assert len(results) == len(ref_results)
        for i in range(len(results)):
            res = results[i].asnumpy()
            ref_res = ref_results[i].asnumpy()
            assert res.shape == ref_res.shape
            tvm.testing.assert_allclose(res, ref_res, rtol=1e-3, atol=1e-3)

    def test_conv2d(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), groups=1, padding=(0, 0), strides=(1, 1), dilation=(1, 1)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
        out = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4], groups=groups, padding=padding, strides=strides, dilation=dilation)
        f = relay.Function([x, kernel], out)
        return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']

    def test_conv2d_const_weights(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), groups=1, padding=(0, 0), strides=(1, 1), dilation=(1, 1)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.const(np.ones(k_shape).astype("float32"))
        out = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4], groups=groups, padding=padding, strides=strides, dilation=dilation)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []
    
    def test_dense(x_shape=(1, 16), k_shape=(32, 16)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
        # Dense requires constant weights in TensorRT, so the weights are transposed by us.
        out = relay.nn.dense(x, kernel, units=k_shape[0])
        f = relay.Function([x, kernel], out)
        return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']
    
    def test_bias_add(x_shape=(1, 16), channels=16):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        bias = relay.var('bias', shape=(channels,), dtype='float32')
        out = relay.nn.bias_add(x, bias)
        f = relay.Function([x, bias], out)
        return f, {'x': x_shape, 'bias': (channels,)}, ['bias']
    
    def test_pool2d(op, x_shape=(1, 3, 32, 32), pool_size=(2, 2), strides=(2, 2), padding=(0, 0), ceil_mode=False, count_include_pad=None):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        if count_include_pad is not None:
            out = op(x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        else:
            out = op(x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_global_pool2d(op, x_shape=(1, 3, 32, 32)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = op(x)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_batch_flatten(x_shape=(1, 3, 4, 6)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.nn.batch_flatten(x)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []
    
    def test_expand_dims(x_shape=(1, 3), axis=1, num_newaxis=1):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.expand_dims(x, axis, num_newaxis)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_squeeze(x_shape, axis):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.squeeze(x, axis=axis)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []
    
    def test_concatenate(input_shapes, axis):
        concat_inputs = []
        shapes_dict = {}
        for i in range(len(input_shapes)):
            name = 'input_{}'.format(i)
            concat_inputs.append(relay.var(name, shape=(input_shapes[i]), dtype='float32'))
            shapes_dict[name] = input_shapes[i]
        out = relay.concatenate(concat_inputs, axis)
        f = relay.Function(concat_inputs, out)
        return f, shapes_dict, []
    
    def test_conv2d_transpose(x_shape=(1, 32, 8, 8), k_shape=(32, 16, 3, 3), groups=1, padding=(0, 0), strides=(1, 1)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
        out = relay.nn.conv2d_transpose(x, kernel, channels=k_shape[1], kernel_size=k_shape[2:4], groups=groups, padding=padding, strides=strides)
        f = relay.Function([x, kernel], out)
        return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']
    
    def test_reshape(x_shape, new_shape):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.reshape(x, new_shape)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []
    
    def test_transpose(x_shape, order):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.transpose(x, order)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_transpose_weights_conv2d(x_shape=(1, 32, 9, 9), k_shape=(3, 3, 32, 16), order=(3, 2, 0, 1)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
        kernel_t = relay.transpose(kernel, order)
        # Conv2d requires constant weights in TensorRT, so the weights are transposed by us.
        out = relay.nn.conv2d(x, kernel_t, channels=k_shape[order[0]], kernel_size=(3, 3))
        f = relay.Function([x, kernel], out)
        return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']

    def test_transpose_weights_dense(x_shape=(1, 16), k_shape=(16, 32)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
        kernel_t = relay.transpose(kernel, (1, 0))
        # Dense requires constant weights in TensorRT, so the weights are transposed by us.
        out = relay.nn.dense(x, kernel_t)
        f = relay.Function([x, kernel], out)
        return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']

    def test_dense_from_pytorch(x_shape=(1, 16), k_shape=(32, 16)):
        # FixPyTorchAddmm will fold away the tranpose -> mult -> transpose.
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
        kernel_t = relay.transpose(kernel, (1, 0))
        beta = relay.const(1, dtype='float32')
        kernel_t = relay.multiply(kernel_t, beta)
        kernel_t = relay.transpose(kernel_t, (1, 0))
        # Dense requires constant weights in TensorRT, so the weights are transposed by us.
        out = relay.nn.dense(x, kernel_t)
        f = relay.Function([x, kernel], out)
        return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']

    def test_float_const(x_shape=(1, 16)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        beta = relay.const(1, dtype='float32')
        out = relay.multiply(x, beta)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_pad(x_shape, pad_width):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.nn.pad(x, pad_width=pad_width)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_softmax(x_shape, axis):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.nn.softmax(x, axis=axis)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_batch_norm(x_shape, param_shape, axis=1, epsilon=1e-5):
        x = relay.var("x", shape=(x_shape), dtype='float32')
        beta = relay.var("beta", shape=(param_shape), dtype='float32')
        gamma = relay.var("gamma",  shape=(param_shape), dtype='float32')
        moving_mean = relay.var("moving_mean", shape=(param_shape), dtype='float32')
        moving_var = relay.var("moving_var", shape=(param_shape), dtype='float32')
        out, _, _ = relay.nn.batch_norm(x, gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var,
                                        axis=axis, center=True, scale=True, epsilon=epsilon)
        f = relay.Function([x, gamma, beta, moving_mean, moving_var], out)
        return f, {'x': x_shape, 'beta': param_shape, 'gamma': param_shape,
                   'moving_mean': param_shape, 'moving_var': param_shape}, ['beta', 'gamma', 'moving_mean', 'moving_var']

    def test_unary(op, x_shape=(1, 8, 3, 3)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = op(x)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_clip(x_shape=(1, 8, 3, 3)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.clip(x, a_min=-0.2, a_max=0.4)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_leaky_relu(x_shape=(1, 8, 3, 3)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.nn.leaky_relu(x, alpha=0.1)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []
    
    def test_binary(op, x_shape, y_shape, y_is_const=False):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        if y_is_const:
            y = relay.const(np.ones(y_shape).astype('float32'))
            out = op(x, y)
            f = relay.Function([x], out)
            return f, {'x': x_shape}, []
        y = relay.var('y', shape=(y_shape), dtype='float32')
        out = op(x, y)
        f = relay.Function([x, y], out)
        return f, {'x': x_shape, 'y': y_shape}, []

    def test_reduce(op, x_shape=(1, 2, 3, 4), axis=(2, 3), keepdims=False):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = op(x, axis=axis, keepdims=keepdims)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_strided_slice(x_shape, begin, end, strides=None):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.strided_slice(x, begin, end, strides)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_adaptive_pool2d(op, x_shape=(1, 3, 32, 32), out_size=(1, 1)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = op(x, out_size)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_resize(x_shape=(1, 3, 16, 16), out_size=(32, 32), layout='NCHW', method='nearest_neighbor', coordinate_transformation_mode='align_corners'):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        out = relay.image.resize(x, out_size, layout=layout, method=method, coordinate_transformation_mode=coordinate_transformation_mode)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_multiple_outputs():
        x = relay.var('x', shape=(1, 3), dtype='float32')
        y = relay.var('y', shape=(1, 3), dtype='float32')
        z = relay.add(x, y)
        w = relay.add(z, y)
        out = relay.Tuple((z, w))
        f = relay.Function([x, y], out)
        return f, {'x': (1, 3), 'y': (1, 3)}, []

    def test_conv3d(x_shape=(1, 32, 8, 8, 8), k_shape=(16, 32, 3, 3, 3), groups=1, padding=(0, 0, 0), strides=(1, 1, 1), dilation=(1, 1, 1)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
        out = relay.nn.conv3d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:], groups=groups, padding=padding, strides=strides, dilation=dilation)
        f = relay.Function([x, kernel], out)
        return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']

    def test_pool3d(op, x_shape=(1, 3, 8, 32, 32), pool_size=(2, 2, 2), strides=(2, 2, 2), padding=(0, 0, 0), ceil_mode=False, count_include_pad=None):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        if count_include_pad is not None:
            out = op(x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        else:
            out = op(x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode)
        f = relay.Function([x], out)
        return f, {'x': x_shape}, []

    def test_conv3d_transpose(x_shape=(1, 32, 8, 8, 8), k_shape=(32, 16, 3, 3, 3), groups=1, padding=(0, 0, 0), strides=(1, 1, 1), output_padding=(0, 0, 0)):
        x = relay.var('x', shape=(x_shape), dtype='float32')
        kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
        out = relay.nn.conv3d_transpose(x, kernel, channels=k_shape[1], kernel_size=k_shape[2:5], groups=groups, padding=padding, strides=strides, output_padding=output_padding)
        f = relay.Function([x, kernel], out)
        return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']

    run_and_verify(test_float_const())
    run_and_verify(test_multiple_outputs())
    run_and_verify(test_clip())
    run_and_verify(test_leaky_relu())
    run_and_verify(test_batch_norm((1, 64, 56, 56), (64,)))
    run_and_verify(test_batch_norm((1, 56, 56, 64), (64,), axis=3, epsilon=1.001e-05))
    run_and_verify(test_softmax((1, 1000), axis=1))
    run_and_verify(test_softmax((1, 1000), axis=-1))
    run_and_verify(test_softmax((1, 3, 4), axis=-2))
    run_and_verify(test_softmax((1, 3, 4), axis=1))
    for k_shape, groups in [((16, 32, 3, 3), 1), ((32, 1, 3, 3), 32)]:
        for padding in [(0, 0), (1, 1)]:
            for strides in [(1, 1), (2, 2)]:
                for dilation in [(1, 1), (2, 2)]:
                    run_and_verify(test_conv2d(k_shape=k_shape, groups=groups, padding=padding,
                                               strides=strides, dilation=dilation))
    # Disabled due to incorrect results from TVM.
    run_and_verify(test_conv2d_const_weights())
    run_and_verify(test_dense())
    run_and_verify(test_dense_from_pytorch())
    run_and_verify(test_bias_add())
    run_and_verify(test_bias_add((1, 6, 3, 4), 6))
    for op in [relay.add, relay.subtract, relay.multiply, relay.divide, relay.power]:
        # Disabled y_is_const=True due to incorrect results from TVM.
        for y_is_const in [True, False]:
            run_and_verify(test_binary(op, (1, 8, 3, 3), (1, 8, 3, 3), y_is_const))
            run_and_verify(test_binary(op, (1, 8, 1, 3), (1, 8, 3, 1), y_is_const))
            run_and_verify(test_binary(op, (1, 10), (10,), y_is_const))
            run_and_verify(test_binary(op, (1, 1, 1, 10), (10,), y_is_const))
            run_and_verify(test_binary(op, (1, 1, 1), (3,), y_is_const))
    for pool_size in [(2, 2), (3, 3)]:
        for strides in [(1, 1), (2, 2)]:
            for padding in [(0, 0), (1, 1), (0, 0, 1, 1)]:
                for ceil_mode in [False, True]:
                    # Skip "the padding size is larger than or equal to the filter size for exclusive-counting pooling"
                    if pool_size == (2, 2) and padding == (0, 0, 1, 1):
                        continue
                    for count_include_pad in [False, True]:
                        # Skip "inclusive-counted blended or average pooling is not supported in combination with asymmetric padding"
                        if count_include_pad and (padding == (0, 0, 1, 1) or strides == (2, 2)):
                            continue
                        run_and_verify(test_pool2d(relay.nn.avg_pool2d, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad))
                    run_and_verify(test_pool2d(relay.nn.max_pool2d, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode))
    for op in [relay.nn.global_max_pool2d, relay.nn.global_max_pool2d]:
        run_and_verify(test_global_pool2d(op))
    for op in [relay.nn.relu, relay.sigmoid, relay.tanh, relay.exp, relay.log, relay.sqrt,
               relay.abs, relay.negative, relay.sin, relay.cos, relay.atan, relay.ceil, relay.floor]:
        run_and_verify(test_unary(op))
    run_and_verify(test_batch_flatten())
    run_and_verify(test_expand_dims())
    run_and_verify(test_squeeze((1, 5, 1, 1), (2, 3)))
    run_and_verify(test_squeeze((1, 3, 1), (-1,)))
    run_and_verify(test_concatenate([(1, 2, 6, 6), (1, 3, 6, 6)], axis=1))
    for padding in [(0, 0), (1, 1)]:
        for strides in [(1, 1), (2, 2)]:           
                run_and_verify(test_conv2d_transpose(padding=padding, strides=strides))
    run_and_verify(test_transpose((1, 16, 7, 7), [0, 2, 3, 1]))
    run_and_verify(test_transpose((1, 7, 7, 16), [0, 3, 1, 2]))
    run_and_verify(test_transpose_weights_conv2d())
    run_and_verify(test_transpose_weights_conv2d((1, 32, 9, 9), (3, 3, 16, 32), (2, 3, 0, 1)))
    run_and_verify(test_transpose_weights_dense())
    run_and_verify(test_reshape((1, 1, 1, 10), (-1, 10)))
    run_and_verify(test_reshape((1, 10, 2, 3), (1, -1)))
    run_and_verify(test_reshape((1, 1, 2, 3), (1, 6)))
    run_and_verify(test_pad((1, 64, 56, 56), [[0, 0], [0, 0], [0, 0], [0, 0]]))
    run_and_verify(test_pad((1, 64, 56, 56), [[0, 0], [0, 0], [1, 1], [1, 1]]))
    run_and_verify(test_pad((1, 56, 56, 64), [[0, 0], [1, 1], [1, 1], [0, 0]]))
    for op in [relay.sum, relay.prod, relay.max, relay.min, relay.mean]:
        for keepdims in [True, False]:
            run_and_verify(test_reduce(op, axis=(1), keepdims=keepdims))
            run_and_verify(test_reduce(op, axis=(2, 3), keepdims=keepdims))
            run_and_verify(test_reduce(op, axis=(1, 2), keepdims=keepdims))
            run_and_verify(test_reduce(op, axis=(1, 2, 3), keepdims=keepdims))
    run_and_verify(test_strided_slice((1, 3, 6, 7), (0, 0, 0, 0), (1, 1, 6, 7)))
    run_and_verify(test_strided_slice((1, 3, 6, 7), (0, 1, 0, 0), (1, 2, 6, 6)))
    run_and_verify(test_strided_slice((1, 10), (0, 0), (1, 10), (1, 2)))
    for op in [relay.nn.adaptive_max_pool2d, relay.nn.adaptive_avg_pool2d]:
        run_and_verify(test_adaptive_pool2d(op))
    # for x_shape, layout in [((1, 3, 16, 16), 'NCHW'), ((1, 16, 16, 3), 'NHWC')]:
    #     for out_size in [(32, 32), (40, 40), (5, 21)]:
    #         for method in ['nearest_neighbor', 'bilinear']:
    #             for coordinate_transformation_mode in ['asymmetric']:
    #                 # TODO(trevmorr): 'align_corners' gives incorrect results. 'half_pixel' not supported?
    #                 run_and_verify(test_resize(x_shape, out_size, layout, method, coordinate_transformation_mode))
    run_and_verify(test_conv3d())
    run_and_verify(test_conv3d(padding=(0, 0, 0, 1, 1, 1)))
    run_and_verify(test_pool3d(relay.nn.avg_pool3d))
    run_and_verify(test_pool3d(relay.nn.max_pool3d))
    run_and_verify(test_pool3d(relay.nn.max_pool3d, padding=(0, 0, 0, 1, 1, 1)))
    run_and_verify(test_pool3d(relay.nn.max_pool3d, strides=(1, 1, 1)))
    run_and_verify(test_conv3d_transpose())
    # Verify that op is not converted because output_padding isn't supported by TRT.
    run_and_verify(test_conv3d_transpose(output_padding=(1, 1, 1, 1, 1, 1)))

def test_tensorrt_integration(test_all_models=False):
    if should_skip():
        return
    
    def test_model(model, i_data, input_shape, dtype, use_trt=True, num_iteration=1):
        import mxnet as mx
        from mxnet.gluon.model_zoo.vision import get_model
        def check_trt_used(graph):
            import json
            graph = json.loads(graph)
            num_trt_subgraphs = sum([1 for n in graph['nodes'] if n.get('attrs', {}).get('func_name', '') == 'tensorrt_0'])
            assert num_trt_subgraphs == 1

        block = get_model(model, pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)

        if use_trt:
            mod = relay.tensorrt.EnableTrt(mod, params)
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(mod, "cuda")
            check_trt_used(graph)
        else:
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(mod, "cuda", params=params)

        mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
        mod.set_input(**params)
        # Warmup
        for i in range(10):
            mod.run(data=i_data)

        # Time
        times = []
        for i in range(num_iteration):
            start_time = time.time()
            mod.run(data=i_data)
            res = mod.get_output(0)
            times.append(time.time() - start_time)
        latency = 1000.0 * np.mean(times)
        print(model, latency)
        return latency, res

    latency = {}
    models = [
        'alexnet',
        'resnet18_v1',
        'resnet18_v2',
        'squeezenet1.0',
        'mobilenet0.25',
        'mobilenetv2_0.25',
        'vgg11',
        'densenet121',
    ]
    additional_models = [
        'resnet34_v1',
        'resnet50_v1',
        'resnet101_v1',
        'resnet152_v1',
        'resnet34_v2',
        'resnet50_v2',
        'resnet101_v2',
        'resnet152_v2',
        'mobilenet0.5',
        'mobilenet0.75',
        'mobilenet1.0',
        'mobilenetv2_0.5',
        'mobilenetv2_0.75',
        'mobilenetv2_1.0',
        'vgg16',
        'densenet169',
        'densenet201']
    if test_all_models:
        models.extend(additional_models)
    
    dtype = 'float32'
    input_shape = (1, 3, 224, 224)
    i_data = np.random.uniform(-1, 1, input_shape).astype(dtype)
    for model in models:
        latency[model], res = test_model(model, i_data, input_shape, dtype, use_trt=True)
        _, ref_res = test_model(model, i_data, input_shape, dtype, use_trt=False, num_iteration=1)
        tvm.testing.assert_allclose(res.asnumpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-3)
    
    for model in models:
        print(model, latency[model])

def test_tensorrt_serialize():
    if should_skip():
        return
    import mxnet
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model('resnet18_v1', pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': (1, 3, 224, 224)}, dtype='float32')
    # Compile
    mod = relay.tensorrt.EnableTrt(mod, params)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "cuda")
    # Serialize
    with open('compiled.json', 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open('compiled.params', 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))
    lib.export_library('compiled.so')
    # Deserialize
    with open('compiled.json', 'r') as f_graph_json:
        graph = f_graph_json.read()
    with open('compiled.params', 'rb') as f_params:
        params = bytearray(f_params.read())
    lib = tvm.runtime.load_module("compiled.so")
    # Run
    mod = graph_runtime.create(graph, lib, ctx=tvm.gpu(0))
    mod.load_params(params)
    i_data = np.random.uniform(0, 1, (1, 3, 224, 224)).astype('float32')
    for i in range(10):
        mod.run(data=i_data)

if __name__ == '__main__':
    test_tensorrt_ops()
    test_tensorrt_simple()
    test_tensorrt_not_compatible()
    test_tensorrt_integration()
    test_tensorrt_serialize()
