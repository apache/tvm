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
"""CLML integration operator tests."""

import tvm
import numpy as np
from tvm import relay
from tvm.relay.op.contrib import clml
from tvm.relay import testing
from tvm.ir import IRModule
from tvm.contrib import utils
from test_clml.infrastructure import (
    build_and_run,
    Device,
    skip_codegen_test,
    verify_codegen,
    build_module,
    get_cpu_op_count,
)
import pytest


def _get_conv_model(
    shape,
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    groups,
    dtype,
    channels,
    var,
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Return a model and any parameters it may have"""
    a = relay.var(next(iter(var)), shape=shape, dtype=dtype)
    input_arr = var[next(iter(var))]
    if has_pad:
        p = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        a = relay.nn.pad(a, pad_width=p)
        padding = (0, 0, 0, 0)
    else:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        shape = (shape[0], shape[1], shape[2] + padding[0] * 2, shape[3] + padding[1] * 2)
    is_depthwise = shape[1] == channels == groups

    weight_format = "OIHW"
    weight_shape = (channels, shape[1] // groups, kernel_h, kernel_w)

    w = tvm.nd.array(np.random.uniform(-1, 1, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=(kernel_h, kernel_w),
        data_layout="NCHW",
        kernel_layout=weight_format,
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype=dtype,
    )
    params = {"w": w}
    if has_bias:
        bias_shape = (weight_shape[0],)
        b = tvm.nd.array(np.random.uniform(-1, 1, bias_shape).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.nn.bias_add(out, biasc, axis=1)
        params["b"] = b

    if has_activation:
        out = relay.nn.relu(out)

    return out, params


def _get_conv_expected_codegen(
    shape,
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    groups,
    dtype,
    channels,
    has_bias=False,
    has_activation=False,
):
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    output_height = ((shape[2] - kernel_h + padding[0] + padding[2]) / strides[0]) + 1
    output_width = ((shape[3] - kernel_w + padding[1] + padding[3]) / strides[1]) + 1
    output_shape = (1, channels, int(output_height), int(output_width))
    out_dtype = dtype
    is_depthwise = shape[1] == channels == groups

    weight_format = "IOHW" if is_depthwise else "OIHW"
    if weight_format == "OIHW":
        weight_shape = (channels, shape[1] // groups, kernel_h, kernel_w)
    else:
        weight_shape = (shape[1] // groups, channels, kernel_h, kernel_w)

    if is_depthwise:
        name = "nn.depthwise_conv2d"
    else:
        name = "nn.conv2d"

    node = {
        "op": "kernel",
        "name": name,
        "inputs": [],
        "attrs": {
            "groups": [[str(groups)]],
            "num_outputs": "1",
            "data_layout": [["NCHW"]],
            "kernel_layout": [[weight_format]],
            "channels": [[str(channels)]],
            "dilation": [[str(dilation[0]), str(dilation[1])]],
            "out_layout": [[""]],
            "out_dtype": [[out_dtype]],
            "kernel_size": [[str(kernel_h), str(kernel_w)]],
            "shape": [[list(output_shape)]],
            "dtype": [[dtype]],
            "padding": [[str(p) for p in padding]],
            "strides": [[str(s) for s in strides]],
        },
    }

    if has_activation:
        node["attrs"]["activation_type"] = [["relu"]]

    inputs = [
        {"op": "input", "name": "", "attrs": {"shape": [[list(shape)]], "dtype": [[str(dtype)]]}},
        {
            "op": "const",
            "name": "",
            "attrs": {"shape": [[list(weight_shape)]], "dtype": [[str(dtype)]]},
        },
    ]

    if has_bias:
        bias_dtype = dtype
        inputs.append(
            {
                "op": "const",
                "name": "",
                "attrs": {
                    "shape": [[[1, weight_shape[1] if is_depthwise else weight_shape[0], 1, 1]]],
                    "dtype": [[bias_dtype]],
                },
            }
        )

    input_idx = 0
    for _ in range(len(inputs)):
        node["inputs"].append([input_idx, 0, 0])
        input_idx += 1
    node["attrs"]["num_inputs"] = str(len(inputs))
    inputs.append(node)
    return inputs


@pytest.mark.parametrize("dtype", ["float32"])
@tvm.testing.requires_openclml
def test_conv2d(device, dtype):
    trials = [
        # Normal convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, False, False), False],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (15, 16, 12), (True, False, True), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, True), False],
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, False, False), False],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (16, 12, 15), (False, False, True), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False), False],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (16, 12, 15), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False), False],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True), False],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False), False],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (14, 10, 10), (False, True, True), False],
        # Depth-wise convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, True), True],
        [5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), (False, True, False), True],
        [3, 3, (2, 2), (2, 2), (1, 1), 14, (14, 10, 10), (False, False, False), True],
        [5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, False), True],
        [3, 3, (1, 1), (2, 2), (1, 1), 14, (14, 10, 10), (False, True, True), True],
    ]

    for (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape,
        composite,
        is_depthwise,
    ) in trials:
        shape = (1, *shape)
        if is_depthwise:
            groups = shape[1]
        else:
            groups = 1
        outputs = []
        inputs = {
            "a": tvm.nd.array(np.random.uniform(-1, 1, shape).astype(dtype)),
        }

        func, params = _get_conv_model(
            shape,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            groups,
            dtype,
            out_channels,
            inputs,
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        opencl_out = build_and_run(func, inputs, 1, params, device, enable_clml=False)[0]
        clml_out = build_and_run(func, inputs, 1, params, device, enable_clml=True)[0]

        tvm.testing.assert_allclose(
            clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-5, atol=1e-5
        )
        args = (shape, kernel_h, kernel_w, pad, stride, dilation, groups, dtype, out_channels)
        exp_codegen = _get_conv_expected_codegen(
            *args, has_bias=composite[1], has_activation=composite[2]
        )
        verify_codegen(func, exp_codegen, device, params)


@pytest.mark.parametrize("dtype", ["float16"])
@tvm.testing.requires_openclml
def test_batchnorm(device, dtype):
    if tvm.support.libinfo().get("TVM_CLML_VERSION", 2) < 3:
        print("Skip due to unsupported CLML version")
        return
    in_shape = (1, 8, 64, 64)
    channels = 8

    input_arr = tvm.nd.array(np.random.uniform(-1, 1, in_shape).astype(dtype))
    inp = relay.var("a", shape=in_shape, dtype=dtype)
    gamma_arr = tvm.nd.array(np.random.uniform(-1, 1, (channels)).astype(dtype))
    beta_arr = tvm.nd.array(np.random.uniform(-1, 1, (channels)).astype(dtype))
    gamma = relay.const(gamma_arr, dtype)
    beta = relay.const(beta_arr, dtype)

    mean_arr = tvm.nd.array(np.mean(input_arr.asnumpy(), axis=(0, 2, 3), keepdims=False))
    mean = relay.const(mean_arr)
    variance_arr = tvm.nd.array(np.var(input_arr.asnumpy(), axis=(0, 2, 3), keepdims=False))
    variance = relay.const(variance_arr)

    params = {}

    func = relay.nn.batch_norm(inp, gamma, beta, mean, variance, axis=1, epsilon=0.0001)[0]
    mod = IRModule.from_expr(func)

    inputs = {
        "a": input_arr,
    }

    opencl_out = build_and_run(mod, inputs, 1, params, device, enable_clml=False)[0]
    clml_out = build_and_run(mod, inputs, 1, params, device, enable_clml=True)[0]

    tvm.testing.assert_allclose(
        clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dtype", ["float16"])
@tvm.testing.requires_openclml
def test_concat(device, dtype):
    in_shape_1 = (1, 16, 16, 16)
    in_shape_2 = (1, 16, 16, 16)
    a = relay.var("input_1", shape=in_shape_1, dtype=dtype)
    b = relay.var("input_2", shape=in_shape_2, dtype=dtype)
    low, high = -1, 1
    inputs = {
        "input_1": tvm.nd.array(np.random.uniform(-1, 1, in_shape_1).astype(dtype)),
        "input_2": tvm.nd.array(np.random.uniform(-1, 1, in_shape_2).astype(dtype)),
    }

    params = {}
    func = relay.concatenate((a, b), axis=1)
    mod = IRModule.from_expr(func)

    opencl_out = build_and_run(mod, inputs, 1, params, device, enable_clml=False)[0]
    clml_out = build_and_run(mod, inputs, 1, params, device, enable_clml=True)[0]

    tvm.testing.assert_allclose(
        clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-3, atol=1e-3
    )
    exp_codegen = [
        {
            "attrs": {
                "dtype": [[dtype]],
                "shape": [[list(in_shape_1)]],
            },
            "name": "",
            "op": "input",
        },
        {
            "attrs": {
                "dtype": [[dtype]],
                "shape": [[list(in_shape_2)]],
            },
            "name": "",
            "op": "input",
        },
        {
            "attrs": {
                "axis": [["1"]],
                "dtype": [[dtype]],
                "num_inputs": "2",
                "num_outputs": "1",
                "shape": [[list(clml_out[0].shape)]],
            },
            "inputs": [[0, 0, 0], [1, 0, 0]],
            "name": "concatenate",
            "op": "kernel",
        },
    ]
    verify_codegen(func, exp_codegen, device, params)


def _get_pool_expected_codegen(input_shape, pool_size, stride, padding, pool_type, dtype):
    import math

    pool_height = math.floor(((input_shape[2] + padding[2] - pool_size[0]) / stride[0]) + 1)
    pool_width = math.floor(((input_shape[3] + padding[3] - pool_size[1]) / stride[1]) + 1)
    output_shape = [input_shape[0], input_shape[1], pool_height, pool_width]
    attrs = {
        "ceil_mode": [["0"]],
        "dilation": [["1", "1"]],
        "layout": [["NCHW"]],
        "num_inputs": "1",
        "num_outputs": "1",
        "out_layout": [[""]],
        "padding": [[str(p) for p in padding]],
        "pool_size": [[str(p) for p in pool_size]],
        "shape": [[list(output_shape)]],
        "dtype": [[dtype]],
        "strides": [[str(s) for s in stride]],
    }
    if sum(padding):
        attrs["count_include_pad"] = [["0"]]

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [[list(input_shape)]], "dtype": [[str(dtype)]]},
        },
        {
            "op": "kernel",
            "name": "nn.avg_pool2d" if pool_type == "avg" else "nn.max_pool2d",
            "inputs": [[0, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


@pytest.mark.parametrize("dtype", ["float16"])
@tvm.testing.requires_openclml
def test_pool(device, dtype):
    trials = [
        # input size         pool_size stride  paading
        [(1, 64, 147, 147), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 192, 71, 71), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 288, 35, 35), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 768, 17, 17), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 2048, 17, 17), (3, 3), (2, 2), (0, 0, 0, 0), "max"],
        [(1, 192, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
        [(1, 256, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
        [(1, 288, 35, 35), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
        [(1, 768, 17, 17), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
        [(1, 1280, 8, 8), (3, 3), (1, 1), (0, 0, 1, 1), "avg"],
    ]
    params = {}
    for (
        input_shape,
        pool_size,
        stride,
        padding,
        pooling_type,
    ) in trials:
        a = relay.var("input_1", shape=input_shape, dtype=dtype)
        input_arr = tvm.nd.array(np.random.uniform(-1, 1, input_shape).astype(dtype))
        inputs = {
            "input_1": input_arr,
        }

        if pooling_type == "max":
            func = relay.nn.max_pool2d(a, pool_size=pool_size, strides=stride, padding=padding)
        else:
            func = relay.nn.avg_pool2d(a, pool_size=pool_size, strides=stride, padding=padding)
        mod = IRModule.from_expr(func)

        opencl_out = build_and_run(mod, inputs, 1, params, device, enable_clml=False)[0]
        clml_out = build_and_run(mod, inputs, 1, params, device, enable_clml=True)[0]
        tvm.testing.assert_allclose(
            clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-3, atol=1e-3
        )

        args = (input_shape, pool_size, stride, padding, pooling_type, dtype)
        exp_codegen = _get_pool_expected_codegen(*args)
        verify_codegen(func, exp_codegen, device, params)


@pytest.mark.parametrize("dtype", ["float32"])
@tvm.testing.requires_openclml
def test_dense(device, dtype):
    def _get_model(x_shape, k_shape, has_bias=False):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
        out = relay.nn.dense(x, kernel, units=k_shape[0])
        params = {"kernel": tvm.nd.array(np.random.uniform(-1, 1, k_shape).astype(dtype))}
        inputs = {"x": tvm.nd.array(np.random.uniform(-1, 1, x_shape).astype(dtype))}
        exp_codegen = [
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[list(x_shape)]],
                },
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[list(k_shape)]],
                },
                "name": "",
                "op": "const",
            },
        ]
        if has_bias:
            bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
            out = relay.nn.bias_add(out, bias)
            bias_node = {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[list((1, k_shape[0]))]],
                },
                "name": "",
                "op": "const",
            }
            exp_codegen.append(bias_node)
            params["bias"] = tvm.nd.array(np.random.uniform(-1, 1, (k_shape[0],)).astype(dtype))

        dense_node = {
            "attrs": {
                "num_inputs": "3" if has_bias else "2",
                "num_outputs": "1",
                "dtype": [[dtype]],
                "out_dtype": [[""]],
                "shape": [[[x_shape[0], k_shape[0]]]],
                "units": [[str(k_shape[0])]],
            },
            "inputs": [[0, 0, 0], [1, 0, 0], [2, 0, 0]] if has_bias else [[0, 0, 0], [1, 0, 0]],
            "name": "nn.dense",
            "op": "kernel",
        }
        exp_codegen.append(dense_node)
        return out, params, inputs, exp_codegen

    def _verify(out, params, inputs, exp_codegen):
        mod = IRModule.from_expr(out)
        opencl_out = build_and_run(mod, inputs, 1, params, device, enable_clml=False)[0]
        clml_out = build_and_run(mod, inputs, 1, params, device, enable_clml=True)[0]
        tvm.testing.assert_allclose(
            clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-3, atol=1e-3
        )
        verify_codegen(out, exp_codegen, device, params)

    _verify(*(_get_model((1, 16), (32, 16))))
    _verify(*(_get_model((1, 16), (32, 16), True)))


@pytest.mark.parametrize("dtype", ["float32"])
@tvm.testing.requires_openclml
def test_binary_ops(device, dtype):
    def _get_model(a_shape, b_shape, op):
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        b = relay.var("b", shape=(b_shape), dtype=dtype)
        out = op(a, b)
        inputs = {
            "a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype)),
            "b": tvm.nd.array(np.random.uniform(-1, 1, b_shape).astype(dtype)),
        }
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        opencl_out = build_and_run(mod, inputs, 1, params, device, enable_clml=False)[0]
        clml_out = build_and_run(mod, inputs, 1, params, device, enable_clml=True)[0]
        tvm.testing.assert_allclose(
            clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-3, atol=1e-3
        )

        # Check to make sure these ops are offloaded to CLML instead of TVM.
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            mod = clml.partition_for_clml(mod, params)
            tvm_op_count = get_cpu_op_count(mod)
            assert tvm_op_count == 0, "Got {} TVM Native Compute partitions, expected 0".format(
                tvm_op_count
            )

    _verify(*(_get_model((1, 16), (1, 16), relay.add)))
    _verify(*(_get_model((1, 16), (1, 16), relay.subtract)))
    _verify(*(_get_model((1, 16), (1, 16), relay.multiply)))
    _verify(*(_get_model((1, 16), (1, 16), relay.divide)))
    _verify(*(_get_model((1, 16), (1, 16), relay.minimum)))
    _verify(*(_get_model((1, 16), (1, 16), relay.maximum)))


@pytest.mark.parametrize("dtype", ["float32"])
@tvm.testing.requires_openclml
def test_unary_ops(device, dtype):
    def _get_model(a_shape, op):
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        out = op(a)
        inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype))}
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        opencl_out = build_and_run(mod, inputs, 1, params, device, enable_clml=False)[0]
        clml_out = build_and_run(mod, inputs, 1, params, device, enable_clml=True)[0]
        tvm.testing.assert_allclose(
            clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-3, atol=1e-3
        )

        # Check to make sure these ops are offloaded to CLML instead of TVM.
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            mod = clml.partition_for_clml(mod, params)
            tvm_op_count = get_cpu_op_count(mod)
            assert tvm_op_count == 0, "Got {} TVM Native Compute partitions, expected 0".format(
                tvm_op_count
            )

    _verify(*(_get_model((1, 16), relay.nn.softmax)))
    _verify(*(_get_model((1, 16), relay.nn.relu)))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@tvm.testing.requires_openclml
def test_depth_to_space(device, dtype):
    def _get_model(a_shape, block_size):
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        out = relay.nn.depth_to_space(a, block_size)
        inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype))}
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        opencl_out = build_and_run(mod, inputs, 1, params, device, enable_clml=False)[0]
        clml_out = build_and_run(mod, inputs, 1, params, device, enable_clml=True)[0]
        tvm.testing.assert_allclose(
            clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-3, atol=1e-3
        )

        # Check to make sure these ops are offloaded to CLML instead of TVM.
        exp_codegen = [
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[list(inputs["a"].shape)]],
                },
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "block_size": [[str(int(out.attrs.block_size))]],
                    "layout": [["NCHW"]],
                    "mode": [["DCR"]],
                    "dtype": [[dtype]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "shape": [[list(clml_out[0].shape)]],
                },
                "inputs": [[0, 0, 0]],
                "name": "nn.depth_to_space",
                "op": "kernel",
            },
        ]
        verify_codegen(out, exp_codegen, device, params)

    _verify(*(_get_model((1, 64, 8, 8), 4)))
    _verify(*(_get_model((1, 64, 8, 8), 8)))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@tvm.testing.requires_openclml
def test_resize_bilinear(device, dtype):
    def _get_model(a_shape, scale, align_corners):
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        out = relay.nn.upsampling(
            a, scale_h=scale[0], scale_w=scale[1], method="bilinear", align_corners=align_corners
        )
        inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype))}
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        opencl_out = build_and_run(mod, inputs, 1, params, device, enable_clml=False)[0]
        clml_out = build_and_run(mod, inputs, 1, params, device, enable_clml=True)[0]
        tvm.testing.assert_allclose(
            clml_out[0].asnumpy(), opencl_out[0].asnumpy(), rtol=1e-3, atol=1e-3
        )

        # Check to make sure these ops are offloaded to CLML instead of TVM.
        exp_codegen = [
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[list(inputs["a"].shape)]],
                },
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "scale_h": [[str(int(out.attrs.scale_h))]],
                    "scale_w": [[str(int(out.attrs.scale_w))]],
                    "layout": [["NCHW"]],
                    "method": [[out.attrs.method]],
                    "align_corners": [[str(out.attrs.align_corners)]],
                    "dtype": [[dtype]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "shape": [[list(clml_out[0].shape)]],
                },
                "inputs": [[0, 0, 0]],
                "name": "nn.upsampling",
                "op": "kernel",
            },
        ]
        verify_codegen(out, exp_codegen, device, params)

    _verify(*(_get_model((1, 16, 8, 8), (2, 2), False)))
    _verify(*(_get_model((1, 16, 7, 7), (2, 2), True)))


if __name__ == "__main__":
    tvm.testing.main()
