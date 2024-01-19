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
    build_and_run_vm,
    verify_codegen,
)
import pytest


executor_type = tvm.testing.parameter("ge", "vm")


def _build_and_run_network(remote, mod, params, input_data, target, executor_type, tvm_log=""):
    """Helper function to build and run a network."""

    outputs = []
    for clml in [True, False]:
        if executor_type == "ge":
            outputs.append(
                build_and_run(
                    remote,
                    mod,
                    params,
                    input_data,
                    target,
                    enable_clml=clml,
                    stat_file=tvm_log,
                )
            )
        else:
            outputs.append(
                build_and_run_vm(
                    remote,
                    mod,
                    params,
                    input_data,
                    target,
                    enable_clml=clml,
                    stat_file=tvm_log,
                )
            )
    return outputs


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


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
        # Normal convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False), False],
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (16, 10, 10), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (16, 10, 10), (False, False, False), False],
        [5, 5, (1, 1), (1, 1), (1, 1), 4, (6, 256, 256), (True, True, True), False],
        [3, 3, (0, 0), (1, 1), (1, 1), 4, (4, 512, 512), (False, True, False), False],
        [3, 3, (1, 1), (1, 1), (1, 1), 8, (6, 512, 512), (False, True, False), False],
        [1, 3, (0, 0), (1, 1), (1, 1), 16, (16, 20, 20), (False, False, True), False],
        [3, 1, (0, 0), (1, 1), (1, 1), 64, (64, 20, 20), (False, False, True), False],
        # [3, 3, (1, 1), (1, 1), (1, 1), 128, (128, 16, 16), (False, True, False), False],
        # [3, 3, (1, 1), (2, 2), (1, 1), 256, (128, 16, 16), (False, True, True), False],
        # Depth-wise convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 11, (11, 20, 20), (False, False, True), True],
        [5, 5, (2, 2), (1, 1), (1, 1), 32, (32, 20, 20), (False, True, False), True],
        [3, 3, (2, 2), (2, 2), (1, 1), 128, (128, 8, 8), (False, False, False), True],
        [5, 5, (0, 0), (1, 1), (1, 1), 64, (64, 32, 32), (False, False, False), True],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (16, 256, 256), (False, True, True), True],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d(remote, dtype, target, trials, executor_type):
    np.random.seed(0)

    (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape,
        composite,
        is_depthwise,
    ) = trials

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
    outputs = _build_and_run_network(remote, func, params, inputs, target, executor_type)
    out_tol = 1e-1 if dtype == "float16" else 1e-5
    tvm.testing.assert_allclose(
        outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
    )
    args = (shape, kernel_h, kernel_w, pad, stride, dilation, groups, dtype, out_channels)
    exp_codegen = _get_conv_expected_codegen(
        *args, has_bias=composite[1], has_activation=composite[2]
    )
    verify_codegen(remote, func, params, exp_codegen, target)


def _get_conv2d_transpose_expected_codegen(
    dshape, kshape, channels, kernel_size, strides, padding, dilation, dtype, output_shape
):
    attrs = {
        "channels": [[str(channels)]],
        "data_layout": [["NCHW"]],
        "kernel_layout": [["OIHW"]],
        "groups": [["1"]],
        "dilation": [[str(p) for p in dilation]],
        "num_inputs": "2",
        "num_outputs": "1",
        "padding": [[str(p) for p in padding]],
        "kernel_size": [[str(p) for p in kernel_size]],
        "shape": [[list(output_shape)]],
        "dtype": [[dtype]],
        "strides": [[str(s) for s in strides]],
        "out_dtype": [[""]],
        "out_layout": [[""]],
        "output_padding": [["0", "0"]],
    }

    kshape = [kshape[1], kshape[0], kshape[2], kshape[3]]

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [[list(dshape)]], "dtype": [[str(dtype)]]},
        },
        {
            "op": "const",
            "name": "",
            "attrs": {"shape": [[list(kshape)]], "dtype": [[str(dtype)]]},
        },
        {
            "op": "kernel",
            "name": "nn.conv2d_transpose",
            "inputs": [[0, 0, 0], [1, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 64, 200, 200), (64, 64, 4, 4), 64, (4, 4), (2, 2), (1, 1, 1, 1)],
        [(1, 64, 400, 400), (64, 16, 4, 4), 16, (4, 4), (2, 2), (1, 1, 1, 1)],
        [(1, 16, 32, 32), (16, 16, 3, 3), 16, (3, 3), (1, 1), (1, 1, 1, 1)],
        # [(1, 256, 100, 100), (256, 64, 4, 4), 64, (4, 4), (2, 2), (1, 1, 1, 1)],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_transpose(remote, dtype, target, trials, executor_type):
    np.random.seed(0)
    (dshape, kshape, channels, kernel_size, strides, padding) = trials
    x = relay.var("input", shape=dshape, dtype=dtype)
    input_arr = tvm.nd.array(np.random.uniform(-1, 1, dshape).astype(dtype))
    w = relay.var("wt", shape=kshape, dtype=dtype)
    weight_arr = tvm.nd.array(np.random.uniform(-1, 1, kshape).astype(dtype))
    inputs = {
        "input": input_arr,
    }
    params = {
        "wt": weight_arr,
    }
    y = relay.nn.conv2d_transpose(
        x,
        w,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_layout="IOHW",
        data_layout="NCHW",
    )
    func = relay.Function([x, w], y)
    mod = IRModule.from_expr(func)
    outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
    out_tol = 1e-1 if dtype == "float16" else 1e-5
    tvm.testing.assert_allclose(
        outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
    )
    args = (
        dshape,
        kshape,
        channels,
        kernel_size,
        strides,
        padding,
        (1, 1),
        dtype,
        outputs[0].shape,
    )
    exp_codegen = _get_conv2d_transpose_expected_codegen(*args)
    verify_codegen(remote, mod, params, exp_codegen, target)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("trials", [[1, 64, 8, 8], [1, 16, 64, 64]])
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_batchnorm(remote, dtype, target, trials, executor_type):
    if clml.clml_sdk_version() < 3:
        print("Skip due to unsupported CLML version:", clml.clml_sdk_version())
        return
    in_shape = trials
    channels = in_shape[1]

    np.random.seed(0)

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

    func = relay.nn.batch_norm(inp, gamma, beta, mean, variance, axis=1, epsilon=0.0003)[0]
    mod = IRModule.from_expr(func)
    inputs = {
        "a": input_arr,
    }
    outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
    out_tol = 1e-3 if dtype == "float16" else 1e-5
    tvm.testing.assert_allclose(
        outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
    )
    exp_codegen = [
        {
            "attrs": {"dtype": [[dtype]], "shape": [[list(inputs["a"].shape)]]},
            "name": "",
            "op": "input",
        },
        {"attrs": {"dtype": [[dtype]], "shape": [[[channels]]]}, "name": "", "op": "const"},
        {"attrs": {"dtype": [[dtype]], "shape": [[[channels]]]}, "name": "", "op": "const"},
        {"attrs": {"dtype": [[dtype]], "shape": [[[channels]]]}, "name": "", "op": "const"},
        {"attrs": {"dtype": [[dtype]], "shape": [[[channels]]]}, "name": "", "op": "const"},
        {
            "attrs": {
                "axis": [["1"]],
                "center": [["1"]],
                "dtype": [[dtype]],
                "epsilon": [["0.00029999999999999997"]],
                "num_inputs": "5",
                "num_outputs": "1",
                "scale": [["1"]],
                "shape": [[list(outputs[0].shape)]],
            },
            "inputs": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            "name": "nn.batch_norm",
            "op": "kernel",
        },
    ]
    verify_codegen(remote, mod, params, exp_codegen, target)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 64, 64, 40), (1, 64, 64, 40)],
        [(1, 1280, 32, 32), (1, 640, 32, 32)],
        [(1, 64), (1, 32)],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_concat(remote, dtype, target, trials, executor_type):
    np.random.seed(0)
    in_shape_1 = trials[0]
    in_shape_2 = trials[1]
    a = relay.var("input_1", shape=in_shape_1, dtype=dtype)
    b = relay.var("input_2", shape=in_shape_2, dtype=dtype)
    low, high = -1, 1
    inputs = {
        "input_1": tvm.nd.array(np.random.uniform(-1, 1, in_shape_1).astype(dtype)),
        "input_2": tvm.nd.array(np.random.uniform(-1, 1, in_shape_2).astype(dtype)),
    }

    params = {}
    func = relay.concatenate((a, b), axis=1)

    outputs = _build_and_run_network(remote, func, params, inputs, target, executor_type)
    out_tol = 1e-2 if dtype == "float16" else 1e-5
    tvm.testing.assert_allclose(
        outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
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
                "shape": [[list(outputs[0].shape)]],
            },
            "inputs": [[0, 0, 0], [1, 0, 0]],
            "name": "concatenate",
            "op": "kernel",
        },
    ]
    verify_codegen(remote, func, params, exp_codegen, target)


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


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
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
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_pool(remote, dtype, target, trials, executor_type):
    np.random.seed(0)
    params = {}
    (
        input_shape,
        pool_size,
        stride,
        padding,
        pooling_type,
    ) = trials
    a = relay.var("input_1", shape=input_shape, dtype=dtype)
    input_arr = tvm.nd.array(np.random.uniform(-1, 1, input_shape).astype(dtype))
    inputs = {
        "input_1": input_arr,
    }
    if pooling_type == "max":
        func = relay.nn.max_pool2d(a, pool_size=pool_size, strides=stride, padding=padding)
    else:
        func = relay.nn.avg_pool2d(a, pool_size=pool_size, strides=stride, padding=padding)

    outputs = _build_and_run_network(remote, func, params, inputs, target, executor_type)
    out_tol = 1e-2 if dtype == "float16" else 1e-5
    tvm.testing.assert_allclose(
        outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
    )
    args = (input_shape, pool_size, stride, padding, pooling_type, dtype)
    exp_codegen = _get_pool_expected_codegen(*args)
    verify_codegen(remote, func, params, exp_codegen, target)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
        [(5, 16), (32, 16), False],
        [(320, 64), (320, 64), False],
        [(256, 256), (256, 256), False],
        [(512, 512), (512, 512), False],
        [(1, 256), (100, 256), False],
        [(1, 16), (32, 16), True],
        [(1, 512), (512, 512), True],
        [(1, 5), (4, 5), True],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_dense(remote, dtype, target, trials, executor_type):
    def _get_model(x_shape, k_shape, has_bias=False):
        np.random.seed(0)
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
        input_nodes = [[0, 0, 0], [1, 0, 0]]
        num_inputs = 2
        if has_bias:
            bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
            out = relay.nn.bias_add(out, bias)
            bias_data_node = {
                "attrs": {
                    "dtype": [[dtype]],
                    "shape": [[list((1, k_shape[0]))]],
                },
                "name": "",
                "op": "const",
            }
            exp_codegen.append(bias_data_node)
            input_nodes.append([2, 0, 0])
            num_inputs += 1
            params["bias"] = tvm.nd.array(np.random.uniform(-1, 1, (k_shape[0],)).astype(dtype))

        dense_node = {
            "attrs": {
                "num_inputs": str(num_inputs),
                "num_outputs": "1",
                "dtype": [[dtype]],
                "out_dtype": [[""]],
                "shape": [[[x_shape[0], k_shape[0]]]],
                "units": [[str(k_shape[0])]],
            },
            "inputs": input_nodes,
            "name": "nn.dense",
            "op": "kernel",
        }
        exp_codegen.append(dense_node)

        return out, params, inputs, exp_codegen

    def _verify(out, params, inputs, exp_codegen):
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-1 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(*(_get_model(trials[0], trials[1], trials[2])))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_binary_ops(remote, dtype, target, executor_type):
    def _get_model(a_shape, b_shape, op_func):
        np.random.seed(0)
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        b = relay.var("b", shape=(b_shape), dtype=dtype)
        out = op_func(a, b)
        inputs = {
            "a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype)),
            "b": tvm.nd.array(np.random.uniform(-1, 1, b_shape).astype(dtype)),
        }
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-2 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )
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
                    "dtype": [[dtype]],
                    "shape": [[list(inputs["b"].shape)]],
                },
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "num_inputs": "2",
                    "num_outputs": "1",
                    "shape": [[list(outputs[0].shape)]],
                },
                "inputs": [[0, 0, 0], [1, 0, 0]],
                "name": str(out.op.name),
                "op": "kernel",
            },
        ]
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(*(_get_model((1, 16), (1, 16), relay.add)))
    _verify(*(_get_model((1, 18), (1, 18), relay.subtract)))
    _verify(*(_get_model((1, 256), (1, 256), relay.multiply)))
    _verify(*(_get_model((1, 10), (1, 10), relay.divide)))
    _verify(*(_get_model((1, 16), (1, 16), relay.minimum)))
    _verify(*(_get_model((1, 512), (1, 512), relay.maximum)))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_unary_ops(remote, dtype, target, executor_type):
    def _get_model(a_shape, op):
        np.random.seed(0)
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        out = op(a)
        inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype))}
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-2 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )

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
                    "dtype": [[dtype]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "shape": [[list(outputs[0].shape)]],
                },
                "inputs": [[0, 0, 0]],
                "name": "nn.relu",
                "op": "kernel",
            },
        ]
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(*(_get_model((1, 16), relay.nn.relu)))
    _verify(*(_get_model((1, 256), relay.nn.relu)))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depth_to_space(remote, dtype, target, executor_type):
    def _get_model(a_shape, block_size):
        np.random.seed(0)
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        out = relay.nn.depth_to_space(a, block_size)
        inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype))}
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-2 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )

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
                    "shape": [[list(outputs[0].shape)]],
                },
                "inputs": [[0, 0, 0]],
                "name": "nn.depth_to_space",
                "op": "kernel",
            },
        ]
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(*(_get_model((1, 64, 8, 8), 4)))
    _verify(*(_get_model((1, 64, 8, 8), 8)))
    _verify(*(_get_model((1, 512, 8, 8), 8)))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_resize_bilinear(remote, dtype, target, executor_type):
    def _get_model(a_shape, scale, align_corners):
        np.random.seed(0)
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        out = relay.nn.upsampling(
            a, scale_h=scale[0], scale_w=scale[1], method="bilinear", align_corners=align_corners
        )
        inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype))}
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-2 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )

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
                    "shape": [[list(outputs[0].shape)]],
                },
                "inputs": [[0, 0, 0]],
                "name": "nn.upsampling",
                "op": "kernel",
            },
        ]
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(*(_get_model((1, 16, 8, 8), (2, 2), False)))
    _verify(*(_get_model((1, 16, 7, 7), (2, 2), True)))
    _verify(*(_get_model((1, 64, 8, 8), (2, 2), True)))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 512, 32), (1, 512, 32), False, True],
        [(1, 128, 32), (1, 128, 32), False, True],
        [(1, 128, 128), (1, 32, 128), False, True],
        [(1, 64, 40), (1, 64, 40), False, True],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_batch_matmul(remote, dtype, target, executor_type, trials):
    def _get_model(a_shape, b_shape, a_transpose, b_transpose):
        np.random.seed(0)
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        b = relay.var("b", shape=(b_shape), dtype=dtype)
        out = relay.nn.batch_matmul(a, b, transpose_a=a_transpose, transpose_b=b_transpose)
        inputs = {
            "a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype)),
            "b": tvm.nd.array(np.random.uniform(-1, 1, b_shape).astype(dtype)),
        }
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-1 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )

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
                    "dtype": [[dtype]],
                    "shape": [[list(inputs["b"].shape)]],
                },
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "transpose_a": [[str(int(out.attrs.transpose_a))]],
                    "transpose_b": [[str(int(out.attrs.transpose_b))]],
                    "out_dtype": [[""]],
                    "dtype": [[dtype]],
                    "num_inputs": "2",
                    "num_outputs": "1",
                    "shape": [[list(outputs[0].shape)]],
                },
                "inputs": [[0, 0, 0], [1, 0, 0]],
                "name": "nn.batch_matmul",
                "op": "kernel",
            },
        ]
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(*(_get_model(trials[0], trials[1], trials[2], trials[3])))


def _get_softmax_exp_codegen(inputs, dtype, output_shape, axis):

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
                "axis": [[str(axis)]],
                "dtype": [[dtype]],
                "num_inputs": "1",
                "num_outputs": "1",
                "shape": [[list(output_shape)]],
            },
            "inputs": [[0, 0, 0]],
            "name": "nn.softmax",
            "op": "kernel",
        },
    ]
    return exp_codegen


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_softmax(remote, dtype, target, executor_type):
    def _get_model(a_shape, axis):
        np.random.seed(0)
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype))}
        out = relay.nn.softmax(a, axis)
        params = {}
        return out, params, inputs, axis

    def _verify(out, params, inputs, axis, out_tol):
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].numpy(), rtol=out_tol, atol=out_tol
        )
        args = (inputs, dtype, outputs[0].shape, axis)
        exp_codegen = _get_softmax_exp_codegen(*args)
        verify_codegen(remote, mod, params, exp_codegen, target)

    # 2D Tensor  TEST CASES
    _verify(*(_get_model((1, 5), 1)), 1e-3)
    _verify(*(_get_model((1, 16), 1)), 1e-3)
    _verify(*(_get_model((1, 1000), -1)), 1e-3)

    # 4D Tensor  TEST CASES  layout = NCHW
    _verify(*(_get_model((1, 100, 64, 100), 1)), 1e-3)
    _verify(*(_get_model((1, 64, 64, 64), 1)), 1e-3)
    _verify(*(_get_model((1, 5, 3, 4), 1)), 1e-3)

    # 4D Tensor  TEST CASES  layout = NHWC
    _verify(*(_get_model((1, 64, 100, 100), 3)), 1e-1)
    _verify(*(_get_model((1, 100, 100, 100), 3)), 1e-1)
    _verify(*(_get_model((1, 64, 5, 32), -1)), 1e-1)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 1, 2, 2), 2, 1],
        [(1, 16, 2, 2), 4, 4],
        [(1, 8, 4, 4), 3, 2],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_upsampling(remote, dtype, target, executor_type, trials):
    def _verify(in_shape, scale_h, scale_w):
        np.random.seed(0)
        a = relay.var("a", shape=in_shape, dtype=dtype)
        inputs = {
            "a": tvm.nd.array(np.random.uniform(-1, 1, in_shape).astype(dtype)),
        }
        params = {}
        func = relay.nn.upsampling(
            a, scale_h, scale_w, layout="NCHW", method="bilinear", align_corners=False
        )
        mod = IRModule.from_expr(func)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-2 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )
        exp_codegen = [
            {
                "attrs": {"dtype": [[dtype]], "shape": [[list(inputs["a"].shape)]]},
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "align_corners": [["0"]],
                    "dtype": [[dtype]],
                    "layout": [["NCHW"]],
                    "method": [["bilinear"]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "scale_h": [[str(scale_h)]],
                    "scale_w": [[str(scale_w)]],
                    "shape": [[list(outputs[0].shape)]],
                },
                "inputs": [[0, 0, 0]],
                "name": "nn.upsampling",
                "op": "kernel",
            },
        ]
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(trials[0], trials[1], trials[2])


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 40, 64, 64), (1, 40, 4096)],
        [(1, 77, 768), (1, 1, -1, 768)],
        [(1, 80, 32, 32), (1, 80, 1024)],
        [(1, 2, 3, 4), (1, 0, -1)],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_reshape(remote, dtype, target, executor_type, trials):
    def _verify(shape, newshape):
        np.random.seed(0)
        x = relay.var("x", shape=(shape), dtype=dtype)
        # Defined the test case with unary operator
        # Single reshape op is failing in native OpenCL with vm executor type
        # Empty TVM mod in VM doesn't pick appropriate cross compiler
        out = relay.nn.relu(x)
        out = relay.reshape(out, newshape)

        inputs = {"x": tvm.nd.array(np.random.uniform(-1, 1, shape).astype(dtype))}
        params = {}
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-3 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )
        exp_codegen = [
            {
                "attrs": {"dtype": [[dtype]], "shape": [[list(inputs["x"].shape)]]},
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "shape": [[list(inputs["x"].shape)]],
                },
                "inputs": [[0, 0, 0]],
                "name": "nn.relu",
                "op": "kernel",
            },
            {
                "attrs": {
                    "allowzero": [["0"]],
                    "dtype": [[dtype]],
                    "newshape": [[str(ele) for ele in list(newshape)]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "shape": [[list(outputs[0].shape)]],
                },
                "inputs": [[1, 0, 0]],
                "name": "reshape",
                "op": "kernel",
            },
        ]
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(trials[0], trials[1])


def _get_pool_global_expected_codegen(input_shape, pool_type, dtype, out_shape):

    exp_codegen = [
        {
            "attrs": {
                "dtype": [[str(dtype)]],
                "shape": [[list(input_shape)]],
            },
            "name": "",
            "op": "input",
        },
        {
            "attrs": {
                "dtype": [[str(dtype)]],
                "layout": [["NCHW"]],
                "num_inputs": "1",
                "num_outputs": "1",
                "out_layout": [[""]],
                "shape": [[list(out_shape)]],
            },
            "inputs": [[0, 0, 0]],
            "name": "nn.global_avg_pool2d" if pool_type == "avg" else "nn.global_max_pool2d",
            "op": "kernel",
        },
    ]
    return exp_codegen


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "trials",
    [
        [(1, 3, 32, 32), "avg"],
        [(1, 64, 147, 147), "max"],
        [(1, 192, 71, 71), "max"],
        [(1, 288, 35, 35), "max"],
        [(1, 768, 17, 17), "max"],
        [(1, 2048, 17, 17), "max"],
        [(1, 192, 35, 35), "avg"],
        [(1, 256, 35, 35), "avg"],
        [(1, 288, 35, 35), "avg"],
        [(1, 768, 17, 17), "avg"],
        [(1, 1280, 8, 8), "avg"],
    ],
)
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_pool_global(remote, dtype, target, executor_type, trials):
    params = {}
    (input_shape, pooling_type) = trials
    np.random.seed(0)
    a = relay.var("a", shape=input_shape, dtype=dtype)
    inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, input_shape).astype(dtype))}
    if pooling_type == "max":
        func = relay.nn.global_max_pool2d(a)
    else:
        func = relay.nn.global_avg_pool2d(a)
    mod = IRModule.from_expr(func)
    outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
    out_tol = 1e-3 if dtype == "float16" else 1e-5
    tvm.testing.assert_allclose(
        outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
    )
    args = (input_shape, pooling_type, dtype, outputs[0].shape)
    exp_codegen = _get_pool_global_expected_codegen(*args)
    verify_codegen(remote, mod, params, exp_codegen, target)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@tvm.testing.requires_openclml
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_batch_flatten(remote, dtype, target, executor_type):
    def _get_model(a_shape):
        a = relay.var("a", shape=(a_shape), dtype=dtype)
        # Defined the test case with unary operator
        # Single batch_flatten op is failing in native OpenCL
        # Empty TVM mod in VM doesn't pick appropriate cross compiler
        np.random.seed(0)
        out = relay.nn.relu(a)
        out = relay.nn.batch_flatten(out)
        inputs = {"a": tvm.nd.array(np.random.uniform(-1, 1, a_shape).astype(dtype))}
        params = {}
        return out, params, inputs

    def _verify(out, params, inputs):
        mod = IRModule.from_expr(out)
        outputs = _build_and_run_network(remote, mod, params, inputs, target, executor_type)
        out_tol = 1e-3 if dtype == "float16" else 1e-5
        tvm.testing.assert_allclose(
            outputs[0].asnumpy(), outputs[1].asnumpy(), rtol=out_tol, atol=out_tol
        )
        exp_codegen = [
            {
                "attrs": {"dtype": [[dtype]], "shape": [[list(inputs["a"].shape)]]},
                "name": "",
                "op": "input",
            },
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "shape": [[list(inputs["a"].shape)]],
                },
                "inputs": [[0, 0, 0]],
                "name": "nn.relu",
                "op": "kernel",
            },
            {
                "attrs": {
                    "dtype": [[dtype]],
                    "num_inputs": "1",
                    "num_outputs": "1",
                    "shape": [[list(outputs[0].shape)]],
                },
                "inputs": [[1, 0, 0]],
                "name": "nn.batch_flatten",
                "op": "kernel",
            },
        ]
        verify_codegen(remote, mod, params, exp_codegen, target)

    _verify(*(_get_model((1, 3, 2))))
    _verify(*(_get_model((1, 4, 3, 2))))
    _verify(*(_get_model((1, 64, 8, 8))))
    _verify(*(_get_model((1, 128, 4, 4))))


if __name__ == "__main__":
    tvm.testing.main()
