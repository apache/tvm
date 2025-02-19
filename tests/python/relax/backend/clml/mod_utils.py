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
import pytest
import numpy as np
import tvm
import tvm.testing
import json

from tvm import relax, rpc
from tvm.script import relax as R
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder
from tvm.relax.backend.adreno import clml


def get_relax_conv2d_mod(
    data_shape,
    weight_shape,
    stride,
    dilation,
    padding,
    weight_layout="OIHW",
    groups=1,
    dtype="float32",
    has_bias=False,
    has_bn=False,
    has_activation=False,
    has_pad=False,
    is_depthwise=False,
):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            if has_pad:
                p = (0, 0, 0, 0, padding[0], padding[0], padding[1], padding[1])
                orig_data = R.arg("data", R.Tensor(data_shape, dtype))
                data = R.nn.pad(orig_data, pad_width=p, pad_value=0.0)
                padding = (0, 0, 0, 0)
            else:
                data = R.arg("data", R.Tensor(data_shape, dtype))
            weight = R.arg("weight", R.Tensor(weight_shape, dtype))
            if has_bias:
                bias = R.arg("bias", R.Tensor((1, weight_shape[0], 1, 1), dtype))

            is_depthwise = data_shape[1] == weight_shape[0] == groups

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d(
                        data,
                        weight,
                        out_dtype=dtype,
                        strides=stride,
                        dilation=dilation,
                        padding=padding,
                        data_layout="NCHW",
                        kernel_layout=weight_layout,
                        groups=groups,
                    )
                )
                if has_bias:
                    output = R.emit(output + bias)
                if has_bn:
                    gamma = R.arg("gamma", R.Tensor((weight_shape[0],), dtype))
                    beta = R.arg("beta", R.Tensor((weight_shape[0],), dtype))
                    mean = R.arg("mean", R.Tensor((weight_shape[0],), dtype))
                    variance = R.arg("variance", R.Tensor((weight_shape[0],), dtype))
                    output = R.emit(
                        R.nn.batch_norm(output, gamma, beta, mean, variance, axis=1, epsilon=1e-5)[
                            0
                        ]
                    )
                if has_activation:
                    output = R.emit(R.nn.relu(output))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def get_clml_conv2d_codegen(
    data_shape,
    weight_shape,
    stride,
    dilation,
    padding,
    weight_layout="OIHW",
    groups=1,
    dtype="float32",
    has_bias=False,
    has_bn=False,
    has_activation=False,
    has_pad=False,
    is_depthwise=False,
):
    kernel_h, kernel_w = weight_shape[2], weight_shape[3]
    channels = weight_shape[0]
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    output_height = ((data_shape[2] - kernel_h + padding[0] + padding[2]) / stride[0]) + 1
    output_width = ((data_shape[3] - kernel_w + padding[1] + padding[3]) / stride[1]) + 1
    output_shape = (1, channels, int(output_height), int(output_width))
    out_dtype = dtype
    is_depthwise = data_shape[1] == channels == groups

    weight_layout = "IOHW" if is_depthwise else "OIHW"
    if weight_layout == "OIHW":
        weight_shape = (channels, data_shape[1] // groups, kernel_h, kernel_w)
    else:
        weight_shape = (data_shape[1] // groups, channels, kernel_h, kernel_w)

    if is_depthwise:
        name = "openclml.nn.depthwise_conv2d"
    else:
        name = "openclml.nn.conv2d"

    node = {
        "op": "kernel",
        "name": "",
        "inputs": [],
        "attrs": {
            "groups": [[str(groups)]],
            "num_outputs": "1",
            "data_layout": [["NCHW"]],
            "kernel_layout": [[weight_layout]],
            "dilation": [[str(dilation[0]), str(dilation[1])]],
            "out_layout": [["NCHW"]],
            "out_dtype": [[out_dtype]],
            "shape": [[list(output_shape)]],
            "dtype": [[dtype]],
            "padding": [[str(p) for p in padding]],
            "strides": [[str(s) for s in stride]],
        },
    }

    if has_activation:
        node["attrs"]["activation_type"] = [["relu"]]

    nodes = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [[list(data_shape)]], "dtype": [[str(dtype)]]},
        },
    ]

    nodes.append(
        {
            "op": "const",
            "name": "",
            "attrs": {"shape": [[list(weight_shape)]], "dtype": [[str(dtype)]]},
        }
    )

    if has_bias:
        bias_dtype = dtype
        nodes.append(
            {
                "op": "const",
                "name": "",
                "attrs": {
                    "shape": [[[1, weight_shape[1] if is_depthwise else weight_shape[0], 1, 1]]],
                    "dtype": [[bias_dtype]],
                },
            }
        )

    if has_bn:
        bn_shape = [[1, weight_shape[0], 1, 1]]
        # conv2d + bn --> conv2d + Add due to OptimizeBatchNorm transformation Pass
        nodes.append(
            {
                "name": "",
                "op": "const",
                "attrs": {"dtype": [[dtype]], "shape": [[[1, weight_shape[0], 1, 1]]]},
            },
        )

    input_idx = 0
    for _ in range(len(nodes)):
        node["inputs"].append([input_idx, 0, 0])
        input_idx += 1
    node["attrs"]["num_inputs"] = str(len(nodes))
    nodes.append(node)
    return nodes


def get_relax_conv2d_transpose_mod(
    data_shape,
    weight_shape,
    channels,
    stride,
    padding,
    dtype="float32",
):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            data = R.arg("data", R.Tensor(data_shape, dtype))
            weight = R.arg("weight", R.Tensor(weight_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d_transpose(
                        data,
                        weight,
                        groups=1,
                        strides=stride,
                        padding=padding,
                        kernel_layout="OIHW",
                        data_layout="NCHW",
                    )
                )
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def get_conv2d_transpose_expected_codegen(
    dshape, kshape, channels, kernel_size, strides, padding, dilation, dtype, output_shape
):
    attrs = {
        "data_layout": [["NCHW"]],
        "kernel_layout": [["OIHW"]],
        "groups": [["1"]],
        "clml_version": [["3"]],
        "dilation": [[str(p) for p in dilation]],
        "num_inputs": "2",
        "num_outputs": "1",
        "padding": [[str(p) for p in padding]],
        "shape": [[list(output_shape)]],
        "dtype": [[dtype]],
        "strides": [[str(s) for s in strides]],
        "out_dtype": [[""]],
        "out_layout": [["NCHW"]],
        "output_padding": [["0", "0"]],
    }

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
            "name": "",
            "inputs": [[0, 0, 0], [1, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


def get_batchnorm_mod(data_shape, channels, axis, epsilon, dtype):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            data = R.arg("data", R.Tensor(data_shape, dtype))
            gamma = R.arg("gamma", R.Tensor((channels,), dtype))
            beta = R.arg("beta", R.Tensor((channels,), dtype))
            mean = R.arg("moving_mean", R.Tensor((channels,), dtype))
            variance = R.arg("moving_var", R.Tensor((channels,), dtype))
            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.batch_norm(data, gamma, beta, mean, variance, axis, epsilon)[0]
                )
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

        func = builder.get()
        return tvm.IRModule({"main": func})


def get_binary_op_mod(a_shape, b_shape, op, dtype):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            a = R.arg("a", R.Tensor(a_shape, dtype))
            b = R.arg("b", R.Tensor(b_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(op(a, b))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()

    low, high = 0, 1
    a_data = np.random.uniform(low, high, size=(a_shape)).astype(dtype)
    b_data = np.random.uniform(low, high, size=(b_shape)).astype(dtype)

    return (tvm.IRModule({"main": func}), (a_data, b_data))


def get_unary_op_mod(a_shape, op, dtype):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            a = R.arg("a", R.Tensor(a_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(op(a))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()

    low, high = 0, 1
    a_data = np.random.uniform(low, high, size=(a_shape)).astype(dtype)

    return (tvm.IRModule({"main": func}), (a_data,))


def get_relax_maxpool_mod(
    data_shape, dtype, pool_size, stride=None, dilation=(1, 1), padding=(0, 0), has_pad=False
):
    """
    Args:
        data_shape (tuple): Input tensor shape
        pool_size (tuple): Pooling window size (height, width)
        stride (tuple, optional): Stride of pooling operation. Defaults to pool_size.
        dilation (tuple, optional): Dilation rate. Defaults to (1, 1).
        padding (tuple, optional): Padding for the input tensor. Defaults to (0, 0).
        dtype (str, optional): Data type. Defaults to "float32".
        has_pad (bool, optional): Whether to apply explicit padding. Defaults to False.

    Returns:
        tvm.IRModule: Relax MaxPool module
    """
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")

            if has_pad:
                p = (0, 0, 0, 0, padding[0], padding[1], padding[0], padding[1])
                orig_data = R.arg("data", R.Tensor(data_shape, dtype))
                data = R.nn.pad(orig_data, pad_width=p, pad_value=float("-inf"))
                padding = (0, 0)
            else:
                data = R.arg("data", R.Tensor(data_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.max_pool2d(
                        data,
                        pool_size=pool_size,
                        strides=stride,
                        dilation=dilation,
                        padding=padding,
                        layout="NCHW",
                    )
                )
                R.output(output)
            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def get_maxpool_expected_codegen(input_shape, pool_size, stride, padding, pool_type, dtype):
    import math

    adjusted_input_shape = [
        input_shape[0],
        input_shape[1],
        input_shape[2] + padding[0] + padding[1],
        input_shape[3] + padding[2] + padding[3],
    ]

    pool_height = math.floor(((adjusted_input_shape[2] - pool_size[0]) / stride[0]) + 1)
    pool_width = math.floor(((adjusted_input_shape[3] - pool_size[1]) / stride[1]) + 1)
    output_shape = [adjusted_input_shape[0], adjusted_input_shape[1], pool_height, pool_width]

    attrs = {
        "ceil_mode": [["0"]],
        "clml_version": [["3"]],
        "dilation": [["1", "1"]],
        "layout": [["NCHW"]],
        "num_inputs": "1",
        "num_outputs": "1",
        "out_layout": [["NCHW"]],
        "padding": [[str(0) for p in padding]],
        "pool_size": [[str(p) for p in pool_size]],
        "shape": [[list(output_shape)]],
        "dtype": [[dtype]],
        "strides": [[str(s) for s in stride]],
        "count_include_pad": [["0"]],
    }
    if sum(padding):
        attrs["count_include_pad"] = [["0"]]

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [[list(adjusted_input_shape)]], "dtype": [[str(dtype)]]},
        },
        {
            "op": "kernel",
            "name": "",
            "inputs": [[0, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


def get_relax_avgpool_mod(data_shape, dtype, pool_size, stride, dilation, padding, has_pad):
    """
    Args:
        data_shape (tuple): Input tensor shape
        pool_size (tuple): Pooling window size (height, width)
        stride (tuple, optional): Stride of pooling operation. Defaults to pool_size.
        dilation (tuple, optional): Dilation rate. Defaults to (1, 1).
        padding (tuple, optional): Padding for the input tensor. Defaults to (0, 0).
        dtype (str, optional): Data type. Defaults to "float32".
        has_pad (bool, optional): Whether to apply explicit padding. Defaults to False.
        count_include_pad (bool, optional): Whether to include padding in averaging. Defaults to True.

    Returns:
        tvm.IRModule: Relax AvgPool module
    """
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")

            if has_pad:
                p = (0, 0, 0, 0, padding[0], padding[1], padding[0], padding[1])
                orig_data = R.arg("data", R.Tensor(data_shape, dtype))
                data = R.nn.pad(orig_data, pad_width=p, pad_value=0.0)
                padding = (0, 0)
            else:
                data = R.arg("data", R.Tensor(data_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.avg_pool2d(
                        data,
                        pool_size=pool_size,
                        strides=stride,
                        dilation=dilation,
                        padding=padding,
                        layout="NCHW",
                    )
                )
                R.output(output)
            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def get_avgpool_expected_codegen(input_shape, pool_size, stride, padding, pool_type, dtype):
    import math

    adjusted_input_shape = [
        input_shape[0],
        input_shape[1],
        input_shape[2] + padding[0] + padding[1],
        input_shape[3] + padding[2] + padding[3],
    ]

    pool_height = math.floor(((adjusted_input_shape[2] - pool_size[0]) / stride[0]) + 1)
    pool_width = math.floor(((adjusted_input_shape[3] - pool_size[1]) / stride[1]) + 1)
    output_shape = [adjusted_input_shape[0], adjusted_input_shape[1], pool_height, pool_width]

    attrs = {
        "ceil_mode": [["0"]],
        "clml_version": [["3"]],
        "dilation": [["1", "1"]],
        "layout": [["NCHW"]],
        "num_inputs": "1",
        "num_outputs": "1",
        "out_layout": [["NCHW"]],
        "padding": [[str(0) for p in padding]],
        "pool_size": [[str(p) for p in pool_size]],
        "shape": [[list(output_shape)]],
        "dtype": [[dtype]],
        "strides": [[str(s) for s in stride]],
        "count_include_pad": [["0"]],
    }
    if sum(padding):
        attrs["count_include_pad"] = [["0"]]

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [[list(adjusted_input_shape)]], "dtype": [[str(dtype)]]},
        },
        {
            "op": "kernel",
            "name": "",
            "inputs": [[0, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


def get_relax_reshape_mod(input_shape, output_shape, dtype):
    """
    Args:
        input_shape (tuple): Input tensor shape
        output_shape (tuple): Desired output tensor shape
        dtype (str, optional): Data type. Defaults to "float32".

    Returns:
        tvm.IRModule: Relax Reshape module
    """
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            data = R.arg("data", R.Tensor(input_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(R.reshape(data, output_shape))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def get_relax_reshape_codegen(input_shape, output_shape, dtype):
    def compute_output_shape(input_shape, output_shape):
        input_elements = np.prod(input_shape)
        specified_elements = np.prod([dim for dim in output_shape if dim != -1])
        missing_dim = input_elements // specified_elements
        return [int(dim) if dim != -1 else int(missing_dim) for dim in output_shape]

    expected_output_shape = compute_output_shape(input_shape, output_shape)

    expected_codegen_str = [
        {
            "attrs": {
                "dtype": [[dtype]],
                "shape": [[list(input_shape)]],
            },
            "name": "",
            "op": "input",
        },
        {
            "attrs": {
                "clml_version": [["3"]],
                "dtype": [[dtype]],
                "num_inputs": "1",
                "num_outputs": "1",
                "shape": [[expected_output_shape]],
            },
            "inputs": [[0, 0, 0]],
            "name": "",
            "op": "kernel",
        },
    ]
    return expected_codegen_str


def get_relax_global_avgpool_mod(data_shape, keepdims, dtype):
    """
    Create a Relax module for Global Average Pooling (GAP).

    Args:
        data_shape (tuple): Input tensor shape (N, C, H, W)
        dtype (str): Data type

    Returns:
        tvm.IRModule: Relax GAP module
    """
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            data = R.arg("data", R.Tensor(data_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(R.mean(data, axis=[2, 3], keepdims=keepdims))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def get_global_avgpool_expected_codegen(input_shape, keep_dims, dtype):
    """
    Generate expected codegen for Global Average Pooling.

    Args:
        input_shape (tuple): Input shape (N, C, H, W)
        dtype (str): Data type

    Returns:
        dict: Expected codegen output
    """
    output_shape = (
        [input_shape[0], input_shape[1]]
        if not keep_dims
        else [input_shape[0], input_shape[1], 1, 1]
    )
    attrs = {
        "num_inputs": "1",
        "num_outputs": "1",
        "clml_version": [["3"]],
        "shape": [[list(output_shape)]],
        "dtype": [[dtype]],
        "axis": [["2", "3"]],
        "keepdims": [["1" if keep_dims else "0"]],
    }

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [[list(input_shape)]], "dtype": [[str(dtype)]]},
        },
        {"op": "kernel", "name": "", "inputs": [[0, 0, 0]], "attrs": attrs},
    ]
    return exp_codegen


def get_relax_global_maxpool_mod(data_shape, keepdims, dtype):
    """
    Create a Relax module for Global Average Pooling (GAP).

    Args:
        data_shape (tuple): Input tensor shape (N, C, H, W)
        dtype (str): Data type

    Returns:
        tvm.IRModule: Relax GAP module
    """
    N, C, H, W = data_shape
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            data = R.arg("data", R.Tensor(data_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.max_pool2d(
                        data, pool_size=(H, W), strides=(1, 1), padding=(0, 0), layout="NCHW"
                    )
                )
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def get_global_maxpool_expected_codegen(input_shape, pool_size, stride, padding, pool_type, dtype):
    import math

    adjusted_input_shape = [
        input_shape[0],
        input_shape[1],
        input_shape[2] + padding[0] + padding[1],
        input_shape[3] + padding[2] + padding[3],
    ]

    output_shape = [adjusted_input_shape[0], adjusted_input_shape[1], 1, 1]

    attrs = {
        "ceil_mode": [["0"]],
        "clml_version": [["3"]],
        "dilation": [["1", "1"]],
        "layout": [["NCHW"]],
        "num_inputs": "1",
        "num_outputs": "1",
        "out_layout": [["NCHW"]],
        "padding": [[str(0) for p in padding]],
        "pool_size": [[str(p) for p in pool_size]],
        "shape": [[list(output_shape)]],
        "dtype": [[dtype]],
        "strides": [[str(s) for s in stride]],
        "count_include_pad": [["0"]],
    }
    if sum(padding):
        attrs["count_include_pad"] = [["0"]]

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [[list(adjusted_input_shape)]], "dtype": [[str(dtype)]]},
        },
        {
            "op": "kernel",
            "name": "",
            "inputs": [[0, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen
