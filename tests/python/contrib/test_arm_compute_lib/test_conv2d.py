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
"""Arm Compute Library integration conv2d tests."""

import numpy as np
import pytest

import tvm
from tvm import relay

from test_arm_compute_lib.infrastructure import (
    QNN_DTYPES,
    get_low_high_atol_rtol,
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    verify,
    verify_codegen,
)
from test_arm_compute_lib.infrastructure import Device


def _get_model(
    shape,
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    groups,
    dtype,
    channels,
    var_names,
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    if has_pad:
        p = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
        a = relay.nn.pad(a, pad_width=p)
        padding = (0, 0, 0, 0)
    else:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        shape = (shape[0], shape[1] + padding[0] * 2, shape[2] + padding[1] * 2, shape[3])
    is_depthwise = shape[3] == channels == groups
    weight_format = "HWOI" if is_depthwise else "HWIO"
    if weight_format == "HWIO":
        weight_shape = (kernel_h, kernel_w, shape[3] // groups, channels)
    else:
        weight_shape = (kernel_h, kernel_w, channels, shape[3] // groups)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=(kernel_h, kernel_w),
        data_layout="NHWC",
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
        bias_shape = weight_shape[2] if is_depthwise else weight_shape[3]
        b = tvm.nd.array(np.random.uniform(-128, 127, bias_shape).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.nn.bias_add(out, biasc, axis=3)
        params["b"] = b
    if has_activation:
        out = relay.nn.relu(out)
    return out, params


def _get_qnn_params(input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, channels):
    """Get output qnn parameters given input and kernel parameters."""
    input_max = input_sc * (255 - input_zp)
    input_min = -input_sc * input_zp
    kernel_max = kernel_sc * (255 - kernel_zp)
    kernel_min = -kernel_sc * kernel_zp
    output_limits = [
        kernel_max * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_min,
        kernel_max * kernel_h * kernel_w * channels * input_min,
    ]
    output_max = max(output_limits)
    output_min = min(output_limits)
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return output_zp, output_sc


def _get_qnn_model(
    shape,
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    groups,
    dtype,
    channels,
    input_zp,
    input_sc,
    kernel_zp,
    kernel_sc,
    output_zp,
    output_sc,
    var_names,
    has_bias=False,
    has_activation=False,
    has_pad=False,
):
    """Return a model and any parameters it may have."""
    low, high, _, _ = get_low_high_atol_rtol(dtype)

    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    if has_pad:
        p = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
        a = relay.nn.pad(a, pad_width=p, pad_value=input_zp, pad_mode="constant")
        padding = (0, 0, 0, 0)
    else:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        shape = (shape[0], shape[1] + padding[0] * 2, shape[2] + padding[1] * 2, shape[3])
    is_depthwise = shape[3] == channels == groups
    weight_format = "HWOI" if is_depthwise else "HWIO"
    if weight_format == "HWIO":
        weight_shape = (kernel_h, kernel_w, shape[3] // groups, channels)
    else:
        weight_shape = (kernel_h, kernel_w, channels, shape[3] // groups)
    w = tvm.nd.array(np.random.uniform(low, high, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.qnn.op.conv2d(
        a,
        weights,
        input_zero_point=relay.const(input_zp, "int32"),
        kernel_zero_point=relay.const(kernel_zp, "int32"),
        input_scale=relay.const(input_sc, "float32"),
        kernel_scale=relay.const(kernel_sc, "float32"),
        kernel_size=(kernel_h, kernel_w),
        data_layout="NHWC",
        kernel_layout=weight_format,
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype="int32",
    )
    params = {"w": w}
    if has_bias:
        bias_shape = weight_shape[2] if is_depthwise else weight_shape[3]
        b = tvm.nd.array(np.random.uniform(-128, 127, bias_shape).astype("int32"))
        biasc = relay.const(b, "int32")
        out = relay.nn.bias_add(out, biasc, axis=3)
        params["b"] = b
    if has_activation:
        out = relay.nn.relu(out)
    req = relay.qnn.op.requantize(
        out,
        relay.const(input_sc * kernel_sc, "float32"),  # input scale
        relay.const(0, "int32"),  # input zero point
        relay.const(output_sc, "float32"),  # output scale
        relay.const(output_zp, "int32"),  # output zero point
        out_dtype=dtype,
    )
    return req, params


def _get_expected_codegen(
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
    output_height = ((shape[1] - kernel_h + padding[0] + padding[2]) / strides[0]) + 1
    output_width = ((shape[2] - kernel_w + padding[1] + padding[3]) / strides[1]) + 1
    output_shape = (1, int(output_height), int(output_width), channels)
    out_dtype = "int32" if dtype in QNN_DTYPES else "float32"
    is_depthwise = shape[3] == channels == groups
    weight_format = "IHWO" if is_depthwise else "OHWI"
    if weight_format == "IHWO":
        weight_shape = (shape[3] // groups, kernel_h, kernel_w, channels)
    else:
        weight_shape = (channels, kernel_h, kernel_w, shape[3] // groups)
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
            "data_layout": [["NHWC"]],
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

    # qnn.conv2d params, input and kernel
    if dtype in QNN_DTYPES:
        node["name"] = "qnn." + node["name"].split(".")[1]
        for param_dtype in ["int32", "float32"]:
            for _ in range(2):
                inputs.append(
                    {
                        "op": "const",
                        "name": "",
                        "attrs": {"shape": [[[]]], "dtype": [[param_dtype]]},
                    }
                )

    if has_bias:
        bias_dtype = "int32" if dtype in QNN_DTYPES else "float32"
        inputs.append(
            {
                "op": "const",
                "name": "",
                "attrs": {
                    "shape": [[[1, 1, 1, weight_shape[3] if is_depthwise else weight_shape[0]]]],
                    "dtype": [[bias_dtype]],
                },
            }
        )

    # qnn.conv2d params, output
    if dtype in QNN_DTYPES:
        for param_dtype in ["float32", "int32"]:
            inputs.append(
                {"op": "const", "name": "", "attrs": {"shape": [[[]]], "dtype": [[param_dtype]]}}
            )

    input_idx = 0
    for _ in range(len(inputs)):
        node["inputs"].append([input_idx, 0, 0])
        input_idx += 1
    node["attrs"]["num_inputs"] = str(len(inputs))
    inputs.append(node)
    return inputs


def test_conv2d():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    dtype = "float32"
    trials = [
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, False, False), False],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (12, 15, 16), (False, False, True), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, True, False), False],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (12, 15, 16), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True), False],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False), False],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (10, 10, 14), (False, True, True), False],
        # Depth-wise convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, True), True],
        [5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), (False, True, False), True],
        [3, 3, (2, 2), (2, 2), (1, 1), 14, (10, 10, 14), (True, False, False), True],
        [5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, False), True],
        [3, 3, (1, 1), (2, 2), (1, 1), 14, (10, 10, 14), (False, True, True), True],
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
            groups = shape[3]
        else:
            groups = 1
        outputs = []
        inputs = {
            "a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype)),
        }

        func, params = _get_model(
            shape,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            groups,
            dtype,
            out_channels,
            iter(inputs),
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1, params, device, enable_acl=acl)[0])

        config = {
            "shape": shape,
            "groups": groups,
            "kernel size": (kernel_h, kernel_w),
            "padding": pad,
            "stride": stride,
            "dilation": dilation,
            "out channels": out_channels,
            "composite operators (pad, bias, activation)": composite,
        }
        verify(outputs, atol=0.002, rtol=0.01, config=config)


def test_codegen_conv2d():
    if skip_codegen_test():
        return

    dtype = "float32"
    trials = [
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, False, False), False],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (12, 15, 16), (False, False, True), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, True, False), False],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (12, 15, 16), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True), False],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False), False],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (10, 10, 14), (False, True, True), False],
        # Depth-wise convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, True), True],
        [5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), (False, True, False), True],
        [3, 3, (2, 2), (2, 2), (1, 1), 14, (10, 10, 14), (True, False, False), True],
        [5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, False), True],
        [3, 3, (1, 1), (2, 2), (1, 1), 14, (10, 10, 14), (False, True, True), True],
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
            groups = shape[3]
        else:
            groups = 1
        inputs = {"a"}

        args = (shape, kernel_h, kernel_w, pad, stride, dilation, groups, dtype, out_channels)

        func, params = _get_model(
            *args,
            var_names=iter(inputs),
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        exp_codegen = _get_expected_codegen(
            *args, has_bias=composite[1], has_activation=composite[2]
        )
        verify_codegen(func, exp_codegen, 1)


@pytest.mark.parametrize("dtype", QNN_DTYPES)
def test_qnn_conv2d(dtype):
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    trials = [
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, False, False), False],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (12, 15, 16), (False, False, True), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, True, False), False],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (12, 15, 16), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True), False],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False), False],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (10, 10, 14), (False, True, True), False],
        # Depth-wise convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, True), True],
        [5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), (False, True, False), True],
        [3, 3, (2, 2), (2, 2), (1, 1), 14, (10, 10, 14), (True, False, False), True],
        [5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, False), True],
        [3, 3, (1, 1), (2, 2), (1, 1), 14, (10, 10, 14), (False, True, True), True],
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
            groups = shape[3]
        else:
            groups = 1
        outputs = []
        inputs = {"a": tvm.nd.array(np.random.uniform(0, 255, shape).astype(dtype))}

        input_zp = 100
        input_sc = 0.5
        kernel_zp = 25
        kernel_sc = 0.03
        output_zp, output_sc = _get_qnn_params(
            input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, shape[3]
        )

        func, params = _get_qnn_model(
            shape,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            groups,
            dtype,
            out_channels,
            input_zp,
            input_sc,
            kernel_zp,
            kernel_sc,
            output_zp,
            output_sc,
            iter(inputs),
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        for acl in [False, True]:
            outputs.append(build_and_run(func, inputs, 1, params, device, enable_acl=acl)[0])

        config = {
            "shape": shape,
            "groups": groups,
            "kernel size": (kernel_h, kernel_w),
            "padding": pad,
            "stride": stride,
            "dilation": dilation,
            "out channels": out_channels,
            "composite operators (pad, bias, activation)": composite,
            "input scale": input_sc,
            "input zero point": input_zp,
            "kernel scale": kernel_sc,
            "kernel zero point": kernel_zp,
            "output scale": output_sc,
            "output zero point": output_zp,
        }

        atol = 2 if is_depthwise else 1
        verify(outputs, atol=atol, rtol=0, config=config, verify_saturation=True)


@pytest.mark.parametrize("dtype", QNN_DTYPES)
def test_codegen_qnn_conv2d(dtype):
    if skip_codegen_test():
        return

    trials = [
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, False, False), False],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (12, 15, 16), (False, False, True), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (10, 10, 14), (False, True, False), False],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (12, 15, 16), (False, False, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True), False],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False), False],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (10, 10, 14), (True, False, False), False],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False), False],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (10, 10, 14), (False, True, True), False],
        # Depth-wise convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, True), True],
        [5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), (False, True, False), True],
        [3, 3, (2, 2), (2, 2), (1, 1), 14, (10, 10, 14), (True, False, False), True],
        [5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, False), True],
        [3, 3, (1, 1), (2, 2), (1, 1), 14, (10, 10, 14), (False, True, True), True],
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
            groups = shape[3]
        else:
            groups = 1
        inputs = {"a"}

        input_zp = 100
        input_sc = 0.5
        kernel_zp = 25
        kernel_sc = 0.03
        output_zp, output_sc = _get_qnn_params(
            input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, shape[3]
        )

        args = (shape, kernel_h, kernel_w, pad, stride, dilation, groups, dtype, out_channels)

        func, params = _get_qnn_model(
            *args,
            input_zp=input_zp,
            input_sc=input_sc,
            kernel_zp=kernel_zp,
            kernel_sc=kernel_sc,
            output_zp=output_zp,
            output_sc=output_sc,
            var_names=iter(inputs),
            has_pad=composite[0],
            has_bias=composite[1],
            has_activation=composite[2],
        )
        exp_codegen = _get_expected_codegen(
            *args, has_bias=composite[1], has_activation=composite[2]
        )
        verify_codegen(func, exp_codegen, 1)


if __name__ == "__main__":
    test_conv2d()
    test_qnn_conv2d()
    test_codegen_conv2d()
    test_codegen_qnn_conv2d()
