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

import tvm
from tvm import relay

from .infrastructure import (
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    verify,
    verify_codegen,
    generate_trials,
)
from .infrastructure import Device


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
    weight_shape = (kernel_h, kernel_w, shape[3] // groups, channels)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=(kernel_h, kernel_w),
        data_layout="NHWC",
        kernel_layout="HWIO",
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype=dtype,
    )
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.uniform(-128, 127, weight_shape[3]).astype(dtype))
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
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    if has_pad:
        p = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
        a = relay.nn.pad(a, pad_width=p, pad_value=input_zp, pad_mode="constant")
        padding = (0, 0, 0, 0)
    else:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        shape = (shape[0], shape[1] + padding[0] * 2, shape[2] + padding[1] * 2, shape[3])
    weight_shape = (kernel_h, kernel_w, shape[3] // groups, channels)
    w = tvm.nd.array(np.random.uniform(0, 255, weight_shape).astype(dtype))
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
        kernel_layout="HWIO",
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype="int32",
    )
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.uniform(0, 255, weight_shape[3]).astype("int32"))
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
        out_dtype="uint8",
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
    weight_shape = (channels, kernel_h, kernel_w, shape[3] // groups)
    output_height = ((shape[1] - kernel_h + padding[0] + padding[2]) / strides[0]) + 1
    output_width = ((shape[2] - kernel_w + padding[1] + padding[3]) / strides[1]) + 1
    output_shape = (1, int(output_height), int(output_width), channels)
    out_dtype = "int32" if dtype == "uint8" else "float32"

    node = {
        "op": "kernel",
        "name": "nn.conv2d",
        "inputs": [],
        "attrs": {
            "groups": [["1"]],
            "num_outputs": "1",
            "data_layout": [["NHWC"]],
            "kernel_layout": [["OHWI"]],
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
    if dtype == "uint8":
        node["name"] = "qnn.conv2d"
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
        bias_dtype = "int32" if dtype == "uint8" else "float32"
        inputs.append(
            {
                "op": "const",
                "name": "",
                "attrs": {"shape": [[[weight_shape[0]]]], "dtype": [[bias_dtype]]},
            }
        )

    # qnn.conv2d params, output
    if dtype == "uint8":
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

    kernel_hs = [1, 2, 3, 5]
    kernel_ws = [1, 2, 3, 5]
    pad = [(1, 1), (2, 2), (2, 1)]
    strides = [(1, 1), (2, 2)]
    dilation = [(1, 1)]
    out_channels = [4, 7, 16]
    input_shapes = [(10, 10, 14), (12, 15, 16), (20, 20, 20)]
    # composite operator (pad, bias, activation)
    composite = [
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (True, False, False),
    ]
    dtype = "float32"
    trials = generate_trials(
        [kernel_hs, kernel_ws, pad, strides, dilation, out_channels, input_shapes, composite], 3
    )

    for kernel_h, kernel_w, pad, stride, dilation, out_channels, input_shapes, composite in trials:
        groups = 1
        shape = (1, *input_shapes)
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

    np.random.seed(0)

    kernel_hs = [1, 2, 3, 5]
    kernel_ws = [1, 2, 3, 5]
    pad = [(1, 1), (2, 2), (2, 1)]
    strides = [(1, 1), (2, 2)]
    dilation = [(1, 1)]
    out_channels = [4, 7, 16]
    input_shapes = [(10, 10, 14), (12, 15, 16), (20, 20, 20)]
    # composite operator (pad, bias, activation)
    composite = [
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (True, False, False),
    ]
    dtype = "float32"
    trials = generate_trials(
        [kernel_hs, kernel_ws, pad, strides, dilation, out_channels, input_shapes, composite], 3
    )

    for kernel_h, kernel_w, pad, stride, dilation, out_channels, input_shapes, composite in trials:
        groups = 1
        shape = (1, *input_shapes)
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


def test_qnn_conv2d():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    kernel_hs = [1, 2, 3, 5]
    kernel_ws = [1, 2, 3, 5]
    pad = [(1, 1), (2, 2)]
    strides = [(1, 1), (2, 2)]
    dilation = [(1, 1)]
    out_channels = [4, 7, 16]
    input_shapes = [(10, 10, 14), (12, 15, 16), (20, 20, 20)]
    # composite operator (pad, bias, activation)
    composite = [
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (True, False, False),
    ]
    dtype = "uint8"
    trials = generate_trials(
        [kernel_hs, kernel_ws, pad, strides, dilation, out_channels, input_shapes, composite], 3
    )

    for kernel_h, kernel_w, pad, stride, dilation, out_channels, input_shapes, composite in trials:
        groups = 1
        shape = (1, *input_shapes)
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
        verify(outputs, atol=1, rtol=0, config=config, verify_saturation=True)


def test_codegen_qnn_conv2d():
    if skip_codegen_test():
        return

    kernel_hs = [1, 2, 3, 5]
    kernel_ws = [1, 2, 3, 5]
    pad = [(1, 1), (2, 2), (2, 1)]
    strides = [(1, 1), (2, 2)]
    dilation = [(1, 1)]
    out_channels = [4, 7, 16]
    input_shapes = [(10, 10, 14), (12, 15, 16), (20, 20, 20)]
    # composite operator (pad, bias, activation)
    composite = [
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (True, False, False),
    ]
    dtype = "uint8"
    trials = generate_trials(
        [kernel_hs, kernel_ws, pad, strides, dilation, out_channels, input_shapes, composite], 3
    )

    for kernel_h, kernel_w, pad, stride, dilation, out_channels, input_shapes, composite in trials:
        groups = 1
        shape = (1, *input_shapes)
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
