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
"""CLML integration conv2d tests."""

import numpy as np

np.random.seed(0)

import tvm
from tvm import testing
from tvm import relay
from tvm.ir import IRModule

from test_clml.infrastructure import (
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    Device,
)


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
        p = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
        a = relay.nn.pad(a, pad_width=p)
        padding = (0, 0, 0, 0)
    else:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        shape = (shape[0], shape[1], shape[2] + padding[0] * 2, shape[3] + padding[1] * 2)
    is_depthwise = shape[1] == channels == groups

    weight_format = "OIHW" if is_depthwise else "OIHW"
    if weight_format == "IOHW":
        weight_shape = (shape[1] // groups, channels, kernel_h, kernel_w)
    else:
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
        bias_shape = weight_shape[2] if is_depthwise else weight_shape[0]
        b = tvm.nd.array(np.random.uniform(-1, 1, bias_shape).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.nn.bias_add(out, biasc, axis=1)
        params["b"] = b

    if has_activation:
        out = relay.nn.relu(out)

    print("Out:", out)

    return out, params


def test_conv2d():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    dtype = "float32"

    trials = [
        # Normal convolution
        [3, 3, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (15, 16, 12), (False, False, True)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, True)],
        # Normal convolution
        [2, 2, (1, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [2, 1, (2, 2), (1, 1), (1, 1), 7, (16, 12, 15), (False, False, True)],
        [3, 3, (2, 1), (1, 1), (1, 1), 4, (14, 10, 10), (False, True, False)],
        [3, 3, (1, 1), (1, 1), (1, 1), 16, (16, 12, 15), (False, False, False)],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [1, 3, (1, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, True)],
        [2, 2, (2, 2), (1, 1), (1, 1), 4, (20, 20, 20), (False, True, False)],
        [5, 5, (1, 1), (2, 2), (1, 1), 4, (14, 10, 10), (False, False, False)],
        [3, 3, (2, 1), (1, 1), (1, 1), 7, (20, 20, 20), (False, False, False)],
        [3, 3, (1, 1), (2, 2), (1, 1), 16, (14, 10, 10), (False, True, True)],
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
    ) in trials:
        shape = (1, *shape)
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


def test_batchnorm():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    dtype = "float32"

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


if __name__ == "__main__":
    test_conv2d()
    test_batchnorm()
