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
"""BNNS integration conv2d tests."""

import numpy as np

import tvm
from tvm import relay

from .infrastructure import Device
from .infrastructure import (
    skip_runtime_test,
    skip_codegen_test,
    build_and_run,
    verify,
    generate_trials,
)

# TODO: Missed cases
#   1. Bias as add with 3d const tensor. Lead to additional unsqueeze op between
#   2. Check unsupported cases of fusion. Like bias add with axis != 1, add with broadcast by spatial dims
#   3. Check if bias/weights is not constants. Should fallback into LLVM or decompose it
#   4. Check if bias/weights is constants expr. Should works somehow.

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
    bias_type='none',
    activation_type='none',
):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    weight_shape = (channels, shape[1] // groups, kernel_h, kernel_w)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=(kernel_h, kernel_w),
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype=dtype,
    )
    params = {"w": w}
    if bias_type == "bias_add":
        b = tvm.nd.array(np.random.uniform(-10, 10, weight_shape[0]).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.nn.bias_add(out, biasc, axis=1)
        params["b"] = b
    elif bias_type == "add_3d" or bias_type == "add_4d":
        bias_shape = (weight_shape[0], 1, 1) if bias_type == "add_3d" else (1, weight_shape[0], 1, 1)
        b = tvm.nd.array(np.random.uniform(-10, 10, bias_shape).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.add(out, biasc)
        params["b"] = b

    if activation_type == "relu":
        out = relay.nn.relu(out)
    return out, params


def test_conv2d():
    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    kernel_hs = [1, 2, 3, 5]
    kernel_ws = [1, 2, 3, 5]
    pad = [(1, 1), (2, 2), (2, 1)]
    strides = [(1, 1), (2, 2)]
    dilation = [(1, 1)]
    out_channels = [4, 8, 16]
    input_shapes = [(10, 10, 14), (12, 15, 16), (20, 20, 20)]
    batches = [1, 2]
    groups = [1, 2]
    bias_kind = ['none', 'add_3d', 'add_4d', 'bias.add']
    activation_kind = ['none', 'relu']
    dtype = "float32"
    trials = generate_trials(
        [kernel_hs, kernel_ws, pad, strides, dilation, out_channels, input_shapes,
         groups, batches, bias_kind, activation_kind], 3
    )

    for kernel_h, kernel_w, pad, stride, dilation, out_channels, input_shapes, group, batch, bias, activation in trials:
        shape = (batch, *input_shapes)
        outputs = []
        inputs = {
            "a": tvm.nd.array(np.random.uniform(0, 127, shape).astype(dtype)),
        }

        func, params = _get_model(
            shape,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            group,
            dtype,
            out_channels,
            iter(inputs),
            bias_type=bias,
            activation_type=activation,
        )
        for bnns in [False, True]:
            outputs.append(build_and_run(func, inputs, 1, params, device, enable_bnns=bnns)[0])

        config = {
            "shape": shape,
            "group": group,
            "kernel size": (kernel_h, kernel_w),
            "padding": pad,
            "stride": stride,
            "dilation": dilation,
            "out channels": out_channels,
            "bias": bias,
            "activation": activation
        }
        verify(outputs, atol=0.002, rtol=0.007, config=config)


if __name__ == "__main__":
    test_conv2d()
