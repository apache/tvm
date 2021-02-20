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
import pytest
import tvm
from tvm import relay

from .infrastructure import skip_runtime_test, compare_inference_with_ref, generate_trials

# TODO: Missed cases
#   1. Bias as add with 3d const tensor. Lead to additional unsqueeze op between
#   2. Check unsupported cases of fusion. Like bias add with axis != 1, add with broadcast by spatial dims
#   3. Check if bias/weights is not constants. Should fallback into LLVM or decompose it
#   4. Check if bias/weights is constants expr. Should works somehow.


def _get_model(
    shape,
    kernel=(3, 3),
    padding=(1, 1),
    strides=(1, 1),
    dilation=(1, 1),
    groups=1,
    dtype="float32",
    channels=-1,  # -1 means same as input channels
    bias_type="none",
    activation_type="none",
):
    """Return a model and any parameters it may have"""
    if channels == -1:
        channels = shape[1]

    a = relay.var("a", shape=shape, dtype=dtype)
    weight_shape = (channels, shape[1] // groups, *kernel)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=kernel,
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
        bias_shape = (
            (weight_shape[0], 1, 1) if bias_type == "add_3d" else (1, weight_shape[0], 1, 1)
        )
        b = tvm.nd.array(np.random.uniform(-10, 10, bias_shape).astype(dtype))
        biasc = relay.const(b, dtype)
        out = relay.add(out, biasc)
        params["b"] = b

    if activation_type == "relu":
        out = relay.nn.relu(out)
    elif activation_type == "sigmoid":
        out = relay.op.sigmoid(out)
    return out, params


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_conv2d():
    np.random.seed(0)

    kernel_hs = [1, 2, 3, 5]
    kernel_ws = [1, 2, 3, 5]
    pad = [(1, 1), (2, 2), (2, 1)]
    strides = [(1, 1), (2, 2)]
    dilation = [(1, 1)]
    out_channels = [1, 4, 8, 16]
    input_shapes = [(10, 10, 14), (12, 15, 16), (20, 20, 20)]
    batches = [1, 2]
    groups = [1, 2]
    bias_kind = ["none", "add_3d", "add_4d", "bias.add"]
    activation_kind = ["none", "relu", "sigmoid"]
    trials = generate_trials(
        [
            kernel_hs,
            kernel_ws,
            pad,
            strides,
            dilation,
            out_channels,
            input_shapes,
            groups,
            batches,
            bias_kind,
            activation_kind,
        ],
        3,
    )

    for (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        input_shapes,
        group,
        batch,
        bias,
        activation,
    ) in trials:
        if out_channels % group != 0:
            continue
        func, params = _get_model(
            shape=(batch, *input_shapes),
            kernel=(kernel_h, kernel_w),
            padding=pad,
            strides=stride,
            dilation=dilation,
            groups=group,
            channels=out_channels,
            bias_type=bias,
            activation_type=activation,
        )
        compare_inference_with_ref(func, params)


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_conv2d_dw():
    if skip_runtime_test():
        return

    np.random.seed(0)
    shape = [4, 5, 5]

    for batch in [1, 2]:
        mod, params = _get_model(shape=(batch, *shape), groups=shape[0])
        compare_inference_with_ref(mod, params)


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_conv2d_with_oc1():
    if skip_runtime_test():
        return

    np.random.seed(0)
    shape = [3, 5, 5]

    for batch in [1, 2]:
        for bias in ["none", "add_4d"]:
            mod, params = _get_model(shape=(batch, *shape), channels=1, bias_type=bias)
            compare_inference_with_ref(mod, params)


if __name__ == "__main__":
    test_conv2d()
    test_conv2d_dw()
    test_conv2d_with_oc1()
