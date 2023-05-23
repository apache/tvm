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

"""Arm(R) Ethos(TM)-N integration conv2d tests"""

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.testing import requires_ethosn

from . import infrastructure as tei


def _get_model(
    shape,
    kernel_h,
    kernel_w,
    input_zp,
    input_sc,
    kernel_zp,
    kernel_sc,
    output_zp,
    output_sc,
    pad,
    strides,
    dilation,
    groups,
    dtype,
    out_channels,
    weight_format,
):
    """Return a model and any parameters it may have"""
    a = relay.var("a", shape=shape, dtype=dtype)
    if pad in ("op", "both"):
        p = tei.get_same_padding((shape[1], shape[2]), (kernel_h, kernel_w), dilation, strides)
        a = relay.nn.pad(
            a,
            pad_width=[(0, 0), (p[0], p[2]), (p[1], p[3]), (0, 0)],
            pad_value=input_zp,
            pad_mode="constant",
        )
        shape = (shape[0], shape[1] + p[0] + p[2], shape[2] + p[1] + p[3], shape[3])

    p = tei.get_same_padding((shape[1], shape[2]), (kernel_h, kernel_w), dilation, strides)
    if weight_format == "HWIO":
        weight_shape = (kernel_h, kernel_w, shape[3] // groups, out_channels)
    else:
        weight_shape = (kernel_h, kernel_w, out_channels, 1)
    weights_array = tvm.nd.array(
        np.random.randint(
            np.iinfo(dtype).min, high=np.iinfo(dtype).max + 1, size=weight_shape, dtype=dtype
        )
    )
    weights = relay.const(weights_array, dtype)
    conv = relay.qnn.op.conv2d(
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
        groups=groups,
        channels=out_channels,
        padding=p if pad in ("attr", "both") else (0, 0, 0, 0),
        out_dtype="int32",
    )
    bias_data = tvm.nd.array(
        np.random.randint(
            np.iinfo(dtype).min, high=np.iinfo(dtype).max + 1, size=(out_channels,), dtype="int32"
        )
    )
    biasc = relay.const(bias_data, "int32")
    bias = relay.nn.bias_add(conv, biasc, axis=3)
    if isinstance(kernel_sc, tvm.runtime.ndarray.NDArray):
        req_input_sc = [sc * input_sc for sc in kernel_sc.numpy()]
    else:
        req_input_sc = input_sc * kernel_sc
    req = relay.qnn.op.requantize(
        bias,
        relay.const(req_input_sc, "float32"),  # input zero scale
        relay.const(0, "int32"),  # input zero point
        relay.const(output_sc, "float32"),  # output zero scale
        relay.const(output_zp, "int32"),  # output zero point
        out_dtype=dtype,
    )
    params = {"w": weights_array, "b": bias_data}
    return req, params


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,qnn_per_channel", [("uint8", False), ("int8", False), ("int8", True)]
)
@pytest.mark.parametrize("pad,stride", [("attr", (2, 2)), ("none", (2, 2)), ("op", (1, 1))])
@pytest.mark.parametrize(
    "shape,out_channels,kernel_size",
    [
        [(1, 17, 20, 26), 4, (3, 1)],
        [(1, 9, 20, 30), 7, (1, 5)],
        [(1, 21, 21, 22), 8, (2, 2)],
    ],
)
def test_conv2d(
    dtype,
    shape,
    out_channels,
    kernel_size,
    pad,
    stride,
    qnn_per_channel,
):
    """Compare Conv2D output with TVM."""
    np.random.seed(0)

    dilation = (1, 1)
    groups = 1
    weight_format = "HWIO"

    outputs = []
    inputs = {
        "a": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min,
                np.iinfo(dtype).max + 1,
                size=shape,
                dtype=dtype,
            )
        ),
    }
    input_zp = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max)
    input_sc = np.random.random() * 2
    if qnn_per_channel:
        kernel_sc = tvm.nd.array(
            np.random.uniform(low=0, high=2, size=(out_channels,)).astype(np.float32)
        )
    else:
        kernel_sc = np.random.random() * 2
    kernel_zp = (
        0 if dtype == "int8" else np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max)
    )
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, kernel_zp, kernel_sc, kernel_size[0], kernel_size[1], shape[3]
    )
    model, params = _get_model(
        shape,
        kernel_size[0],
        kernel_size[1],
        input_zp,
        input_sc,
        kernel_zp,
        kernel_sc,
        output_zp,
        output_sc,
        pad,
        stride,
        dilation,
        groups,
        dtype,
        out_channels,
        weight_format,
    )
    for npu in [False, True]:
        mod = tei.make_module(model, params)
        outputs.append(tei.build_and_run(mod, inputs, 1, params, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,qnn_per_channel", [("uint8", False), ("int8", False), ("int8", True)]
)
@pytest.mark.parametrize("pad,stride", [("attr", (2, 2)), ("none", (2, 2)), ("op", (1, 1))])
@pytest.mark.parametrize(
    "shape,kernel_size",
    [
        [(1, 17, 20, 28), (3, 3)],
        [(1, 9, 20, 30), (5, 5)],
        [(1, 21, 21, 22), (2, 2)],
    ],
)
def test_conv2d_depthwise(
    dtype,
    shape,
    kernel_size,
    pad,
    stride,
    qnn_per_channel,
):
    """Compare Conv2D output with TVM."""
    np.random.seed(0)

    dilation = (1, 1)
    out_channels = shape[3]
    groups = out_channels
    weight_format = "HWOI"

    outputs = []
    inputs = {
        "a": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min,
                np.iinfo(dtype).max + 1,
                size=shape,
                dtype=dtype,
            )
        ),
    }
    input_zp = np.random.randint(0, np.iinfo(dtype).max)
    input_sc = np.random.random() * 2
    if qnn_per_channel:
        kernel_sc = tvm.nd.array(
            np.random.uniform(low=0, high=2, size=(out_channels,)).astype(np.float32)
        )
    else:
        kernel_sc = np.random.random() * 2
    kernel_zp = (
        0 if dtype == "int8" else np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max)
    )
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, kernel_zp, kernel_sc, kernel_size[0], kernel_size[1], shape[3]
    )
    model, params = _get_model(
        shape,
        kernel_size[0],
        kernel_size[1],
        input_zp,
        input_sc,
        kernel_zp,
        kernel_sc,
        output_zp,
        output_sc,
        pad,
        stride,
        dilation,
        groups,
        dtype,
        out_channels,
        weight_format,
    )
    for npu in [False, True]:
        mod = tei.make_module(model, params)
        outputs.append(tei.build_and_run(mod, inputs, 1, params, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,pad,stride,dilation,err_msg",
    [
        (
            (1, 4, 4, 4),
            "both",
            (1, 1),
            (1, 1),
            "both op and attr padding exist, must be either op/attr only or no padding",
        ),
        (
            (1, 4, 4, 4),
            "none",
            (1, 1, 1),
            (1, 1),
            "stride size=3, stride size must = 2",
        ),
        (
            (1, 4, 4, 4),
            "none",
            (1, 1),
            (2, 1),
            "dilation=[2, 1], dilation must = [1, 1]",
        ),
        (
            (2, 4, 4, 4),
            "none",
            (1, 1),
            (1, 1),
            "batch size=2, batch size must = 1",
        ),
    ],
)
def test_conv2d_failure(shape, pad, stride, dilation, err_msg):
    """Check Conv2D error messages."""
    np.random.seed(0)

    kernel_size = (2, 2)
    groups = 1
    dtype = "uint8"
    out_channels = 8
    weight_format = "HWIO"

    model, _ = _get_model(
        shape,
        kernel_size[0],
        kernel_size[1],
        0,
        1,
        0,
        1,
        0,
        1,
        pad,
        stride,
        dilation,
        groups,
        dtype,
        out_channels,
        weight_format,
    )
    model = tei.make_ethosn_composite(model, "ethos-n.qnn_conv2d")
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)


@requires_ethosn
def test_conv2d_out_of_range_scale():
    """Check Conv2D scale out of range error."""
    np.random.seed(0)

    input_sc = 1024
    kernel_sc = 1024
    output_sc = 1

    model, _ = _get_model(
        (1, 4, 4, 4),
        1,
        1,
        0,
        input_sc,
        0,
        kernel_sc,
        0,
        output_sc,
        "none",
        (1, 1),
        (1, 1),
        1,
        "uint8",
        8,
        "HWIO",
    )
    model = tei.make_ethosn_composite(model, "ethos-n.qnn_conv2d")
    mod = tei.make_ethosn_partition(model)

    expected_err_msg = (
        "Overall scale (of the input * weights / output) should be in the range (2^-32, 65536)"
    )
    tei.test_error(mod, {}, expected_err_msg)
