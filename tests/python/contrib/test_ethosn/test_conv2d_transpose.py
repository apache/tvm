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

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.relay.op.contrib import ethosn_api_version
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
    stride,
    dilation,
    groups,
    kernel_layout,
    dtype,
    out_channels,
    bias,
):
    """Return a model and any parameters it may have"""
    a = relay.var("a", shape=shape, dtype=dtype)
    p = tei.get_same_padding((shape[1], shape[2]), (kernel_h, kernel_w), dilation, stride)
    weight_shape = (shape[3], out_channels // groups, kernel_h, kernel_w)

    weight_data = tvm.nd.array(
        np.random.randint(
            np.iinfo(dtype).min,
            high=(np.iinfo(dtype).max + 1),
            size=weight_shape,
            dtype=dtype,
        )
    )
    weights = relay.const(weight_data, dtype)
    op = relay.qnn.op.conv2d_transpose(
        a,
        weights,
        input_zero_point=relay.const(input_zp, "int32"),
        input_scale=relay.const(input_sc, "float32"),
        kernel_zero_point=relay.const(kernel_zp, "int32"),
        kernel_scale=relay.const(kernel_sc, "float32"),
        kernel_size=(kernel_h, kernel_w),
        padding=p,
        strides=stride,
        dilation=dilation,
        data_layout="NHWC",
        kernel_layout=kernel_layout,
        out_dtype="int32",
        channels=out_channels,
        groups=groups,
    )
    if bias:
        bias_data = tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min,
                high=np.iinfo(dtype).max + 1,
                size=(out_channels,),
                dtype="int32",
            )
        )
        biasc = relay.const(bias_data, "int32")
        op = relay.nn.bias_add(op, biasc, axis=3)

    if isinstance(kernel_sc, tvm.runtime.ndarray.NDArray):
        req_input_sc = [sc * input_sc for sc in kernel_sc.numpy()]
    else:
        req_input_sc = input_sc * kernel_sc

    op = relay.qnn.op.requantize(
        op,
        input_zero_point=relay.const(input_zp, "int32"),
        input_scale=relay.const(req_input_sc, "float32"),
        output_zero_point=relay.const(output_zp, "int32"),
        output_scale=relay.const(output_sc, "float32"),
        axis=3,
        rounding="UPWARD",
        out_dtype=dtype,
    )
    params = {"w": weight_data}
    if bias:
        params["b"] = bias_data
    return op, params


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "ifm_shape,strides,kernel_size,out_channels,bias",
    [
        ((1, 2, 2, 1), (2, 2), (1, 1), 1, False),
        ((1, 2, 2, 5), (2, 2), (3, 5), 4, False),
        ((1, 7, 7, 4), (2, 2), (7, 7), 8, True),
    ],
)
def test_conv2d_transpose(ifm_shape, strides, kernel_size, out_channels, dtype, bias):
    """Check transpose convolution output with TVM."""
    np.random.seed(0)

    kernel_layout = "IOHW"
    dilation = (1, 1)
    groups = 1

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max

    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    kernel_zp = np.random.randint(data_min, data_max)
    kernel_sc = np.random.random() * 4
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, kernel_zp, kernel_sc, ifm_shape[1], ifm_shape[2], ifm_shape[3]
    )

    model, params = _get_model(
        shape=ifm_shape,
        kernel_h=kernel_size[0],
        kernel_w=kernel_size[1],
        input_zp=input_zp,
        input_sc=input_sc,
        kernel_zp=kernel_zp,
        kernel_sc=kernel_sc,
        output_zp=output_zp,
        output_sc=output_sc,
        stride=strides,
        dilation=dilation,
        groups=groups,
        kernel_layout=kernel_layout,
        dtype=dtype,
        out_channels=out_channels,
        bias=bias,
    )

    outputs = []
    inputs = {
        "a": tvm.nd.array(np.random.randint(data_min, data_max + 1, size=ifm_shape, dtype=dtype))
    }

    for npu in [False, True]:
        mod = tei.make_module(model, params)
        outputs.append(tei.build_and_run(mod, inputs, 1, params, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "ifm_shape,strides,kernel_size,out_channels,bias",
    [
        ((1, 10, 20, 3), (1, 1), (8, 5), 4, False),
        ((1, 10, 10, 2), (2, 2), (7, 9), 8, True),
    ],
)
def test_conv2d_transpose_kernel_size_gt_8(
    ifm_shape, strides, kernel_size, out_channels, dtype, bias
):
    """Check transpose convolution for big kernel sizes."""
    if ethosn_api_version() in ["3.2.0", "3.1.0"]:
        pytest.skip("Skipping because NPU driver 22.11 fails to interpret zp used in the test.")

    np.random.seed(0)

    kernel_layout = "IOHW"
    dilation = (1, 1)
    groups = 1

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max

    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    kernel_zp = np.random.randint(data_min, data_max)
    kernel_sc = np.random.random() * 4
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, kernel_zp, kernel_sc, ifm_shape[1], ifm_shape[2], ifm_shape[3]
    )

    model, params = _get_model(
        shape=ifm_shape,
        kernel_h=kernel_size[0],
        kernel_w=kernel_size[1],
        input_zp=input_zp,
        input_sc=input_sc,
        kernel_zp=kernel_zp,
        kernel_sc=kernel_sc,
        output_zp=output_zp,
        output_sc=output_sc,
        stride=strides,
        dilation=dilation,
        groups=groups,
        kernel_layout=kernel_layout,
        dtype=dtype,
        out_channels=out_channels,
        bias=bias,
    )

    outputs = []
    inputs = {
        "a": tvm.nd.array(np.random.randint(data_min, data_max + 1, size=ifm_shape, dtype=dtype))
    }

    for npu in [False, True]:
        mod = tei.make_module(model, params)
        outputs.append(tei.build_and_run(mod, inputs, 1, params, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape, stride, dilation, groups, err_msg",
    [
        (
            (1, 4, 4, 4),
            (1, 1, 1),
            (1, 1),
            1,
            "stride size=3, stride size must = 2",
        ),
        (
            (1, 4, 4, 4),
            (2, 2),
            (2, 2),
            2,
            "dilation=[2, 2], dilation must = [1, 1]",
        ),
        (
            (2, 4, 4, 4),
            (1, 1),
            (1, 1),
            1,
            "batch size=2, batch size must = 1",
        ),
    ],
)
def test_conv2d_transpose_failure(
    shape,
    stride,
    dilation,
    groups,
    err_msg,
    dtype,
):
    """
    Test transpose_conv2d error messages.
    """
    np.random.seed(0)
    out_channels = 8

    model, _ = _get_model(
        shape=shape,
        kernel_h=1,
        kernel_w=1,
        input_zp=0,
        input_sc=1,
        kernel_zp=0,
        kernel_sc=1,
        output_zp=0,
        output_sc=1,
        stride=stride,
        dilation=dilation,
        groups=groups,
        kernel_layout="IOHW",
        dtype=dtype,
        out_channels=out_channels,
        bias=False,
    )
    model = tei.make_ethosn_composite(model, "ethos-n.qnn_conv2d_transpose")
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)
