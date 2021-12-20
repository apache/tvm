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
import pytest

pytest.importorskip("ethosu.vela")

from tvm import te
import tvm.contrib.ethosu.cascader as cs
from tvm.relay.backend.contrib.ethosu.te.convolution import match_ethosu_conv2d, conv2d_compute

import numpy as np


def _make_matrices(kernel, stride, dilation, padding, ifm_channels, ifm_layout, ofm_layout):
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    nhwc_to_nhcwb16 = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1 / 16, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 16],
        [0, 0, 0, 0, 1],
    ]
    nhcwb16_to_nhwc = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 16, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    ifm_matrix = [
        [1, 0, 0, 0, 0],
        [0, stride_h, 0, 0, (dilated_kernel_h - stride_h)],
        [0, 0, stride_w, 0, (dilated_kernel_w - stride_w)],
        [0, 0, 0, 0, ifm_channels],
        [0, 0, 0, 0, 1],
    ]
    weight_matrix = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, kernel_h],
        [0, 0, 0, 0, kernel_w],
        [0, 0, 0, 0, ifm_channels],
        [0, 0, 0, 0, 1],
    ]
    scale_bias_matrix = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 10],
        [0, 0, 0, 0, 1],
    ]
    if ofm_layout == "NHCWB16":
        ifm_matrix = np.matmul(ifm_matrix, nhcwb16_to_nhwc).tolist()
        weight_matrix = np.matmul(weight_matrix, nhcwb16_to_nhwc).tolist()
        scale_bias_matrix = np.matmul(scale_bias_matrix, nhcwb16_to_nhwc).tolist()
    if ifm_layout == "NHCWB16":
        ifm_matrix = np.matmul(nhwc_to_nhcwb16, ifm_matrix).tolist()

    ifm_offset = (
        [0, -padding[0], -padding[1], 0]
        if ifm_layout == "NHWC"
        else [0, -padding[0], 0, -padding[1], 0]
    )
    weight_offset = [0, 0, 0, 0]
    scale_bias_offset = [0, 0]
    return (
        ifm_matrix,
        ifm_offset,
        weight_matrix,
        weight_offset,
        scale_bias_matrix,
        scale_bias_offset,
    )


@pytest.mark.parametrize("kernel", [(3, 3), (2, 1), (3, 5)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("dilation", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("padding", [(0, 0, 0, 0), (3, 2, 3, 2), (2, 1, 0, 1)])
@pytest.mark.parametrize("ifm_channels", [8, 57])
@pytest.mark.parametrize("ifm_layout", ["NHWC", "NHCWB16"])
@pytest.mark.parametrize("ofm_layout", ["NHWC", "NHCWB16"])
def test_ethosu_conv2d_matcher(
    kernel, stride, dilation, padding, ifm_channels, ifm_layout, ofm_layout
):
    if ifm_layout == "NHWC":
        ifm_shape = (1, 12, 15, ifm_channels)
    else:
        ifm_shape = (1, 12, 1 + ((ifm_channels - 1) // 16), 15, 16)
    ofm_channels = 8
    kernel_h, kernel_w = kernel
    ifm = te.placeholder(ifm_shape, dtype="int8")
    weight = te.placeholder((ofm_channels, kernel_h, kernel_w, ifm_channels), dtype="int8")
    scale_bias = te.placeholder((ofm_channels, 10), dtype="uint8")
    lut = te.placeholder((), dtype="uint8")
    out = conv2d_compute(
        ifm=ifm,
        weight=weight,
        scale_bias=scale_bias,
        lut=lut,
        ifm_scale=1,
        ifm_zero_point=0,
        ofm_scale=1,
        ofm_zero_point=0,
        weight_zero_point=0,
        strides=stride,
        padding=padding,
        dilation=dilation,
        activation="NONE",
        clip_min=0,
        clip_max=0,
        upscale="NONE",
        rounding_mode="TFL",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    (
        ifm_transform,
        ifm_offset,
        weight_transform,
        weight_offset,
        scale_bias_transform,
        scale_bias_offset,
    ) = _make_matrices(
        kernel,
        stride,
        dilation,
        padding,
        ifm_channels,
        ifm_layout,
        ofm_layout,
    )

    part = match_ethosu_conv2d(out)

    assert isinstance(part, cs.EthosuPart)
    assert len(part.propagators) == 3
    assert part.propagators[0].transform == ifm_transform
    assert part.propagators[0].offset == ifm_offset
    assert part.propagators[1].transform == weight_transform
    assert part.propagators[1].offset == weight_offset
    assert part.propagators[2].transform == scale_bias_transform
    assert part.propagators[2].offset == scale_bias_offset


if __name__ == "__main__":
    pytest.main([__file__])
