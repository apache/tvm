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

import numpy as np

from tvm import te
import tvm.contrib.ethosu.cascader as cs
from tvm.relay.backend.contrib.ethosu.te.depthwise import (
    match_ethosu_depthwise_conv2d,
    depthwise_conv2d_compute,
)
from .infra import make_matrices


@pytest.mark.parametrize("kernel", [(3, 3), (2, 1), (3, 5)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("dilation", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("padding", [(0, 0, 0, 0), (3, 2, 3, 2), (2, 1, 0, 1)])
@pytest.mark.parametrize("ifm_layout", ["NHWC", "NHCWB16"])
@pytest.mark.parametrize("ofm_layout", ["NHWC", "NHCWB16"])
def test_ethosu_depthwise2d_matcher(kernel, stride, dilation, padding, ifm_layout, ofm_layout):
    ofm_channels = 57
    if ifm_layout == "NHWC":
        ifm_shape = (1, 12, 15, ofm_channels)
    else:
        ifm_shape = (1, 12, 1 + ((ofm_channels - 1) // 16), 15, 16)
    kernel_h, kernel_w = kernel
    ifm = te.placeholder(ifm_shape, dtype="int8")
    weight = te.placeholder((ofm_channels, kernel_h, kernel_w, 1), dtype="int8")
    scale_bias = te.placeholder((ofm_channels, 10), dtype="uint8")
    lut = te.placeholder((), dtype="uint8")
    out = depthwise_conv2d_compute(
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
        rounding_mode="TFL",
        upscale="NONE",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
        ofm_dtype=ifm.dtype,
    )
    (
        ifm_transform,
        ifm_offset,
        weight_transform,
        weight_offset,
        scale_bias_transform,
        scale_bias_offset,
    ) = make_matrices(
        "ethosu_depthwise_conv2d",
        kernel,
        stride,
        padding,
        ifm_layout,
        ofm_layout,
        dilation,
        ofm_channels=ofm_channels,
    )

    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    part = match_ethosu_depthwise_conv2d(out, device_config)

    assert isinstance(part, cs.EthosuPart)
    assert len(part.propagators) == 3
    assert part.propagators[0].transform == ifm_transform
    assert part.propagators[0].offset == ifm_offset
    assert part.propagators[1].transform == weight_transform
    assert part.propagators[1].offset == weight_offset
    assert part.propagators[2].transform == scale_bias_transform
    assert part.propagators[2].offset == scale_bias_offset


if __name__ == "__main__":
    tvm.testing.main()
