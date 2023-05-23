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

from .infra import make_matrices


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
    ) = make_matrices(
        "ethosu_conv2d",
        kernel,
        stride,
        padding,
        ifm_layout,
        ofm_layout,
        dilation,
        ifm_channels,
        ofm_channels,
    )

    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    part = match_ethosu_conv2d(out, device_config)

    assert isinstance(part, cs.EthosuPart)
    assert len(part.propagators) == 3
    assert part.propagators[0].transform == ifm_transform
    assert part.propagators[0].offset == ifm_offset
    assert part.propagators[1].transform == weight_transform
    assert part.propagators[1].offset == weight_offset
    assert part.propagators[2].transform == scale_bias_transform
    assert part.propagators[2].offset == scale_bias_offset


@pytest.mark.parametrize(
    "ifm_layout, ofm_layout, ifm_channels, expected_cycles",
    [
        ("NHWC", "NHWC", 24, 2304),
        ("NHCWB16", "NHWC", 12, 2352),
        ("NHWC", "NHCWB16", 38, 7056),
        ("NHCWB16", "NHCWB16", 55, 4608),
    ],
)
def test_ethosu_conv2d_block_config_from_matcher(
    ifm_layout, ofm_layout, ifm_channels, expected_cycles
):
    ofm_channels = 10
    ifm_height = 123
    ifm_width = 155

    ifm_shape = (
        (1, ifm_height, ifm_width, ifm_channels)
        if ifm_layout == "NHWC"
        else (1, ifm_height, 1 + ((ifm_channels - 1) // 16), ifm_width, 16)
    )
    weight_shape = (ofm_channels, 3, 3, ifm_channels)
    scale_bias_shape = (ofm_channels, 10)

    ifm = te.placeholder(ifm_shape, dtype="int8")
    weight = te.placeholder(weight_shape, dtype="int8")
    scale_bias = te.placeholder(scale_bias_shape, dtype="uint8")
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
        strides=(1, 1),
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        activation="NONE",
        clip_min=0,
        clip_max=0,
        upscale="NONE",
        rounding_mode="TFL",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )

    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    part = match_ethosu_conv2d(out, device_config)

    ofm_shape = [int(i) for i in part.subgraph.output_tensor.shape]

    # Add inputs and outputs to the part
    input_tensor = cs.Tensor(ifm_shape, "int8")
    part.set_input(0, input_tensor)
    weight_tensor = cs.Tensor(weight_shape, "int8")
    part.set_input(1, weight_tensor)
    scale_bias_tensor = cs.Tensor(scale_bias_shape, "int8")
    part.set_input(2, scale_bias_tensor)
    output_tensor = cs.Tensor(ofm_shape, "int8")
    part.set_output(output_tensor)

    # Create a stripe of a size of the output tensor
    order = [1, 2, 3, 4] if ofm_layout == "NHWC" else [1, 2, 4, 3, 0]
    stripes = [1] * len(order)
    offset = [0] * len(order)

    stripe_config = cs.StripeConfig(ofm_shape, ofm_shape, ofm_shape, order, stripes, offset)

    block = part.get_block_config(stripe_config)

    # Since we dont know the values of the variables we passed to the get_valid_block_configs in
    # the matcher, best we can do is to verify the compute cycle count since the channels have a
    # significant effect on it
    assert block.compute_cycles == expected_cycles


if __name__ == "__main__":
    tvm.testing.main()
