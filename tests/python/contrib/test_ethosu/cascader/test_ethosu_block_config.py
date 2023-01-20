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
import math

import tvm
import tvm.contrib.ethosu.cascader as cs
from tvm.relay.backend.contrib.ethosu.te.common import get_layout_transform_matrices

from .infra import make_matrices


@pytest.mark.parametrize(
    "test_id, op_type, activation, kernel, stride, dilation, padding, in_shape, out_shape",
    [
        # Conv2D
        (
            0,
            "ethosu_conv2d",
            "NONE",
            (34, 19),
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
            (1, 266, 111, 15),
            (1, 117, 47, 15),
        ),
        (
            1,
            "ethosu_conv2d",
            "NONE",
            (14, 14),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 125, 63, 64),
            (1, 112, 50, 128),
        ),
        (
            2,
            "ethosu_conv2d",
            "NONE",
            (7, 1),
            (2, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 13, 4, 12),
            (1, 4, 4, 511),
        ),
        (
            3,
            "ethosu_conv2d",
            "NONE",
            (5, 5),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 96, 16, 276),
            (1, 92, 12, 16),
        ),
        (
            4,
            "ethosu_conv2d",
            "NONE",
            (5, 5),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 96, 16, 276),
            (1, 92, 12, 1),
        ),
        (
            5,
            "ethosu_conv2d",
            "NONE",
            (3, 3),
            (1, 1),
            (2, 2),
            (0, 0, 0, 0),
            (1, 62, 94, 32),
            (1, 58, 90, 16),
        ),
        # Depthwise Conv2D
        (
            6,
            "ethosu_depthwise_conv2d",
            "NONE",
            (3, 5),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 77, 23, 18),
            (1, 75, 19, 18),
        ),
        (
            7,
            "ethosu_depthwise_conv2d",
            "NONE",
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1, 1, 1),
            (1, 25, 10, 276),
            (1, 13, 5, 276),
        ),
        # Pooling
        (
            8,
            "ethosu_pooling",
            "NONE",
            (13, 5),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 13, 5, 276),
            (1, 1, 1, 276),
        ),
        (
            9,
            "ethosu_pooling",
            "NONE",
            (7, 3),
            (2, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 317, 14, 21),
            (1, 156, 12, 21),
        ),
    ],
)
@pytest.mark.parametrize(
    "layouts",
    [
        ("NHWC", "NHWC"),
        ("NHCWB16", "NHCWB16"),
        ("NHWC", "NHCWB16"),
        ("NHCWB16", "NHWC"),
    ],
)
@pytest.mark.parametrize(
    "acc_config, expected_block_configs",
    [
        (
            "ethos-u55-32",
            [
                # Conv2D
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                ((1, 4, 4, 96), (1, 4, 6, 4, 16)),
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 10, 6, 4), (1, 5, 1, 12, 4), (1, 8, 1, 4, 16)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                # Depthwise Conv2D
                ((1, 6, 10, 16), (1, 4, 1, 12, 16)),
                ((1, 8, 5, 16), (1, 6, 1, 5, 16)),
                # Pooling
                ((1, 1, 1, 128), (1, 1, 4, 1, 16)),
                ((1, 9, 6, 16), (1, 8, 1, 4, 16)),
            ],
        ),
        (
            "ethos-u55-64",
            [
                # Conv2D
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                ((1, 4, 4, 96), (1, 4, 6, 4, 16)),
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 10, 6, 8), (1, 8, 1, 4, 16)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                # Depthwise Conv2D
                ((1, 6, 10, 16), (1, 4, 1, 12, 16)),
                ((1, 8, 5, 16), (1, 6, 1, 5, 16)),
                # Pooling
                ((1, 1, 1, 128), (1, 1, 4, 1, 16)),
                ((1, 9, 6, 16), (1, 8, 1, 4, 16)),
            ],
        ),
        (
            "ethos-u55-128",
            [
                # Conv2D
                ((1, 7, 6, 16), (1, 7, 1, 6, 16)),
                ((1, 5, 8, 16), (1, 5, 1, 8, 16)),
                ((1, 4, 4, 128), (1, 4, 8, 4, 16)),
                ((1, 16, 4, 16), (1, 16, 1, 4, 16)),
                ((1, 8, 12, 8), (1, 10, 1, 6, 16)),
                ((1, 10, 6, 16), (1, 10, 1, 6, 16), (1, 6, 1, 6, 16)),
                # Depthwise Conv2D
                ((1, 7, 10, 16), (1, 7, 1, 10, 16), (1, 6, 1, 10, 16)),
                ((1, 10, 6, 16), (1, 10, 1, 6, 16), (1, 6, 1, 6, 16)),
                # Pooling
                # ((1, 1, 2, 16), (1, 1, 1, 2, 16)),
                ((1, 1, 2, 128), (1, 1, 4, 2, 16)),
                ((1, 10, 6, 16), (1, 9, 1, 6, 16)),
            ],
        ),
        (
            "ethos-u55-256",
            [
                # Conv2D
                ((1, 14, 8, 16), (1, 14, 1, 8, 16)),
                ((1, 16, 8, 16), (1, 16, 1, 8, 16)),
                ((1, 4, 4, 128), (1, 4, 8, 4, 16)),
                ((1, 32, 4, 16), (1, 10, 12, 16), (1, 32, 1, 4, 16), (1, 10, 1, 12, 16)),
                ((1, 20, 12, 8), (1, 10, 1, 12, 16)),
                ((1, 12, 10, 16), (1, 12, 1, 10, 16)),
                # Depthwise Conv2D
                ((1, 8, 20, 16), (1, 6, 1, 20, 16), (1, 6, 2, 20, 16)),
                ((1, 14, 6, 16), (1, 12, 1, 6, 16)),
                # Pooling
                # ((1, 2, 2, 16), (1, 2, 1, 2, 16)),
                ((1, 2, 2, 128), (1, 2, 6, 2, 16)),
                ((1, 10, 12, 16), (1, 10, 1, 12, 16)),
            ],
        ),
    ],
)
def test_best_block_config(
    test_id,
    op_type,
    activation,
    kernel,
    stride,
    dilation,
    padding,
    in_shape,
    out_shape,
    layouts,
    acc_config,
    expected_block_configs,
):
    ofm_channels = out_shape[3]
    ifm_channels = in_shape[3]

    nhwc_to_nhcwb16, _ = get_layout_transform_matrices(ofm_channels)

    ifm_matrix, ifm_offset, weight_matrix, weight_offset, _, _ = make_matrices(
        op_type,
        kernel,
        stride,
        padding,
        layouts[0],
        layouts[1],
        dilation,
        ifm_channels,
        ofm_channels,
    )

    if layouts[0] == "NHCWB16":
        in_shape = [
            int(math.ceil(n)) for n in np.matmul(nhwc_to_nhcwb16, in_shape + (1,)).tolist()[:-1]
        ]
    if layouts[1] == "NHCWB16":
        out_shape = [
            int(math.ceil(n)) for n in np.matmul(nhwc_to_nhcwb16, out_shape + (1,)).tolist()[:-1]
        ]

    propagator = cs.Propagator(ifm_matrix, ifm_offset)
    weight_propagator = cs.Propagator(weight_matrix, weight_offset)

    subkernels = ((kernel[0] + 7) // 8) * ((kernel[1] + 7) // 8)

    op_attrs = {
        "op": op_type,
        "activation": activation,
        "stride_h": stride[0],
        "stride_w": stride[1],
        "dilation_h": dilation[0],
        "dilation_w": dilation[1],
    }

    device_config = cs.EthosuDeviceConfig(acc_config)
    block_configs = device_config.get_valid_block_configs(
        propagator,
        op_attrs,
        out_shape,
        ofm_channels,
        ifm_channels,
        layouts[1],
        layouts[0],
        "int8",
        "int8",
        kernel[0],
        kernel[1],
    )

    output_quantum = [1, 1, 2, 8]
    if layouts[1] == "NHCWB16":
        output_quantum = [1, 1, 1, 2, 8]

    # Create EthosUPart
    te_subgraph = cs.TESubgraph([], None)
    part = cs.EthosuPart(
        te_subgraph,
        [propagator, weight_propagator],
        output_quantum,
        subkernels,
        block_configs,
        1,
    )
    # Add tensors
    input_tensor = cs.Tensor(in_shape, "int8")
    part.set_input(0, input_tensor)
    if op_type == "ethosu_conv2d":
        weight_tensor = cs.Tensor([ofm_channels, kernel[0], kernel[1], ifm_channels], "int8")
        part.set_input(1, weight_tensor)
    elif op_type == "ethosu_depthwise_conv2d":
        weight_tensor = cs.Tensor([ofm_channels, kernel[0], kernel[1], 1], "int8")
        part.set_input(1, weight_tensor)

    output_tensor = cs.Tensor(out_shape, "int8")
    part.set_output(output_tensor)

    order = [1, 2, 3, 4] if layouts[1] == "NHCWB16" else [1, 2, 4, 3, 0]
    stripes = [1] * len(output_quantum)
    offset = [0] * len(output_quantum)

    stripe_config = cs.StripeConfig(out_shape, out_shape, out_shape, order, stripes, offset)

    block = part.get_block_config(stripe_config)
    block_shape = tuple(int(a) for a in block.output_shape)

    assert block_shape in expected_block_configs[test_id]


@pytest.mark.parametrize(
    "ofm_layout, block_config_str, expected_block_shape",
    [
        ("NHWC", "4x4x8", [1, 4, 4, 8]),
        ("NHCWB16", "4x4x8", [1, 4, 1, 4, 16]),
        ("NHCWB16", "4x4x24", [1, 4, 2, 4, 16]),
    ],
)
def test_force_block_config_kernelwise(ofm_layout, block_config_str, expected_block_shape):
    op_type = "ethosu_pooling"
    activation = "NONE"
    kernel = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    dilation = (1, 1)
    ifm_channels = 32
    out_shape = (1, 8, 10, 16)

    ifm_matrix, ifm_offset, _, _, _, _ = make_matrices(
        op_type, kernel, stride, padding, "NHWC", ofm_layout, dilation, ifm_channels
    )

    ofm_channels = out_shape[3]

    propagator = cs.Propagator(ifm_matrix, ifm_offset)

    op_attrs = {
        "op": op_type,
        "activation": activation,
        "stride_h": stride[0],
        "stride_w": stride[1],
        "dilation_h": dilation[0],
        "dilation_w": dilation[1],
    }

    config = {
        "enable_cascader": True,
        "dev_force_block_config": block_config_str,
    }
    with tvm.transform.PassContext(config={"relay.ext.ethos-u.options": config}):
        device_config = cs.EthosuDeviceConfig("ethos-u55-128")
        block_configs = device_config.get_valid_block_configs(
            propagator,
            op_attrs,
            out_shape,
            ofm_channels,
            ifm_channels,
            ofm_layout,
            "NHWC",
            "int8",
            "int8",
            kernel[0],
            kernel[1],
        )

    assert len(block_configs) == 1
    assert block_configs[0].output_shape == expected_block_shape


@pytest.mark.parametrize(
    "ofm_layout, block_config_str, expected_block_shape",
    [
        ("NHWC", "4x4x8", [1, 4, 4, 8]),
        ("NHCWB16", "4x4x8", [1, 4, 1, 4, 16]),
        ("NHCWB16", "4x4x24", [1, 4, 2, 4, 16]),
    ],
)
def test_force_block_config_elementwise(ofm_layout, block_config_str, expected_block_shape):
    op_type = "ethosu_elementwise_unary"
    op_str = "ABS"
    activation = "NONE"
    ofm_shape = (1, 8, 10, 16)
    ifm_matrix = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    ifm_offset = [0, 0, 0, 0]

    propagator = cs.Propagator(ifm_matrix, ifm_offset)

    op_attrs = {
        "op": op_type,
        "operator_type": op_str,
        "activation": activation,
        "clip_min": 0,
        "clip_max": 0,
        "rounding_mode": "TFL",
    }

    config = {
        "enable_cascader": True,
        "dev_force_block_config": block_config_str,
    }
    with tvm.transform.PassContext(config={"relay.ext.ethos-u.options": config}):
        device_config = cs.EthosuDeviceConfig("ethos-u55-128")
        block_configs = device_config.get_elementwise_block_config(
            propagator,
            None,
            op_attrs,
            ofm_shape,
            ofm_layout,
            "NWHC",
            None,
            "int8",
            "int8",
        )

    assert len(block_configs) == 1
    assert block_configs[0].output_shape == expected_block_shape


if __name__ == "__main__":
    tvm.testing.main()
