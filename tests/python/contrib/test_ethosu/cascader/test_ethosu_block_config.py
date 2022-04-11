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
                ((1, 4, 4, 16), (1, 4, 1, 4, 16)),
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 10, 6, 4), (1, 5, 1, 12, 4), (1, 10, 1, 6, 4)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                # Depthwise Conv2D
                ((1, 6, 10, 16), (1, 6, 1, 10, 16)),
                ((1, 7, 5, 16), (1, 7, 1, 5, 16)),
                # Pooling
                ((1, 1, 1, 16), (1, 1, 1, 1, 16)),
                ((1, 9, 6, 16), (1, 9, 1, 6, 16)),
            ],
        ),
        (
            "ethos-u55-64",
            [
                # Conv2D
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                ((1, 4, 4, 16), (1, 4, 1, 4, 16)),
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 10, 6, 8), (1, 10, 1, 6, 8)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                # Depthwise Conv2D
                ((1, 6, 10, 16), (1, 6, 1, 10, 16)),
                ((1, 7, 5, 16), (1, 7, 1, 5, 16)),
                # Pooling
                ((1, 1, 1, 16), (1, 1, 1, 1, 16)),
                ((1, 9, 6, 16), (1, 9, 1, 6, 16)),
            ],
        ),
        (
            "ethos-u55-128",
            [
                # Conv2D
                ((1, 7, 6, 16), (1, 7, 1, 6, 16)),
                ((1, 5, 8, 16), (1, 5, 1, 8, 16)),
                ((1, 4, 4, 16), (1, 4, 1, 4, 16)),
                ((1, 16, 4, 16), (1, 16, 1, 4, 16)),
                ((1, 8, 12, 8), (1, 8, 1, 12, 8)),
                ((1, 10, 6, 16), (1, 10, 1, 6, 16)),
                # Depthwise Conv2D
                ((1, 7, 10, 16), (1, 7, 1, 10, 16)),
                ((1, 7, 6, 16), (1, 7, 1, 6, 16)),
                # Pooling
                ((1, 1, 2, 80), (1, 1, 5, 2, 16)),
                ((1, 10, 6, 16), (1, 10, 1, 6, 16)),
            ],
        ),
        (
            "ethos-u55-256",
            [
                # Conv2D
                ((1, 14, 8, 16), (1, 14, 1, 8, 16)),
                ((1, 16, 8, 16), (1, 16, 1, 8, 16)),
                ((1, 4, 4, 16), (1, 4, 1, 4, 16)),
                ((1, 32, 4, 16), (1, 10, 12, 16), (1, 32, 1, 4, 16), (1, 10, 1, 12, 16)),
                ((1, 20, 12, 8), (1, 20, 1, 12, 8)),
                ((1, 12, 10, 16), (1, 12, 1, 10, 16)),
                # Depthwise Conv2D
                ((1, 8, 20, 16), (1, 8, 1, 20, 16)),
                ((1, 14, 6, 16), (1, 14, 1, 6, 16)),
                # Pooling
                ((1, 2, 2, 48), (1, 2, 3, 2, 16)),
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


if __name__ == "__main__":
    pytest.main([__file__])
