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

from .infra import make_matrices


@pytest.mark.parametrize(
    "id, op_type, activation, kernel, stride, dilation, padding, in_shape, out_shape",
    [
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
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                ((1, 4, 4, 16), (1, 4, 1, 4, 16)),
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 10, 6, 4), (1, 16, 1, 4, 4)),
                ((1, 10, 3, 16), (1, 10, 1, 3, 16)),
            ],
        ),
        (
            "ethos-u55-64",
            [
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 6, 5, 16), (1, 6, 1, 5, 16)),
                ((1, 4, 4, 16), (1, 4, 1, 4, 16)),
                ((1, 8, 4, 16), (1, 8, 1, 4, 16)),
                ((1, 10, 6, 8), (1, 16, 1, 4, 8)),
                ((1, 10, 3, 16), (1, 10, 1, 3, 16)),
            ],
        ),
        (
            "ethos-u55-128",
            [
                ((1, 7, 6, 16), (1, 7, 1, 6, 16)),
                ((1, 5, 8, 16), (1, 5, 1, 8, 16)),
                ((1, 4, 4, 16), (1, 4, 1, 4, 16)),
                ((1, 16, 4, 16), (1, 16, 1, 4, 16)),
                ((1, 8, 12, 8), (1, 8, 1, 12, 8)),
                ((1, 10, 6, 16), (1, 10, 1, 6, 16)),
            ],
        ),
        (
            "ethos-u55-256",
            [
                ((1, 14, 8, 16), (1, 14, 1, 8, 16)),
                ((1, 16, 8, 16), (1, 16, 1, 8, 16)),
                ((1, 4, 4, 16), (1, 4, 1, 4, 16)),
                ((1, 32, 4, 16), (1, 32, 1, 4, 16)),
                ((1, 20, 12, 8), (1, 20, 1, 12, 8)),
                ((1, 20, 6, 16), (1, 20, 1, 6, 16)),
            ],
        ),
    ],
)
def test_best_block_config(
    id,
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
        [0, 0, 16, 0, 1, -16],
        [0, 0, 0, 0, 0, 1],
    ]
    ifm_matrix, ifm_offset, weight_matrix, weight_offset, _, _ = make_matrices(
        kernel, stride, dilation, padding, in_shape[3], layouts[0], layouts[1]
    )

    ofm_channels = out_shape[3]
    ifm_channels = in_shape[3]

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

    order = [1, 2, 3, 4] if layouts[1] == "NHCWB16" else [1, 2, 4, 3, 0]
    stripes = [1] * len(output_quantum)
    offset = [0] * len(output_quantum)

    stripe_config = cs.StripeConfig(out_shape, out_shape, out_shape, order, stripes, offset)

    block = part.get_block_config(stripe_config)
    block_shape = tuple(int(a) for a in block.output_shape)
    if layouts[1] == "NHCWB16":
        assert block_shape == expected_block_configs[id][1]
    else:
        assert block_shape == expected_block_configs[id][0]


if __name__ == "__main__":
    pytest.main([__file__])
