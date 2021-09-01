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
import numpy as np
import math

import tvm.contrib.ethosu.cascader as cs
from tvm.contrib.ethosu.cascader.stripe_config import StripeConfig, count_stripes

from .infra import make_matrices


@pytest.mark.parametrize(
    "op_type, activation, kernel, stride, dilation, padding, in_shape, out_shape, expected",
    [
        (
            "ethosu_conv2d",
            "NONE",
            (34, 19),
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
            (1, 266, 111, 15),
            (1, 117, 47, 15),
            (1, 7, 6, 16),
        ),
        (
            "ethosu_conv2d",
            "NONE",
            (14, 14),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 125, 63, 64),
            (1, 112, 50, 128),
            (1, 5, 8, 16),
        ),
        (
            "ethosu_conv2d",
            "NONE",
            (7, 1),
            (2, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 13, 4, 12),
            (1, 4, 4, 511),
            (1, 4, 4, 16),
        ),
        (
            "ethosu_conv2d",
            "NONE",
            (5, 5),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 96, 16, 276),
            (1, 92, 12, 16),
            (1, 16, 4, 16),
        ),
        (
            "ethosu_conv2d",
            "NONE",
            (5, 5),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 96, 16, 276),
            (1, 92, 12, 1),
            (1, 8, 12, 8),
        ),
        (
            "ethosu_conv2d",
            "NONE",
            (3, 3),
            (1, 1),
            (2, 2),
            (0, 0, 0, 0),
            (1, 62, 94, 32),
            (1, 58, 90, 16),
            (1, 10, 6, 16),
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
def test_best_block_config(
    op_type,
    activation,
    kernel,
    stride,
    dilation,
    padding,
    in_shape,
    out_shape,
    expected,
    layouts,
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

    device_config = cs.EthosuDeviceConfig("ethos-u55-128")
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
    if layouts[1] == "NHCWB16":
        block_shape = tuple(
            int(math.ceil(n))
            for n in np.matmul(
                nhcwb16_to_nhwc,
                [int(x) for x in block.output_shape]
                + [
                    1,
                ],
            ).tolist()[:-1]
        )
    else:
        block_shape = tuple(int(a) for a in block.output_shape)

    assert block_shape == expected


if __name__ == "__main__":
    pytest.main([__file__])
