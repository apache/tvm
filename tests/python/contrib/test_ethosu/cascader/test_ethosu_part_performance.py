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

from functools import reduce
import numpy as np
import math

import tvm.contrib.ethosu.cascader as cs
from tvm.contrib.ethosu.cascader.device_config import _Shape

from .infra import make_matrices


@pytest.mark.parametrize(
    "acc_config, expected",
    [
        ("ethos-u55-256", (1, 0.125, 0.75, 0.375, 0.75)),
        ("ethos-u55-128", (1, 0.25, 1.5, 0.75, 0.75)),
        ("ethos-u55-64", (1, 0.5, 3, 1.5, 1.5)),
        ("ethos-u55-32", (2, 1, 6, 3, 3)),
    ],
)
def test_device_config_cycles(acc_config, expected):
    device_config = cs.EthosuDeviceConfig(acc_config)

    conv_type = "ethosu_conv2d"
    conv_str = None
    conv_ifm_dtype = "int8"
    conv_ofm_dtype = "int8"
    conv_activation = "LUT"
    conv_cycles = device_config._get_output_cycles(
        conv_type, conv_str, conv_ifm_dtype, conv_ofm_dtype, conv_activation
    )
    assert conv_cycles == expected[0]

    pool_type = "ethosu_pooling"
    pool_str = "MAX"
    pool_ifm_dtype = "int8"
    pool_ofm_dtype = "int8"
    pool_activation = "NONE"
    pool_cycles = device_config._get_output_cycles(
        pool_type, pool_str, pool_ifm_dtype, pool_ofm_dtype, pool_activation
    )
    assert pool_cycles == expected[1]

    add_type = "ethosu_binary_elementwise"
    add_str = "ADD"
    add_ifm_dtype = "int8"
    add_ofm_dtype = "int8"
    add_activation = "NONE"
    add_cycles = device_config._get_output_cycles(
        add_type, add_str, add_ifm_dtype, add_ofm_dtype, add_activation
    )
    assert add_cycles == expected[2]

    mul_type = "ethosu_binary_elementwise"
    mul_str = "MUL"
    mul_ifm_dtype = "int8"
    mul_ofm_dtype = "int8"
    mul_activation = "NONE"
    mul_cycles = device_config._get_output_cycles(
        mul_type, mul_str, mul_ifm_dtype, mul_ofm_dtype, mul_activation
    )
    assert mul_cycles == expected[3]

    mul_32_type = "ethosu_binary_elementwise"
    mul_32_str = "MUL"
    mul_32_ifm_dtype = "int8"
    mul_32_ofm_dtype = "int32"
    mul_32_activation = "NONE"
    mul_32_cycles = device_config._get_output_cycles(
        mul_32_type, mul_32_str, mul_32_ifm_dtype, mul_32_ofm_dtype, mul_32_activation
    )
    assert mul_32_cycles == expected[4]


@pytest.mark.parametrize(
    "accelerator, op_type, activation, kernel, stride, dilation, padding, in_shape, out_shape, block_shape, input_block_shape, expected",
    [
        (
            "ethos-u55-128",
            "ethosu_conv2d",
            "NONE",
            (3, 3),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 16, 16, 96),
            (1, 16, 16, 96),
            (1, 8, 8, 16),
            (1, 10, 10, 32),
            167733,
        ),
        (
            "ethos-u55-128",
            "ethosu_conv2d",
            "NONE",
            (10, 4),
            (2, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 58, 13, 1),
            (1, 25, 10, 276),
            (1, 6, 10, 32),
            (1, 18, 14, 8),
            174105,
        ),
        (
            "ethos-u55-128",
            "ethosu_depthwise_conv2d",
            "NONE",
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1, 1, 1),
            (1, 25, 10, 276),
            (1, 13, 5, 276),
            (1, 7, 6, 16),
            (1, 15, 14, 16),
            17590,
        ),
        (
            "ethos-u55-128",
            "ethosu_depthwise_conv2d",
            "NONE",
            (4, 9),
            (1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 28, 81, 42),
            (1, 25, 73, 41),
            (1, 4, 16, 16),
            (1, 7, 24, 16),
            173414,
        ),
    ],
)
def test_conv_performance(
    accelerator,
    op_type,
    activation,
    kernel,
    stride,
    dilation,
    padding,
    in_shape,
    out_shape,
    block_shape,
    input_block_shape,
    expected,
):
    ifm_channels = in_shape[3]
    ifm_matrix, ifm_offset, weight_matrix, weight_offset, _, _ = make_matrices(
        op_type,
        kernel,
        stride,
        padding,
        "NHWC",
        "NHWC",
        dilation,
        ifm_channels,
    )

    propagator = cs.Propagator(ifm_matrix, ifm_offset)
    weight_propagator = cs.Propagator(weight_matrix, weight_offset)

    subkernels = ((kernel[0] + 7) // 8) * ((kernel[1] + 7) // 8)

    device_config = cs.EthosuDeviceConfig(accelerator)

    output_cycles = device_config._get_output_cycles(op_type, "", "int8", "int8", activation)
    output_cycles *= reduce(lambda a, b: a * b, block_shape, 1)
    is_partkernel = device_config.is_partkernel(
        op_type, ifm_channels, "int8", kernel[0] * kernel[1]
    )
    compute_cycles = device_config._estimate_compute_cycles_per_block(
        op_type,
        _Shape(block_shape),
        _Shape(input_block_shape),
        kernel[0],
        kernel[1],
        ifm_channels,
        "int8",
        is_partkernel,
    )
    block_configs = [
        cs.BlockConfig(input_block_shape, block_shape, compute_cycles, int(output_cycles))
    ]

    output_quantum = [1, 1, 2, 8]
    te_subgraph = cs.TESubgraph([], None)
    part = cs.EthosuPart(
        te_subgraph,
        [propagator, weight_propagator],
        output_quantum,
        subkernels,
        block_configs,
        1,
    )
    part.set_input(0, cs.Tensor(in_shape, "int8"))
    part.set_input(1, cs.Tensor([ifm_channels, kernel[0], kernel[1], out_shape[-1]], "int8"))
    part.set_output(cs.Tensor(out_shape, "int8"))

    stripes = [1] * len(output_quantum)
    offset = [0] * len(output_quantum)
    order = [1, 2, 3, 4]

    stripe_config = cs.StripeConfig(out_shape, out_shape, out_shape, order, stripes, offset)

    compute_cycles = part.get_performance_info(stripe_config, cs.BufferMode.ROLLING).compute_cycles
    tolerance = expected * 0.1

    assert expected - tolerance <= compute_cycles <= expected + tolerance


if __name__ == "__main__":
    tvm.testing.main()
