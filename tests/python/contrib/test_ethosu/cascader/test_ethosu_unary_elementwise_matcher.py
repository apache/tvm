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

from tvm import te
import tvm.contrib.ethosu.cascader as cs
from tvm.relay.backend.contrib.ethosu.te.unary_elementwise import (
    match_ethosu_unary_elementwise,
    unary_elementwise_compute,
)
from tvm.relay.backend.contrib.ethosu.te.common import get_layout_transform_matrices


def _make_matrices(ifm_layout, ofm_layout, ofm_channels):
    nhwc_to_nhcwb16, nhcwb16_to_nhwc = get_layout_transform_matrices(ofm_channels)
    ifm_matrix = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    if ofm_layout == "NHCWB16":
        ifm_matrix = np.matmul(ifm_matrix, nhcwb16_to_nhwc).tolist()
    if ifm_layout == "NHCWB16":
        ifm_matrix = np.matmul(nhwc_to_nhcwb16, ifm_matrix).tolist()

    return ifm_matrix


@pytest.mark.parametrize(
    "ofm_shape",
    [
        [1, 12, 15, 128],
        [1, 16, 16, 16],
        [1, 1, 1, 1024],
        [1, 53, 91, 7],
        [1, 182, 12, 72],
    ],
)
@pytest.mark.parametrize("ifm_layout", ["NHWC", "NHCWB16"])
@pytest.mark.parametrize("ofm_layout", ["NHWC", "NHCWB16"])
@pytest.mark.parametrize("op_type", ["ABS", "CLZ"])
def test_ethosu_unary_elementwise_matcher(ofm_shape, ifm_layout, ofm_layout, op_type):
    ifm_shape = ofm_shape.copy()
    ofm_channels = ofm_shape[3]
    nhwc_to_nhcwb16, _ = get_layout_transform_matrices(ofm_channels)
    if ifm_layout == "NHCWB16":
        ifm_shape = [
            int(math.ceil(n))
            for n in np.matmul(
                nhwc_to_nhcwb16,
                ifm_shape
                + [
                    1,
                ],
            ).tolist()[:-1]
        ]
    if ofm_layout == "NHCWB16":
        ofm_shape = [
            int(math.ceil(n))
            for n in np.matmul(
                nhwc_to_nhcwb16,
                ofm_shape
                + [
                    1,
                ],
            ).tolist()[:-1]
        ]
        order = [1, 2, 4, 3, 0]
    else:
        order = [1, 2, 3, 4]

    ifm = te.placeholder(ifm_shape, dtype="int8")
    lut = te.placeholder((), dtype="uint8")
    out = unary_elementwise_compute(
        ifm=ifm,
        lut=lut,
        operator_type=op_type,
        ifm_scale=1,
        ifm_zero_point=0,
        ofm_scale=1,
        ofm_zero_point=0,
        ofm_channels=ofm_channels,
        activation="NONE",
        clip_min=0,
        clip_max=0,
        rounding_mode="TFL",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    ifm_propagator = out.op.attrs["ifm_propagator"]

    offset = [0] * len(ofm_shape)
    stripes = [0] * len(ofm_shape)
    output_stripe_config = cs.StripeConfig(ofm_shape, ofm_shape, ofm_shape, order, stripes, offset)

    ifm_transform = _make_matrices(ifm_layout, ofm_layout, ofm_channels)

    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    part = match_ethosu_unary_elementwise(out, device_config)

    assert isinstance(part, cs.EthosuPart)
    assert len(part.propagators) == 1
    assert part.propagators[0].transform == ifm_transform

    propagated_ifm = ifm_propagator.propagate(output_stripe_config).shape

    # The layout transforms that have the exact number of output channels in them
    # will lose no information about the number of channels
    assert ifm_shape == propagated_ifm


if __name__ == "__main__":
    tvm.testing.main()
