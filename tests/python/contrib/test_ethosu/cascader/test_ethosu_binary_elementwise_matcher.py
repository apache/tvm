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
from tvm.relay.backend.contrib.ethosu.te.binary_elementwise import (
    match_ethosu_binary_elementwise,
    binary_elementwise_compute,
)
from tvm.relay.backend.contrib.ethosu.te.common import get_layout_transform_matrices


def _make_matrices(broadcast, ifm_layout, ifm2_layout, ofm_layout, ofm_channels):
    broadcast_h, broadcast_w, broadcast_c = broadcast
    nhwc_to_nhcwb16, nhcwb16_to_nhwc = get_layout_transform_matrices(ofm_channels)
    ifm_matrix = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    ifm2_matrix = [
        [1, 0, 0, 0, 0],
        [0, (1 - broadcast_h), 0, 0, broadcast_h],
        [0, 0, (1 - broadcast_w), 0, broadcast_w],
        [0, 0, 0, (1 - broadcast_c), broadcast_c],
        [0, 0, 0, 0, 1],
    ]
    if ofm_layout == "NHCWB16":
        ifm_matrix = np.matmul(ifm_matrix, nhcwb16_to_nhwc).tolist()
        ifm2_matrix = np.matmul(ifm2_matrix, nhcwb16_to_nhwc).tolist()
    if ifm_layout == "NHCWB16":
        ifm_matrix = np.matmul(nhwc_to_nhcwb16, ifm_matrix).tolist()
    if ifm2_layout == "NHCWB16":
        ifm2_matrix = np.matmul(nhwc_to_nhcwb16, ifm2_matrix).tolist()

    return (ifm_matrix, ifm2_matrix)


@pytest.mark.parametrize(
    "ofm_shape",
    [
        [1, 12, 15, 128],
        [1, 16, 16, 16],
        [1, 1, 1, 1024],
        [1, 73, 51, 20],
        [1, 124, 172, 5],
    ],
)
@pytest.mark.parametrize("ifm2_broadcast", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
@pytest.mark.parametrize("ifm_layout", ["NHWC", "NHCWB16"])
@pytest.mark.parametrize("ifm2_layout", ["NHWC", "NHCWB16"])
@pytest.mark.parametrize("ofm_layout", ["NHWC", "NHCWB16"])
@pytest.mark.parametrize("op_type", ["MUL", "ADD", "MIN"])
def test_ethosu_binary_elementwise_matcher(
    ofm_shape, ifm2_broadcast, ifm_layout, ifm2_layout, ofm_layout, op_type
):
    ifm_shape = ofm_shape.copy()
    ifm2_shape = [1] + [1 if (b == 1) else a for a, b in zip(ofm_shape[1:], ifm2_broadcast)]
    ifm_channels = ifm_shape[3]
    ifm2_channels = ifm2_shape[3]
    ofm_channels = ofm_shape[3]
    nhwc_to_nhcwb16, _ = get_layout_transform_matrices(ofm_channels)
    broadcast = [1 if a == 1 else 0 for a in ifm2_shape[1:]]
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
    if ifm2_layout == "NHCWB16":
        ifm2_shape = [
            int(math.ceil(n))
            for n in np.matmul(
                nhwc_to_nhcwb16,
                ifm2_shape
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
    ifm2 = te.placeholder(ifm2_shape, dtype="int8")
    lut = te.placeholder((), dtype="uint8")
    out = binary_elementwise_compute(
        ifm=ifm,
        ifm2=ifm2,
        lut=lut,
        operator_type=op_type,
        ifm_scale=1,
        ifm_zero_point=0,
        ifm2_scale=1,
        ifm2_zero_point=0,
        ofm_scale=1,
        ofm_zero_point=0,
        ifm_channels=ifm_channels,
        ifm2_channels=ifm2_channels,
        reversed_operands=False,
        activation="NONE",
        clip_min=0,
        clip_max=0,
        rounding_mode="TFL",
        ifm_layout=ifm_layout,
        ifm2_layout=ifm2_layout,
        ofm_layout=ofm_layout,
        ofm_dtype="int8",
        use_rescale=False,
        rescale_scale=0,
        rescale_shift=0,
    )
    ifm_propagator = out.op.attrs["ifm_propagator"]
    ifm2_propagator = out.op.attrs["ifm2_propagator"]

    offset = [0] * len(ofm_shape)
    stripes = [0] * len(ofm_shape)
    output_stripe_config = cs.StripeConfig(ofm_shape, ofm_shape, ofm_shape, order, stripes, offset)

    (ifm_transform, ifm2_transform) = _make_matrices(
        broadcast, ifm_layout, ifm2_layout, ofm_layout, ofm_channels
    )

    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    part = match_ethosu_binary_elementwise(out, device_config)

    assert isinstance(part, cs.EthosuPart)
    assert len(part.propagators) == 2
    assert part.propagators[0].transform == ifm_transform
    assert part.propagators[1].transform == ifm2_transform

    propagated_ifm = ifm_propagator.propagate(output_stripe_config).shape
    propagated_ifm2 = ifm2_propagator.propagate(output_stripe_config).shape

    # The layout transforms that have the exact number of output channels in them
    # will lose no information about the number of channels
    assert ifm_shape == propagated_ifm
    assert ifm2_shape == propagated_ifm2


if __name__ == "__main__":
    tvm.testing.main()
