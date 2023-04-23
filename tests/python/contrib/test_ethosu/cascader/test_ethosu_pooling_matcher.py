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
from tvm.relay.backend.contrib.ethosu.te.pooling import match_ethosu_pooling, pooling_compute
from .infra import make_matrices


@pytest.mark.parametrize("pool_shape", [(3, 3), (2, 1), (3, 5)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("padding", [(0, 0, 0, 0), (3, 2, 3, 2), (2, 1, 0, 1)])
@pytest.mark.parametrize("ifm_layout", ["NHWC", "NHCWB16"])
@pytest.mark.parametrize("ofm_layout", ["NHWC", "NHCWB16"])
def test_ethosu_pooling_matcher(pool_shape, stride, padding, ifm_layout, ofm_layout):
    ofm_channels = 21
    if ifm_layout == "NHWC":
        ifm_shape = (1, 12, 15, ofm_channels)
    else:
        ifm_shape = (1, 12, 1 + ((ofm_channels - 1) // 16), 15, 16)
    ifm = te.placeholder(ifm_shape, dtype="int8")
    lut = te.placeholder((), dtype="uint8")
    out = pooling_compute(
        ifm=ifm,
        lut=lut,
        pooling_type="MAX",
        ifm_scale=1,
        ifm_zero_point=0,
        ofm_scale=1,
        ofm_zero_point=0,
        pool_shape=pool_shape,
        ofm_channels=ofm_channels,
        ofm_dtype="int8",
        strides=stride,
        padding=padding,
        activation="NONE",
        clip_min=0,
        clip_max=0,
        rounding_mode="TFL",
        upscale="NONE",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    (ifm_transform, ifm_offset, _, _, _, _) = make_matrices(
        "ethosu_pooling",
        pool_shape,
        stride,
        padding,
        ifm_layout,
        ofm_layout,
        ofm_channels=ofm_channels,
    )

    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    part = match_ethosu_pooling(out, device_config)

    assert isinstance(part, cs.EthosuPart)
    assert len(part.propagators) == 1
    assert part.propagators[0].transform == ifm_transform
    assert part.propagators[0].offset == ifm_offset


if __name__ == "__main__":
    tvm.testing.main()
