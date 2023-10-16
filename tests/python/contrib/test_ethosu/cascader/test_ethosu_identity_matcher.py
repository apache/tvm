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
from tvm.relay.backend.contrib.ethosu.te.identity import match_ethosu_identity, identity_compute
from .infra import make_matrices


def test_ethosu_identity_matcher():
    ofm_channels = 21
    ifm_shape = (1, 12, 15, ofm_channels)
    ifm = te.placeholder(ifm_shape, dtype="int8")
    lut = te.placeholder((), dtype="uint8")
    out = identity_compute(
        ifm=ifm,
        lut=lut,
        ifm_scale=1,
        ifm_zero_point=0,
        ofm_scale=1,
        ofm_zero_point=0,
        activation="NONE",
        rounding_mode="TFL",
    )

    length = len(ifm.shape)
    ifm_transform = np.identity(length + 1).tolist()
    ifm_offset = np.zeros(length, dtype="int64").tolist()

    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    part = match_ethosu_identity(out, device_config)

    assert isinstance(part, cs.EthosuPart)
    assert len(part.propagators) == 1
    assert part.propagators[0].transform == ifm_transform
    assert part.propagators[0].offset == ifm_offset


if __name__ == "__main__":
    tvm.testing.main()
