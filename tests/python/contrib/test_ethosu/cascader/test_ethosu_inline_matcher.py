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
from tvm.topi.transform import reshape
import tvm.contrib.ethosu.cascader as cs
from tvm.relay.backend.contrib.ethosu.te.inline import match_ethosu_inline


def test_ethosu_inline_matcher():
    ifm_shape = (2, 5, 6)
    new_shape = (2, 30)
    ifm = te.placeholder(ifm_shape, dtype="int8")
    out = reshape(ifm, new_shape)
    ifm_transform = [
        [0, 0, ifm_shape[0]],
        [0, 0, ifm_shape[1]],
        [0, 0, ifm_shape[2]],
        [0, 0, 1],
    ]
    ifm_offset = [0, 0, 0]

    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    part = match_ethosu_inline(out, device_config)

    assert isinstance(part, cs.InlinePart)
    assert len(part.propagators) == 1
    assert part.propagators[0].transform == ifm_transform
    assert part.propagators[0].offset == ifm_offset


if __name__ == "__main__":
    tvm.testing.main()
