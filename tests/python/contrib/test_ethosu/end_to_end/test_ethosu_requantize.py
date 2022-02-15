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
# pylint: disable=invalid-name, unused-argument
import pytest

pytest.importorskip("ethosu.vela")

import numpy as np

import tvm
from tvm import relay

from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tests.python.relay.aot.aot_test_utils import generate_ref_data
from tests.python.contrib.test_ethosu.end_to_end import comparison_infra


ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32", "ethos-u65-256"]


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,ifm_scale,ifm_zp,ofm_scale,ofm_zp",
    [
        [(1, 8, 8, 3), 1.0, 0, 1.0, 0],
        [(1, 20, 30, 3), 1.345, 34, 0.32, -23],
    ],
)
def test_ethosu_requantize(accel_type, ifm_shape, ifm_scale, ifm_zp, ofm_scale, ofm_zp):
    dtype = "int8"

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        requantize = relay.qnn.op.requantize(
            ifm,
            relay.const(ifm_scale, dtype="float32"),
            relay.const(ifm_zp, dtype="int32"),
            relay.const(ofm_scale, dtype="float32"),
            relay.const(ofm_zp, dtype="int32"),
        )
        return tvm.IRModule.from_expr(relay.Function([ifm], requantize))

    cpu_mod = create_model()
    input_data = {"ifm": np.random.randint(-128, high=127, size=ifm_shape, dtype=dtype)}
    output_data = generate_ref_data(cpu_mod, input_data)
    ethosu_mod = partition_for_ethosu(cpu_mod)

    comparison_infra._compare_ethosu_with_reference(ethosu_mod, input_data, output_data, accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
