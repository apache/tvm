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
import tflite.Model

import tvm
import tensorflow as tf
from tvm import relay

from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.backend.contrib.ethosu import preprocess

from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tests.python.relay.aot.aot_test_utils import generate_ref_data

from tests.python.contrib.test_ethosu import infra
from tests.python.contrib.test_ethosu.test_end_to_end import comparison_infra

ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32", "ethos-u65-256"]


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("constant", [np.ones((1, 1, 1, 1)), np.array(1)])
def test_elementwise_add_from_constant_scalar(accel_type, dtype, constant):
    ifm_shape = (1, 4, 4, 8)

    def create_relay_graph():
        inp = relay.var("input", shape=ifm_shape, dtype=dtype)
        scalar = relay.const(constant, dtype=dtype)
        add = relay.qnn.op.add(
            inp,
            scalar,
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
        )
        return tvm.IRModule.from_expr(relay.Function(relay.analysis.free_vars(add), add))

    cpu_mod = create_relay_graph()
    ethosu_mod = partition_for_ethosu(cpu_mod)

    # Generate reference data
    input_data = {
        "input": np.random.randint(
            low=np.iinfo(dtype).min, high=np.iinfo(dtype).max, size=ifm_shape, dtype=dtype
        ),
    }
    output_data = generate_ref_data(cpu_mod, input_data)

    comparison_infra._compare_ethosu_with_reference(
        ethosu_mod, input_data, output_data, accel_type, output_tolerance=0
    )


if __name__ == "__main__":
    pytest.main([__file__])
