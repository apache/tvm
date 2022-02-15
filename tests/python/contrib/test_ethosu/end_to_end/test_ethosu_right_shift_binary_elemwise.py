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
from tests.python.contrib.test_ethosu.end_to_end import comparison_infra

ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32", "ethos-u65-256"]


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape, reversed_operands, ofm_dtype",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], False, "int8"),
        ([1, 2, 3, 1], [1, 1, 3, 1], False, "int32"),
        ([1, 1, 3, 1], [1, 2, 3, 1], True, "int32"),
    ],
)
def test_ethosu_right_shift_binary_elemwise(
    ifm_shape, ifm2_shape, reversed_operands, accel_type, ofm_dtype
):
    dtype = "int32"

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
        ifm2 = relay.var("ifm2", shape=ifm2_shape, dtype=dtype)
        shr_op = infra.make_ethosu_binary_elementwise(
            ifm, ifm2, ifm_shape[3], ifm2_shape[3], "SHR", ofm_dtype, reversed_operands
        )
        return tvm.IRModule.from_expr(relay.Function([ifm, ifm2], shr_op))

    def generate_output_data(input_data):
        lhs = input_data["ifm"]
        rhs = input_data["ifm2"]
        if reversed_operands:
            lhs = np.broadcast_to(lhs, ifm2_shape)
            lhs, rhs = rhs, lhs
        else:
            rhs = np.broadcast_to(rhs, ifm_shape)

        def rounding_right_shift(lhs, rhs):
            r = 1 << (rhs - 1)
            return (lhs + r) >> rhs

        return [
            np.array([rounding_right_shift(x[0], x[1]) for x in zip(lhs.flat, rhs.flat)]).astype(
                ofm_dtype
            )
        ]

    cpu_mod = create_model()

    # Generate reference data
    in_min, in_max = util.get_range_for_dtype_str(dtype)
    in_min, in_max = 18, 19
    lhs = np.random.randint(in_min, high=in_max, size=ifm_shape, dtype=dtype)
    rhs = np.random.randint(1, high=2, size=ifm2_shape, dtype=dtype)
    input_data = {
        "ifm": lhs,
        "ifm2": rhs,
    }
    output_data = generate_output_data(input_data)
    ethosu_mod = comparison_infra._create_ethosu_partition(cpu_mod)

    comparison_infra._compare_ethosu_with_reference(ethosu_mod, input_data, output_data, accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
