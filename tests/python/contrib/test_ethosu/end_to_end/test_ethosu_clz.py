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
def test_ethosu_clz(accel_type):
    ifm_shape = (1, 42, 5, 4)

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int32")
        clz = infra.make_ethosu_unary_elementwise(ifm, 4, "CLZ")
        return tvm.IRModule.from_expr(relay.Function([ifm], clz))

    def generate_output_data(input_data):
        def clz_comp(n):
            n_bin = np.binary_repr(n)
            if n_bin[0] == "-":
                return 0
            else:
                return 32 - len(n_bin)

        return [
            np.array([clz_comp(i) for i in input_data["ifm"].ravel()])
            .reshape(ifm_shape)
            .astype("int32")
        ]

    cpu_mod = create_model()
    input_data = {"ifm": np.random.randint(-500000, high=500000, size=ifm_shape, dtype="int32")}
    output_data = generate_output_data(input_data)
    ethosu_mod = comparison_infra._create_ethosu_partition(cpu_mod)

    comparison_infra._compare_ethosu_with_reference(ethosu_mod, input_data, output_data, accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
