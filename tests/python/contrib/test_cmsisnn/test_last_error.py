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

"""CMSIS-NN integration tests: test if the model builds in case debug_last_error is enabled"""

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn
from tvm.testing.aot import get_dtype_range, generate_ref_data, AOTTestModel, compile_and_run


from .utils import (
    skip_if_no_reference_system,
    make_module,
    make_qnn_relu,
    assert_partitioned_function,
    create_test_runner,
)


def generate_variable(name, dtype="int8"):
    return relay.var(name, shape=(1, 16, 16, 3), dtype=dtype)


def make_model(
    op,
    input_0,
    input_1,
    input_0_scale,
    input_0_zero_point,
    input_1_scale,
    input_1_zero_point,
    relu_type="NONE",
    out_scale=1.0 / 256,
    out_zero_point=-128,
):
    """Create a Relay Function / network model"""
    binary_op = op(
        input_0,
        input_1,
        relay.const(input_0_scale, "float32"),
        relay.const(input_0_zero_point, "int32"),
        relay.const(input_1_scale, "float32"),
        relay.const(input_1_zero_point, "int32"),
        relay.const(out_scale, "float32"),
        relay.const(out_zero_point, "int32"),
    )
    return make_qnn_relu(binary_op, relu_type, out_scale, out_zero_point, "int8")


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("debug_last_error", [True, False])
def test_last_error(debug_last_error):
    """Tests debug_last_error"""
    op = relay.qnn.op.add
    relu_type = "NONE"
    input_0_scale, input_0_zero_point = (0.256, 33)
    input_1_scale, input_1_zero_point = (0.256, 33)
    compiler_cpu = "cortex-m55"
    cpu_flags = ""

    interface_api = "c"
    use_unpacked_api = True

    dtype = "int8"
    shape = [1, 16, 16, 3]
    model = make_model(
        op,
        generate_variable("input_0"),
        generate_variable("input_1"),
        input_0_scale,
        input_0_zero_point,
        input_1_scale,
        input_1_zero_point,
        relu_type,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    in_min, in_max = get_dtype_range(dtype)
    inputs = {
        "input_0": np.random.randint(in_min, high=in_max, size=shape, dtype=dtype),
        "input_1": np.random.randint(in_min, high=in_max, size=shape, dtype=dtype),
    }
    output_list = generate_ref_data(orig_mod["main"], inputs)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            output_tolerance=1,
        ),
        create_test_runner(compiler_cpu, cpu_flags, debug_last_error=debug_last_error),
        interface_api,
        use_unpacked_api,
    )
