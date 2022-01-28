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

"""CMSIS-NN integration tests: binary ops"""

import itertools
import sys

import numpy as np
from enum import Enum
import pytest

import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn

from utils import skip_if_no_reference_system, make_module, count_num_calls, get_range_for_dtype_str
from tests.python.relay.aot.aot_test_utils import (
    AOTTestModel,
    AOT_CORSTONE300_RUNNER,
    generate_ref_data,
    compile_and_run,
)


class BinaryOpInputType(Enum):
    Variable = 0
    TensorConstant = 1
    ScalarConstant = 2


def make_model(
    op,
    shape,
    input_0_dtype,
    input_1_dtype,
    input_0_scale,
    input_0_zero_point,
    input_1_scale,
    input_1_zero_point,
    out_scale=1.0 / 256,
    out_zero_point=-128,
    input_0_type=BinaryOpInputType.Variable,
    input_1_type=BinaryOpInputType.Variable,
):
    """Create a Relay Function / network model"""

    def create_input(name, input_type, shape, dtype):
        if input_type == BinaryOpInputType.Variable:
            return relay.var(name, shape=shape, dtype=dtype)
        if input_type == BinaryOpInputType.TensorConstant:
            rng = np.random.default_rng(12321)
            values = tvm.nd.array(
                rng.integers(np.iinfo(dtype).min, high=np.iinfo(dtype).max, size=shape, dtype=dtype)
            )
            return relay.const(values, dtype)
        if input_type == BinaryOpInputType.ScalarConstant:
            return relay.const(tvm.nd.array(np.array(-32).astype(dtype)), dtype)

    return op(
        create_input("input_0", input_0_type, shape, input_0_dtype),
        create_input("input_1", input_1_type, shape, input_1_dtype),
        relay.const(input_0_scale, "float32"),
        relay.const(input_0_zero_point, "int32"),
        relay.const(input_1_scale, "float32"),
        relay.const(input_1_zero_point, "int32"),
        relay.const(out_scale, "float32"),
        relay.const(out_zero_point, "int32"),
    )


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.qnn.op.mul, relay.qnn.op.add])
@pytest.mark.parametrize(
    [
        "input_0_scale",
        "input_0_zero_point",
        "input_1_scale",
        "input_1_zero_point",
    ],
    [[0.256, 33, 0.256, 33], [0.0128, -64, 0.0128, -64], [0.0128, -64, 0.256, 33]],
)
def test_op_int8(op, input_0_scale, input_0_zero_point, input_1_scale, input_1_zero_point):
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_CORSTONE300_RUNNER

    dtype = "int8"
    shape = [1, 16, 16, 3]
    model = make_model(
        op,
        shape,
        dtype,
        dtype,
        input_0_scale,
        input_0_zero_point,
        input_1_scale,
        input_1_zero_point,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    attrs = [
        cmsisnn_mod[var.name_hint].attrs
        for var in cmsisnn_mod.get_global_vars()
        if cmsisnn_mod[var.name_hint].attrs
    ]
    assert any(attrs), "At least one function with external attributes was expected."

    compilers = [
        key == "Compiler" and value == "cmsis-nn" for attr in attrs for key, value in attr.items()
    ]
    assert any(compilers), "Module does not contain function for cmsisnn target."

    assert count_num_calls(orig_mod) == count_num_calls(
        cmsisnn_mod
    ), "Number of calls changed during partitioning"

    # validate the output
    in_min, in_max = get_range_for_dtype_str(dtype)
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
        test_runner,
        interface_api,
        use_unpacked_api,
    )


# At least one of the inputs is a constant
def parameterize_for_constant_inputs(test):
    op = [relay.qnn.op.mul, relay.qnn.op.add]
    input_0_type = [
        BinaryOpInputType.Variable,
        BinaryOpInputType.TensorConstant,
        BinaryOpInputType.ScalarConstant,
    ]
    input_1_type = [
        BinaryOpInputType.Variable,
        BinaryOpInputType.TensorConstant,
        BinaryOpInputType.ScalarConstant,
    ]
    all_combinations = itertools.product(op, input_0_type, input_1_type)
    all_combinations = filter(
        lambda parameters: not (
            (
                parameters[1] == BinaryOpInputType.Variable
                and parameters[2] == BinaryOpInputType.Variable
            )
            or (
                parameters[1] == BinaryOpInputType.ScalarConstant
                and parameters[2] == BinaryOpInputType.ScalarConstant
            )
        ),
        all_combinations,
    )
    return pytest.mark.parametrize(
        ["op", "input_0_type", "input_1_type"],
        all_combinations,
    )(test)


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@parameterize_for_constant_inputs
def test_constant_input_int8(op, input_0_type, input_1_type):
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_CORSTONE300_RUNNER

    dtype = "int8"
    shape = [1, 16, 16, 3]
    input_0_scale = 0.256
    input_0_zero_point = 33
    input_1_scale = 0.128
    input_1_zero_point = -24
    model = make_model(
        op,
        shape,
        dtype,
        dtype,
        input_0_scale,
        input_0_zero_point,
        input_1_scale,
        input_1_zero_point,
        input_0_type=input_0_type,
        input_1_type=input_1_type,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    attrs = [
        cmsisnn_mod[var.name_hint].attrs
        for var in cmsisnn_mod.get_global_vars()
        if cmsisnn_mod[var.name_hint].attrs
    ]
    assert any(attrs), "At least one function with external attributes was expected."

    compilers = [
        key == "Compiler" and value == "cmsis-nn" for attr in attrs for key, value in attr.items()
    ]
    assert any(compilers), "Module does not contain function for cmsisnn target."

    assert count_num_calls(orig_mod) == count_num_calls(
        cmsisnn_mod
    ), "Number of calls changed during partitioning"

    # validate the output
    in_min, in_max = get_range_for_dtype_str(dtype)
    inputs = {}
    if input_0_type == BinaryOpInputType.Variable:
        inputs.update({"input_0": np.random.randint(in_min, high=in_max, size=shape, dtype=dtype)})
    if input_1_type == BinaryOpInputType.Variable:
        inputs.update({"input_1": np.random.randint(in_min, high=in_max, size=shape, dtype=dtype)})
    output_list = generate_ref_data(orig_mod["main"], inputs)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            output_tolerance=1,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.qnn.op.mul, relay.qnn.op.add])
def test_both_scalar_inputs_int8(
    op,
):
    input_scale = 0.256
    input_zero_point = 33
    dtype = "int8"
    model = make_model(
        op,
        [1, 16, 16, 3],
        dtype,
        dtype,
        input_scale,
        input_zero_point,
        input_scale,
        input_zero_point,
        input_0_type=BinaryOpInputType.ScalarConstant,
        input_1_type=BinaryOpInputType.ScalarConstant,
    )

    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    attrs = [
        cmsisnn_mod[var.name_hint].attrs
        for var in cmsisnn_mod.get_global_vars()
        if cmsisnn_mod[var.name_hint].attrs
    ]
    assert not any(attrs), "No function should have an external attribute."


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.qnn.op.mul, relay.qnn.op.add])
@pytest.mark.parametrize(["input_dtype"], [["uint8"], ["int16"]])
def test_invalid_parameters(
    op,
    input_dtype,
):
    input_scale = 0.256
    input_zero_point = 33
    model = make_model(
        op,
        [1, 16, 16, 3],
        input_dtype,
        input_dtype,
        input_scale,
        input_zero_point,
        input_scale,
        input_zero_point,
    )

    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    attrs = [
        cmsisnn_mod[var.name_hint].attrs
        for var in cmsisnn_mod.get_global_vars()
        if cmsisnn_mod[var.name_hint].attrs
    ]
    assert not any(attrs), "No function should have an external attribute."


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
