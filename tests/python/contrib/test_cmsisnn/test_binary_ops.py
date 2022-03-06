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

from utils import (
    skip_if_no_reference_system,
    make_module,
    get_range_for_dtype_str,
    assert_partitioned_function,
    assert_no_external_function,
)
from tests.python.relay.aot.aot_test_utils import (
    AOTTestModel,
    AOT_CORSTONE300_RUNNER,
    AOT_USMP_CORSTONE300_RUNNER,
    generate_ref_data,
    compile_and_run,
)


def generate_tensor_constant():
    rng = np.random.default_rng(12321)
    dtype = "int8"
    shape = (1, 16, 16, 3)
    values = tvm.nd.array(
        rng.integers(np.iinfo(dtype).min, high=np.iinfo(dtype).max, size=shape, dtype=dtype)
    )
    return relay.const(values, dtype)


def generate_scalar_constant():
    dtype = "int8"
    return relay.const(-30, dtype)


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
    out_scale=1.0 / 256,
    out_zero_point=-128,
):
    """Create a Relay Function / network model"""
    return op(
        input_0,
        input_1,
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
    test_runner = AOT_USMP_CORSTONE300_RUNNER

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
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

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


# At least one of the inputs is a constant, both can't be variables, both can't be scalars
def parameterize_for_constant_inputs(test):
    op = [relay.qnn.op.mul, relay.qnn.op.add]
    input_0 = [generate_variable("input_0"), generate_tensor_constant(), generate_scalar_constant()]
    input_1 = [generate_variable("input_1"), generate_tensor_constant(), generate_scalar_constant()]
    all_combinations = itertools.product(op, input_0, input_1)
    all_combinations = filter(
        lambda parameters: not (
            (
                isinstance(parameters[1], tvm.relay.expr.Var)
                and isinstance(parameters[2], tvm.relay.expr.Var)
            )
            or (
                isinstance(parameters[1], tvm.relay.expr.Constant)
                and isinstance(parameters[2], tvm.relay.expr.Constant)
                and parameters[1].data.numpy().ndim == 0
                and parameters[2].data.numpy().ndim == 0
            )
        ),
        all_combinations,
    )
    return pytest.mark.parametrize(
        ["op", "input_0", "input_1"],
        all_combinations,
    )(test)


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@parameterize_for_constant_inputs
def test_constant_input_int8(op, input_0, input_1):
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    dtype = "int8"
    shape = [1, 16, 16, 3]
    input_0_scale = 0.256
    input_0_zero_point = 33
    input_1_scale = 0.128
    input_1_zero_point = -24
    model = make_model(
        op,
        input_0,
        input_1,
        input_0_scale,
        input_0_zero_point,
        input_1_scale,
        input_1_zero_point,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    in_min, in_max = get_range_for_dtype_str(dtype)
    inputs = {}
    if isinstance(input_0, tvm.relay.expr.Var):
        inputs.update({"input_0": np.random.randint(in_min, high=in_max, size=shape, dtype=dtype)})
    if isinstance(input_1, tvm.relay.expr.Var):
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
        generate_scalar_constant(),
        generate_scalar_constant(),
        input_scale,
        input_zero_point,
        input_scale,
        input_zero_point,
    )

    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)
    assert_no_external_function(cmsisnn_mod)


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
        generate_variable("input_0", input_dtype),
        generate_variable("input_1", input_dtype),
        input_scale,
        input_zero_point,
        input_scale,
        input_zero_point,
    )

    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)
    assert_no_external_function(cmsisnn_mod)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
