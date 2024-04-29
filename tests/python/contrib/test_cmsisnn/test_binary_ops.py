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

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn
from tvm.testing.aot import get_dtype_range, generate_ref_data, AOTTestModel, compile_and_run
from tvm.micro.testing.aot_test_utils import (
    AOT_USMP_CORSTONE300_RUNNER,
)

from .utils import (
    skip_if_no_reference_system,
    make_module,
    make_qnn_relu,
    assert_partitioned_function,
    assert_no_external_function,
    create_test_runner,
)


def generate_tensor_constant():
    rng = np.random.default_rng(12321)
    dtype = "int8"
    shape = (1, 16, 16, 3)
    in_min, in_max = get_dtype_range(dtype)
    values = tvm.nd.array(rng.integers(in_min, high=in_max, size=shape, dtype=dtype))
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
@pytest.mark.parametrize("op", [relay.qnn.op.mul, relay.qnn.op.add])
@pytest.mark.parametrize("relu_type", ["RELU", "NONE"])
@pytest.mark.parametrize(
    [
        "input_0_scale",
        "input_0_zero_point",
        "input_1_scale",
        "input_1_zero_point",
    ],
    [[0.256, 33, 0.256, 33], [0.0128, -64, 0.0128, -64], [0.0128, -64, 0.256, 33]],
)
@pytest.mark.parametrize(
    "compiler_cpu, cpu_flags", [("cortex-m55", "+nomve"), ("cortex-m55", ""), ("cortex-m7", "")]
)
def test_op_int8(
    op,
    relu_type,
    input_0_scale,
    input_0_zero_point,
    input_1_scale,
    input_1_zero_point,
    compiler_cpu,
    cpu_flags,
):
    """Tests QNN binary operator for CMSIS-NN"""
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
        create_test_runner(compiler_cpu, cpu_flags),
        interface_api,
        use_unpacked_api,
    )


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.qnn.op.mul, relay.qnn.op.add])
@pytest.mark.parametrize("relu_type", ["RELU", "NONE"])
@pytest.mark.parametrize(
    [
        "input_0_scale",
        "input_1_scale",
        "output_scale",
    ],
    [
        [0.256, 0.256, 0.256],
        [0.0128, 0.0128, 0.0128],
        [0.0128, 0.256, 0.256],
    ],
)
@pytest.mark.parametrize(
    "compiler_cpu, cpu_flags", [("cortex-m55", "+nomve"), ("cortex-m55", ""), ("cortex-m7", "")]
)
def test_op_int16(
    op,
    relu_type,
    input_0_scale,
    input_1_scale,
    output_scale,
    compiler_cpu,
    cpu_flags,
):
    """Tests QNN 16bit binary operators for CMSIS-NN"""
    interface_api = "c"
    use_unpacked_api = True

    dtype = "int16"
    shape = [1, 16, 16, 3]
    model = make_model(
        op,
        generate_variable("input_0", dtype),
        generate_variable("input_1", dtype),
        input_0_scale,
        0,
        input_1_scale,
        0,
        relu_type,
        output_scale,
        0,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

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
        create_test_runner(compiler_cpu, cpu_flags),
        interface_api,
        use_unpacked_api,
    )


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.qnn.op.mul, relay.qnn.op.add])
@pytest.mark.parametrize("relu_type", ["RELU", "NONE"])
@pytest.mark.parametrize(
    [
        "input_0_scale",
        "input_0_zero_point",
        "input_1_scale",
        "input_1_zero_point",
        "output_scale",
        "output_zero_point",
    ],
    [
        [0.256, 0, 0.256, 33, 0.256, 33],
        [0.0128, -64, 0.0128, 0, 0.0128, -64],
        [0.0128, -64, 0.256, 33, 0.256, 0],
    ],
)
def test_op_int16_cannot_partition(
    op,
    relu_type,
    input_0_scale,
    input_0_zero_point,
    input_1_scale,
    input_1_zero_point,
    output_scale,
    output_zero_point,
):
    """Tests QNN 16bit binary operators for CMSIS-NN in the edge case of
    non-zero zero points"""

    model = make_model(
        op,
        generate_variable("input_0", "int16"),
        generate_variable("input_1", "int16"),
        input_0_scale,
        input_0_zero_point,
        input_1_scale,
        input_1_zero_point,
        relu_type,
        output_scale,
        output_zero_point,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # arm_elementwise_(mul|add)_s16 does not support non-zero shifts in any
    # argument
    assert_no_external_function(cmsisnn_mod)


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.qnn.op.mul, relay.qnn.op.add])
@pytest.mark.parametrize("relu_type", ["RELU", "NONE"])
def test_same_input_to_binary_op(op, relu_type):
    """Tests QNN binary operator for CMSIS-NN where both inputs are the same"""
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    dtype = "int8"
    shape = [1, 16, 16, 3]
    input_ = generate_variable("input")
    input_scale = 0.256
    input_zero_point = 33

    model = make_model(
        op,
        input_,
        input_,
        input_scale,
        input_zero_point,
        input_scale,
        input_zero_point,
        relu_type,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # Check if the number of internal function parameter is 1
    cmsisnn_global_func = cmsisnn_mod["tvmgen_default_cmsis_nn_main_0"]
    assert (
        isinstance(cmsisnn_global_func.body, tvm.relay.expr.Call)
        and len(cmsisnn_global_func.body.args) == 1
    ), "Composite function for the binary op should have only 1 parameter."

    # validate the output
    in_min, in_max = get_dtype_range(dtype)
    inputs = {
        "input": np.random.randint(in_min, high=in_max, size=shape, dtype=dtype),
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


def parameterize_for_constant_inputs(test):
    """Generates parameters in such a way so that at least one of the inputs is a constant,
    both can't be variables, both can't be scalars.
    """
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
    """Tests binary ops where one of the operands is a constant"""
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
    in_min, in_max = get_dtype_range(dtype)
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
    """Tests binary ops where both operands are scalars"""
    input_scale = 0.256
    input_zero_point = 33
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
@pytest.mark.parametrize(["input_dtype"], [["uint8"], ["uint16"]])
def test_invalid_parameters(
    op,
    input_dtype,
):
    """Tests binary ops for non int8 dtypes"""
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
    tvm.testing.main()
