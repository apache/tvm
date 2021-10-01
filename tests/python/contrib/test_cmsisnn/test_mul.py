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

"""CMSIS-NN integration tests: mul"""

import sys

import numpy as np
import pytest

from tvm import relay
from tvm.relay.op.contrib import cmsisnn

from utils import skip_if_no_reference_system, make_module, count_num_calls, get_range_for_dtype_str
from tests.python.relay.aot.aot_test_utils import (
    AOTTestModel,
    AOT_CORSTONE300_RUNNER,
    generate_ref_data,
    compile_and_run,
)


def make_model(
    shape,
    input_0_dtype,
    input_1_dtype,
    input_0_scale,
    input_0_zero_point,
    input_1_scale,
    input_1_zero_point,
    out_scale=1.0 / 256,
    out_zero_point=-128,
):
    """Create a Relay Function / network model"""

    return relay.qnn.op.mul(
        relay.var("input_0", shape=shape, dtype=input_0_dtype),
        relay.var("input_1", shape=shape, dtype=input_1_dtype),
        relay.const(input_0_scale, "float32"),
        relay.const(input_0_zero_point, "int32"),
        relay.const(input_1_scale, "float32"),
        relay.const(input_1_zero_point, "int32"),
        relay.const(out_scale, "float32"),
        relay.const(out_zero_point, "int32"),
    )


@skip_if_no_reference_system
@pytest.mark.parametrize(
    [
        "input_0_scale",
        "input_0_zero_point",
        "input_1_scale",
        "input_1_zero_point",
        "output_tolerance",
    ],
    [[0.256, 33, 0.256, 33, 0], [0.0128, -64, 0.0128, -64, 1], [0.0128, -64, 0.256, 33, 0]],
)
def test_mul_int8(
    input_0_scale, input_0_zero_point, input_1_scale, input_1_zero_point, output_tolerance
):
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_CORSTONE300_RUNNER

    dtype = "int8"
    shape = [1, 16, 16, 3]
    model = make_model(
        shape, dtype, dtype, input_0_scale, input_0_zero_point, input_1_scale, input_1_zero_point
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
        key == "Compiler" and value == "cmsisnn" for attr in attrs for key, value in attr.items()
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
            output_tolerance=output_tolerance,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@pytest.mark.parametrize(["input_dtype"], [["uint8"], ["int16"]])
def test_invalid_parameters(
    input_dtype,
):
    input_scale = 0.256
    input_zero_point = 33
    model = make_model(
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
