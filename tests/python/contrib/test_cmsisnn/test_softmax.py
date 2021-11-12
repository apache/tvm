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

"""CMSIS-NN integration tests: softmax"""

import sys
import itertools

import numpy as np
import pytest

import tvm.testing
from tvm import relay
from tvm.relay.op.contrib import cmsisnn

from utils import (
    skip_if_no_reference_system,
    make_module,
    count_num_calls,
    get_range_for_dtype_str,
)
from tests.python.relay.aot.aot_test_utils import (
    AOTTestModel,
    AOT_CORSTONE300_RUNNER,
    generate_ref_data,
    compile_and_run,
)


def make_model(
    shape, in_dtype, out_dtype, in_zero_point, in_scale, out_zero_point=-128, out_scale=1.0 / 256
):
    """Create a Relay Function / network model"""
    a = relay.var("in0", shape=shape, dtype=in_dtype)
    dequantize = relay.qnn.op.dequantize(
        a,
        input_scale=relay.const(in_scale, "float32"),
        input_zero_point=relay.const(in_zero_point, "int32"),
    )
    softmax = relay.nn.softmax(dequantize)
    model = relay.qnn.op.quantize(
        softmax,
        output_scale=relay.const(out_scale, "float32"),
        output_zero_point=relay.const(out_zero_point, "int32"),
        out_dtype=out_dtype,
    )
    return model


@skip_if_no_reference_system
@pytest.mark.parametrize(["zero_point", "scale"], [[33, 0.256], [-64, 0.0128]])
@tvm.testing.requires_cmsisnn
def test_softmax_int8(zero_point, scale):
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_CORSTONE300_RUNNER

    dtype = "int8"
    shape = [1, 16, 16, 3]
    model = make_model(shape, dtype, dtype, zero_point, scale)
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
    np.random.seed(0)
    input_data = np.random.randint(in_min, high=in_max, size=shape, dtype=dtype)
    inputs = {"in0": input_data}
    params = {}
    output_list = generate_ref_data(orig_mod["main"], inputs, params)
    compile_and_run(
        AOTTestModel(module=cmsisnn_mod, inputs=inputs, outputs=output_list, params=params),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


def parameterize_for_invalid_model(test):
    in_dtype = ["uint8", "int8"]
    out_dtype = ["uint8", "int8"]
    zero_point = [-128, 64]
    scale = [1.0 / 256, 0.2]
    out_zero_point = [-128, 33]
    out_scale = [1.0 / 256, 0.2]
    all_combinations = itertools.product(
        in_dtype, out_dtype, zero_point, scale, out_zero_point, out_scale
    )
    all_combinations = filter(
        lambda parameters: not (
            parameters[0] == "int8"
            and parameters[1] == "int8"
            and parameters[4] == -128
            and parameters[5] == 1.0 / 256
        ),
        all_combinations,
    )
    return pytest.mark.parametrize(
        ["in_dtype", "out_dtype", "zero_point", "scale", "out_zero_point", "out_scale"],
        all_combinations,
    )(test)


@parameterize_for_invalid_model
@tvm.testing.requires_cmsisnn
def test_invalid_softmax(in_dtype, out_dtype, zero_point, scale, out_zero_point, out_scale):
    model = make_model(
        [1, 16, 16, 3], in_dtype, out_dtype, zero_point, scale, out_zero_point, out_scale
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
