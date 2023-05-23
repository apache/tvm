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

"""CMSIS-NN integration tests: Tests invalid graphs"""
import numpy as np
import tvm

from tvm.testing.aot import AOTTestModel, get_dtype_range, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import (
    AOT_USMP_CORSTONE300_RUNNER,
)
from .utils import (
    skip_if_no_reference_system,
)


@skip_if_no_reference_system
@tvm.testing.requires_cmsisnn
def test_empty_function():
    """Test partitioned function without composite function"""
    original_model = """
#[version = "0.0.5"]
def @main(%data : Tensor[(16, 29), int8]) -> Tensor[(16, 29), int8] {
    add(%data, %data)
}
"""
    cmsisnn_model = """
#[version = "0.0.5"]
def @tvmgen_default_cmsis_nn_main_1(%i1: Tensor[(16, 29), int8], Inline=1, Compiler="cmsis-nn", global_symbol="tvmgen_default_cmsis_nn_main_1", Primitive=1) -> Tensor[(16, 29), int8] {
  add(%i1, %i1)
}
def @main(%data : Tensor[(16, 29), int8]) -> Tensor[(16, 29), int8] {
  %1 = @tvmgen_default_cmsis_nn_main_1(%data) /* ty=Tensor[(16, 29), int8] */;
  %1
}
"""
    orig_mod = tvm.relay.fromtext(original_model)
    cmsisnn_mod = tvm.relay.fromtext(cmsisnn_model)
    params = {}

    # validate the output
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER
    dtype = "int8"
    in_min, in_max = get_dtype_range(dtype)
    rng = np.random.default_rng(12345)
    inputs = {"data": rng.integers(in_min, high=in_max, size=(16, 29), dtype=dtype)}
    outputs = generate_ref_data(orig_mod["main"], inputs, params)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=outputs,
            params=params,
            output_tolerance=0,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )
