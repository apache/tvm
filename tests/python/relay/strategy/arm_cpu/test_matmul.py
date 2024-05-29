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

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm import meta_schedule
from tvm.testing.aot import (
    AOTTestModel,
    AOTCompiledTestModel,
    run_and_check,
    generate_ref_data,
)
from tvm.micro.testing.aot_test_utils import AOT_APROFILE_AEM_RUNNER
from tvm.target.codegen import llvm_version_major
from tvm.relay.op.strategy.arm_cpu import arm_cpu_tir_strategy
from scalable_utils import calculate_extra_workspace_size_from_scalable_extents


@pytest.mark.skipif(
    llvm_version_major() < 17, reason="SME is not supported in earlier versions of LLVM"
)
@tvm.testing.requires_aprofile_aem_fvp
@pytest.mark.parametrize(
    "data_shape,weight_shape,transpose_a,transpose_b,in_dtype",
    [
        ((4, 63), (63, 10), False, False, "float32"),
        ((64, 32), (32, 32), False, True, "float32"),
        ((96, 64), (64, 32), False, False, "float32"),
        ((62, 3), (3, 3), False, False, "float32"),
        ((4, 5), (79, 5), False, True, "float32"),
        ((134, 36), (36, 111), False, False, "float32"),
        ((3, 10), (10, 72), False, False, "float32"),
        ((4, 63), (10, 63), False, True, "float16"),
        ((96, 64), (32, 64), False, True, "float16"),
        ((62, 3), (3, 3), False, True, "float16"),
        ((4, 5), (79, 5), False, True, "float16"),
        ((134, 36), (111, 36), False, True, "float16"),
        # Tensorization does not work when the reduction axis has unit iters.
        # See https://github.com/apache/tvm/issues/16566
        # ((5, 1), (1, 5), False, False),
    ],
)
def test_sme_matmul_with_const_b(data_shape, weight_shape, transpose_a, transpose_b, in_dtype):
    """
    Execution tests for matmul Scalable Matrix Extension (SME) schedule.
    """
    np.random.seed(0)
    out_dtype = "float32"

    input_data = np.random.uniform(size=data_shape).astype(in_dtype)
    inp = relay.var("data", shape=data_shape, dtype=in_dtype)
    weight_data = np.random.uniform(size=weight_shape).astype(in_dtype)
    weight = relay.const(weight_data, dtype=in_dtype)

    matmul = relay.nn.matmul(
        inp, weight, out_dtype=out_dtype, transpose_a=transpose_a, transpose_b=transpose_b
    )
    func = relay.Function(relay.analysis.free_vars(matmul), matmul)

    ir_mod = tvm.IRModule.from_expr(func)
    ir_mod = tvm.relay.transform.InferType()(ir_mod)

    inputs = {"data": input_data}
    params = {}
    ref_outputs = generate_ref_data(ir_mod, inputs, params)

    target = tvm.target.Target("llvm -mtriple=aarch64-none-elf -mattr=+v9.2a,+sme")
    runtime = tvm.relay.backend.Runtime("crt", {"system-lib": True})
    executor = tvm.relay.backend.Executor(
        "aot",
        {
            "interface-api": "packed",
            "unpacked-api": False,
        },
    )
    with tvm.transform.PassContext(
        opt_level=3, config=AOT_APROFILE_AEM_RUNNER.pass_config
    ), target, meta_schedule.database.ScheduleFnDatabase(arm_cpu_tir_strategy):
        executor_factory = tvm.relay.build(
            ir_mod,
            target=target,
            executor=executor,
            runtime=runtime,
            params=params,
        )
    generated_func = executor_factory.lowered_ir_mods.items()[0][1][
        "tvmgen_default_fused_nn_matmul"
    ]
    extra_memory_in_bytes = calculate_extra_workspace_size_from_scalable_extents(generated_func, 4)

    test_model = AOTTestModel(
        ir_mod, inputs, ref_outputs, params=params, extra_memory_in_bytes=extra_memory_in_bytes
    )
    compiled = AOTCompiledTestModel(test_model, executor_factory)

    assembly = executor_factory.module.imported_modules[0].imported_modules[0].get_source("asm")
    assert "fmopa" in assembly

    assert run_and_check(
        models=[compiled],
        interface_api="packed",
        runner=AOT_APROFILE_AEM_RUNNER,
        print_output_on_mismatch=True,
    )


if __name__ == "__main__":
    tvm.testing.main()
