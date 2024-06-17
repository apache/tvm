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
import tvm.testing
from tvm import relay
from tvm import meta_schedule
from tvm.testing.aot import (
    AOTTestModel,
    AOTCompiledTestModel,
    compile_and_run,
    run_and_check,
    generate_ref_data,
)
from tvm.micro.testing.aot_test_utils import AOT_CORSTONE300_RUNNER, AOT_APROFILE_AEM_RUNNER
from tvm.target.codegen import llvm_version_major
from tvm.relay.op.strategy.arm_cpu import arm_cpu_tir_strategy
from scalable_utils import calculate_extra_workspace_size_from_scalable_extents


class BasicDenseTests:
    @tvm.testing.requires_corstone300
    def test_dense(self, shape, weight_shape, dtype, schedule_name, enable_bias):
        """Test a subgraph with a single dense operator."""
        ishape = shape
        wshape = weight_shape
        out_dtype = "int32"
        units = weight_shape[0]
        weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)
        if enable_bias:
            bias_data = np.random.randint(low=-10, high=10, size=(wshape[0]), dtype=out_dtype)

        input = relay.var("input", relay.TensorType(ishape, dtype))
        weight = relay.const(weight_data)
        dense = relay.op.nn.dense(
            input,
            weight,
            units=units,
            out_dtype=out_dtype,
        )
        if enable_bias:
            bias = relay.const(bias_data)
            relay_op = relay.op.nn.bias_add(dense, bias)
        else:
            relay_op = dense

        inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
        ref_mod = tvm.IRModule.from_expr(relay.Function([input], relay_op))
        output_list = generate_ref_data(ref_mod, inputs)

        mod = tvm.IRModule.from_expr(relay.Function([input], relay_op))
        compile_and_run(
            AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
            runner=AOT_CORSTONE300_RUNNER,
            interface_api="c",
            use_unpacked_api=True,
            target_opts={
                "-keys": "arm_cpu",
                "-mcpu": "cortex-m7",
            },
            schedule_name=schedule_name,
        )


class TestDense(BasicDenseTests):
    """This test is for dense_dsp schedule."""

    shape, weight_shape = tvm.testing.parameters(
        ((8, 128), (32, 128)),
        ((32, 32), (32, 32)),
        ((1, 64), (1, 64)),
        ((11, 2), (2, 2)),
        ((1, 32), (64, 32)),
        ((3, 12), (10, 12)),
    )
    dtype = tvm.testing.parameter("int8", "int16")
    schedule_name = tvm.testing.parameter("dense_dsp.arm_cpu")
    enable_bias = tvm.testing.parameter(False, True)


@pytest.mark.skipif(
    llvm_version_major() < 17, reason="SME is not supported in earlier versions of LLVM"
)
@tvm.testing.requires_aprofile_aem_fvp
@pytest.mark.parametrize(
    "data_shape,weight_shape,enable_bias",
    [
        ((32, 32), (32, 32), False),
        ((2, 35), (6, 35), False),
        ((3, 3), (68, 3), False),
        ((79, 65), (152, 65), True),
    ],
)
@pytest.mark.parametrize("in_dtype", ["float32", "float16"])
def test_sme_dense(data_shape, weight_shape, enable_bias, in_dtype):
    np.random.seed(0)
    out_dtype = "float32"

    input_data = np.random.uniform(size=data_shape).astype(in_dtype)
    inp = relay.var("data", shape=data_shape, dtype=in_dtype)
    weight_data = np.random.uniform(size=weight_shape).astype(in_dtype)
    weight = relay.const(weight_data, dtype=in_dtype)

    relay_op = relay.nn.dense(inp, weight, out_dtype=out_dtype)

    if enable_bias:
        bias_data = np.random.uniform(size=weight_shape[0]).astype(out_dtype)
        bias = relay.const(bias_data, dtype=out_dtype)
        relay_op = relay.nn.bias_add(relay_op, bias)

    func = relay.Function(relay.analysis.free_vars(relay_op), relay_op)

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

    bias_postfix = "_add" if enable_bias else ""
    generated_func = executor_factory.lowered_ir_mods.items()[0][1][
        f"tvmgen_default_fused_nn_matmul{bias_postfix}"
    ]
    extra_memory_in_bytes = calculate_extra_workspace_size_from_scalable_extents(generated_func, 4)

    test_model = AOTTestModel(
        ir_mod, inputs, ref_outputs, params=params, extra_memory_in_bytes=extra_memory_in_bytes
    )
    compiled = AOTCompiledTestModel(test_model, executor_factory)

    assembly = (
        compiled.executor_factory.module.imported_modules[0].imported_modules[0].get_source("asm")
    )
    assert "fmopa" in assembly

    assert run_and_check(
        models=[compiled],
        interface_api="packed",
        runner=AOT_APROFILE_AEM_RUNNER,
        print_output_on_mismatch=True,
    )


class TestGemmDense:
    """This test is for dense_gemm schedule."""


@pytest.mark.parametrize(
    "data_shape,weight_shape,enable_bias",
    [
        ((32, 32), (32, 32), False),
        ((2, 35), (6, 35), False),
        ((3, 3), (68, 3), False),
        ((79, 65), (152, 65), True),
    ],
)
@pytest.mark.parametrize("in_dtype", ["float32", "float16"])
def test_gemm_dense(data_shape, weight_shape, enable_bias, in_dtype):
    np.random.seed(0)
    in_np = np.random.uniform(size=(data_shape)).astype(in_dtype)
    w1 = np.random.uniform(size=(weight_shape)).astype(in_dtype)

    w = relay.const(w1)
    d = relay.var("data", shape=data_shape, dtype=in_dtype)
    y = relay.nn.dense(d, w)

    mod = tvm.IRModule()

    mod["main"] = relay.Function([d], y)

    target = "llvm -mtriple=aarch64-linux-gnu -device=arm_cpu -mattr=+v8.2a,+neon"

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=None)

    out_np = np.array(np.matmul(in_np, w1.T))

    dev = tvm.cpu(0)
    input_buf = tvm.nd.array(in_np, device=dev)
    rt = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    rt.set_input("data", input_buf)
    rt.run()
    out = rt.get_output(0)

    tvm.testing.assert_allclose(out.numpy(), out_np, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
