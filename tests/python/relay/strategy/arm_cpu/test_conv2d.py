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
"""Tests for arm_cpu schedules for regular conv2d."""

import pytest
import numpy as np

import tvm
import tvm.topi.testing
from tvm import relay
from test_generalized_conv2d import GeneralizedConv2dTests
from tvm.testing import fixture, main, parameter, parameters
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from tvm.target.codegen import llvm_version_major
from tvm.testing.aot import AOTTestModel, AOTCompiledTestModel, run_and_check, generate_ref_data
from tvm.micro.testing.aot_test_utils import AOT_APROFILE_AEM_RUNNER
from tvm.relay.op.strategy.arm_cpu import arm_cpu_tir_strategy
from scalable_utils import calculate_extra_workspace_size_from_scalable_extents


class Conv2dTests(GeneralizedConv2dTests):
    """Helper for constructing regular Conv2ds. Always sets groups to 1. We set the reference
    kernel layout here as we must pick something, but the x86 implementation supports several."""

    @fixture
    def groups(self):
        """Using a fixture instead of a parameter stops Pytest from adding the (redundant) number of
        groups to the name of each test."""
        return 1

    def setup_method(self):
        self.ref_kernel_layout = "HWIO"


class TestConv2d_NHWC_DSP(Conv2dTests):
    """This test is for conv2d_nhwc_dsp.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = parameters(
        # TODO(mehrdadh): Fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 32, 32, 1), (3, 3), 12, 1, 0, 1),
        # ((1, 32, 10, 3), (3, 3), 16, 1, 0, 1),
        # ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
        # from Keyword Spotting model from MLPerfTiny models
        # TODO(mehrdad): Fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 49, 10, 1), (10, 4), 64, (2, 2), (4, 1, 5, 1), 1),
        # from Visual Wake Word model from MLPerfTiny models
        # TODO(mehrdadh): fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 96, 96, 3), (3, 3), 8, (2, 2), (0, 0, 1, 1), 1),
        # from Image Classification model from MLPerfTiny models
        ((1, 16, 16, 32), (1, 1), 64, (2, 2), 0, 1),
        ((4, 16, 16, 8), (5, 5), 8, 2, (0, 4, 4, 0), 1),
        ((4, 16, 16, 8), (5, 5), 16, 2, (0, 4, 4, 0), 1),
        ((4, 16, 16, 8), (5, 5), 8, 2, 0, 1),
        ((4, 16, 16, 8), (5, 5), 16, 2, 0, 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 0, 1, 1), 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (1, 1, 2, 2), 1),
        ((1, 16, 16, 8), (5, 5), 16, 2, (3, 3, 2, 2), 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 1, 2, 3), 1),
    )
    in_dtype = parameter("int8", "int16")

    data_layout = parameter("NHWC")
    kernel_layout = parameter("HWOI")
    out_layout = parameter("NHWC")
    schedule_name = parameter("conv2d_nhwc_dsp.arm_cpu")


class TestConv2d_NHWC_Spatial_Pack(Conv2dTests):
    """This test is for conv2d_nhwc_spatial_pack.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = parameters(
        ((1, 32, 32, 1), (3, 3), 12, 1, 0, 1),
        ((1, 32, 10, 3), (3, 3), 16, 1, 0, 1),
        ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )
    in_dtype = parameter("int8", "int16")

    data_layout = parameter("NHWC")
    kernel_layout = parameter("HWIO")
    out_layout = parameter("NHWC")
    schedule_name = parameter("conv2d_nhwc_spatial_pack.arm_cpu")


class TestConv2d_NCHW_Spatial_Pack(Conv2dTests):
    """This test is for conv2d_nchw_spatial_pack.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation, in_dtype = parameters(
        ((1, 32, 32, 16), (3, 3), 12, 1, 0, 1, "int8"),
        ((1, 32, 32, 16), (3, 3), 12, 1, 0, 1, "int16"),
        ((1, 16, 16, 32), (3, 3), 12, 1, 0, 1, "int16"),
    )
    data_layout = parameter("NCHW")
    kernel_layout = parameter("OIHW")
    out_layout = parameter("NCHW")
    schedule_name = parameter("conv2d_nchw_spatial_pack.arm_cpu")


in_dtype = tvm.testing.parameter("float16", "float32")
out_dtype = tvm.testing.parameter("float32")

batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation = tvm.testing.parameters(
    # Pad M, N, K
    (1, 1, 1, 1, 1, 1, "SAME", 1),
    (1, 1, 3, 15, 1, 1, "SAME", 1),
    # Pad M, K
    (1, 3, 9, 16, 3, 1, "SAME", 1),
    # Pad M, N
    (1, 2, 9, 15, 4, 1, "SAME", 1),
    # Pad K, N
    (1, 7, 4, 15, 3, 1, "SAME", 1),
    # Pad M
    (1, 2, 9, 16, 4, 1, "SAME", 1),
    # Pad K
    (1, 7, 4, 16, 3, 1, "SAME", 1),
    # Pad N
    (1, 2, 4, 15, 4, 1, "SAME", 1),
    (1, 2, 4, 20, 1, 1, "SAME", 1),
    # Large workloads
    (1, 128, 32, 128, 3, 1, "SAME", 1),
    (4, 64, 16, 64, 5, 2, "SAME", 1),
    (1, 128, 32, 128, 3, 1, "VALID", 1),
    (4, 64, 16, 64, 5, 2, "VALID", 1),
    (1, 64, 16, 64, 3, 2, (0, 0, 1, 1), 1),
    (1, 64, 16, 64, 3, 2, (1, 1, 2, 2), 1),
    (1, 64, 16, 64, 5, 2, (3, 3, 2, 2), 1),
    (1, 64, 16, 64, 3, 2, (0, 1, 2, 3), 1),
    (1, 64, 32, 64, 3, 1, "SAME", 2),
    (1, 64, 32, 64, 3, 1, (1, 1, 2, 2), 2),
)


@tvm.testing.fixture()
def ref_data(
    in_dtype, out_dtype, batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation
):
    np.random.seed(0)
    in_height = in_width = in_size
    a_shape = (batch, in_height, in_width, in_channel)
    w_shape = (kernel, kernel, in_channel, num_filter)

    a_np = np.random.uniform(size=a_shape).astype(in_dtype)
    w_np = np.random.uniform(size=w_shape).astype(in_dtype)
    dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
    b_np = tvm.topi.testing.conv2d_nhwc_python(
        a_np.astype(out_dtype), dw_np.astype(out_dtype), stride, padding
    ).astype(out_dtype)
    return a_np, w_np, dw_np, b_np


@pytest.mark.skipif(
    llvm_version_major() < 16, reason="SME is not supported in earlier versions of LLVM"
)
@tvm.testing.requires_aprofile_aem_fvp
def test_conv2d_sme(target, ref_data, in_dtype, out_dtype, stride, padding, dilation):
    a_np, w_np, dw_np, b_np = ref_data

    kernel_size = get_const_tuple(w_np.shape[:2])
    out_channels = w_np.shape[3]

    x = relay.var("data", shape=a_np.shape, dtype=in_dtype)
    weight = relay.const(w_np, dtype=in_dtype)
    conv2d = relay.nn.conv2d(
        x,
        weight,
        channels=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        dilation=dilation,
        padding=get_pad_tuple(padding, dw_np.shape[:2]),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype=out_dtype,
    )

    func = relay.Function(relay.analysis.free_vars(conv2d), conv2d)

    ir_mod = tvm.IRModule.from_expr(func)
    ir_mod = tvm.relay.transform.InferType()(ir_mod)

    inputs = {"data": a_np}
    params = {}
    ref_outputs = {"output": b_np}

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
    ), target, tvm.meta_schedule.database.ScheduleFnDatabase(arm_cpu_tir_strategy):
        executor_factory = tvm.relay.build(
            ir_mod,
            target=target,
            executor=executor,
            runtime=runtime,
            params=params,
        )

    if in_dtype == "float16":
        func_name = "tvmgen_default_fused_nn_contrib_conv2d_gemm_without_weight_transform"
    else:
        func_name = "tvmgen_default_fused_nn_conv2d"
    generated_func = executor_factory.lowered_ir_mods.items()[0][1][func_name]
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


if __name__ == "__main__":
    tvm.testing.main()
