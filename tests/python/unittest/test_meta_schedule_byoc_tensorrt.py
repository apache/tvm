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
""" Test Meta Schedule Builder """
import sys
import pytest
import itertools
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay.op.contrib import tensorrt
import numpy as np
from typing import List
from tvm._ffi import register_func
from tvm.target import Target
from tvm.runtime import Module
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder, BuilderResult
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    LocalRunner,
    RunnerInput,
)

from tvm.tir import FloatImm
from tvm.meta_schedule.testing import get_network

has_tensorrt_codegen = pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.tensorrt", True), reason="TensorRT codegen not available"
)
has_tensorrt_runtime = pytest.mark.skipif(
    not tensorrt.is_tensorrt_runtime_enabled(), reason="TensorRT runtime not available"
)


# conv2d+relu network
def get_conv2d_relu(
    data_shape,
    out_channels,
    kernel_size,
    strides,
    padding,
    dilation,
    groups,
    data_layout,
    kernel_layout,
    dtype,
):

    data = relay.var("data", relay.TensorType(data_shape, dtype))
    weight = relay.var("weight")

    net = relay.nn.conv2d(
        data=data,
        weight=weight,  # conv kernel
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        channels=out_channels,
        kernel_size=kernel_size,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )
    net = relay.add(net, net)
    net = relay.nn.relu(net)

    inputs = relay.analysis.free_vars(net)
    return relay.Function(inputs, net)


def verify_meta_schedule_with_tensorrt(
    mod, params, data_shape, use_meta_sched: bool = True, use_trt: bool = True, mode: str = "vm"
):
    if use_meta_sched:
        # With meta_schedule
        dev = "cuda"

        # Build
        if use_trt:
            from tvm.meta_schedule.testing import relay_build_with_tensorrt

            builder = LocalBuilder(f_build=relay_build_with_tensorrt)
        else:

            def relay_build_without_tensorrt(
                mod: Module,
                target: Target,
                params: dict,
            ) -> List[BuilderResult]:
                return tvm.relay.build_module._build_module_no_factory(mod, "cuda", "llvm", params)

            builder = LocalBuilder(f_build=relay_build_without_tensorrt)

        builder_input = BuilderInput(mod, Target(dev, host="llvm"), params)

        (builder_result,) = builder.build([builder_input])
        assert builder_result.error_msg is None
        assert builder_result.artifact_path is not None

        # Run
        evaluator_config = EvaluatorConfig(
            number=5,
            repeat=2,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )

        runner_input = RunnerInput(
            builder_result.artifact_path, "cuda", [TensorInfo("float32", data_shape)]
        )

        def eval_func(rt_mod, device, evaluator_config, repeated_args):
            rt_mod = tvm.contrib.graph_executor.GraphModule(rt_mod["default"](device))

            eval = rt_mod.module.time_evaluator(
                func_name="run",
                dev=device,
                number=evaluator_config.number,
                repeat=evaluator_config.repeat,
                min_repeat_ms=evaluator_config.min_repeat_ms,
                f_preproc="cache_flush_cpu_non_first_arg"
                if evaluator_config.enable_cpu_cache_flush
                else "",
            )
            repeated_costs: List[List[float]] = []
            for args in repeated_args:
                profile_result = eval(*args)
                repeated_costs.append(profile_result.results)

            costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
            return costs

        runner = LocalRunner(
            evaluator_config=evaluator_config,
            f_run_evaluator=eval_func,
        )

        # Run the module
        (runner_future,) = runner.run([runner_input])
        runner_result = runner_future.result()
        assert runner_result is not None
        assert runner_result.run_secs is not None
        assert runner_result.error_msg is None

        for result in runner_result.run_secs:
            if isinstance(result, FloatImm):
                result = result.value
            assert isinstance(result, float)
            assert result >= 0.0

    else:
        # Without meta_schedule
        if use_trt:
            mod, config = tensorrt.partition_for_tensorrt(mod)
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.tensorrt.options": config}
            ):
                func = relay.create_executor(
                    mode, mod=mod, device=tvm.cuda(0), target="cuda"
                ).evaluate()
        else:
            with tvm.transform.PassContext(opt_level=3):
                func = relay.create_executor(
                    mode, mod=mod, device=tvm.cuda(0), target="cuda", params=params
                ).evaluate()


@tvm.testing.requires_cuda
@has_tensorrt_codegen
@has_tensorrt_runtime
def test_conv2d_relu():
    data_shape = (1, 1280, 14, 14)
    out_channels = 256
    kernel_size, strides, padding, dilation, groups = (1, 1), (1, 1), (0, 0, 0, 0), (1, 1), 1
    data_layout, kernel_layout = "NCHW", "OIHW"
    dtype = "float32"

    f = get_conv2d_relu(
        data_shape,
        out_channels,
        kernel_size,
        strides,
        padding,
        dilation,
        groups,
        data_layout,
        kernel_layout,
        dtype,
    )

    mod, params = testing.create_workload(f)
    verify_meta_schedule_with_tensorrt(mod, params, data_shape)


@tvm.testing.requires_cuda
@has_tensorrt_codegen
@has_tensorrt_runtime
@pytest.mark.parametrize(
    "model_name",
    ["resnet-50", "mobilenet"],
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("use_meta_sched", [True])
@pytest.mark.parametrize("use_trt", [True, False])
def test_relay_model(model_name: str, batch_size: int, use_meta_sched: bool, use_trt: bool):

    mod, params, input_shape, output_shape = get_network(name=model_name, batch_size=batch_size)
    verify_meta_schedule_with_tensorrt(
        mod, params, input_shape, use_meta_sched=use_meta_sched, use_trt=use_trt, mode="vm"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
