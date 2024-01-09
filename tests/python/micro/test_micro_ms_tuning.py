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
import numpy as np
import pytest
from types import MappingProxyType
import pathlib
import json
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.contrib import graph_executor, utils
from tvm import meta_schedule as ms


@tvm.testing.requires_micro
def test_micro_tuning_with_meta_schedule():
    from tests.micro.zephyr.test_ms_tuning import create_relay_module
    from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
    from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro

    platform = "crt"
    target = tvm.target.target.micro(model="host")
    options = {}

    work_dir = utils.tempdir()
    mod, params, model_info = create_relay_module()
    input_name = model_info["in_tensor"]
    input_shape = model_info["in_shape"]
    input_dtype = model_info["in_dtype"]
    data_sample = np.random.rand(*input_shape).astype(input_dtype)

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": True})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    builder = get_local_builder_micro()

    with ms.Profiler() as profiler:
        with get_rpc_runner_micro(
            platform=platform, options=options, session_timeout_sec=120
        ) as runner:
            db: ms.Database = ms.relay_integration.tune_relay(
                mod=mod,
                params=params,
                target=target,
                builder=builder,
                runner=runner,
                strategy="evolutionary",
                num_trials_per_iter=2,
                max_trials_per_task=10,
                max_trials_global=100,
                work_dir=str(work_dir.path),
                module_equality="ignore-ndarray",
            )

        #  Build model using meta_schedule logs
        ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
            database=db,
            mod=mod,
            target=target,
            params=params,
            pass_config=MappingProxyType(
                {
                    "relay.backend.use_meta_schedule": True,
                    "relay.backend.tir_converter": "default",
                    "tir.disable_vectorize": True,
                }
            ),
            executor=executor,
            runtime=runtime,
        )
    print(profiler.table())

    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects(platform)),
        ms_mod,
        str(work_dir / "project"),
        options=options,
    )
    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
        aot_executor.get_input(0).copyfrom(data_sample)
        result = aot_executor.module.time_evaluator("run", session.device, number=3)()
        output = aot_executor.get_output(0).numpy()

    # Build reference model (without tuning)
    dev = tvm.cpu()
    target = tvm.target.target.micro(model="host")
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
    ):
        ref_mod = relay.build(
            mod,
            target=target,
            params=params,
            runtime=runtime,
        )
    ref_mod.export_library(work_dir / "compiled_lib2.so")
    mod2: tvm.runtime.Module = tvm.runtime.load_module(work_dir / "compiled_lib2.so")
    graph_mod = graph_executor.GraphModule(mod2["default"](dev))
    graph_mod.set_input(input_name, data_sample)
    graph_mod.run()
    ref_output = graph_mod.get_output(0).numpy()

    assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"
    work_dir.remove()


if __name__ == "__main__":
    tvm.testing.main()
