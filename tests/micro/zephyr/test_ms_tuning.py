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
from tvm import relay
import tvm.micro.testing
from tvm.relay.backend import Executor
from tvm.contrib import graph_executor
from tvm import meta_schedule as ms
from tvm.contrib.micro.meta_schedule.local_builder_micro import get_local_builder_micro
from tvm.contrib.micro.meta_schedule.rpc_runner_micro import get_rpc_runner_micro


def create_relay_module():
    data_shape = (1, 3, 16, 16)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    weight_sample = np.random.rand(
        weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
    ).astype("float32")
    params = {mod["main"].params[1].name_hint: weight_sample}

    model_info = {
        "in_tensor": "data",
        "in_shape": data_shape,
        "in_dtype": "float32",
    }

    return mod, params, model_info


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521", "mps3_an547", "nucleo_f746zg", "stm32f746g_disco"])
def test_ms_tuning_conv2d(workspace_dir, board, microtvm_debug, use_fvp, serial_number):
    """Test meta-schedule tuning for microTVM Zephyr"""

    mod, params, model_info = create_relay_module()
    input_name = model_info["in_tensor"]
    input_shape = model_info["in_shape"]
    input_dtype = model_info["in_dtype"]
    data_sample = np.random.rand(*input_shape).astype(input_dtype)

    platform = "zephyr"
    project_options = {
        "board": board,
        "verbose": microtvm_debug,
        "project_type": "host_driven",
        "use_fvp": bool(use_fvp),
        "serial_number": serial_number,
        "config_main_stack_size": 4096,
    }
    if isinstance(serial_number, list):
        project_options["serial_number"] = serial_number[0]  # project_api expects an string.
        serial_numbers = serial_number
    else:
        if serial_number is not None:  # use a single device in tuning
            serial_numbers = [serial_number]
        else:  # use two dummy serial numbers (for testing with QEMU)
            serial_numbers = [str(i) for i in range(2)]

    boards_file = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")) / "boards.json"
    with open(boards_file) as f:
        boards = json.load(f)
    target = tvm.micro.testing.get_target("zephyr", board)

    runtime = relay.backend.Runtime("crt", {"system-lib": True})
    executor = Executor("aot", {"link-params": True})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    builder = get_local_builder_micro()
    with ms.Profiler() as profiler:
        with get_rpc_runner_micro(
            platform=platform,
            options=project_options,
            session_timeout_sec=120,
            serial_numbers=serial_numbers,
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
                work_dir=str(workspace_dir),
                module_equality="ignore-ndarray",
            )

        #  Build model using meta_schedule logs
        opt_mod, opt_params = relay.optimize(mod, target)
        ms_mod: tvm.runtime.Module = ms.relay_integration.compile_relay(
            database=db,
            mod=opt_mod,
            target=target,
            params=opt_params,
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
        str(workspace_dir / "project"),
        options=project_options,
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
    target = tvm.micro.testing.get_target("crt")
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
    ):
        ref_mod = relay.build(
            mod,
            target=target,
            params=params,
            runtime=runtime,
        )
    ref_mod.export_library(workspace_dir / "compiled_lib2.so")
    mod2: tvm.runtime.Module = tvm.runtime.load_module(workspace_dir / "compiled_lib2.so")
    graph_mod = graph_executor.GraphModule(mod2["default"](dev))
    graph_mod.set_input(input_name, data_sample)
    graph_mod.run()
    ref_output = graph_mod.get_output(0).numpy()

    assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"


if __name__ == "__main__":
    tvm.testing.main()
