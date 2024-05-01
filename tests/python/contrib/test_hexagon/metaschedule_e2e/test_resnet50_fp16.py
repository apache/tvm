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
"""Test Resnet50 float16 with MetaSchedule"""

import os
import tempfile

import pytest
import numpy as np

import tvm.testing
from tvm import relay
from tvm import meta_schedule as ms
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm.relay.backend import Executor

from ..infrastructure import get_hexagon_target


def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


@pytest.mark.skip("End-to-end tuning is skipped on CI.")
@tvm.testing.requires_hexagon
def test_resnet50(hexagon_launcher):
    """Test Resnet50."""
    model_json = "resnet50_fp16.json"
    target_llvm = tvm.target.Target("llvm")
    target_hexagon = get_hexagon_target("v69")
    model_params = "resnet50_fp16.params"

    if not os.path.exists(model_json):
        pytest.skip("Run python export_models.py first.")

    with open(model_json, "r") as file:
        mod = tvm.ir.load_json(file.read())

    with open(model_params, "rb") as file:
        params = relay.load_param_dict(file.read())

    mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "HWIO"]})

    inp = np.random.randn(1, 3, 224, 224).astype("float32")
    input_name = "image"

    executor = Executor("graph", {"link-params": True})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=target_hexagon,
            params=params,
            work_dir=work_dir,
            # for faster tuning
            max_trials_global=20000,
            max_trials_per_task=8,
            num_trials_per_iter=8,
            strategy="replay-trace",
            # max_trials_global=20000,
            # num_trials_per_iter=32,
            # max_trials_per_task=128,
            # strategy="evolutionary",
            builder=get_hexagon_local_builder(),
            runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
            # Without this, the same workloads with different constant weights
            # are treated as distinct tuning tasks.
            module_equality="ignore-ndarray",
        )

        hexagon_lowered = ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=target_hexagon,
            params=params,
        )

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            params=params,
        )

        llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
        llvm_graph_mod.set_input(input_name, inp.copy())
        llvm_graph_mod.run()
        ref_result = llvm_graph_mod.get_output(0).numpy()

    with hexagon_launcher.create_session() as session:
        graph_mod = session.get_executor_from_factory(hexagon_lowered)
        graph_mod.set_input(input_name, inp.copy())

        graph_mod.run()
        hexagon_output = graph_mod.get_output(0).numpy()

        # Example output: max and mean abs difference with the reference: 0.1406 0.0126
        print(
            "max and mean abs difference with the reference:",
            np.max(np.abs(ref_result - hexagon_output)),
            np.mean(np.abs(ref_result - hexagon_output)),
        )
        tvm.testing.assert_allclose(ref_result, hexagon_output, atol=2e-1)

        time_ms = graph_mod.benchmark(session.device, number=1, repeat=20).mean * 1e3

        print("time elapsed: ", time_ms)

        debug_ex = session.get_graph_debug_executor(
            hexagon_lowered.get_graph_json(), hexagon_lowered.lib
        )
        print(debug_ex.profile(input_name=inp.copy()))


if __name__ == "__main__":
    tvm.testing.main()
