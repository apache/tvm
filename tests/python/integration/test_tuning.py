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
# pylint: disable=missing-docstring
import logging
import tempfile
from typing import List, Optional

import numpy as np  # type: ignore
import pytest
import tvm
from tvm import meta_schedule as ms
from tvm import relay
from tvm.contrib import graph_executor
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.tune_utils import generate_input_data
from tvm.target.target import Target

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@pytest.mark.skip("Integration test")
@pytest.mark.parametrize(
    "model_name, input_shape, data_type, target, layout",
    [
        ("resnet_18", [1, 3, 224, 224], "float32", "llvm --num-cores=12", "NHWC"),
        ("resnet_18", [1, 3, 224, 224], "float32", "nvidia/geforce-rtx-3070", "NHWC"),
    ],
)
def test_meta_schedule_tune_relay(
    model_name: str,
    input_shape: List[int],
    data_type: str,
    target: str,
    layout: Optional[str],
):
    dev = tvm.cpu() if str(target).startswith("llvm") else tvm.cuda()
    data = generate_input_data(input_shape, data_type)

    mod, params, (input_name, _, _) = get_network(
        name=model_name,
        input_shape=input_shape,
        layout=layout,
    )

    target = Target(target)
    with tempfile.TemporaryDirectory() as work_dir:
        with ms.Profiler() as profiler:
            database = ms.relay_integration.tune_relay(
                mod=mod,
                target=target,
                params=params,
                work_dir=work_dir,
                max_trials_global=2048,
            )
            rt_mod1 = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
            )
        print(profiler.table())

    def get_output(data, lib, dev):
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_name, tvm.nd.array(data, device=dev))
        module.run()
        return module.get_output(0).numpy()

    # Check correctness
    actual_output = get_output(data, rt_mod1, dev)
    print(
        f"{model_name} finished tuning and running on {Target(target).kind.name}. "
        "Running baseline...",
        flush=True,
    )

    # Compile without meta-schedule for correctness check
    baseline_target = "llvm -num-cores=1"
    with tvm.transform.PassContext(opt_level=0):
        rt_mod2 = relay.build(mod, target=baseline_target, params=params)

    expected_output = get_output(data, rt_mod2, tvm.cpu())
    print(
        f"Basline finished running on {Target(baseline_target).kind.name}. "
        "Verifying correctness...",
        flush=True,
    )

    assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)
    print(
        f"Correctness verified for {model_name} on {Target(target).kind.name}.",
        flush=True,
    )


if __name__ == """__main__""":
    test_meta_schedule_tune_relay(
        "resnet_18", [1, 3, 224, 224], "float32", "llvm --num-cores=12", "NHWC"
    )
    test_meta_schedule_tune_relay(
        "resnet_18", [1, 3, 224, 224], "float32", "nvidia/geforce-rtx-3070", None
    )
