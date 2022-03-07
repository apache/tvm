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
from typing import List

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.ir import IRModule
from tvm.meta_schedule import ReplayTraceConfig
from tvm.meta_schedule.database import PyDatabase, TuningRecord, Workload
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.tune import tune_relay
from tvm.meta_schedule.utils import derived_object
from tvm.target.target import Target

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@derived_object
class DummyDatabase(PyDatabase):
    def __init__(self):
        super().__init__()
        self.records = []
        self.workload_reg = []

    def has_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return True
        return False

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.records.append(record)

    def commit_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return workload
        workload = Workload(mod)
        self.workload_reg.append(workload)
        return workload

    def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
        return list(
            filter(
                lambda x: x.workload == workload,
                sorted(self.records, key=lambda x: sum(x.run_secs) / len(x.run_secs)),
            )
        )[: int(top_k)]

    def __len__(self) -> int:
        return len(self.records)

    def print_results(self) -> None:
        print("\n".join([str(r) for r in self.records]))


@pytest.mark.skip("Integration test")
@pytest.mark.parametrize(
    "model_name, input_shape, target",
    [
        ("resnet_18", [1, 3, 224, 224], "llvm --num-cores=16"),
        ("resnet_18", [1, 3, 224, 224], "nvidia/geforce-rtx-3070"),
        ("mobilenet_v2", [1, 3, 224, 224], "llvm --num-cores=16"),
        ("mobilenet_v2", [1, 3, 224, 224], "nvidia/geforce-rtx-3070"),
        ("bert_base", [1, 64], "llvm --num-cores=16"),
        ("bert_base", [1, 64], "nvidia/geforce-rtx-3070"),
    ],
)
def test_meta_schedule_tune_relay(
    model_name: str,
    input_shape: List[int],
    target: str,
):
    dev = tvm.cpu() if str(target).startswith("llvm") else tvm.cuda()
    if model_name.startswith("bert"):
        data = tvm.nd.array(np.random.randint(0, 30521, size=input_shape), dev)  # embedding size
    else:
        data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), dev)

    mod, params, (input_name, _, _) = get_network(name=model_name, input_shape=input_shape)
    target = Target(target)
    with tempfile.TemporaryDirectory() as work_dir:
        database = DummyDatabase()
        rt_mod: tvm.runtime.Module = tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=ReplayTraceConfig(
                num_trials_per_iter=32,
                num_trials_total=32,
            ),
            work_dir=work_dir,
            database=database,
        )
        # Compile without meta-scheduler for correctness check
        with tvm.transform.PassContext(opt_level=0):
            rt_mod2 = relay.build(mod, target=Target("llvm"), params=params)

        def get_output(data, lib):
            module = graph_executor.GraphModule(lib["default"](dev))
            module.set_input(input_name, data)
            module.run()
            return module.get_output(0).numpy()

        # Check correctness
        actual_output = get_output(data, rt_mod)
        expected_output = get_output(tvm.nd.array(data.numpy(), device=tvm.cpu()), rt_mod2)
        assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)


if __name__ == """__main__""":
    test_meta_schedule_tune_relay("resnet_18", [1, 3, 224, 224], "llvm --num-cores=16")
    test_meta_schedule_tune_relay("resnet_18", [1, 3, 224, 224], "nvidia/geforce-rtx-3070")
    test_meta_schedule_tune_relay("mobilenet_v2", [1, 3, 224, 224], "llvm --num-cores=16")
    test_meta_schedule_tune_relay("mobilenet_v2", [1, 3, 224, 224], "nvidia/geforce-rtx-3070")
    test_meta_schedule_tune_relay("bert_base", [1, 64], "llvm --num-cores=16")
    test_meta_schedule_tune_relay("bert_base", [1, 64], "nvidia/geforce-rtx-3070")
