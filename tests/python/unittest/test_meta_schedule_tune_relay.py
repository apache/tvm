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
import pytest
import numpy as np
from typing import Tuple, List

from tvm.meta_schedule.utils import derived_object

try:
    import torch
except ModuleNotFoundError:
    pass

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.runtime.ndarray import cpu, cuda
from tvm.target.target import Target
from tvm.contrib import graph_executor
from tvm.meta_schedule import ReplayTraceConfig
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord
from tvm.meta_schedule.testing import MODEL_TYPE, MODEL_TYPES, get_torch_model
from tvm.meta_schedule.tune import tune_relay

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
@pytest.mark.parametrize("model_name", ["resnet18", "mobilenet_v2", "bert_base"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("target", ["llvm --num-cores=16", "nvidia/geforce-rtx-3070"])
def test_meta_schedule_tune_relay(model_name: str, batch_size: int, target: str):
    if model_name == "inception_v3" and batch_size == 1:
        pytest.skip("inception_v3 does not handle batch_size of 1")

    input_shape: Tuple[int, ...]
    input_name = "input0"
    dev = tvm.cpu() if str(target).startswith("llvm") else cuda()
    if MODEL_TYPES[model_name] == MODEL_TYPE.TEXT_CLASSIFICATION:
        seq_length = 128
        input_name = "input_ids"
        input_shape = (batch_size, seq_length)
        data = tvm.nd.array(np.random.randint(0, 30521, size=input_shape), dev)  # embedding size
    else:
        if MODEL_TYPES[model_name] == MODEL_TYPE.IMAGE_CLASSIFICATION:
            input_shape = (batch_size, 3, 299, 299)
        elif MODEL_TYPES[model_name] == MODEL_TYPE.SEGMENTATION:
            input_shape = (batch_size, 3, 299, 299)
        elif MODEL_TYPES[model_name] == MODEL_TYPE.OBJECT_DETECTION:
            input_shape = (1, 3, 300, 300)
        elif MODEL_TYPES[model_name] == MODEL_TYPE.VIDEO_CLASSIFICATION:
            input_shape = (batch_size, 3, 3, 299, 299)
        else:
            raise ValueError("Unsupported model: " + model_name)
        data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), dev)

    output_shape: Tuple[int, int] = (batch_size, 1000)

    mod, params = get_torch_model(
        model_name=model_name,
        input_shape=input_shape,
        output_shape=output_shape,
        dtype="float32",
    )

    with tempfile.TemporaryDirectory() as work_dir:
        target = Target(target)
        database = DummyDatabase()
        rt_mod: tvm.module = tune_relay(
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
            rt_mod2 = relay.build(mod, target=target, params=params)

        def get_output(data, lib):
            module = graph_executor.GraphModule(lib["default"](dev))
            module.set_input(input_name, data)
            module.run()
            return module.get_output(0).numpy()

        # Check correctness
        actual_output = get_output(data, rt_mod)
        expected_output = get_output(data, rt_mod2)
        assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)


if __name__ == """__main__""":
    test_meta_schedule_tune_relay("resnet18", 1, "llvm --num-cores=16")
    test_meta_schedule_tune_relay("resnet18", 1, "nvidia/geforce-rtx-3070")
    test_meta_schedule_tune_relay("mobilenet_v2", 1, "llvm --num-cores=16")
    test_meta_schedule_tune_relay("mobilenet_v2", 1, "nvidia/geforce-rtx-3070")
    test_meta_schedule_tune_relay("bert_base", 1, "llvm --num-cores=16")
    test_meta_schedule_tune_relay("bert_base", 1, "nvidia/geforce-rtx-3070")
