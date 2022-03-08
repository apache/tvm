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
import sys
from typing import List

import pytest
import tvm
from tvm import meta_schedule as ms
from tvm.ir.module import IRModule
from tvm.meta_schedule.database import PyDatabase, TuningRecord, Workload
from tvm.meta_schedule.integration import (
    ApplyHistoryBest,
    ExtractedTask,
    MetaScheduleContext,
    TaskExtraction,
)
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.utils import derived_object
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule

# pylint: disable=no-member,line-too-long,too-many-nested-blocks,unbalanced-tuple-unpacking,no-self-argument,missing-docstring,invalid-name


@tvm.script.ir_module
class MockModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:  # type: ignore
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        for i in T.serial(0, 16):
            with T.block("matmul"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]


# pylint: enable=no-member,line-too-long,too-many-nested-blocks,unbalanced-tuple-unpacking,no-self-argument


def _has_torch():
    import importlib.util  # pylint: disable=unused-import,import-outside-toplevel

    spec = importlib.util.find_spec("torch")
    return spec is not None


requires_torch = pytest.mark.skipif(not _has_torch(), reason="torch is not installed")


def _check_mock_task(tasks: List[ExtractedTask], mod: IRModule):
    (task,) = tasks
    assert isinstance(task, ExtractedTask)
    assert task.task_name == "mock-task"
    tvm.ir.assert_structural_equal(task.mod, mod)
    (tir_mod,) = task.dispatched
    tvm.ir.assert_structural_equal(tir_mod, MockModule)


@requires_torch
def test_meta_schedule_integration_task_extraction_query():
    mod, _, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    env = TaskExtraction()
    env.query(task_name="mock-task", mod=mod, target=Target("llvm"), dispatched=[MockModule])
    _check_mock_task(env.tasks, mod)


def test_meta_schedule_integration_current():
    env = TaskExtraction()
    with env:
        assert MetaScheduleContext.current() == env


def test_meta_schedule_integration_no_current():
    assert MetaScheduleContext.current() is None


def test_meta_schedule_integration_multiple_current():
    env = TaskExtraction()
    with env:
        with pytest.raises(ValueError):
            with env:
                ...


@requires_torch
def test_meta_schedule_integration_query_inside_with_scope():
    mod, _, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    env = TaskExtraction()
    with env:
        MetaScheduleContext.query_inside_with_scope(
            task_name="mock-task",
            mod=mod,
            target=Target("llvm"),
            dispatched=[MockModule],
        )
    _check_mock_task(env.tasks, mod)


@requires_torch
def test_meta_schedule_integration_extract_from_resnet():
    mod, params, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    extracted_tasks = ms.integration.extract_task_from_relay(mod, target="llvm", params=params)
    expected_task_names = [
        "vm_mod_fused_" + s
        for s in [
            "nn_max_pool2d",
            "nn_adaptive_avg_pool2d",
            "nn_dense_add",
            "nn_conv2d_add",
            "nn_conv2d_add_1",
            "nn_conv2d_add_2",
            "nn_conv2d_add_add_nn_relu",
            "nn_conv2d_add_add_nn_relu_1",
            "nn_conv2d_add_nn_relu",
            "nn_conv2d_add_nn_relu_1",
            "nn_conv2d_add_nn_relu_2",
            "nn_conv2d_add_nn_relu_3",
            "nn_conv2d_add_nn_relu_4",
            "nn_conv2d_add_nn_relu_5",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_1",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu",
            "nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1",
            # The two tasks below are purely spatial and are ruled out by AutoScheduler
            "layout_transform",
            "layout_transform_reshape_squeeze",
        ]
    ]

    assert len(extracted_tasks) == 20
    for t in extracted_tasks:
        assert t.task_name in expected_task_names, t.task_name


@requires_torch
def test_meta_schedule_integration_apply_history_best():
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

    mod, _, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    database = DummyDatabase()
    env = ApplyHistoryBest(database)
    target = Target("llvm")
    workload = database.commit_workload(MockModule)
    database.commit_tuning_record(
        TuningRecord(Schedule(MockModule).trace, [1.0], workload, target, [])
    )
    mod = env.query(task_name="mock-task", mod=mod, target=target, dispatched=[MockModule])
    mod = IRModule({"main": mod})
    assert tvm.ir.structural_equal(mod, workload.mod)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
