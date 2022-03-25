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
import numpy as np

import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm import meta_schedule as ms
from tvm.ir.module import IRModule
from tvm.meta_schedule.database import PyDatabase, TuningRecord, Workload
from tvm.meta_schedule.integration import (
    ApplyHistoryBest,
    ExtractedTask,
    MetaScheduleContext,
)
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.utils import derived_object
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule
from tvm.meta_schedule.testing import DummyDatabase
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
from tvm.meta_schedule.tune import extract_task_from_relay, Parse

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


def test_meta_schedule_integration_no_current():
    assert MetaScheduleContext.current() is None


@requires_torch
def test_meta_schedule_integration_extract_from_resnet():
    mod, params, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    extracted_tasks = ms.integration.extract_task_from_relay(mod, target="llvm", params=params)
    expected_task_names = [
        "fused_" + s
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
    mod, _, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    database = DummyDatabase()
    env = ApplyHistoryBest(database)
    target = Target("llvm")
    workload = database.commit_workload(MockModule)
    database.commit_tuning_record(
        TuningRecord(Schedule(MockModule).trace, [1.0], workload, target, [])
    )
    mod = env.query(task_name="mock-task", mod=mod, target=target, dispatched=[MockModule])
    assert tvm.ir.structural_equal(mod, workload.mod)


@pytest.mark.skip("Too slow on CI")
def extract_task_qbert():
    mod, params, _ = load_quantized_bert_base(batch_size=1, seq_len=128)
    target = "llvm -mcpu=cascadelake"
    extracted_tasks = extract_task_from_relay(mod, target, params)
    tune_tasks = list(
        filter(
            lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
            extracted_tasks,
        )
    )
    # three int8 dense, two int8 bmm, and one fp32 dense
    assert len(tune_tasks) == 6

    for task in tune_tasks:
        relay_func = list(task.mod.functions.values())[0]
        out_type = relay_func.body.checked_type

        if out_type.dtype == "float32":
            continue

        mod = Parse._mod(task.dispatched[0])
        sch = tvm.tir.Schedule(mod)
        block = sch.get_block("compute")
        annotations = sch.get(block).annotations

        assert "schedule_rule" in annotations
        assert "vnni" in annotations["schedule_rule"]


@tvm.testing.skip_if_32bit(reason="Apparently the LLVM version on i386 image is too old")
def test_extract_task_arm_conv2d_nchwc():
    data_shape = (1, 64, 128, 128)
    weight_shape = (32, 64, 1, 1)
    bias_shape = (weight_shape[0],)
    padding = (1, 1)

    data = relay.var("data", shape=data_shape, dtype="int8")
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    bias = relay.var("bias", shape=bias_shape, dtype="int32")
    conv2d = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=weight_shape[2:],
        channels=weight_shape[0],
        padding=padding,
        strides=(1, 1),
        out_dtype="int32",
    )
    bias_add = relay.nn.bias_add(conv2d, bias)
    relay_mod = tvm.IRModule.from_expr(bias_add)

    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    bias_np = np.random.uniform(1, 10, size=bias_shape).astype("int32")

    params = {"weight": weight_np, "bias": bias_np}

    target = "llvm -device arm_cpu -mtriple aarch64-linux-gnu -mattr=+neon"
    extracted_tasks = extract_task_from_relay(relay_mod, target, params)
    tune_tasks = list(
        filter(
            lambda task: "conv2d" in task.task_name,
            extracted_tasks,
        )
    )

    assert len(tune_tasks) == 1

    relay_func = list(tune_tasks[0].mod.functions.values())[0]
    out_type = relay_func.body.checked_type

    # Check that the output is in NCHWc layout
    assert list(out_type.shape) == [1, 8, 130, 130, 4]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
