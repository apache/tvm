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

"""Test search policy"""

import numpy as np
import tempfile

import tvm
import tvm.testing
from tvm import auto_scheduler
from tvm.auto_scheduler.utils import get_const_tuple
from tvm.testing.auto_scheduler import (
    matmul_auto_scheduler_test,
    zero_rank_compute_auto_scheduler_test,
    zero_rank_reduce_auto_scheduler_test,
)


def test_search_task_add_task_input():
    auto_scheduler.search_task.TASK_INPUT_BUFFER_TABLE.clear()
    N = 64
    target = "llvm"
    test_input_0 = tvm.runtime.ndarray.empty((64, 64))
    test_input_1 = tvm.runtime.ndarray.empty((10, 20))
    test_input_2 = tvm.runtime.ndarray.empty((30, 40, 50))
    task = auto_scheduler.SearchTask(
        func="matmul_auto_scheduler_test",
        args=(N, N, N),
        target=target,
        task_inputs={
            "test_input_0": test_input_0,
            "test_input_1": test_input_1,
            "test_input_2": test_input_2,
        },
        task_inputs_overwrite=True,
    )

    assert len(task.task_input_names) == 3
    assert task.task_input_names[0] == "test_input_0"
    assert task.task_input_names[1] == "test_input_1"
    assert task.task_input_names[2] == "test_input_2"


def test_search_task_record():
    auto_scheduler.search_task.TASK_INPUT_BUFFER_TABLE.clear()
    N = 64
    target = "llvm"

    # Log with no task input
    task = auto_scheduler.SearchTask(
        func="matmul_auto_scheduler_test", args=(N, N, N), target=target
    )
    task_record = auto_scheduler._ffi_api.SerializeSearchTask(task)
    new_task = auto_scheduler._ffi_api.DeserializeSearchTask(task_record)
    # TODO(jcf94): Check the compute dag & hardware parameter
    assert task.workload_key == new_task.workload_key
    assert str(task.target) == str(new_task.target)
    assert str(task.target.host) == str(new_task.target.host)
    assert task.layout_rewrite_option == new_task.layout_rewrite_option

    # Log with 1 task input
    test_input_0 = tvm.runtime.ndarray.empty((64, 64))
    task = auto_scheduler.SearchTask(
        func="matmul_auto_scheduler_test",
        args=(N, N, N),
        target=target,
        task_inputs={"test_input_0": test_input_0},
        task_inputs_overwrite=True,
    )
    task_record = auto_scheduler._ffi_api.SerializeSearchTask(task)
    new_task = auto_scheduler._ffi_api.DeserializeSearchTask(task_record)
    assert task.workload_key == new_task.workload_key
    assert str(task.target) == str(new_task.target)
    assert str(task.target.host) == str(new_task.target.host)
    assert task.layout_rewrite_option == new_task.layout_rewrite_option
    assert len(new_task.task_input_names) == 1
    assert new_task.task_input_names[0] == "test_input_0"

    # Log with multiple task inputs
    test_input_1 = tvm.runtime.ndarray.empty((64, 64))
    task = auto_scheduler.SearchTask(
        func="matmul_auto_scheduler_test",
        args=(N, N, N),
        target=target,
        task_inputs={
            "test_input_0": test_input_0,
            "test_input_1": test_input_1,
        },
        task_inputs_overwrite=True,
    )
    task_record = auto_scheduler._ffi_api.SerializeSearchTask(task)
    new_task = auto_scheduler._ffi_api.DeserializeSearchTask(task_record)
    assert task.workload_key == new_task.workload_key
    assert str(task.target) == str(new_task.target)
    assert str(task.target.host) == str(new_task.target.host)
    assert task.layout_rewrite_option == new_task.layout_rewrite_option
    assert len(new_task.task_input_names) == 2
    assert new_task.task_input_names[0] == "test_input_0"
    assert new_task.task_input_names[1] == "test_input_1"

    # Log with version 0.5
    v5_log = (
        """["[\\\"matmul_auto_scheduler_test\\\", 64, 64, 64]", """
        f'"{str(tvm.target.Target(target))}"'
        """, [6, 64, 64, 0, 0, 0, 0, 0], "", 1]"""
    )
    new_task = auto_scheduler._ffi_api.DeserializeSearchTask(v5_log)
    assert task.workload_key == new_task.workload_key
    assert str(task.target) == str(new_task.target)
    assert str(task.target.host) == str(new_task.target.host)
    assert task.layout_rewrite_option == new_task.layout_rewrite_option
    assert len(new_task.task_input_names) == 0


def test_recover_measure_input_with_task_input():
    auto_scheduler.search_task.TASK_INPUT_BUFFER_TABLE.clear()
    target = "llvm"

    # Since this file is tests for search_task, we only check the search_task here

    # Log with no task input
    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test, args=(512, 512, 512), target=target
    )
    inp = auto_scheduler.measure.MeasureInput(task, task.compute_dag.init_state)
    res = auto_scheduler.measure.MeasureResult([0.1], 0, "", 0.2, 1)
    measure_record = auto_scheduler.measure_record.dump_record_to_string(inp, res)
    measure_log = auto_scheduler.measure_record.load_record_from_string(measure_record)
    new_task = measure_log[0].task
    assert task.workload_key == new_task.workload_key
    assert str(task.target) == str(new_task.target)
    assert str(task.target.host) == str(new_task.target.host)
    assert task.layout_rewrite_option == new_task.layout_rewrite_option

    # Log with 1 task input
    test_input_0 = tvm.runtime.ndarray.empty((64, 64))
    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test,
        args=(512, 512, 512),
        target=target,
        task_inputs={
            "test_input_0": test_input_0,
        },
        task_inputs_overwrite=True,
    )
    inp = auto_scheduler.measure.MeasureInput(task, task.compute_dag.init_state)
    res = auto_scheduler.measure.MeasureResult([0.1], 0, "", 0.2, 1)
    measure_record = auto_scheduler.measure_record.dump_record_to_string(inp, res)
    measure_log = auto_scheduler.measure_record.load_record_from_string(measure_record)
    new_task = measure_log[0].task
    assert task.workload_key == new_task.workload_key
    assert str(task.target) == str(new_task.target)
    assert str(task.target.host) == str(new_task.target.host)
    assert task.layout_rewrite_option == new_task.layout_rewrite_option
    assert len(new_task.task_input_names) == 1
    assert new_task.task_input_names[0] == "test_input_0"

    # Log with multiple task inputs
    test_input_1 = tvm.runtime.ndarray.empty((64, 64))
    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test,
        args=(512, 512, 512),
        target=target,
        task_inputs={
            "test_input_0": test_input_0,
            "test_input_1": test_input_1,
        },
        task_inputs_overwrite=True,
    )
    inp = auto_scheduler.measure.MeasureInput(task, task.compute_dag.init_state)
    res = auto_scheduler.measure.MeasureResult([0.1], 0, "", 0.2, 1)
    measure_record = auto_scheduler.measure_record.dump_record_to_string(inp, res)
    measure_log = auto_scheduler.measure_record.load_record_from_string(measure_record)
    new_task = measure_log[0].task
    assert task.workload_key == new_task.workload_key
    assert str(task.target) == str(new_task.target)
    assert str(task.target.host) == str(new_task.target.host)
    assert task.layout_rewrite_option == new_task.layout_rewrite_option
    assert len(new_task.task_input_names) == 2
    assert new_task.task_input_names[0] == "test_input_0"
    assert new_task.task_input_names[1] == "test_input_1"

    # Log with version 0.5
    v5_log = (
        """{"i": [["[\\\"matmul_auto_scheduler_test\\\", 512, 512, 512]", """
        f'"{str(tvm.target.Target(target))}"'
        """, [6, 64, 64, 0, 0, 0, 0, 0], "", 1], [[], []]], "r": [[0.1], 0, 0.2, 1], "v": "v0.6"}"""
    )
    measure_log = auto_scheduler.measure_record.load_record_from_string(v5_log)
    new_task = measure_log[0].task
    assert task.workload_key == new_task.workload_key
    assert str(task.target) == str(new_task.target)
    assert str(task.target.host) == str(new_task.target.host)
    assert task.layout_rewrite_option == new_task.layout_rewrite_option
    assert len(new_task.task_input_names) == 0


if __name__ == "__main__":
    test_search_task_add_task_input()
    test_search_task_record()
    test_recover_measure_input_with_task_input()
