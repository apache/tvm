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
from tvm.meta_schedule.integration import (
    ExtractedTask,
    MetaScheduleContext,
    TaskExtraction,
)
from tvm.meta_schedule.testing import get_network
from tvm.script import tir as T

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring,unbalanced-tuple-unpacking


@tvm.script.ir_module
class MockModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        for i in T.serial(0, 16):
            with T.block("matmul"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring,unbalanced-tuple-unpacking


def _check_mock_task(tasks: List[ExtractedTask], mod: IRModule):
    (task,) = tasks
    assert isinstance(task, ExtractedTask)
    assert task.task_name == "mock-task"
    tvm.ir.assert_structural_equal(task.mod, mod)
    (tir_mod,) = task.dispatched
    tvm.ir.assert_structural_equal(tir_mod, MockModule)


def test_meta_schedule_integration_task_extraction_query():
    mod, _, _, _ = get_network(
        name="resnet-18",
        batch_size=1,
        layout="NHWC",
        dtype="float32",
    )
    env = TaskExtraction()
    env.query(task_name="mock-task", mod=mod, dispatched=[MockModule])
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


def test_meta_schedule_integration_query_inside_with_scope():
    mod, _, _, _ = get_network(
        name="resnet-18",
        batch_size=1,
        layout="NHWC",
        dtype="float32",
    )
    env = TaskExtraction()
    with env:
        MetaScheduleContext.query_inside_with_scope(
            task_name="mock-task",
            mod=mod,
            dispatched=[MockModule],
        )
    _check_mock_task(env.tasks, mod)


def test_meta_schedule_integration_extract_from_resnet():
    mod, params, _, _ = get_network(
        name="resnet-18",
        batch_size=1,
        layout="NHWC",
        dtype="float32",
    )
    extracted_tasks = ms.integration.extract_task(mod, target="llvm", params=params)
    assert len(extracted_tasks) == 30


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
