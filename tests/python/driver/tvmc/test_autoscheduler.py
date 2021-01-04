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
import json
import pytest
import os
import tarfile

from os import path

from tvm import auto_scheduler
from tvm.driver import tvmc


def _get_tasks(model):
    mod, params = tvmc.frontends.load_model(model)
    tasks, weights = tvmc.autotuner.autoscheduler_get_tuning_tasks(mod, params, "llvm")
    return (tasks, weights)


def _autoscheduler_test_helper(
    model, tmpdir_name, tasks_weights=None, early_stopping=1, tuning_records=None
):
    tasks, weights = tasks_weights if tasks_weights else _get_tasks(model)
    log_file = os.path.join(tmpdir_name, "autoscheduler.json")

    tuning_options = auto_scheduler.TuningOptions(
        num_measure_trials=1,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        runner="local",
        builder="local",
        verbose=0,
        early_stopping=early_stopping,
    )

    tvmc.autotuner.schedule_tasks(tasks[:1], weights[:1], tuning_options, tuning_records)

    # testing whether the log file was produced
    assert path.exists(log_file), "autoscheduler log file should exist"

    with auto_scheduler.ApplyHistoryBest(log_file) as best:
        assert isinstance(
            best, auto_scheduler.dispatcher.ApplyHistoryBest
        ), "unable to load the best results of tuning"

    return log_file


def test_get_tuning_tasks(onnx_resnet50):
    pytest.importorskip("onnx")

    tasks, weights = _get_tasks(onnx_resnet50)
    expected_task_type = auto_scheduler.SearchTask

    assert type(tasks) is list
    assert len(tasks) > 0
    assert all([type(x) is expected_task_type for x in tasks]) is True


def test_tune_tasks(onnx_resnet50, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _autoscheduler_test_helper(onnx_resnet50, tmpdir_name)


def test_tune_tasks__tuning_records(onnx_resnet50, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    output_log_phase_1 = _autoscheduler_test_helper(onnx_resnet50, tmpdir_name)

    # Exercises transfer learning by making sure a previous log exists
    _autoscheduler_test_helper(onnx_resnet50, tmpdir_name, tuning_records=output_log_phase_1)


def test_tune_tasks__no_early_stopping(onnx_resnet50, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _autoscheduler_test_helper(onnx_resnet50, tmpdir_name, tasks_weights=None, early_stopping=None)


def test_tune_tasks__no_tuning_records(onnx_resnet50, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _autoscheduler_test_helper(onnx_resnet50, tmpdir_name, tasks_weights=None, tuning_records=None)
