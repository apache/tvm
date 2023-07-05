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
import platform
import pytest
import os

from unittest import mock

from os import path
from pathlib import Path

import tvm
import tvm.testing
from tvm import autotvm, auto_scheduler
from tvm.driver import tvmc
from tvm.driver.tvmc.autotuner import filter_tasks, gen_task_list


def _get_tasks(model):
    tvmc_model = tvmc.frontends.load_model(model)
    return tvmc.autotuner.autotvm_get_tuning_tasks(tvmc_model.mod, tvmc_model.params, "llvm")


def _get_measure_options():
    return autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner="local"
    )


def _tuner_test_helper(model, tuner_name, tmpdir_name, early_stopping=1, prior_records=None):
    tvmc_model = tvmc.frontends.load_model(model)
    log_file = os.path.join(tmpdir_name, "log_{}.txt".format(tuner_name))

    tvmc.tune(
        tvmc_model,
        target="llvm",
        tuning_records=log_file,
        prior_records=prior_records,
        tuner=tuner_name,
        trials=4,
        early_stopping=early_stopping,
    )

    # testing whether the log file was produced
    assert path.exists(log_file), "tuning log file should exist"

    with autotvm.apply_history_best(log_file) as best:
        assert isinstance(
            best, autotvm.task.dispatcher.ApplyHistoryBest
        ), "unable to load the best results of tuning"

    return log_file


def test_get_tuning_tasks(onnx_mnist):
    pytest.importorskip("onnx")

    sut = _get_tasks(onnx_mnist)
    expected_task_type = autotvm.task.Task

    assert type(sut) is list
    assert len(sut) > 0
    assert all([type(x) is expected_task_type for x in sut]) is True


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_tune_tasks__tuner__xgb(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "xgb", tmpdir_name)


def test_tune_tasks__tuner__xgb_knob(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "xgb_knob", tmpdir_name)


def test_tune_tasks__tuner__ga(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "ga", tmpdir_name)


def test_tune_tasks__tuner__random(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "random", tmpdir_name)


def test_tune_tasks__tuner__gridsearch(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "gridsearch", tmpdir_name)


def test_tune_tasks__tuner__gridsearch__tuning_records(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    output_log_phase_1 = _tuner_test_helper(onnx_mnist, "gridsearch", tmpdir_name)

    # Exercises transfer learning by making sure a previous log exists
    _tuner_test_helper(onnx_mnist, "gridsearch", tmpdir_name, prior_records=output_log_phase_1)


def test_tune_tasks__tuner__ga__empty_tasks(tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    log_file = os.path.join(tmpdir_name, "log_{}.txt".format("ga"))

    tvmc.autotuner.tune_tasks(
        tasks=[],
        log_file=log_file,
        measure_option=_get_measure_options(),
        tuner="ga",
        trials=1,
        early_stopping=1,
    )


def test_tune_tasks__tuner__xgb__no_early_stopping(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "xgb", tmpdir_name, early_stopping=None)


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_tune_tasks__tuner__xgb__no_tuning_records(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "xgb", tmpdir_name, prior_records=None)


def test_tune_tasks__invalid_tuner(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tasks = _get_tasks(onnx_mnist)
    log_file = os.path.join(tmpdir_factory.mktemp("data"), "log2.txt")

    with pytest.raises(tvmc.TVMCException):
        tvmc.autotuner.tune_tasks(tasks, log_file, _get_measure_options(), "invalid_tuner", 1, 1)


@mock.patch("tvm.driver.tvmc.autotuner.auto_scheduler.HardwareParams", return_value=None)
@mock.patch("tvm.driver.tvmc.autotuner.tune_model", return_value=None)
@mock.patch("tvm.driver.tvmc.frontends.load_model", return_value=None)
def test_tune_rpc_tracker_parsing(mock_load_model, mock_tune_model, mock_auto_scheduler):
    cli_args = mock.MagicMock()
    cli_args.rpc_tracker = "10.0.0.1:9999"
    # FILE is not used but it's set to a valid value here to avoid it being set
    # by mock to a MagicMock class, which won't pass the checks for valid FILE.
    fake_input_file = "./fake_input_file.tflite"
    Path(fake_input_file).touch()
    cli_args.FILE = fake_input_file

    tvmc.autotuner.drive_tune(cli_args)

    os.remove(fake_input_file)

    mock_tune_model.assert_called_once()

    # inspect the mock call, to search for specific arguments
    _, _, kwargs = mock_tune_model.mock_calls[0]
    assert "hostname" in kwargs
    assert "10.0.0.1" == kwargs["hostname"]
    assert "port" in kwargs
    assert 9999 == kwargs["port"]


@mock.patch("tvm.transform.PassContext", return_value=tvm.transform.PassContext())
def test_autotune_pass_context(mock_pc, onnx_mnist, tmpdir_factory):
    """
    Check that the pass context while tuning is as expected.
    """
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "gridsearch", tmpdir_name)

    # AutoTVM overrides the pass context later in the pipeline to disable AlterOpLayout
    assert mock_pc.call_count == 2
    assert mock_pc.call_args_list[0][1]["opt_level"] == 3


def test_filter_tasks_valid():
    filter_tasks(list(range(10)), "list") == ([], True)
    filter_tasks(list(range(10)), "help") == ([], True)
    filter_tasks(list(range(10)), "all") == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], False)
    filter_tasks(list(range(10)), "5") == ([5], False)
    filter_tasks(list(range(10)), "1-5") == ([1, 2, 3, 4, 5], False)
    filter_tasks(list(range(10)), "-5") == ([0, 1, 2, 3, 4, 5], False)
    filter_tasks(list(range(10)), "6-") == ([6, 7, 8, 9], False)
    filter_tasks(list(range(10)), "0,1-3,all") == ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], False)
    filter_tasks(list(range(10)), "0,4-5,9,list") == ([0, 4, 5, 9], True)


@pytest.mark.parametrize(
    "value,err_msg",
    [
        ("10", "Task index out of range"),
        ("5,10", "Task index out of range"),
        ("1-10", "Right-hand side expression out of range"),
        ("-10", "Right-hand side expression out of range"),
        ("-", "Missing lhs or rhs for range expression"),
        ("-10-", "Malformed range expression"),
        ("--", "Malformed range expression"),
    ],
)
def test_filter_tasks_invalid(value, err_msg):
    with pytest.raises(AssertionError, match=err_msg):
        filter_tasks(list(range(10)), value)


@pytest.mark.parametrize(
    "enable_autoscheduler,expected",
    [
        (
            False,
            """Available Tasks for tuning:
  0. Task(func_name=taskA, args=[], kwargs={}, workload=('taskA',)) (len=?)
  1. Task(func_name=taskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBta... (len=?)
  2. Task(func_name=taskC, args=[], kwargs={}, workload=('taskC',)) (len=?)""",
        ),
        (
            True,
            """Available Tasks for tuning:
  0. taskA
  1. taskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBtaskBta...
  2. Unnamed""",
        ),
    ],
)
def test_print_task_list(enable_autoscheduler, expected):
    if enable_autoscheduler:
        auto_scheduler.search_task.TASK_INPUT_BUFFER_TABLE.clear()
        N = 64
        target = "llvm"
        test_input_0 = tvm.runtime.ndarray.empty((64, 64))
        test_input_1 = tvm.runtime.ndarray.empty((10, 20))
        test_input_2 = tvm.runtime.ndarray.empty((30, 40, 50))
        task_inputs = {
            "test_input_0": test_input_0,
            "test_input_1": test_input_1,
            "test_input_2": test_input_2,
        }
        task1 = auto_scheduler.SearchTask(
            func="matmul_auto_scheduler_test",
            args=(N, N, N),
            target=target,
            task_inputs=task_inputs,
            task_inputs_overwrite=True,
            desc="taskA",
        )
        task2 = auto_scheduler.SearchTask(
            func="matmul_auto_scheduler_test",
            args=(N, N, N),
            target=target,
            task_inputs=task_inputs,
            task_inputs_overwrite=True,
            desc="taskB" * 20,  # very long name
        )
        task3 = auto_scheduler.SearchTask(
            func="matmul_auto_scheduler_test",
            args=(N, N, N),
            target=target,
            task_inputs=task_inputs,
            task_inputs_overwrite=True,
            # missing description
        )
    else:
        task1 = autotvm.task.Task("taskA", [])
        task2 = autotvm.task.Task("taskB" * 20, [])  # very long name
        task3 = autotvm.task.Task("taskC", [])
    tasks = [task1, task2, task3]
    out = gen_task_list(tasks, enable_autoscheduler)
    assert out == expected


if __name__ == "__main__":
    tvm.testing.main()
