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
import pytest
import os

from os import path

from tvm import autotvm
from tvm.driver import tvmc


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


def test_tune_tasks__tuner__xgb__no_tuning_records(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _tuner_test_helper(onnx_mnist, "xgb", tmpdir_name, prior_records=None)


def test_tune_tasks__invalid_tuner(onnx_mnist, tmpdir_factory):
    pytest.importorskip("onnx")

    tasks = _get_tasks(onnx_mnist)
    log_file = os.path.join(tmpdir_factory.mktemp("data"), "log2.txt")

    with pytest.raises(tvmc.common.TVMCException):
        tvmc.autotuner.tune_tasks(tasks, log_file, _get_measure_options(), "invalid_tuner", 1, 1)
