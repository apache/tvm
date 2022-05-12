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

from os import path

from tvm import auto_scheduler
from tvm.driver import tvmc


def _get_tasks(model):
    tvmc_model = tvmc.frontends.load_model(model)
    tasks, weights = tvmc.autotuner.autoscheduler_get_tuning_tasks(
        tvmc_model.mod, tvmc_model.params, "llvm"
    )
    return (tasks, weights)


def _autoscheduler_test_helper(model, tmpdir_name, early_stopping=1, prior_records=None):
    tvmc_model = tvmc.frontends.load_model(model)
    log_file = os.path.join(tmpdir_name, "autoscheduler.json")

    hardware_params = auto_scheduler.HardwareParams(num_cores=4, target="llvm")

    tvmc.tune(
        tvmc_model,
        target="llvm",
        tuning_records=log_file,
        prior_records=prior_records,
        early_stopping=early_stopping,
        enable_autoscheduler=True,
        trials=2,
        hardware_params=hardware_params,
    )

    # testing whether the log file was produced
    assert path.exists(log_file), "autoscheduler log file should exist"

    with auto_scheduler.ApplyHistoryBest(log_file) as best:
        assert isinstance(
            best, auto_scheduler.dispatcher.ApplyHistoryBest
        ), "unable to load the best results of tuning"

    return log_file


def test_get_tuning_tasks(keras_simple):
    pytest.importorskip("tensorflow")

    tasks, weights = _get_tasks(keras_simple)
    expected_task_type = auto_scheduler.SearchTask

    assert type(tasks) is list
    assert len(tasks) > 0
    assert all([type(x) is expected_task_type for x in tasks]) is True


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_tune_tasks(keras_simple, tmpdir_factory):
    pytest.importorskip("tensorflow")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _autoscheduler_test_helper(keras_simple, tmpdir_name)


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_tune_tasks__tuning_records(keras_simple, tmpdir_factory):
    pytest.importorskip("tensorflow")

    tmpdir_name = tmpdir_factory.mktemp("data")
    output_log_phase_1 = _autoscheduler_test_helper(keras_simple, tmpdir_name)

    # Exercises transfer learning by making sure a previous log exists
    _autoscheduler_test_helper(keras_simple, tmpdir_name, prior_records=output_log_phase_1)


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_tune_tasks__no_early_stopping(keras_simple, tmpdir_factory):
    pytest.importorskip("tensorflow")

    tmpdir_name = tmpdir_factory.mktemp("data")
    _autoscheduler_test_helper(keras_simple, tmpdir_name, early_stopping=None)
