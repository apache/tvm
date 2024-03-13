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
import time

import multiprocessing
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm import MeasureInput, MeasureResult
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel

from tvm.testing.autotvm import get_sample_task, get_sample_records


def test_fit():
    task, target = get_sample_task()
    records = get_sample_records(n=500)

    base_model = XGBoostCostModel(task, feature_type="itervar", loss_type="reg")
    base_model.fit_log(records, plan_size=32)

    upper_model = XGBoostCostModel(task, feature_type="itervar", loss_type="reg")
    upper_model.load_basemodel(base_model)

    xs = np.arange(10)
    ys = np.arange(10)

    upper_model.fit(xs, ys, plan_size=32)

    # feature lengths are not guaranteed to always be the same
    upper_model.predict(np.ones(12))
    upper_model.predict(np.ones(8))


def fit_spawn():
    assert multiprocessing.get_start_method(False) == "spawn"
    test_fit()


def test_fit_spawn():
    # Subprocesses inherit the spawn method of their parents
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=test_fit)
    p.start()
    p.join()


def test_tuner():
    task, target = get_sample_task()
    records = get_sample_records(n=10)

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.load_history(records, min_seed_records=10)
    # Confirm that loading history successfully loaded a
    # base_model.
    assert tuner.cost_model.base_model is not None

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.load_history(records, min_seed_records=11)
    # Confirm that loading history did not load base_model
    # when not enough records according to `min_seed_records`
    # are provided
    assert tuner.cost_model.base_model is None


def test_update():
    task, target = get_sample_task()
    tuner = autotvm.tuner.XGBTuner(task)
    n_records = 5
    records = get_sample_records(n=n_records)
    tuner.update([inp for inp, _ in records], [res for _, res in records])
    assert len(tuner.xs) == n_records
    assert len(tuner.ys) == n_records
    assert len(tuner.visited) == n_records
    assert all(x in tuner.visited for x in tuner.xs)


if __name__ == "__main__":
    test_fit()
    test_fit_spawn()
    test_tuner()
    test_update()
