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

import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm import MeasureInput, MeasureResult
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel

from test_autotvm_common import get_sample_task, get_sample_records


def test_fit():
    task, target = get_sample_task()
    records = get_sample_records(n=500)

    base_model = XGBoostCostModel(task, feature_type='itervar', loss_type='rank')
    base_model.fit_log(records, plan_size=32)

    upper_model = XGBoostCostModel(task, feature_type='itervar', loss_type='rank')
    upper_model.load_basemodel(base_model)

    xs = np.arange(10)
    ys = np.arange(10)

    upper_model.fit(xs, ys, plan_size=32)


def test_tuner():
    task, target = get_sample_task()
    records = get_sample_records(n=100)

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.load_history(records)


if __name__ == "__main__":
    test_fit()
    test_tuner()

