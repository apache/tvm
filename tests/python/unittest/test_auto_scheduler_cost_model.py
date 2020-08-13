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

"""Test cost models"""

import numpy as np

import tvm
from tvm import auto_scheduler

from test_auto_scheduler_common import matmul_auto_scheduler_test


def get_sample_records(number):
    """Generate random a list of random MeasureInput and MeasureResult pairs"""
    N = 128
    workload_key = auto_scheduler.make_workload_key(matmul_auto_scheduler_test, (N, N, N))
    dag = auto_scheduler.ComputeDAG(workload_key)
    target = tvm.target.create('llvm')
    task = auto_scheduler.SearchTask(dag, workload_key, target)
    policy = auto_scheduler.SketchPolicy(task, verbose=0)
    states = policy.sample_initial_population(number)

    inputs = [auto_scheduler.MeasureInput(task, s) for s in states]
    results = [auto_scheduler.MeasureResult([np.random.uniform(0.1, 0.2)], 0, "", 0.1, 0)
               for _ in range(len(inputs))]

    return task, dag, inputs, results


def test_random_model():
    task, dag, inputs, results = get_sample_records(100)

    model = auto_scheduler.RandomModel()
    model.update(inputs, results)
    scores = model.predict(task, [x.state for x in inputs])
    assert len(scores) == len(inputs)


def test_xgb_model():
    task, dag, inputs, results = get_sample_records(100)

    model = auto_scheduler.XGBModel()
    model.update(inputs, results)
    scores = model.predict(task, [x.state for x in inputs])
    assert len(scores) == len(inputs)


if __name__ == "__main__":
    test_random_model()
    test_xgb_model()
