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

"""Test the task scheduler """

import tvm
from tvm import ansor

from test_ansor_common import matmul_ansor_test

def test_task_scheduler_basic():
    N = 128
    A, B, C = matmul_ansor_test(N, N, N)
    dag = ansor.ComputeDAG([A, B, C])
    tgt = tvm.target.create("llvm")
    task1 = ansor.SearchTask(dag, "test", tgt)
    task2 = ansor.SearchTask(dag, "test", tgt)

    def objective(costs):
        return sum(costs)

    task_scheduler = ansor.SimpleTaskScheduler([task1, task2], objective)
    tune_option = ansor.TuneOption(n_trials=3, runner='local')

    task_scheduler.tune(tune_option)


if __name__ == "__main__":
    test_task_scheduler_basic()
