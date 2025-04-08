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
"""Test task scheduler"""

import tempfile
import time

import numpy as np

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.tune_context import _normalize_mod
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (1024, 1024), "float32")
    B = T.match_buffer(b, (1024, 1024), "float32")
    C = T.match_buffer(c, (1024, 1024), "float32")
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0  # type: ignore
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_dynamic_gradient_descent():
    slide_window_size = (
        10  # Size of the sliding window used in dynamic gradient search
    )
    max_tuning_time = (
        120  # Maximum tuning time in seconds, 120 is the suggested value
    )
    max_trials = 1000  # Maximum number of measurement trials to perform in dynamic gradient search, use 1000 to get better performance
    n_start = 5  # Number of start points from the initial sampled population
    # Number of samples to generate the initial model, 64 is the suggested
    # value
    init_population_size = 64
    predict_score_threshold_ratio = 0.6  # Threshold for the predict score
    measure_threshold_ratio = 0.6  # Threshold for the measured throughput
    rules = ms.ScheduleRule.create("llvm")
    space_gen = ms.space_generator.PostOrderApply(sch_rules=rules)
    # Initialize tuner
    start_time = time.time()
    with tempfile.TemporaryDirectory() as work_dir:
        tuner = ms.dynamic_gradient_search.DynamicGradientSearchTuner(
            matmul,
            n_start,
            init_population_size,
            slide_window_size,
            max_trials,
            max_tuning_time,
            predict_score_threshold_ratio,
            measure_threshold_ratio,
            space=space_gen,
            target=Target("llvm -mcpu=icelake-server -num-cores 28"),
            task_name="matmul",
            tmpdir=work_dir,
        )

        # Run tuner
        database = tuner.dynamic_gradient_search()
        record = database.query_tuning_record(
            _normalize_mod(matmul),
            Target("llvm -mcpu=icelake-server -num-cores 28"),
            workload_name="main",
        )
        sch = Schedule(record.workload.mod)
        record.trace.apply_to_schedule(sch, remove_postproc=False)
    end_time = time.time()
    search_time = end_time - start_time
    search_time /= 60
    print(f"Total search time: {search_time} minutes", flush=True)
    myfunc = tvm.build(
        sch.mod,
        target=Target("llvm -mcpu=icelake-server -num-cores 28"),
        name="matmul",
    )
    dev = tvm.device("cpu", 0)
    a_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    b_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    c_np = a_np.dot(b_np)
    buff_a = tvm.nd.array(a_np, dev)
    buff_b = tvm.nd.array(b_np, dev)
    buff_c = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
    myfunc(buff_a, buff_b, buff_c)
    tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)
    print("[INFO]*********success!")


if __name__ == "__main__":
    test_dynamic_gradient_descent()
