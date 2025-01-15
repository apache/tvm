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
import os
import time

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.builder import Builder
from tvm.script import tir as T
from tvm.target import Target


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


def parallel_builder_run(inputs):
    builder = Builder.create("local", max_workers=1)
    builder_results = builder.build(inputs)
    return builder_results


def test_buider():
    target = Target("llvm -mcpu=icelake-server -num-cores 28")
    t1 = time.time()
    mod = tvm.IRModule({"main": matmul})
    builder_inputs = [ms.builder.BuilderInput(mod, target)]
    builder = Builder.create("local", max_workers=os.cpu_count() * 2)

    from multiprocessing import Pool

    runner_inputs_2d = list(map(lambda x: [x], builder_inputs * 100))
    t1 = time.time()
    with Pool(8) as pool:
        pool.map(parallel_builder_run, runner_inputs_2d)
    t2 = time.time()

    print("[INFO] Parallel batch build time(s):", t2 - t1)

    builder_inputs_list = builder_inputs * 100
    num_workers = min(os.cpu_count(), len(builder_inputs * 100))
    batch_size = len(builder_inputs * 100) // num_workers  # 平均分配任务

    t1 = time.time()
    with Pool(num_workers) as pool:
        builder_results = pool.map(
            parallel_builder_run,
            [
                builder_inputs_list[i: i + batch_size]
                for i in range(0, len(builder_inputs_list), batch_size)
            ],
        )
    t2 = time.time()

    print("[INFO] Parallel batch build time(s):", t2 - t1)

    t1 = time.time()
    builder.build(builder_inputs * 100)
    t2 = time.time()
    print("[INFO]build time(s): ", t2 - t1)

    t1 = time.time()
    builder.build(builder_inputs)
    t2 = time.time()
    print("[INFO]single build time(s): ", t2 - t1)

    n = len(builder_inputs_list) // 20  # 每行的元素个数
    reshaped_list = [
        builder_inputs_list[i * n: (i + 1) * n] for i in range(5)
    ]
    t1 = time.time()
    from concurrent.futures import ThreadPoolExecutor

    builder_results = []
    with ThreadPoolExecutor(max_workers=len(reshaped_list)) as executor:
        futures = [
            executor.submit(builder.build, batch) for batch in reshaped_list
        ]
        for future in futures:
            records = future.result()
            builder_results.extend(records)
    t2 = time.time()
    print("[INFO]build time(s): ", t2 - t1)


if __name__ == "__main__":
    test_buider()
