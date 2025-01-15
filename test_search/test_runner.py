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
""" Test Meta Schedule Runner """

import os
import time

import tvm
import tvm.testing
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.runner import Runner, RunnerInput
from tvm.script import tir as T
from tvm.target import Target

MATMUL_N = 16
MATMUL_M = 32

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring,unbalanced-tuple-unpacking


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        a: T.handle, b: T.handle, c: T.handle
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        C = T.match_buffer(c, (16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def parallel_runner_run(input_group):
    runner = Runner.create("local", max_workers=os.cpu_count())
    return runner.run(input_group)


def test_meta_schedule_rpc_single_run():
    """Test meta schedule rpc runner for a single run"""
    # Build the module
    mod = MatmulModule
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
    )
    runner_inputs_2d = list(map(lambda x: [x], [runner_input] * 40))

    from multiprocessing import Pool

    t1 = time.time()
    with Pool(60) as pool:
        pool.map(parallel_runner_run, runner_inputs_2d)
    t2 = time.time()
    print(f"[INFO] Execution Time: {t2 - t1} seconds")
    runner = Runner.create("local", max_workers=os.cpu_count())
    runner_inputs = [runner_input] * 40
    t1 = time.time()
    runner.run(runner_inputs)
    t2 = time.time()
    print(f"[INFO] Execution Time: {t2 - t1} seconds")


if __name__ == "__main__":
    test_meta_schedule_rpc_single_run()
