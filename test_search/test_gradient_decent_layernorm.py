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
import logging
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
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN

logging.basicConfig(level=logging.INFO)


@T.prim_func
def layernorm(
    input: T.handle, gamma: T.handle, beta: T.handle, output: T.handle
) -> None:
    input_ = T.match_buffer(input, [64, 100, 4096], dtype="float32")
    gamma_ = T.match_buffer(gamma, [4096], dtype="float32")
    beta_ = T.match_buffer(beta, [4096], dtype="float32")
    output_ = T.match_buffer(output, [64, 100, 4096], dtype="float32")
    input_sum = T.alloc_buffer([64, 100], dtype="float32")
    input_mean = T.alloc_buffer([64, 100], dtype="float32")

    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("input_sum"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSR", [ib, ir, ip])
            with T.init():
                input_sum[ib_0, ir_0] = T.float32(0)

            input_sum[ib_0, ir_0] = (
                input_sum[ib_0, ir_0] + input_[ib_0, ir_0, ip_0]
            )

    for ib, ir in T.grid(64, 100):
        with T.block("input_norm"):
            ib_0, ir_0 = T.axis.remap("SS", [ib, ir])
            input_mean[ib_0, ir_0] = input_sum[ib_0, ir_0] / T.float32(4096)

    input_diff = T.alloc_buffer([64, 100, 4096])
    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("input_diff"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            input_diff[ib_0, ir_0, ip_0] = (
                input_[ib_0, ir_0, ip_0] - input_mean[ib_0, ir_0]
            )

    input_variance = T.alloc_buffer([64, 100], dtype="float32")

    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("input_variance"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSR", [ib, ir, ip])
            with T.init():
                input_variance[ib_0, ir_0] = T.float32(0)
            input_variance[ib_0, ir_0] = (
                input_variance[ib_0, ir_0] + input_diff[ib_0, ir_0, ip_0]
            )

    variance_norm = T.alloc_buffer([64, 100], dtype="float32")
    for ib, ir in T.grid(64, 100):
        with T.block("variance_norm"):
            ib_0, ir_0 = T.axis.remap("SS", [ib, ir])
            variance_norm[ib_0, ir_0] = input_variance[ib_0, ir_0] / 4096

    variance_sqrt = T.alloc_buffer([64, 100], dtype="float32")
    for ib, ir in T.grid(64, 100):
        with T.block("variance_sqrt"):
            ib_0, ir_0 = T.axis.remap("SS", [ib, ir])
            variance_sqrt[ib_0, ir_0] = T.sqrt(variance_norm[ib_0, ir_0])

    diff_input = T.alloc_buffer([64, 100, 4096], dtype="float32")
    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("diff_input"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            diff_input[ib_0, ir_0, ip_0] = (
                input_[ib_0, ir_0, ip_0] - input_mean[ib_0, ir_0]
            )

    diff_gamma = T.alloc_buffer([64, 100, 4096], dtype="float32")
    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("diff_gamma"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            diff_gamma[ib_0, ir_0, ip_0] = (
                input_diff[ib_0, ir_0, ip_0] * gamma_[ip_0]
            )

    diff_div = T.alloc_buffer([64, 100, 4096], dtype="float32")
    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("diff_div"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            diff_div[ib_0, ir_0, ip_0] = diff_gamma[ib_0, ir_0, ip_0] / (
                variance_sqrt[ib_0, ir_0] + T.float32(1e-5)
            )

    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("output"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            output_[ib_0, ir_0, ip_0] = (
                diff_div[ib_0, ir_0, ip_0] + beta_[ip_0]
            )


ActionSpace_CPU = [
    ms.schedule_rule.AutoInline(
        into_producer=False,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=True,
        require_injective=True,
        require_ordered=True,
        disallow_op=["tir.exp"],
    ),
    ms.schedule_rule.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
    ms.schedule_rule.MultiLevelTilingWithIntrin(
        VNNI_INTRIN,
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=ms.schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="global",
        ),
    ),
    ms.schedule_rule.MultiLevelTiling(
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=ms.schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="global",
        ),
    ),
    ms.schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=16,
        max_vectorize_extent=512,
        unroll_max_steps=[0, 16, 64, 512],
        unroll_explicit=True,
    ),
    ms.schedule_rule.RandomComputeLocation(),
    ms.schedule_rule.InlineConstantScalars(),
]


def ref_program(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / (std + eps)
    out = gamma * x_normalized + beta
    return out


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
    init_population_size = 16
    predict_score_threshold_ratio = 0.6  # Threshold for the predict score
    measure_threshold_ratio = 0.6  # Threshold for the measured throughput
    # Initialize tuner

    # postprocs = [
    #     ms.postproc.DisallowDynamicLoop(),
    #     ms.postproc.RewriteParallelVectorizeUnroll(),
    #     ms.postproc.RewriteReductionBlock(),
    #     ms.postproc.RewriteTensorize(vectorize_init_loop=True),
    #     ms.postproc.RewriteLayout(),
    # ]
    # postprocs = None

    space_gen = ms.space_generator.PostOrderApply(
        sch_rules=ActionSpace_CPU[:1]
    )
    start_time = time.time()
    with tempfile.TemporaryDirectory() as work_dir:
        tuner = ms.dynamic_gradient_search.DynamicGradientSearchTuner(
            layernorm,
            n_start,
            init_population_size,
            slide_window_size,
            max_trials,
            max_tuning_time,
            predict_score_threshold_ratio,
            measure_threshold_ratio,
            space=space_gen,
            target=Target("llvm -mcpu=icelake-server -num-cores 28"),
            task_name="layernorm",
            tmpdir=work_dir,
        )

        # Run tuner
        database = tuner.dynamic_gradient_search()
        record = database.query_tuning_record(
            _normalize_mod(layernorm),
            Target("llvm -mcpu=icelake-server -num-cores 28"),
            workload_name="main",
        )
    end_time = time.time()
    search_time = end_time - start_time
    search_time /= 60
    print(f"Total search time: {search_time} minutes", flush=True)
    if record is not None:
        sch = Schedule(record.workload.mod)
        record.trace.apply_to_schedule(sch, remove_postproc=False)

        myfunc = tvm.build(
            sch.mod,
            target=Target("llvm -mcpu=icelake-server -num-cores 28"),
            name="layernorm",
        )
        dev = tvm.device("cpu", 0)
        dtype = "float32"
        input_array = np.random.uniform(size=[64, 100, 4096]).astype(dtype)
        gamma_array = np.random.uniform(size=[4096]).astype(dtype)
        beta_array = np.random.uniform(size=[4096]).astype(dtype)
        output_array = np.random.uniform(size=[64, 100, 4096]).astype(dtype)
        ref_program(input_array, gamma_array, beta_array)
        buff_a = tvm.nd.array(input_array, dev)
        buff_b = tvm.nd.array(gamma_array, dev)
        buff_c = tvm.nd.array(beta_array, dev)
        buff_d = tvm.nd.array(output_array, dev)

        myfunc(buff_a, buff_b, buff_c, buff_d)
        # tvm.testing.assert_allclose(buff_d.numpy(), output, rtol=1e-3)
        print("[INFO]*********success!")
    else:
        print("[INFO]*************Failed!")


def test_meta_schedule():
    start_time = time.time()
    # rules = [ms.schedule_rule.RandomComputeLocation()]
    # postprocs = [
    #     ms.postproc.DisallowDynamicLoop(),
    #     ms.postproc.RewriteParallelVectorizeUnroll(),
    #     ms.postproc.RewriteReductionBlock(),
    #     ms.postproc.RewriteTensorize(vectorize_init_loop=True),
    #     ms.postproc.RewriteLayout(),
    # ]
    space_gen = ms.space_generator.PostOrderApply(
        sch_rules=ActionSpace_CPU[:1]
    )
    # print("[IFNO]**********rule: ", rule)

    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm -mcpu=icelake-server -num-cores 28")
        database = ms.tune_tir(
            mod=layernorm,
            target=target,
            max_trials_global=1000,
            num_trials_per_iter=64,
            work_dir=work_dir,
            space=space_gen,
            runner=ms.runner.LocalRunner(
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=1,
                    repeat=10,
                    min_repeat_ms=10,
                )
            ),
            cost_model=ms.cost_model.XGBModel(
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=True,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
        )
        record = database.query_tuning_record(
            _normalize_mod(layernorm),
            Target("llvm -mcpu=icelake-server -num-cores 28"),
            workload_name="main",
        )
    end_time = time.time()
    search_time = end_time - start_time
    search_time /= 60
    print(f"Total search time: {search_time} minutes", flush=True)
    if record is not None:
        sch = Schedule(record.workload.mod)
        record.trace.apply_to_schedule(sch, remove_postproc=False)

        myfunc = tvm.build(
            sch.mod,
            target=Target("llvm -mcpu=icelake-server -num-cores 28"),
            name="layernorm",
        )
        dev = tvm.device("cpu", 0)
        dtype = "float32"
        input_array = np.random.uniform(size=[64, 100, 4096]).astype(dtype)
        gamma_array = np.random.uniform(size=[4096]).astype(dtype)
        beta_array = np.random.uniform(size=[4096]).astype(dtype)
        output_array = np.random.uniform(size=[64, 100, 4096]).astype(dtype)
        ref_program(input_array, gamma_array, beta_array)
        buff_a = tvm.nd.array(input_array, dev)
        buff_b = tvm.nd.array(gamma_array, dev)
        buff_c = tvm.nd.array(beta_array, dev)
        buff_d = tvm.nd.array(output_array, dev)

        myfunc(buff_a, buff_b, buff_c, buff_d)

        evaluator = myfunc.time_evaluator(myfunc.entry_name, dev, number=100)
        time_ms = evaluator(buff_a, buff_b, buff_c, buff_d).mean * 1e3
        print(f"[INFO]******time_ms: {time_ms} ms")
        print("[INFO]*********success!")
    else:
        print("[INFO]*************Failed!")


if __name__ == "__main__":
    test_dynamic_gradient_descent()
