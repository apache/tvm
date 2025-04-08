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
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN


@T.prim_func
def softmax(
    A: T.Buffer((64, 1280), "float32"),
    T_softmax_norm: T.Buffer((64, 1280), "float32"),
) -> None:
    T_softmax_maxelem = T.alloc_buffer([64], dtype="float32", scope="local")
    T_softmax_exp = T.alloc_buffer([64, 1280], dtype="float32", scope="local")
    T_softmax_expsum = T.alloc_buffer([64], dtype="float32", scope="local")
    for i0, i1 in T.grid(64, 1280):
        with T.block("T_softmax_maxelem"):
            i0_1, k = T.axis.remap("SR", [i0, i1])
            with T.init():
                T_softmax_maxelem[i0_1] = T.min_value("float32")
            T_softmax_maxelem[i0_1] = T.max(
                T_softmax_maxelem[i0_1], A[i0_1, k]
            )
    for i0, i1 in T.grid(64, 1280):
        with T.block("T_softmax_exp"):
            i0_2, i1_1 = T.axis.remap("SS", [i0, i1])
            T_softmax_exp[i0_2, i1_1] = T.exp(
                A[i0_2, i1_1] - T_softmax_maxelem[i0_2], dtype="float32"
            )
    for i0_3, i1 in T.grid(64, 1280):
        with T.block("T_softmax_expsum"):
            i0_4, k = T.axis.remap("SR", [i0_3, i1])
            with T.init():
                T_softmax_expsum[i0_4] = T.float32(0)
            T_softmax_expsum[i0_4] = (
                T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
            )
    for i0_5, i1 in T.grid(64, 1280):
        with T.block("T_softmax_norm"):
            i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
            T_softmax_norm[i0_6, i1_2] = (
                T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]
            )


def ref_program(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


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

postprocs = [
    ms.postproc.DisallowDynamicLoop(),
    ms.postproc.RewriteParallelVectorizeUnroll(),
    ms.postproc.RewriteReductionBlock(),
    ms.postproc.RewriteTensorize(vectorize_init_loop=True),
    ms.postproc.RewriteLayout(),
]


def test_dynamic_gradient_descent():
    start_time = time.time()
    sch_rules = ms.space_generator.PostOrderApply(sch_rules=ActionSpace_CPU)
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm --num-cores=28")
        database = ms.tune_tir(
            mod=softmax,
            target=target,
            max_trials_global=1000,
            num_trials_per_iter=64,
            work_dir=work_dir,
            space=ms.space_generator.PostOrderApply(
                f_block_filter=None,
                sch_rules=sch_rules,
                postprocs=postprocs,
                mutator_probs={},
            ),
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
            _normalize_mod(softmax),
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
            name="softmax",
        )
        dev = tvm.device("cpu", 0)
        a_np = np.random.uniform(size=(64, 1280)).astype("float32")
        c_np = ref_program(a_np)
        buff_a = tvm.nd.array(a_np, dev)
        buff_c = tvm.nd.array(np.zeros((64, 1280), dtype="float32"), dev)
        myfunc(buff_a, buff_c)
        tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)
        evaluator = myfunc.time_evaluator(myfunc.entry_name, dev, number=100)
        time_ms = evaluator(buff_a, buff_c).mean * 1e3
        print(f"[INFO]******time_ms: {time_ms} ms")
        print("[INFO]*********success!")
    else:
        print("[INFO]*************Failed!")


if __name__ == "__main__":
    test_dynamic_gradient_descent()
