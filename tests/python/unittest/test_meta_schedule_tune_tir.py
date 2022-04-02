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
# pylint: disable=missing-docstring
import logging
import tempfile

import pytest
import tvm
from tvm.meta_schedule import ReplayTraceConfig, schedule_rule, tune_tir
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.meta_schedule.testing import te_workload
from tvm.script import tir as T
from tvm.target.target import Target
from tvm.te.operation import create_prim_func
from tvm.tir import Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


# pylint: enable=no-member,invalid-name,unused-variable


@pytest.mark.skip("Integration test")
def test_tune_matmul_cpu():
    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_tir(
            mod=matmul,
            target=Target("llvm --num-cores=16"),
            config=ReplayTraceConfig(
                num_trials_per_iter=32,
                max_trials_per_task=32,
                max_trials_global=32,
            ),
            work_dir=work_dir,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)


@pytest.mark.skip("Integration test")
def test_tune_matmul_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_tir(
            mod=matmul,
            target=Target("nvidia/geforce-rtx-3070"),
            config=ReplayTraceConfig(
                num_trials_per_iter=32,
                max_trials_per_task=32,
                max_trials_global=32,
            ),
            work_dir=work_dir,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)


@pytest.mark.skip("Integeration test")
def test_tune_matmul_cuda_tensor_core():
    n = 512
    mod = create_prim_func(te_workload.matmul_fp16(n, n, n))
    target = Target("nvidia/geforce-rtx-3070")
    config = ReplayTraceConfig(
        num_trials_per_iter=32,
        max_trials_per_task=320,
        max_trials_global=320,
    )

    class DefaultTensorCore:
        @staticmethod
        def _sch_rules():
            from tvm.meta_schedule import (
                schedule_rule as M,  # pylint: disable=import-outside-toplevel
            )

            return [
                M.AutoInline(
                    into_producer=False,
                    into_consumer=True,
                    inline_const_tensor=True,
                    disallow_if_then_else=False,
                    require_injective=False,
                    require_ordered=False,
                    disallow_op=None,
                ),
                M.MultiLevelTiling(
                    structure="SSSRRSRS",
                    tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
                    # use_tensor_core=True,
                    max_innermost_factor=64,
                    vector_load_lens=[1, 2, 3, 4],
                    reuse_read=schedule_rule.ReuseType(
                        req="must",
                        levels=[4],
                        scope="shared",
                    ),
                    reuse_write=schedule_rule.ReuseType(
                        req="no",
                        levels=[],
                        scope="",
                    ),
                ),
                M.AutoInline(
                    into_producer=True,
                    into_consumer=True,
                    inline_const_tensor=True,
                    disallow_if_then_else=False,
                    require_injective=False,
                    require_ordered=False,
                    disallow_op=None,
                ),
                M.ParallelizeVectorizeUnroll(
                    max_jobs_per_core=-1,  # disable parallelize
                    max_vectorize_extent=-1,  # disable vectorize
                    unroll_max_steps=[0, 16, 64, 512, 1024],
                    unroll_explicit=True,
                ),
            ]

        @staticmethod
        def _postproc():
            from tvm.meta_schedule import (
                postproc as M,  # pylint: disable=import-outside-toplevel
            )

            return [
                M.RewriteCooperativeFetch(),
                M.RewriteParallelVectorizeUnroll(),
                M.RewriteReductionBlock(),
                M.RewriteTensorCore(),
                M.VerifyGPUCode(),
            ]

    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_tir(
            mod=mod,
            target=target,
            config=config,
            work_dir=work_dir,
            space=PostOrderApply(),
            sch_rules=DefaultTensorCore._sch_rules,
            postprocs=DefaultTensorCore._postproc,
            num_threads=None,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)

            import numpy as np
            from tvm.contrib import nvcc

            ctx = tvm.gpu(0)
            if nvcc.have_tensorcore(ctx.compute_version):
                with tvm.transform.PassContext():
                    func = tvm.build(sch.mod["main"], [], "cuda")
                    print(sch.mod.script())
                    print(func.imported_modules[0].get_source())
                a_np = np.random.uniform(size=(n, n)).astype("float16")
                b_np = np.random.uniform(size=(n, n)).astype("float16")
                a = tvm.nd.array(a_np, ctx)
                b = tvm.nd.array(b_np, ctx)
                c = tvm.nd.array(np.zeros((n, n), dtype="float32"), ctx)
                evaluator = func.time_evaluator(
                    func.entry_name, ctx, number=3, repeat=1, min_repeat_ms=40
                )
                print("matmul with tensor core: %f ms" % (evaluator(a, b, c).mean * 1e3))

                np.testing.assert_allclose(
                    c.asnumpy(),
                    np.matmul(a_np.astype("float32"), b_np.astype("float32")),
                    rtol=1e-4,
                    atol=1e-4,
                )


if __name__ == """__main__""":
    test_tune_matmul_cpu()
    test_tune_matmul_cuda()
    test_tune_matmul_cuda_tensor_core()
