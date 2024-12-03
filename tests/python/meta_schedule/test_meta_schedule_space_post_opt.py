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
# pylint: disable=missing-docstring,no-member,invalid-name,unused-variable
import logging
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.runner.config import EvaluatorConfig
from tvm.script import tir as T
from tvm.target import Target

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


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


@pytest.mark.skip("Integration test")
@tvm.testing.requires_llvm
def test_tune_matmul_cpu():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm --num-cores=16")
        database = ms.tir_integration.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
            post_optimization=True,
        )
        trials = 32
        database = ms.tune_tir(
            mod=matmul,
            target=target,
            max_trials_global=trials,
            num_trials_per_iter=64,
            work_dir=work_dir,
            runner=ms.runner.LocalRunner(
                evaluator_config=EvaluatorConfig(
                    number=1,
                    repeat=1,
                    min_repeat_ms=100,
                )
            ),
            cost_model=ms.cost_model.XGBModel(
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=False,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
            post_optimization=True,  # testing post optmization
        )
        # +1 because of post optmization
        assert len(database) == trials + 1


@pytest.mark.skip("Integration test")
@tvm.testing.requires_cuda
def test_tune_matmul_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/geforce-rtx-3070")
        trials = 32
        database = ms.tune_tir(
            mod=matmul,
            target=target,
            max_trials_global=trials,
            num_trials_per_iter=64,
            work_dir=work_dir,
            runner=ms.runner.LocalRunner(
                evaluator_config=EvaluatorConfig(
                    number=1,
                    repeat=1,
                    min_repeat_ms=100,
                )
            ),
            cost_model=ms.cost_model.XGBModel(
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=False,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
            post_optimization=True,  # testing post optmization
        )
        # +1 because of post optmization
        assert len(database) == trials + 1


if __name__ == """__main__""":
    test_tune_matmul_cpu()
    test_tune_matmul_cuda()
