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
import os
import re
import shutil
import sys
import tempfile
from typing import List

import numpy as np
import pytest

import tvm
from tvm.meta_schedule.cost_model import PyCostModel, RandomModel
from tvm.meta_schedule.feature_extractor import RandomFeatureExtractor
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.cost_model import XGBModel
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.tune_context import TuneContext
from tvm.meta_schedule.utils import derived_object
from tvm.script import tir as T
from tvm.tir.schedule.schedule import Schedule

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring
@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,disable=unused-argument


def test_meta_schedule_cost_model():
    @derived_object
    class FancyCostModel(PyCostModel):
        def load(self, path: str) -> None:
            pass

        def save(self, path: str) -> None:
            pass

        def update(
            self,
            context: TuneContext,
            candidates: List[MeasureCandidate],
            results: List[RunnerResult],
        ) -> None:
            pass

        def predict(self, context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
            return np.random.rand(10)

    model = FancyCostModel()
    model.save("fancy_test_location")
    model.load("fancy_test_location")
    model.update(TuneContext(), [], [])
    results = model.predict(
        TuneContext(), [MeasureCandidate(Schedule(mod=Matmul), []) for _ in range(10)]
    )
    assert results.shape == (10,)


def test_meta_schedule_cost_model_as_string():
    @derived_object
    class NotSoFancyCostModel(PyCostModel):
        def load(self, path: str) -> None:
            pass

        def save(self, path: str) -> None:
            pass

        def update(
            self,
            context: TuneContext,
            candidates: List[MeasureCandidate],
            results: List[RunnerResult],
        ) -> None:
            pass

        def predict(self, context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
            return np.random.rand(10)

    cost_model = NotSoFancyCostModel()
    pattern = re.compile(r"meta_schedule.NotSoFancyCostModel\(0x[a-f|0-9]*\)")
    assert pattern.match(str(cost_model))


def test_meta_schedule_random_model():
    model = RandomModel()
    model.update(TuneContext(), [], [])
    res = model.predict(TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(10)])
    assert len(res) == 10
    assert min(res) >= 0 and max(res) <= model.max_range


def test_meta_schedule_random_model_reseed():
    model = RandomModel(seed=100)
    res = model.predict(TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(20)])
    new_model = RandomModel(seed=100)
    new_res = new_model.predict(
        TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(20)]
    )
    assert (res == new_res).all()


def test_meta_schedule_random_model_reload():
    model = RandomModel(seed=25973)
    model.predict(
        TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(30)]
    )  # change state
    path = os.path.join(tempfile.mkdtemp(), "test_output_meta_schedule_random_model.npy")
    model.save(path)
    res1 = model.predict(TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(70)])
    model.load(path)
    res2 = model.predict(TuneContext(), [MeasureCandidate(Schedule(Matmul), []) for i in range(70)])
    shutil.rmtree(os.path.dirname(path))
    assert (res1 == res2).all()


def _dummy_candidate():
    return MeasureCandidate(Schedule(Matmul), [])


def _dummy_result(num_samples: int = 4, max_run_sec: int = 10):
    return RunnerResult(list(np.random.rand(num_samples) * max_run_sec + 1e-6), None)


def test_meta_schedule_xgb_model():
    extractor = RandomFeatureExtractor()
    model = XGBModel(extractor=extractor, num_warmup_samples=2)
    update_sample_count = 10
    predict_sample_count = 100
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])


def test_meta_schedule_xgb_model_reload():
    extractor = RandomFeatureExtractor()
    model = XGBModel(extractor=extractor, num_warmup_samples=10)
    update_sample_count = 20
    predict_sample_count = 30
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])
    random_state = model.extractor.random_state  # save feature extractor's random state
    path = os.path.join(tempfile.mkdtemp(), "test_output_meta_schedule_xgb_model.bin")
    cached = (model.cached_features.copy(), model.cached_mean_costs.copy())
    model.save(path)
    res1 = model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])
    model.extractor.random_state = random_state  # load feature extractor's random state
    model.cached_features = None
    model.cached_mean_costs = None
    model.load(path)
    new_cached = (model.cached_features.copy(), model.cached_mean_costs.copy())
    res2 = model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])
    shutil.rmtree(os.path.dirname(path))
    assert (res1 == res2).all()
    # cached feature does not change
    assert len(cached[0]) == len(new_cached[0])
    for i in range(len(cached[0])):
        assert (cached[0][i] == new_cached[0][i]).all()
    # cached meaen cost does not change
    assert (cached[1] == new_cached[1]).all()


def test_meta_schedule_xgb_model_reupdate():
    extractor = RandomFeatureExtractor()
    model = XGBModel(extractor=extractor, num_warmup_samples=2)
    update_sample_count = 60
    predict_sample_count = 100
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.update(
        TuneContext(),
        [_dummy_candidate() for i in range(update_sample_count)],
        [_dummy_result() for i in range(update_sample_count)],
    )
    model.predict(TuneContext(), [_dummy_candidate() for i in range(predict_sample_count)])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
