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
""" Test Meta Schedule SpaceGenerator """
# pylint: disable=missing-function-docstring

import math

import pytest
import tvm
import tvm.testing
from tvm._ffi.base import TVMError
from tvm.meta_schedule.space_generator import (
    PySpaceGenerator,
    ScheduleFn,
    SpaceGeneratorUnion,
)
from tvm.meta_schedule.tune_context import TuneContext
from tvm.meta_schedule.utils import derived_object
from tvm.script import tir as T
from tvm.tir.schedule import Schedule

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    # TODO(@zxybazh): Change to `sample_perfect_tile` after upstreaming
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _check_correct(schedule: Schedule):
    trace = schedule.trace
    for inst in trace.decisions:
        assert math.prod(trace.decisions[inst]) == 1024


def test_meta_schedule_space_generator_schedule_fn():
    mod = Matmul
    space_generator = ScheduleFn(sch_fn=schedule_matmul)
    design_spaces = space_generator.generate_design_space(mod)
    assert len(design_spaces) == 1
    (schedule,) = design_spaces
    _check_correct(schedule)


def test_meta_schedule_design_space_generator_union():
    mod = Matmul
    space_generator = ScheduleFn(sch_fn=schedule_matmul)
    space_generator_union = SpaceGeneratorUnion([space_generator, space_generator])
    design_spaces = space_generator_union.generate_design_space(mod)
    assert len(design_spaces) == 2
    for design_space in design_spaces:
        _check_correct(design_space)


def test_meta_schedule_design_space_generator_NIE():
    @derived_object
    class TestPySpaceGenerator(PySpaceGenerator):
        def __init__(self):
            super().__init__()
            self.sch_rules = []
            self.postprocs = []
            self.mutator_probs = {}

    with pytest.raises(
        TVMError, match="PySpaceGenerator's InitializeWithTuneContext method not implemented!"
    ):
        generator = TestPySpaceGenerator()
        generator._initialize_with_tune_context(TuneContext())


if __name__ == "__main__":
    tvm.testing.main()
