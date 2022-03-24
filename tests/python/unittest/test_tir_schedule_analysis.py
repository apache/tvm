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
from typing import List

from tvm.tir import (
    Evaluate,
    For,
    ForKind,
    IndexMap,
    Var,
    decl_buffer,
    floordiv,
    floormod,
)
from tvm.tir.analysis import expr_deep_equal
from tvm.tir.schedule.analysis import suggest_index_map


def _make_vars(*args: str) -> List[Var]:
    return [Var(arg, dtype="int32") for arg in args]


def _make_loops(loop_vars: List[Var], extents: List[int]) -> List[For]:
    assert len(loop_vars) == len(extents)
    return [
        For(
            loop_var=loop_var,
            min_val=0,
            extent=extent,
            kind=ForKind.SERIAL,
            body=Evaluate(0),
        )
        for loop_var, extent in zip(loop_vars, extents)
    ]


def _assert_equal_index_map(map1: IndexMap, map2: IndexMap) -> None:
    iters_1 = map1.map_indices(map2.initial_indices)
    iters_2 = map2.final_indices
    assert len(iters_1) == len(iters_2)
    for iter1, iter2 in zip(iters_1, iters_2):
        assert expr_deep_equal(iter1, iter2)


def test_suggest_index_map_simple():
    i, j = _make_vars("i", "j")
    index_map = suggest_index_map(
        buffer=decl_buffer(shape=[8, 256]),
        indices=[
            floordiv(i, 16) * 4 + floordiv(j, 16),
            floormod(i, 16) * 16 + floormod(j, 16),
        ],
        loops=_make_loops(
            loop_vars=[i, j],
            extents=[32, 64],
        ),
        predicate=True,
    )
    expected_index_map = IndexMap.from_func(
        lambda x, y: [
            floordiv(x, 4),
            floordiv(y, 16),
            floormod(x, 4),
            floormod(y, 16),
        ],
    )
    _assert_equal_index_map(index_map, expected_index_map)


def test_suggest_index_map_bijective():
    i, j = _make_vars("i", "j")
    index_map = suggest_index_map(
        buffer=decl_buffer(shape=[8]),
        indices=[floormod(j, 4) * 2 + i],
        loops=_make_loops(
            loop_vars=[i, j],
            extents=[2, 32],
        ),
        predicate=True,
    )
    expected_index_map = IndexMap.from_func(
        lambda x: [
            floormod(x, 2),
            floordiv(x, 2),
        ],
    )
    _assert_equal_index_map(index_map, expected_index_map)


if __name__ == "__main__":
    test_suggest_index_map_simple()
    test_suggest_index_map_bijective()
