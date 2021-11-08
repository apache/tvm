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
from collections import defaultdict
import sys

import pytest

from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip


# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 257, 1470))
    B = T.match_buffer(b, (128, 257, 1470))
    for i, j, k in T.grid(128, 257, 1470):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


# pylint: enable=no-member,invalid-name,unused-variable


def test_sample_categorical():
    """Test sample categorical sampling function"""
    n = 1000
    sch = tir.Schedule(elementwise, seed=42, debug_mask="all")
    counter = defaultdict(int)
    candidates = [5, 2, 7, 1]
    probs = [0.15, 0.55, 0.05, 0.25]
    for _ in range(n):
        v = sch.get(sch.sample_categorical(candidates, probs))
        counter[v] += 1
    for i, prob in enumerate(probs):
        assert (prob - 0.07) * n <= counter[candidates[i]] <= (prob + 0.07) * n
    verify_trace_roundtrip(sch, mod=elementwise)


def test_sample_categorical_copy():
    """Check the random variable sampling results after schedule copy"""
    n = 100
    sch = tir.Schedule(elementwise, seed=42, debug_mask="all")
    candidates = [1, 2, 3, 4]
    probs = [0.1, 0.2, 0.3, 0.4]
    rv_decisions = []
    for _ in range(n):
        rv = sch.sample_categorical(candidates, probs)  # pylint: disable=invalid-name
        rv_decisions.append((rv, sch.get(rv)))
    sch_copy = sch.copy()
    for rv, decision in rv_decisions:  # pylint: disable=invalid-name
        decision_copy = sch_copy.get(rv)
        assert int(decision) == int(decision_copy)


def test_sample_categorical_serialize():
    """Check the random variable sampling results after schedule serialization"""
    n = 100
    sch = tir.Schedule(elementwise, seed=42, debug_mask="all")
    candidates = [5, 6, 7, 8]
    probs = [0.23, 0.19, 0.37, 0.21]
    decisions = []
    for _ in range(n):
        rv = sch.get(sch.sample_categorical(candidates, probs))  # pylint: disable=invalid-name
        decisions.append(rv)
    new_sch = verify_trace_roundtrip(sch, mod=elementwise)
    for i, new_inst in enumerate(new_sch.trace.insts):
        assert decisions[i] == candidates[new_sch.trace.decisions[new_inst].value]


def test_sample_perfect_tile_power_of_two():
    sch = tir.Schedule(elementwise, debug_mask="all")
    i, _, _ = sch.get_loops(sch.get_block("B"))
    factors = sch.sample_perfect_tile(i, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 128
    verify_trace_roundtrip(sch, mod=elementwise)


def test_sample_perfect_tile_prime():
    sch = tir.Schedule(elementwise, debug_mask="all")
    _, i, _ = sch.get_loops(sch.get_block("B"))
    factors = sch.sample_perfect_tile(i, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 257
    verify_trace_roundtrip(sch, mod=elementwise)


def test_sample_perfect_tile_composite():
    sch = tir.Schedule(elementwise, debug_mask="all")
    _, _, i = sch.get_loops(sch.get_block("B"))
    factors = sch.sample_perfect_tile(i, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 1470
    verify_trace_roundtrip(sch, mod=elementwise)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
