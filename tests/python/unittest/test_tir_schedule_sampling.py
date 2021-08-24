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
import sys
from collections import defaultdict

import pytest
import tvm
from tvm import tir
from tvm.script import ty
from tvm.tir.schedule.testing import verify_trace_roundtrip
from tvm.tir.schedule import Trace


# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def elementwise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
        B[vi, vj, vk] = A[vi, vj, vk] * 2.0


# pylint: enable=no-member,invalid-name,unused-variable


def test_sample_categorical():
    """Test sample categprical sampling function"""
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
