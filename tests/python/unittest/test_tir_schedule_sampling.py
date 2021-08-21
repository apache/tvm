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
from typing import Union
from collections import defaultdict

import pytest
import tvm
from tvm import tir
from tvm.script import ty
from tvm.ir import IRModule
from tvm.tir import PrimFunc
from tvm.tir.schedule import Trace


# pylint: disable=no-member,invalid-name,unused-variable


def _check_serialization(sch: tir.Schedule, mod: Union[PrimFunc, IRModule]) -> tir.Schedule:
    record = sch.trace.as_json()
    new_sch = tir.Schedule(mod=mod)
    Trace.apply_json_to_schedule(record, sch=new_sch)
    assert tvm.ir.structural_equal(new_sch.mod, sch.mod)
    py_repr = "\n".join(sch.trace.as_python())
    new_py_repr = "\n".join(new_sch.trace.as_python())
    assert py_repr == new_py_repr
    # print(py_repr)
    return new_sch


@tvm.script.tir
def elementwise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
        B[vi, vj, vk] = A[vi, vj, vk] * 2.0


def test_fuse_sample_categorical():
    """Test sample categprical sampling function"""
    n = 1000
    sch = tir.Schedule(elementwise, seed=42)
    counter = defaultdict(int)
    candidates = [5, 2, 7, 1]
    probs = [0.15, 0.55, 0.05, 0.25]
    for _ in range(n):
        v = sch.get(sch.sample_categorical(candidates, probs))
        counter[v] += 1
    for i, prob in enumerate(probs):
        assert (prob - 0.07) * n <= counter[candidates[i]] <= (prob + 0.07) * n
    _check_serialization(sch, mod=elementwise)


@pytest.mark.xfail
def test_fuse_sample_categorical_out_of_range():
    """Test sample categprical sampling function"""
    n = 1000
    sch = tir.Schedule(elementwise, seed=42)
    counter = defaultdict(int)
    candidates = [5, 2, 7, 1]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    for _ in range(n):
        v = sch.get(sch.sample_categorical(candidates, probs))
        counter[v] += 1
    for i, prob in enumerate(probs):
        assert (prob - 0.07) * n <= counter[candidates[i]] <= (prob + 0.07) * n


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
