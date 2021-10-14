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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

from typing import List, Optional

import re

import tvm
from tvm.ir.base import assert_structural_equal
from tvm.script import tir as T

from tvm.meta_schedule.mutator import PyMutator
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.utils import _get_hex_address
from tvm.tir.schedule import Schedule, Trace

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        with T.block([1024, 1024, T.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def test_meta_schedule_mutator():
    class FancyMutator(PyMutator):
        def apply(self, trace: Trace) -> Optional[Trace]:
            return Trace(trace.insts, {})

    mutator = FancyMutator()
    sch = Schedule(Matmul)
    res = mutator.apply(sch.trace)
    assert res is not None
    new_sch = sch.copy()
    res.apply_to_schedule(new_sch, remove_postproc=True)
    assert_structural_equal(sch.mod, new_sch.mod)


def test_meta_schedule_mutator_as_string():
    class YetAnotherFancyMutator(PyMutator):
        def __str__(self) -> str:
            return f"YetAnotherFancyMutator({_get_hex_address(self.handle)})"

    mutator = YetAnotherFancyMutator()
    pattern = re.compile(r"YetAnotherFancyMutator\(0x[a-f|0-9]*\)")
    assert pattern.match(str(mutator))


if __name__ == "__main__":
    test_meta_schedule_mutator()
    test_meta_schedule_mutator_as_string()

