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
"""Test the tune context of meta schedule."""

import sys
import pytest

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.target import Target
from tvm.meta_schedule import TuneContext

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


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


def test_tune_context_create():
    mod = Matmul
    context = TuneContext(mod=mod, target=Target("llvm"), task_name="Test Task")
    assert context.num_threads > 0
    assert context.rand_state != -1
    assert context.task_name == "Test Task"
    assert context.mod == mod or tvm.ir.structural_equal(context.mod, mod)


if __name__ == "__main__":
    tvm.testing.main()
