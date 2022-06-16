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
import tvm
from tvm.meta_schedule.schedule_rule import RandomComputeLocation
from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.script import tir as T
from tvm.target import Target

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm.script.ir_module
class Add:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, [2048, 2048, 2048], dtype="float32")
        B = T.match_buffer(b, [2048, 2048, 2048], dtype="float32")
        A_cached = T.alloc_buffer([2048, 2048, 2048], dtype="float32")
        # body
        for i, j, k in T.grid(2048, 2048, 2048):
            with T.block("move"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([A_cached[vi, vj, vk]])
                A_cached[vi, vj, vk] = A[vi, vj, vk]
        for i0, j0, i1, j1, k0, i2, j2, k1 in T.grid(128, 64, 4, 4, 64, 4, 8, 32):
            with T.block("add"):
                vi = T.axis.spatial(2048, i0 * 16 + i1 * 4 + i2)
                vj = T.axis.spatial(2048, j0 * 32 + j1 * 8 + j2)
                vk = T.axis.spatial(2048, k0 * 32 + k1)
                T.reads([A_cached[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A_cached[vi, vj, vk] + T.float32(1)

# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def _create_context(mod, target, rule):
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    return ctx


def test_random_compute_location():
    expected = [
        [
            'b0 = sch.get_block(name="move", func_name="main")',
            "l1 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l1, preserve_unit_loops=True)",
        ]
    ]
    mod = Add
    target = Target("llvm")
    ctx = _create_context(
        mod=mod,
        target=target,
        rule=RandomComputeLocation(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_random_compute_location()
