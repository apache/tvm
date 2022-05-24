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

from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing.schedule_rule import auto_bind
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.target import Target
from tvm.script import tir as T


@T.prim_func
def element_wise(var_A: T.handle, var_B: T.handle) -> None:
    A = T.match_buffer(var_A, [512, 512], dtype="float32")
    B = T.match_buffer(var_B, [512, 512], dtype="float32")
    for i, j in T.grid(512, 512):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] + 1.0


def _create_context(mod, target, rule) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for sch_rule in ctx.sch_rules:
        sch_rule.initialize_with_tune_context(ctx)
    return ctx


def test_cuda_element_wise():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "l1, l2 = sch.get_loops(block=b0)",
            "l3 = sch.fuse(l1, l2)",
            "v4 = sch.sample_categorical(candidates=[32, 64, 128, 256, 512, 1024], probs=[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666])",
            "l5, l6 = sch.split(loop=l3, factors=[None, v4])",
            'sch.bind(loop=l5, thread_axis="blockIdx.x")',
            'sch.bind(loop=l6, thread_axis="threadIdx.x")',
        ]
    ]
    target = Target("nvidia/geforce-rtx-3080", host="llvm")
    ctx = _create_context(
        element_wise,
        target=target,
        rule=auto_bind(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_cuda_element_wise()
