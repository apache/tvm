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
from tvm.script import tir as T
from tvm.target import Target


@T.prim_func
def element_wise(var_A: T.handle, var_B: T.handle) -> None:
    A = T.match_buffer(var_A, [512, 512], dtype="float32")
    B = T.match_buffer(var_B, [512, 512], dtype="float32")
    for i, j in T.grid(512, 512):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] + 1.0


@T.prim_func
def reduction_loop_only(
    A: T.Buffer[2, "float32"],
    B: T.Buffer[2, "float32"],
    C: T.Buffer[(), "float32"],
) -> None:
    for i0 in T.serial(2):
        with T.block("C"):
            k0 = T.axis.reduce(2, i0)
            T.reads(A[k0], B[k0])
            T.writes(C[()])
            with T.init():
                C[()] = T.float32(1.0)
            C[()] = T.min(C[()], A[k0] / B[k0])


@T.prim_func
def zero_dim_add(
    A: T.Buffer[(), "float32"],
    B: T.Buffer[(), "float32"],
    C: T.Buffer[(), "float32"],
) -> None:
    with T.block("C"):
        vi = T.axis.spatial(1, 0)
        C[()] = A[()] + B[()]


def _create_context(mod, target, rule) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    return ctx


def test_cuda_element_wise():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "l1, l2 = sch.get_loops(block=b0)",
            "l3 = sch.fuse(l1, l2, preserve_unit_iters=True)",
            "v4 = sch.sample_categorical(candidates=[32, 64, 128, 256, 512, 1024], probs=[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666])",
            "l5, l6 = sch.split(loop=l3, factors=[None, v4], preserve_unit_iters=True)",
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


def test_cuda_reduction_loop_only():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "l1, = sch.get_loops(block=b0)",
            "l2 = sch.add_unit_loop(block_or_loop=l1)",
            "l3 = sch.fuse(l2, preserve_unit_iters=True)",
            "l4, l5 = sch.split(loop=l3, factors=[None, 1], preserve_unit_iters=True)",
            'sch.bind(loop=l4, thread_axis="blockIdx.x")',
            'sch.bind(loop=l5, thread_axis="threadIdx.x")',
        ]
    ]
    target = Target("nvidia/geforce-rtx-3080", host="llvm")
    ctx = _create_context(
        reduction_loop_only,
        target=target,
        rule=auto_bind(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_cuda_zero_dim_add():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "l1 = sch.add_unit_loop(block_or_loop=b0)",
            "l2 = sch.fuse(l1, preserve_unit_iters=True)",
            "l3, l4 = sch.split(loop=l2, factors=[None, 1], preserve_unit_iters=True)",
            'sch.bind(loop=l3, thread_axis="blockIdx.x")',
            'sch.bind(loop=l4, thread_axis="threadIdx.x")',
        ]
    ]
    target = Target("nvidia/geforce-rtx-3080", host="llvm")
    ctx = _create_context(
        zero_dim_add,
        target=target,
        rule=auto_bind(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_cuda_element_wise()
    test_cuda_reduction_loop_only()
    test_cuda_zero_dim_add()
