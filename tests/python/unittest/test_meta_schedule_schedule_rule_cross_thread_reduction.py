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
from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.schedule_rule import cross_thread_reduction
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.script import tir as T
from tvm.target import Target
from tvm.te.operation import create_prim_func


@tvm.script.ir_module
class Softmax_mn_after_inline:
    @T.prim_func
    def main(
        A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]
    ) -> None:
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.min_value("float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_expsum"):
                i0_2, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(
                    A[i0_2, k] - T_softmax_maxelem[i0_2], dtype="float32"
                )
        for i0_3, i1 in T.grid(256, 256):
            with T.block("T_softmax_norm"):
                i0_4, i1_1 = T.axis.remap("SS", [i0_3, i1])
                T.block_attr({"axis": 1})
                T_softmax_norm[i0_4, i1_1] = (
                    T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4], dtype="float32")
                    / T_softmax_expsum[i0_4]
                )


def _create_context(mod, target, rule) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    ctx.initialize()
    return ctx


def test_gpu_softmax_mn():
    expected = [
        [],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, l3 = sch.get_loops(block=b1)",
            "v4 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l5, l6 = sch.split(loop=l3, factors=[None, v4])",
            'sch.bind(loop=l6, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l2, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l7, l8, l9 = sch.get_loops(block=b0)",
            "l10, l11 = sch.split(loop=l9, factors=[None, v4])",
            'sch.bind(loop=l11, thread_axis="threadIdx.x")',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, l3 = sch.get_loops(block=b1)",
            "v4 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l5, l6 = sch.split(loop=l3, factors=[None, v4])",
            'sch.bind(loop=l6, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l2, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l7, l8, l9 = sch.get_loops(block=b0)",
            "l10, l11 = sch.split(loop=l9, factors=[None, v4])",
            'sch.bind(loop=l11, thread_axis="threadIdx.x")',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            "b2, = sch.get_consumers(block=b1)",
            "l3, l4 = sch.get_loops(block=b2)",
            "v5 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l6, l7 = sch.split(loop=l4, factors=[None, v5])",
            'sch.bind(loop=l7, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b1, loop=l3, preserve_unit_loops=True)",
            'sch.set_scope(block=b1, buffer_index=0, storage_scope="shared")',
            "l8, l9, l10 = sch.get_loops(block=b1)",
            "l11, l12 = sch.split(loop=l10, factors=[None, v5])",
            'sch.bind(loop=l12, thread_axis="threadIdx.x")',
            "b13, = sch.get_consumers(block=b0)",
            "l14, l15 = sch.get_loops(block=b13)",
            "v16 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l17, l18 = sch.split(loop=l15, factors=[None, v16])",
            'sch.bind(loop=l18, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l14, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l19, l20, l21 = sch.get_loops(block=b0)",
            "l22, l23 = sch.split(loop=l21, factors=[None, v16])",
            'sch.bind(loop=l23, thread_axis="threadIdx.x")',
        ],
    ]
    target = Target("nvidia/geforce-rtx-3090", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.softmax_mn(
                n=256,
                m=256,
            )
        ),
        target=target,
        rule=cross_thread_reduction(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 4
    check_trace(spaces, expected)


def test_gpu_softmax_mn_after_inline():
    expected = [
        [],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            "v1 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l2, l3 = sch.get_loops(block=b0)",
            "l4, l5 = sch.split(loop=l3, factors=[None, v1])",
            'sch.bind(loop=l5, thread_axis="threadIdx.x")',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, l3 = sch.get_loops(block=b1)",
            "v4 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l5, l6 = sch.split(loop=l3, factors=[None, v4])",
            'sch.bind(loop=l6, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l2, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l7, l8, l9 = sch.get_loops(block=b0)",
            "l10, l11 = sch.split(loop=l9, factors=[None, v4])",
            'sch.bind(loop=l11, thread_axis="threadIdx.x")',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            "b2, = sch.get_consumers(block=b1)",
            "l3, l4 = sch.get_loops(block=b2)",
            "v5 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l6, l7 = sch.split(loop=l4, factors=[None, v5])",
            'sch.bind(loop=l7, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b1, loop=l3, preserve_unit_loops=True)",
            'sch.set_scope(block=b1, buffer_index=0, storage_scope="shared")',
            "l8, l9, l10 = sch.get_loops(block=b1)",
            "l11, l12 = sch.split(loop=l10, factors=[None, v5])",
            'sch.bind(loop=l12, thread_axis="threadIdx.x")',
            "b13, b14 = sch.get_consumers(block=b0)",
            "l15, l16, l17, l18 = sch.get_loops(block=b13)",
            "sch.compute_at(block=b0, loop=l15, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l19, l20, l21 = sch.get_loops(block=b0)",
            "l22, l23 = sch.split(loop=l21, factors=[None, v5])",
            'sch.bind(loop=l23, thread_axis="threadIdx.x")',
        ],
    ]
    target = Target("nvidia/geforce-rtx-3090", host="llvm")
    ctx = _create_context(
        mod=Softmax_mn_after_inline,
        target=target,
        rule=cross_thread_reduction(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 4
    check_trace(spaces, expected)


def test_gpu_batch_norm_bmn():
    expected = [
        [],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, = sch.get_loops(block=b1)",
            "v3 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l4, l5 = sch.split(loop=l2, factors=[None, v3])",
            'sch.bind(loop=l5, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l4, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l6, l7, l8, l9 = sch.get_loops(block=b0)",
            "l10 = sch.fuse(l8, l9)",
            "l11, l12 = sch.split(loop=l10, factors=[None, v3])",
            'sch.bind(loop=l12, thread_axis="threadIdx.x")',
        ],
    ]
    target = Target("nvidia/geforce-rtx-3090", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.norm_bmn(
                B=1,
                M=512,
                N=512,
            )
        ),
        target=target,
        rule=cross_thread_reduction(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 2
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_gpu_softmax_mn()
    test_gpu_softmax_mn_after_inline()
    test_gpu_batch_norm_bmn()
