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

import tvm
from tvm import meta_schedule as ms
from tvm.ir import IRModule
from tvm.meta_schedule.testing.conv2d_winograd_cuda import conv2d_winograd_cuda
from tvm.target import Target
from tvm.tir.schedule import Schedule, Trace


def _get_mod():
    # pylint: disable=invalid-name
    def inline(sch: Schedule):
        b125 = sch.get_block(name="A")
        sch.compute_inline(block=b125)
        b126 = sch.get_block(name="B")
        sch.compute_inline(block=b126)

    def input_tile_data_pad(sch: Schedule):
        b115 = sch.get_block(name="input_tile")
        (b116,) = sch.get_consumers(block=b115)
        _, _, _, l120, _, _, _, _ = sch.get_loops(block=b116)
        sch.compute_at(block=b115, loop=l120, preserve_unit_loops=True)
        sch.set_scope(block=b115, buffer_index=0, storage_scope="local")

        b127 = sch.get_block(name="data_pad")
        sch.compute_inline(block=b127)

        b3 = sch.get_block(name="data_pack")
        l25, l26, l27, l28, _, _, _, _ = sch.get_loops(block=b3)
        l33 = sch.fuse(l25, l26, l27, l28)
        v34 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        l35, l36 = sch.split(loop=l33, factors=[None, v34])
        sch.bind(loop=l35, thread_axis="blockIdx.x")
        sch.bind(loop=l36, thread_axis="threadIdx.x")

    def data_pack(sch: Schedule):
        b16 = sch.get_block(name="data_pack")
        l17, l18, l19, l20, l21, l22 = sch.get_loops(block=b16)
        sch.unroll(loop=l17)
        sch.unroll(loop=l18)
        v23, v24 = sch.sample_perfect_tile(
            n=2,
            loop=l19,
            max_innermost_factor=64,
            decision=[3, 3],
        )
        l25, l26 = sch.split(loop=l19, factors=[v23, v24])
        v27, v28 = sch.sample_perfect_tile(
            n=2,
            loop=l20,
            max_innermost_factor=64,
            decision=[64, 2],
        )
        l29, l30 = sch.split(loop=l20, factors=[v27, v28])
        sch.unroll(loop=l21)
        sch.unroll(loop=l22)
        sch.reorder(l25, l29, l26, l30, l17, l18, l21, l22)

    def bgemm(sch: Schedule):
        b31 = sch.get_block(name="bgemm")
        sch.annotate(
            block_or_loop=b31,
            ann_key="meta_schedule.tiling_structure",
            ann_val="SSSRRSRS",
        )
        sch.annotate(
            block_or_loop=b31,
            ann_key="meta_schedule.thread_extent_low_inclusive",
            ann_val=32,
        )
        sch.annotate(
            block_or_loop=b31,
            ann_key="meta_schedule.thread_extent_high_inclusive",
            ann_val=1024,
        )
        b32 = sch.cache_write(block=b31, write_buffer_index=0, storage_scope="local")
        b31, b32 = b32, b31
        l33, l34, l35, l36, l37 = sch.get_loops(block=b32)
        v38, v39, v40, v41, v42 = sch.sample_perfect_tile(
            n=5,
            loop=l33,
            max_innermost_factor=64,
            decision=[1, 1, 1, 1, 6],
        )
        l43, l44, l45, l46, l47 = sch.split(loop=l33, factors=[v38, v39, v40, v41, v42])
        v48, v49, v50, v51, v52 = sch.sample_perfect_tile(
            n=5,
            loop=l34,
            max_innermost_factor=64,
            decision=[1, 1, 1, 3, 2],
        )
        l53, l54, l55, l56, l57 = sch.split(loop=l34, factors=[v48, v49, v50, v51, v52])
        v58, v59, v60, v61, v62 = sch.sample_perfect_tile(
            n=5,
            loop=l35,
            max_innermost_factor=64,
            decision=[3, 1, 1, 1, 3],
        )
        l63, l64, l65, l66, l67 = sch.split(loop=l35, factors=[v58, v59, v60, v61, v62])
        v68, v69, v70, v71, v72 = sch.sample_perfect_tile(
            n=5,
            loop=l36,
            max_innermost_factor=64,
            decision=[4, 2, 1, 4, 4],
        )
        l73, l74, l75, l76, l77 = sch.split(loop=l36, factors=[v68, v69, v70, v71, v72])
        v78, v79, v80 = sch.sample_perfect_tile(
            n=3,
            loop=l37,
            max_innermost_factor=64,
            decision=[32, 1, 4],
        )
        l81, l82, l83 = sch.split(loop=l37, factors=[v78, v79, v80])
        sch.reorder(
            # fmt: off
            l43, l53, l63, l73,
            l44, l54, l64, l74,
            l45, l55, l65, l75,
            l81,
            l82,
            l46, l56, l66, l76,
            l83,
            l47, l57, l67, l77,
            # fmt: on
        )
        l84 = sch.fuse(l43, l53, l63, l73)
        sch.bind(loop=l84, thread_axis="blockIdx.x")
        l85 = sch.fuse(l44, l54, l64, l74)
        sch.bind(loop=l85, thread_axis="vthread.x")
        l86 = sch.fuse(l45, l55, l65, l75)
        sch.bind(loop=l86, thread_axis="threadIdx.x")

        b87 = sch.cache_read(block=b32, read_buffer_index=1, storage_scope="shared")
        sch.compute_at(block=b87, loop=l81, preserve_unit_loops=True)
        _, _, _, _, l92, l93, l94, l95 = sch.get_loops(block=b87)
        sch.fuse(l92, l93, l94, l95)
        v97 = sch.sample_categorical(
            candidates=[1, 2, 3, 4],
            probs=[0.25, 0.25, 0.25, 0.25],
            decision=1,
        )
        sch.annotate(
            block_or_loop=b87,
            ann_key="meta_schedule.cooperative_fetch",
            ann_val=v97,
        )

        b101 = sch.cache_read(block=b32, read_buffer_index=2, storage_scope="shared")
        sch.compute_at(block=b101, loop=l81, preserve_unit_loops=True)
        _, _, _, _, l106, l107, l108, l109 = sch.get_loops(block=b101)
        sch.fuse(l106, l107, l108, l109)
        v110 = sch.sample_categorical(
            candidates=[1, 2, 3, 4],
            probs=[0.25, 0.25, 0.25, 0.25],
            decision=1,
        )
        sch.annotate(
            block_or_loop=b101,
            ann_key="meta_schedule.cooperative_fetch",
            ann_val=v110,
        )

        sch.reverse_compute_at(block=b31, loop=l86, preserve_unit_loops=True)

    def inverse(sch: Schedule):
        b1 = sch.get_block(name="inverse")
        l2, l3, l4, l5, l6, l7 = sch.get_loops(block=b1)
        sch.unroll(loop=l2)
        sch.unroll(loop=l3)
        v8, v9 = sch.sample_perfect_tile(
            n=2,
            loop=l4,
            max_innermost_factor=64,
            decision=[3, 3],
        )
        l10, l11 = sch.split(loop=l4, factors=[v8, v9])
        v12, v13 = sch.sample_perfect_tile(
            n=2,
            loop=l5,
            max_innermost_factor=64,
            decision=[2, 64],
        )
        l14, l15 = sch.split(loop=l5, factors=[v12, v13])
        sch.unroll(loop=l6)
        sch.unroll(loop=l7)
        sch.reorder(l10, l14, l11, l15, l2, l3, l6, l7)
        l59 = sch.fuse(l10, l14, l11, l15)
        v60 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        l61, l62 = sch.split(loop=l59, factors=[None, v60])
        sch.bind(loop=l61, thread_axis="blockIdx.x")
        sch.bind(loop=l62, thread_axis="threadIdx.x")

    def conv2d(sch: Schedule):
        b7 = sch.get_block(name="conv2d_winograd")
        l141, l142, l143, l144 = sch.get_loops(block=b7)
        l145 = sch.fuse(l141, l142, l143, l144)
        v146 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        l147, l148 = sch.split(loop=l145, factors=[None, v146])
        sch.bind(loop=l147, thread_axis="blockIdx.x")
        sch.bind(loop=l148, thread_axis="threadIdx.x")

    def root_anno(sch: Schedule):
        b8 = sch.get_block(name="root", func_name="main")
        v140 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024],
            probs=[
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
            ],
            decision=2,
        )
        sch.annotate(block_or_loop=b8, ann_key="meta_schedule.unroll_explicit", ann_val=v140)

    # pylint: enable=invalid-name

    sch = Schedule(mod=conv2d_winograd_cuda)
    inline(sch)
    data_pack(sch)
    input_tile_data_pad(sch)
    bgemm(sch)
    inverse(sch)
    conv2d(sch)
    root_anno(sch)

    return sch.mod


def test_conv2d_winograd_cuda():
    mod = conv2d_winograd_cuda
    mod = IRModule({"main": mod})
    context = ms.TuneContext(
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        task_name="Custom Search Space Task",
        space_generator=ms.space_generator.PostOrderApply(),
    )
    post_order_apply = context.space_generator
    (sch,) = post_order_apply.generate_design_space(mod)
    decisions = dict(
        zip(
            [i for i in sch.trace.insts if i.kind.name.startswith("Sample")],
            [
                # data_pack
                [3, 3],
                [64, 2],
                2,
                # inverse
                [3, 3],
                [2, 64],
                2,
                # bgemm
                [1, 1, 1, 1, 6],
                [1, 1, 1, 3, 2],
                [3, 1, 1, 1, 3],
                [4, 2, 1, 4, 4],
                [32, 1, 4],
                1,
                1,
                # root anno
                2,
                # conv2d
                2,
            ],
        )
    )
    trace = Trace(sch.trace.insts, decisions=decisions)
    sch = Schedule(mod=mod)
    trace.apply_to_schedule(sch, remove_postproc=False)
    answer = sch.mod
    expected = _get_mod()
    tvm.ir.assert_structural_equal(answer, expected)


if __name__ == "__main__":
    test_conv2d_winograd_cuda()
