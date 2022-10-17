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
from tvm.meta_schedule.testing.conv2d_winograd_cpu import conv2d_winograd_cpu
from tvm.target import Target
from tvm.tir.schedule import Schedule, Trace


def _get_mod():
    # pylint: disable=invalid-name
    def inline(sch: Schedule):
        b1 = sch.get_block(name="A")
        b2 = sch.get_block(name="B")
        sch.compute_inline(block=b1)
        sch.compute_inline(block=b2)

    def input_tile_data_pad(sch: Schedule):
        b78 = sch.get_block(name="input_tile")
        l80 = sch.sample_compute_location(block=b78, decision=4)
        sch.compute_at(block=b78, loop=l80, preserve_unit_loops=True)

        b81 = sch.get_block(name="data_pad")
        l83 = sch.sample_compute_location(block=b81, decision=-2)
        sch.compute_at(block=b81, loop=l83, preserve_unit_loops=True)

    def data_pack(sch: Schedule):
        b18 = sch.get_block(name="data_pack")
        l19, l20, l21, l22, l23, l24 = sch.get_loops(block=b18)
        sch.unroll(loop=l19)
        sch.unroll(loop=l20)
        v25, v26 = sch.sample_perfect_tile(
            n=2,
            loop=l21,
            max_innermost_factor=64,
            decision=[9, 1],
        )
        l27, l28 = sch.split(loop=l21, factors=[v25, v26])
        v29, v30 = sch.sample_perfect_tile(
            n=2,
            loop=l22,
            max_innermost_factor=64,
            decision=[32, 4],
        )
        l31, l32 = sch.split(loop=l22, factors=[v29, v30])
        sch.unroll(loop=l23)
        sch.unroll(loop=l24)
        sch.reorder(l27, l31, l28, l32, l19, l20, l23, l24)

    def bgemm(sch: Schedule):
        bgemm = sch.get_block(name="bgemm")
        write_cache = sch.cache_write(
            block=bgemm,
            write_buffer_index=0,
            storage_scope="global",
        )
        sch.annotate(
            block_or_loop=bgemm,
            ann_key="meta_schedule.tiling_structure",
            ann_val="SSRSRS",
        )
        # b33, b34 = b34, b33
        l35, l36, l37, l38, l39 = sch.get_loops(block=bgemm)
        v40, v41, v42, v43 = sch.sample_perfect_tile(
            n=4,
            loop=l35,
            max_innermost_factor=64,
            decision=[1, 2, 3, 1],
        )
        l44, l45, l46, l47 = sch.split(loop=l35, factors=[v40, v41, v42, v43])
        v48, v49, v50, v51 = sch.sample_perfect_tile(
            n=4,
            loop=l36,
            max_innermost_factor=64,
            decision=[1, 1, 1, 6],
        )
        l52, l53, l54, l55 = sch.split(loop=l36, factors=[v48, v49, v50, v51])
        v56, v57, v58, v59 = sch.sample_perfect_tile(
            n=4,
            loop=l37,
            max_innermost_factor=64,
            decision=[1, 1, 1, 9],
        )
        l60, l61, l62, l63 = sch.split(loop=l37, factors=[v56, v57, v58, v59])
        v64, v65, v66, v67 = sch.sample_perfect_tile(
            n=4,
            loop=l38,
            max_innermost_factor=64,
            decision=[2, 1, 16, 4],
        )
        l68, l69, l70, l71 = sch.split(loop=l38, factors=[v64, v65, v66, v67])
        v72, v73 = sch.sample_perfect_tile(
            n=2,
            loop=l39,
            max_innermost_factor=64,
            decision=[16, 8],
        )
        l74, l75 = sch.split(loop=l39, factors=[v72, v73])
        sch.reorder(
            # fmt: off
                l44, l52, l60, l68,
                l45, l53, l61, l69,
                l74,
                l46, l54, l62, l70,
                l75,
                l47, l55, l63, l71,
            # fmt: on
        )
        sch.reverse_compute_at(block=write_cache, loop=l69, preserve_unit_loops=True)

    def inverse(sch: Schedule):
        b3 = sch.get_block(name="inverse")
        l4, l5, l6, l7, l8, l9 = sch.get_loops(block=b3)
        sch.unroll(loop=l4)
        sch.unroll(loop=l5)
        v10, v11 = sch.sample_perfect_tile(
            n=2,
            loop=l6,
            max_innermost_factor=64,
            decision=[1, 9],
        )
        l12, l13 = sch.split(loop=l6, factors=[v10, v11])
        v14, v15 = sch.sample_perfect_tile(
            n=2,
            loop=l7,
            max_innermost_factor=64,
            decision=[2, 64],
        )
        l16, l17 = sch.split(loop=l7, factors=[v14, v15])
        sch.unroll(loop=l8)
        sch.unroll(loop=l9)
        sch.reorder(l12, l16, l13, l17, l4, l5, l8, l9)

    # pylint: enable=invalid-name

    sch = Schedule(mod=conv2d_winograd_cpu)
    inline(sch)
    data_pack(sch)
    input_tile_data_pad(sch)
    bgemm(sch)
    inverse(sch)
    return sch.mod


def test_conv2d_winograd_cpu():
    mod = conv2d_winograd_cpu
    mod = IRModule({"main": mod})
    target = Target("llvm --num-cores=16")
    context = ms.TuneContext(
        mod=mod,
        target=target,
        task_name="Custom Search Space Task",
        space_generator=ms.space_generator.PostOrderApply(),
    )
    post_order_apply = context.space_generator
    (sch,) = post_order_apply.generate_design_space(mod)
    decisions = dict(
        zip(
            [i for i in sch.trace.insts[:-4] if i.kind.name.startswith("Sample")],
            [
                # data_pack
                [9, 1],
                [32, 4],
                # input_tile
                4,
                # data_pad
                -2,
                # inverse
                [1, 9],
                [2, 64],
                # bgemm
                [1, 2, 3, 1],
                [1, 1, 1, 6],
                [1, 1, 1, 9],
                [2, 1, 16, 4],
                [16, 8],
            ],
        )
    )
    trace = Trace(sch.trace.insts[:-4], decisions=decisions)
    sch = Schedule(mod=mod)
    trace.apply_to_schedule(sch, remove_postproc=False)
    answer = sch.mod
    expected = _get_mod()
    tvm.ir.assert_structural_equal(answer, expected)


if __name__ == "__main__":
    test_conv2d_winograd_cpu()
