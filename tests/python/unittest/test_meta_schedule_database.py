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
"""Test Meta Schedule Database"""
import os.path as osp
import tempfile
from typing import Callable
import pytest

import tvm
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder.builder import BuilderInput
from tvm.meta_schedule.builder.local_builder import LocalBuilder
from tvm.meta_schedule.database.database import TuningRecord
from tvm.meta_schedule.runner.local_runner import LocalRunner
from tvm.meta_schedule.runner.runner import RunnerInput
from tvm.target.target import Target
import tvm.testing
from tvm import tir
from tvm import meta_schedule as ms
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.tir import Schedule


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,unused-variable
# fmt: off
@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulRelu:
    @T.prim_func
    def main(a: T.handle, b: T.handle, d: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        D = T.match_buffer(d, (16, 16), "float32")
        C = T.alloc_buffer((16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(16, 16):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)


@tvm.script.ir_module
class MobileNetConv2d:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 1280, 1, 1), "float32"], placeholder_1: T.Buffer[(1001, 1280, 1, 1), "float32"], placeholder_2: T.Buffer[(1, 1001, 1, 1), "float32"], placeholder_3: T.Buffer[(), "float32"], T_cast: T.Buffer[(1, 1001, 1, 1), "int8"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        pad_temp = T.alloc_buffer([1, 1280, 1, 1], dtype="float32")
        conv2d_nchw = T.alloc_buffer([1, 1001, 1, 1], dtype="float32")
        T_add = T.alloc_buffer([1, 1001, 1, 1], dtype="float32")
        T_divide = T.alloc_buffer([1, 1001, 1, 1], dtype="float32")
        T_round = T.alloc_buffer([1, 1001, 1, 1], dtype="float32")
        compile_engine_const = T.alloc_buffer([], dtype="float32")
        T_add_1 = T.alloc_buffer([1, 1001, 1, 1], dtype="float32")
        compute = T.alloc_buffer([1, 1001, 1, 1], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 1280, 1, 1):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(placeholder[i0_1, i1_1, i2_1, i3_1])
                T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = placeholder[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 1001, 1, 1, 1280, 1, 1):
            with T.block("conv2d_nchw"):
                nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(pad_temp[nn, rc, yy + ry, xx + rx], placeholder_1[ff, rc, ry, rx])
                T.writes(conv2d_nchw[nn, ff, yy, xx])
                T.block_attr({"workload":["conv2d_nchw.cuda", ["TENSOR", [1, 1280, 1, 1], "float32"], ["TENSOR", [1001, 1280, 1, 1], "float32"], [1, 1], [0, 0, 0, 0], [1, 1], "float32"]})
                with T.init():
                    conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
                conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * placeholder_1[ff, rc, ry, rx]
        for i0, i1, i2, i3 in T.grid(1, 1001, 1, 1):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_nchw[ax0, ax1, ax2, ax3], placeholder_2[ax0, ax1, ax2, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = conv2d_nchw[ax0, ax1, ax2, ax3] + placeholder_2[ax0, ax1, ax2, ax3]
        for i0, i1, i2, i3 in T.grid(1, 1001, 1, 1):
            with T.block("T_divide"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[ax0, ax1, ax2, ax3], placeholder_3[()])
                T.writes(T_divide[ax0, ax1, ax2, ax3])
                T_divide[ax0, ax1, ax2, ax3] = T_add[ax0, ax1, ax2, ax3] / placeholder_3[()]
        for i0, i1, i2, i3 in T.grid(1, 1001, 1, 1):
            with T.block("T_round"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_divide[ax0, ax1, ax2, ax3])
                T.writes(T_round[ax0, ax1, ax2, ax3])
                T_round[ax0, ax1, ax2, ax3] = T.round(T_divide[ax0, ax1, ax2, ax3], dtype="float32")
        with T.block("compile_engine_const"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const[()])
            compile_engine_const[()] = T.float32(-51)
        for i0, i1, i2, i3 in T.grid(1, 1001, 1, 1):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_round[ax0, ax1, ax2, ax3], compile_engine_const[()])
                T.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = T_round[ax0, ax1, ax2, ax3] + compile_engine_const[()]
        for i0, i1, i2, i3 in T.grid(1, 1001, 1, 1):
            with T.block("compute"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add_1[i0_2, i1_2, i2_2, i3_2])
                T.writes(compute[i0_2, i1_2, i2_2, i3_2])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max(T.min(T_add_1[i0_2, i1_2, i2_2, i3_2], T.float32(127)), T.float32(-128))
        for i0_3, i1_3, i2_3, i3_3 in T.grid(1, 1001, 1, 1):
            with T.block("T_cast"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                T.reads(compute[ax0, ax1, ax2, ax3])
                T.writes(T_cast[ax0, ax1, ax2, ax3])
                T_cast[ax0, ax1, ax2, ax3] = T.cast(compute[ax0, ax1, ax2, ax3], "int8")

def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_tiles = [1, 1, 2, 512]
    j_tiles = [1, 512, 1, 2]
    k_tiles = [256, 4]
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _schedule_mobile_net_fused_conv2d(sch: Schedule):
    b0 = sch.get_block(name="pad_temp", func_name="main")
    b1 = sch.get_block(name="conv2d_nchw", func_name="main")
    b2 = sch.get_block(name="T_add", func_name="main")
    b3 = sch.get_block(name="T_divide", func_name="main")
    b4 = sch.get_block(name="T_round", func_name="main")
    b5 = sch.get_block(name="compile_engine_const", func_name="main")
    b6 = sch.get_block(name="T_add_1", func_name="main")
    b7 = sch.get_block(name="compute", func_name="main")
    b8 = sch.get_block(name="T_cast", func_name="main")
    b9 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l10, l11, l12, l13, l14, l15, l16 = sch.get_loops(block=b1)
    v17, v18, v19, v20, v21 = sch.sample_perfect_tile(
        loop=l10, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1]
    )
    l22, l23, l24, l25, l26 = sch.split(
        loop=l10, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True
    )
    v27, v28, v29, v30, v31 = sch.sample_perfect_tile(
        loop=l11, n=5, max_innermost_factor=64, decision=[13, 1, 77, 1, 1]
    )
    l32, l33, l34, l35, l36 = sch.split(
        loop=l11, factors=[v27, v28, v29, v30, v31], preserve_unit_iters=True
    )
    v37, v38, v39, v40, v41 = sch.sample_perfect_tile(
        loop=l12, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1]
    )
    l42, l43, l44, l45, l46 = sch.split(
        loop=l12, factors=[v37, v38, v39, v40, v41], preserve_unit_iters=True
    )
    v47, v48, v49, v50, v51 = sch.sample_perfect_tile(
        loop=l13, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1]
    )
    l52, l53, l54, l55, l56 = sch.split(
        loop=l13, factors=[v47, v48, v49, v50, v51], preserve_unit_iters=True
    )
    v57, v58, v59 = sch.sample_perfect_tile(
        loop=l14, n=3, max_innermost_factor=64, decision=[32, 1, 40]
    )
    l60, l61, l62 = sch.split(loop=l14, factors=[v57, v58, v59], preserve_unit_iters=True)
    v63, v64, v65 = sch.sample_perfect_tile(
        loop=l15, n=3, max_innermost_factor=64, decision=[1, 1, 1]
    )
    l66, l67, l68 = sch.split(loop=l15, factors=[v63, v64, v65], preserve_unit_iters=True)
    v69, v70, v71 = sch.sample_perfect_tile(
        loop=l16, n=3, max_innermost_factor=64, decision=[1, 1, 1]
    )
    l72, l73, l74 = sch.split(loop=l16, factors=[v69, v70, v71], preserve_unit_iters=True)
    sch.reorder(
        l22,
        l32,
        l42,
        l52,
        l23,
        l33,
        l43,
        l53,
        l24,
        l34,
        l44,
        l54,
        l60,
        l66,
        l72,
        l61,
        l67,
        l73,
        l25,
        l35,
        l45,
        l55,
        l62,
        l68,
        l74,
        l26,
        l36,
        l46,
        l56,
    )
    l75 = sch.fuse(l22, l32, l42, l52, preserve_unit_iters=True)
    sch.bind(loop=l75, thread_axis="blockIdx.x")
    l76 = sch.fuse(l23, l33, l43, l53, preserve_unit_iters=True)
    sch.bind(loop=l76, thread_axis="vthread.x")
    l77 = sch.fuse(l24, l34, l44, l54, preserve_unit_iters=True)
    sch.bind(loop=l77, thread_axis="threadIdx.x")
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(
        block_or_loop=b1, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024
    )
    b78 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b78, loop=l77, preserve_unit_loops=True)
    b79 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b79, loop=l72, preserve_unit_loops=True)
    l80, l81, l82, l83, l84, l85, l86, l87, l88, l89 = sch.get_loops(block=b79)
    l90 = sch.fuse(l86, l87, l88, l89, preserve_unit_iters=True)
    v91 = sch.sample_categorical(
        candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=2
    )
    sch.annotate(block_or_loop=b79, ann_key="meta_schedule.cooperative_fetch", ann_val=v91)
    b92 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b92, loop=l72, preserve_unit_loops=True)
    l93, l94, l95, l96, l97, l98, l99, l100, l101, l102 = sch.get_loops(block=b92)
    l103 = sch.fuse(l99, l100, l101, l102, preserve_unit_iters=True)
    v104 = sch.sample_categorical(
        candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1
    )
    sch.annotate(block_or_loop=b92, ann_key="meta_schedule.cooperative_fetch", ann_val=v104)
    sch.reverse_compute_inline(block=b8)
    sch.reverse_compute_inline(block=b7)
    sch.compute_inline(block=b5)
    sch.compute_inline(block=b4)
    sch.compute_inline(block=b3)
    sch.compute_inline(block=b2)
    sch.compute_inline(block=b0)
    v105 = sch.sample_categorical(
        candidates=[0, 16, 64, 512, 1024],
        probs=[
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
        ],
        decision=3,
    )
    sch.annotate(block_or_loop=b9, ann_key="meta_schedule.unroll_explicit", ann_val=v105)
    l106, l107, l108, l109 = sch.get_loops(block=b6)
    l110 = sch.fuse(l106, l107, l108, l109, preserve_unit_iters=True)
    v111 = sch.sample_categorical(
        candidates=[32, 64, 128, 256, 512],
        probs=[
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
        ],
        decision=0,
    )
    l112, l113 = sch.split(loop=l110, factors=[None, v111], preserve_unit_iters=True)
    sch.bind(loop=l112, thread_axis="blockIdx.x")
    sch.bind(loop=l113, thread_axis="threadIdx.x")
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b79, ann_key="meta_schedule.cooperative_fetch")
    l114, l115, l116, l117, l118, l119, l120 = sch.get_loops(block=b79)
    l121, l122 = sch.split(loop=l120, factors=[None, 77], preserve_unit_iters=True)
    sch.bind(loop=l122, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b92, ann_key="meta_schedule.cooperative_fetch")
    l123, l124, l125, l126, l127, l128, l129 = sch.get_loops(block=b92)
    l130, l131, l132 = sch.split(loop=l129, factors=[None, 77, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l132)
    sch.bind(loop=l131, thread_axis="threadIdx.x")
    b133 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b133, ann_key="meta_schedule.unroll_explicit")
    b134, b135, b136, b137, b138 = sch.get_child_blocks(b133)
    l139, l140, l141, l142, l143, l144, l145, l146 = sch.get_loops(block=b134)
    sch.annotate(block_or_loop=l139, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l139, ann_key="pragma_unroll_explicit", ann_val=1)
    l147, l148, l149, l150, l151, l152, l153, l154, l155 = sch.get_loops(block=b135)
    sch.annotate(block_or_loop=l147, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l147, ann_key="pragma_unroll_explicit", ann_val=1)
    (
        l156,
        l157,
        l158,
        l159,
        l160,
        l161,
        l162,
        l163,
        l164,
        l165,
        l166,
        l167,
        l168,
        l169,
        l170,
        l171,
        l172,
        l173,
        l174,
        l175,
    ) = sch.get_loops(block=b136)
    sch.annotate(block_or_loop=l156, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l156, ann_key="pragma_unroll_explicit", ann_val=1)
    l176, l177, l178, l179, l180, l181, l182 = sch.get_loops(block=b137)
    sch.annotate(block_or_loop=l176, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l176, ann_key="pragma_unroll_explicit", ann_val=1)
    l183, l184 = sch.get_loops(block=b138)
    sch.annotate(block_or_loop=l183, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l183, ann_key="pragma_unroll_explicit", ann_val=1)
    b185 = sch.get_block(name="conv2d_nchw", func_name="main")
    (
        l186,
        l187,
        l188,
        l189,
        l190,
        l191,
        l192,
        l193,
        l194,
        l195,
        l196,
        l197,
        l198,
        l199,
        l200,
        l201,
        l202,
        l203,
        l204,
        l205,
    ) = sch.get_loops(block=b185)
    b206 = sch.decompose_reduction(block=b185, loop=l189)

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,unused-variable


def _create_schedule(mod: IRModule, sch_fn: Callable[[Schedule], None]) -> Schedule:
    sch = tir.Schedule(mod=mod, debug_mask="all")
    sch_fn(sch)
    return sch


def _create_tmp_database(tmpdir: str) -> ms.database.JSONDatabase:
    path_workload = osp.join(tmpdir, "workloads.json")
    path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    return ms.database.JSONDatabase(path_workload, path_tuning_record)


def _equal_record(a: ms.database.TuningRecord, b: ms.database.TuningRecord):
    assert str(a.trace) == str(b.trace)
    assert str(a.run_secs) == str(b.run_secs)
    # AWAIT(@zxybazh): change to export after fixing "(bool)0"
    assert str(a.target) == str(b.target)
    assert tvm.ir.structural_equal(a.workload.mod, b.workload.mod)
    for arg0, arg1 in zip(a.args_info, b.args_info):
        assert str(arg0.as_json()) == str(arg1.as_json())


def test_meta_schedule_tuning_record_round_trip():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        new_record = ms.database.TuningRecord.from_json(record.as_json(), workload)
        _equal_record(record, new_record)


def test_meta_schedule_database_create():
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        assert osp.exists(database.path_workload)
        assert osp.exists(database.path_tuning_record)


def test_meta_schedule_database_has_workload():
    mod: IRModule = Matmul
    missing_mod: IRModule = MatmulRelu
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        assert len(database) == 1
        assert database.has_workload(mod)
        assert not database.has_workload(missing_mod)


def test_meta_schedule_database_add_entry():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        assert len(database) == 1
        (ret,) = database.get_top_k(workload, 3)
        _equal_record(ret, record)


def test_meta_schedule_database_missing():
    mod: IRModule = Matmul
    mod_2: IRModule = MatmulRelu
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        workload_2 = database.commit_workload(mod_2)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        ret = database.get_top_k(workload_2, 3)
        assert len(ret) == 0


def test_meta_schedule_database_sorting():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        token = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        records = [
            ms.database.TuningRecord(
                trace,
                token,
                [7.0, 8.0, 9.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 2.0, 3.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 5.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.1, 1.2, 600.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 100.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 9.0, 8.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
        ]
        for record in records:
            database.commit_tuning_record(record)
        ret = database.get_top_k(token, 2)
        assert len(ret) == 2
        try:
            _equal_record(ret[0], records[2])
            _equal_record(ret[1], records[1])
        except AssertionError:
            _equal_record(ret[0], records[1])
            _equal_record(ret[1], records[2])


def test_meta_schedule_database_reload():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        token = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        records = [
            ms.database.TuningRecord(
                trace,
                token,
                [7.0, 8.0, 9.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 2.0, 3.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 5.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
        ]
        for record in records:
            database.commit_tuning_record(record)
        new_database = ms.database.JSONDatabase(
            path_workload=database.path_workload,
            path_tuning_record=database.path_tuning_record,
        )
        token = new_database.commit_workload(mod)
        ret = new_database.get_top_k(token, 2)
        assert len(ret) == 2
        try:
            _equal_record(ret[0], records[2])
            _equal_record(ret[1], records[1])
        except AssertionError:
            _equal_record(ret[0], records[1])
            _equal_record(ret[1], records[2])


@tvm.testing.requires_gpu
def test_meta_schedule_trace_loading_datatype():
    mod = ms.default_config.mod(MobileNetConv2d)
    sch = Schedule(mod)
    _schedule_mobile_net_fused_conv2d(sch)
    args_info = [
        TensorInfo("float32", (1, 1280, 1, 1)),
        TensorInfo("float32", (1001, 1280, 1, 1)),
        TensorInfo("float32", (1, 1001, 1, 1)),
        TensorInfo("float32", ()),
        TensorInfo("int8", (1, 1001, 1, 1)),
    ]
    with tempfile.TemporaryDirectory() as work_dir:
        path_workload = osp.join(work_dir, "database_workload.json")
        path_tuning_record = osp.join(work_dir, "database_tuning_record.json")
        database = ms.database.JSONDatabase(path_workload, path_tuning_record)
        workload = database.commit_workload(mod)
        database.commit_tuning_record(
            TuningRecord(
                sch.trace,
                workload,
                run_secs=[1.0],
                args_info=args_info,
            )
        )
        del database
        reload_database = ms.database.JSONDatabase(path_workload, path_tuning_record)
        workload = reload_database.commit_workload(mod)
        records = reload_database.get_top_k(workload, 1)
        assert len(records) == 1
        (record,) = records
        reload_sch = Schedule(mod)
        record.trace.apply_to_schedule(reload_sch, remove_postproc=False)
        builder = LocalBuilder()
        runner = LocalRunner()
        (builder_res,) = builder.build(  # pylint: disable=unbalanced-tuple-unpacking
            [BuilderInput(reload_sch.mod, target=Target("nvidia/geforce-rtx-3070"))]
        )
        assert builder_res.error_msg is None
        (runner_future,) = runner.run(  # pylint: disable=unbalanced-tuple-unpacking
            [
                RunnerInput(
                    builder_res.artifact_path,
                    "cuda",
                    args_info,
                )
            ]
        )
        runner_res = runner_future.result()
        assert runner_res.error_msg is None


if __name__ == "__main__":
    tvm.testing.main()
