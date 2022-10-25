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
import pytest

import tvm
import tvm.meta_schedule as ms
from tvm.script import tir as T
from tvm.tir import Schedule, floormod, floordiv
from tvm.target import Target


@tvm.script.ir_module
class Dense:
    @T.prim_func
    def main(p0: T.Buffer[(128, 128), "float32"], p1: T.Buffer[(128, 128), "float32"], T_matmul_NT: T.Buffer[(128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        for i0, i1, i2 in T.grid(128, 128, 128):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(p0[i, k], p1[j, k])
                T.writes(T_matmul_NT[i, j])
                T.block_attr({"layout_free_placeholders":[]})
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1[j, k]


@tvm.script.ir_module
class DenseAdd:
    @T.prim_func
    def main(p0: T.Buffer[(128, 128), "float32"], p1: T.Buffer[(128, 128), "float32"], T_add: T.Buffer[(128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        T_matmul_NT = T.alloc_buffer([128, 128], dtype="float32")
        compile_engine_const = T.alloc_buffer([], dtype="float32")
        for i0, i1, i2 in T.grid(128, 128, 128):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(p0[i, k], p1[j, k])
                T.writes(T_matmul_NT[i, j])
                T.block_attr({"layout_free_placeholders":[]})
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1[j, k]
        with T.block("compile_engine_const"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const[()])
            compile_engine_const[()] = T.float32(1)
        for i0, i1 in T.grid(128, 128):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_matmul_NT[ax0, ax1], compile_engine_const[()])
                T.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + compile_engine_const[()]


@tvm.script.ir_module
class DenseAdd_scheduled:
    @T.prim_func
    def main(p0: T.Buffer[(128, 128), "float32"], p1: T.Buffer[(128, 128), "float32"], T_add: T.Buffer[(128, 128), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        T_matmul_NT_global = T.alloc_buffer([128, 128], dtype="float32")
        p1_global = T.alloc_buffer([2, 128, 64], dtype="float32")
        for ax0, ax1 in T.grid(128, 128):
            with T.block("p1_global"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(p1[v0, v1])
                T.writes(p1_global[v0 // 64, v1, v0 % 64])
                T.block_attr({"meta_schedule.layout_rewrite_preproc":1})
                p1_global[v0 // 64, v1, v0 % 64] = p1[v0, v1]
        for i0_0_i1_0_fused_fused in T.parallel(4):
            for i0_1, i1_1 in T.grid(8, 1):
                for i0_2_init, i1_2_init, i0_3_init in T.grid(4, 1, 2):
                    for i1_3_fused_init in T.vectorized(64):
                        with T.block("T_matmul_NT_init"):
                            i = T.axis.spatial(128, i0_0_i1_0_fused_fused // 2 * 64 + i0_1 * 8 + i0_2_init * 2 + i0_3_init)
                            j = T.axis.spatial(128, i0_0_i1_0_fused_fused % 2 * 64 + i1_1 * 64 + i1_2_init * 64 + i1_3_fused_init)
                            T.reads()
                            T.writes(T_matmul_NT_global[i, j])
                            T.block_attr({"layout_free_placeholders":[], "meta_schedule.tiling_structure":"SSRSRS"})
                            T_matmul_NT_global[i, j] = T.float32(0)
                for i2_0, i0_2, i1_2, i2_1, i0_3 in T.grid(128, 4, 1, 1, 2):
                    for i1_3_fused in T.vectorized(64):
                        with T.block("T_matmul_NT_update"):
                            i = T.axis.spatial(128, i0_0_i1_0_fused_fused // 2 * 64 + i0_1 * 8 + i0_2 * 2 + i0_3)
                            j = T.axis.spatial(128, i0_0_i1_0_fused_fused % 2 * 64 + i1_1 * 64 + i1_2 * 64 + i1_3_fused)
                            k = T.axis.reduce(128, i2_0 + i2_1)
                            T.reads(T_matmul_NT_global[i, j], p0[i, k], p1_global[j // 64, k, j % 64])
                            T.writes(T_matmul_NT_global[i, j])
                            T.block_attr({"layout_free_placeholders":[], "meta_schedule.tiling_structure":"SSRSRS"})
                            T_matmul_NT_global[i, j] = T_matmul_NT_global[i, j] + p0[i, k] * p1_global[j // 64, k, j % 64]
            for ax0 in T.serial(64):
                for ax1_fused in T.vectorized(64):
                    with T.block("T_matmul_NT_global"):
                        v0 = T.axis.spatial(128, i0_0_i1_0_fused_fused // 2 * 64 + ax0)
                        v1 = T.axis.spatial(128, i0_0_i1_0_fused_fused % 2 * 64 + ax1_fused)
                        T.reads(T_matmul_NT_global[v0, v1])
                        T.writes(T_add[v0, v1])
                        T_add[v0, v1] = T_matmul_NT_global[v0, v1] + T.float32(1)


def test_dense_add_cpu():
    def apply_anchor_trace(sch: Schedule) -> None:
      b0 = sch.get_block(name="T_matmul_NT", func_name="main")
      b1 = sch.get_block(name="root", func_name="main")
      sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
      l2, l3, l4 = sch.get_loops(block=b0)
      v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64, decision=[2, 8, 4, 2])
      l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8], preserve_unit_iters=True)
      v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64, decision=[2, 1, 1, 64])
      l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16], preserve_unit_iters=True)
      v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64, decision=[128, 1])
      l23, l24 = sch.split(loop=l4, factors=[v21, v22], preserve_unit_iters=True)
      sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
      b25 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")
      sch.reverse_compute_at(block=b25, loop=l17, preserve_unit_loops=True, index=-1)
      sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=160)
      sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
      v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
      sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)
      sch.enter_postproc()
      b27 = sch.get_block(name="root", func_name="main")
      sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.parallel")
      sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.vectorize")
      sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.unroll_explicit")
      b28, b29 = sch.get_child_blocks(b27)
      l30, l31, l32, l33, l34, l35, l36, l37, l38, l39 = sch.get_loops(block=b28)
      l40 = sch.fuse(l30, l31, preserve_unit_iters=True)
      sch.parallel(loop=l40)
      l41 = sch.fuse(l39, preserve_unit_iters=True)
      sch.vectorize(loop=l41)
      l42, l43, l44 = sch.get_loops(block=b29)
      l45 = sch.fuse(l42, preserve_unit_iters=True)
      sch.parallel(loop=l45)
      l46 = sch.fuse(l44, preserve_unit_iters=True)
      sch.vectorize(loop=l46)
      b47 = sch.get_block(name="T_matmul_NT", func_name="main")
      l48, l49, l50, l51, l52, l53, l54, l55, l56 = sch.get_loops(block=b47)
      b57 = sch.decompose_reduction(block=b47, loop=l51)
      b58 = sch.get_block(name="T_matmul_NT_update", func_name="main")
      b59 = sch.cache_read(block=b58, read_buffer_index=2, storage_scope="global")
      sch.transform_layout(block=b58, buffer=("read", 2), index_map=tvm.tir.IndexMap.from_func(lambda i0, i1: (floordiv(i0, 64), i1, floormod(i0, 64),), inverse_index_map=lambda i0, i1, i2: (((i0*64) + i2), i1,)), pad_value=None)
      sch.annotate(block_or_loop=b59, ann_key="meta_schedule.layout_rewrite_preproc", ann_val=1)

    anchor_sch = Schedule(Dense)
    apply_anchor_trace(anchor_sch)
    trace = anchor_sch.trace

    sch = Schedule(DenseAdd)
    target = Target("llvm")

    ms.trace_apply.schedule_using_anchor_trace(sch, trace, target)

    tvm.ir.assert_structural_equal(DenseAdd_scheduled, sch.mod)
