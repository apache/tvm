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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    verify_trace_roundtrip,
    assert_structural_equal_ignore_global_symbol,
)

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@T.prim_func
def rowsum_blockized(a: T.handle, b: T.handle) -> None:
    B = T.match_buffer(b, [32, 4])
    A = T.match_buffer(a, [32, 4, 128])
    for i0, i2_0 in T.grid(32, 16):
        with T.block("blockized_B"):
            io, ko = T.axis.remap("SR", [i0, i2_0])
            with T.init():
                for i1 in T.serial(0, 4):
                    with T.block("B_init"):
                        ii_init = T.axis.S(4, i1)
                        B[io, ii_init] = 0.0
            for i1_1, i2_1 in T.grid(4, 8):
                with T.block("B"):
                    ii = T.axis.S(4, i1_1)
                    k = T.axis.R(128, ko * 8 + i2_1)
                    B[io, ii] = B[io, ii] + A[io, ii, k]


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_decompose0(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j in T.grid(128, 128):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = 0.0

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_decompose1(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [32, 4, 128], elem_offset=0, align=64, offset_factor=1)
    B = T.match_buffer(b, [32, 4], elem_offset=0, align=64, offset_factor=1)

    for i0 in T.serial(0, 32):
        with T.block("blockized_B_init"):
            io = T.axis.S(32, i0)
            for i1 in T.serial(0, 4):
                with T.block("B_init"):
                    ii = T.axis.S(4, i1)
                    B[io, ii] = T.float32(0)
    for i0, i2_o in T.grid(32, 16):
        with T.block("blockized_B_update"):
            io, ko = T.axis.remap("SR", [i0, i2_o])
            for i1, i2_i in T.grid(4, 8):
                with T.block("B"):
                    ii = T.axis.S(4, i1)
                    k = T.axis.R(128, ko * 8 + i2_i)
                    B[io, ii] = B[io, ii] + A[io, ii, k]


@T.prim_func
def matmul_decompose2(a: T.handle, b: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=64, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=64, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=64, offset_factor=1)

    for i0, i1 in T.grid(128, 128):
        with T.block("update_init"):
            vi_init, vj_init = T.axis.remap("SS", [i0, i1])
            C[vi_init, vj_init] = T.float32(0)
        for i2 in T.serial(0, 128):
            with T.block("update_update"):
                vi, vj, vk = T.axis.remap("SSR", [i0, i1, i2])
                C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@T.prim_func
def matmul_decompose_fail3(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, k, j in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_decompose4(a: T.handle, b: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=64, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=64, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=64, offset_factor=1)
    # body
    with T.block("root"):
        T.reads([])
        T.writes([])
        for i0_0 in T.serial(0, 16):
            for i0_1_init, i1_init in T.grid(8, 128):
                with T.block("update_init"):
                    vi_init = T.axis.S(128, i0_0 * 8 + i0_1_init)
                    vj_init = T.axis.S(128, i1_init)
                    C[vi_init, vj_init] = T.float32(0)
            for i0_1, i1, i2_0, i2_1 in T.grid(8, 128, 19, 7):
                with T.block("update_update"):
                    T.where((((i2_0 * 7) + i2_1) < 128))
                    vi = T.axis.S(128, i0_0 * 8 + i0_1)
                    vj = T.axis.S(128, i1)
                    vk = T.axis.R(128, i2_0 * 7 + i2_1)
                    C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@T.prim_func
def matmul_with_annotation(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            T.block_attr({"test_annotation": 1})
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_decompose_with_annotation(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j in T.grid(128, 128):
        with T.block("init"):
            T.block_attr({"test_annotation": 1})
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = 0.0

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            T.block_attr({"test_annotation": 1})
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def colsum_with_vectorization(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 32], dtype="float32")
    B = T.match_buffer(b, [32], dtype="float32")
    for k in T.serial(0, 128):
        for i in T.vectorized(0, 32):
            with T.block("B"):
                vk, vi = T.axis.remap("RS", [k, i])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vk, vi]


@T.prim_func
def colsum_decompose_with_vectorization(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 32], dtype="float32")
    B = T.match_buffer(b, [32], dtype="float32")
    for i in T.vectorized(0, 32):
        with T.block("B_init"):
            vi = T.axis.S(32, i)
            B[vi] = T.float32(0)
    for k in T.serial(0, 128):
        for i in T.vectorized(0, 32):
            with T.block("B"):
                vk, vi = T.axis.remap("RS", [k, i])
                B[vi] = B[vi] + A[vk, vi]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg

use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})


def test_reduction_decompose0(use_block_name):
    s = tir.Schedule(matmul, debug_mask="all")
    C = "update" if use_block_name else s.get_block("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, i)
    assert_structural_equal_ignore_global_symbol(matmul_decompose0, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


def test_reduction_decompose1(use_block_name):
    s = tir.Schedule(rowsum_blockized, debug_mask="all")
    blockized_B = "blockized_B" if use_block_name else s.get_block("blockized_B")
    io, ko = s.get_loops(blockized_B)
    s.decompose_reduction(blockized_B, io)
    assert_structural_equal_ignore_global_symbol(matmul_decompose1, s.mod["main"])
    verify_trace_roundtrip(s, mod=rowsum_blockized)


def test_reduction_decompose2():
    s = tir.Schedule(matmul, debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, k)
    assert_structural_equal_ignore_global_symbol(matmul_decompose2, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


def test_reduction_decompose3():
    s = tir.Schedule(matmul_decompose_fail3, debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    with pytest.raises(tvm.tir.ScheduleError):
        s.decompose_reduction(C, k)


def test_reduction_decompose4():
    s = tir.Schedule(matmul, debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    io, ii = s.split(i, factors=[16, 8])
    ko, ki = s.split(k, factors=[19, 7])
    s.decompose_reduction(C, ii)
    assert_structural_equal_ignore_global_symbol(matmul_decompose4, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


def test_reduction_decompose_with_annotation():
    s = tir.Schedule(matmul_with_annotation, debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, i)
    assert_structural_equal_ignore_global_symbol(matmul_decompose_with_annotation, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul_with_annotation)


def test_reduction_decompose_with_different_for_kind():
    s = tir.Schedule(colsum_with_vectorization, debug_mask="all")
    B = s.get_block("B")
    k, _ = s.get_loops(B)
    B_init = s.decompose_reduction(B, k)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], colsum_decompose_with_vectorization)
    assert s.get(B).same_as(s.get(s.get_block("B_update")))
    assert s.get(B_init).same_as(s.get(s.get_block("B_init")))
    verify_trace_roundtrip(s, mod=colsum_with_vectorization)


def test_decompose_reduction_ref_hash_check():
    mod = tvm.IRModule.from_expr(matmul.with_attr("global_symbol", "main"))
    mod_bak = mod
    hash_before = tvm.ir.structural_hash(mod_bak)
    s = tir.Schedule(mod["main"], debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, k)
    hash_after = tvm.ir.structural_hash(mod_bak)
    assert hash_before == hash_after


def test_decompose_reduction_nested_block():
    @T.prim_func
    def nested_block(A: T.Buffer((1, 64), "float32"), B: T.Buffer((1,), "float32")):
        for i, ko in T.grid(1, 2):
            with T.block("outer"):
                vi, vko = T.axis.remap("SR", [i, ko])
                C = T.alloc_buffer((32,), dtype="float32")
                with T.init():
                    B[vi] = T.float32(0)
                for ki in T.serial(32):
                    with T.block("inner_1"):
                        vki = T.axis.remap("S", [ki])
                        C[vki] = A[vi, vko * 32 + vki]
                for ki in T.serial(32):
                    with T.block("inner_2"):
                        vki = T.axis.remap("R", [ki])
                        B[vi] += C[vki]

    @T.prim_func
    def decomposed_nested_block(A: T.Buffer((1, 64), "float32"), B: T.Buffer((1,), "float32")):
        for i in range(1):
            with T.block("outer_init"):
                vi = T.axis.spatial(1, i)
                T.reads()
                T.writes(B[vi])
                B[vi] = T.float32(0)
            for ko in range(2):
                with T.block("outer_update"):
                    vi, vko = T.axis.remap("SR", [i, ko])
                    T.reads(B[vi], A[vi, vko * 32 : vko * 32 + 32])
                    T.writes(B[vi])
                    C = T.alloc_buffer((32,))
                    for ki in range(32):
                        with T.block("inner_1"):
                            vki = T.axis.spatial(32, ki)
                            T.reads(A[vi, vko * 32 + vki])
                            T.writes(C[vki])
                            C[vki] = A[vi, vko * 32 + vki]
                    for ki in range(32):
                        with T.block("inner_2"):
                            vki = T.axis.reduce(32, ki)
                            T.reads(B[vi], C[vki])
                            T.writes(B[vi])
                            B[vi] = B[vi] + C[vki]

    sch = tir.Schedule(nested_block, debug_mask="all")
    outer = sch.get_block("outer")
    i, ko = sch.get_loops(outer)
    sch.decompose_reduction(outer, ko)

    assert_structural_equal_ignore_global_symbol(decomposed_nested_block, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=nested_block)


class TestDecomposeReductionWithThreadBinding(tvm.testing.CompareBeforeAfter):
    def transform(self):
        def func(mod):
            sch = tir.Schedule(mod)
            t, _ = sch.get_loops("B")
            sch.decompose_reduction("B", t)
            return sch.mod

        return func

    @T.prim_func
    def before(A: T.Buffer((32, 16), "float32"), B: T.Buffer((32,), "float32")):
        for t in T.thread_binding(0, 32, thread="threadIdx.x"):
            for r in T.serial(16):
                with T.block("B"):
                    vi, vr = T.axis.remap("SR", [t, r])
                    with T.init():
                        B[vi] = T.float32(0)
                    B[vi] += A[vi, vr]

    @T.prim_func
    def expected(A: T.Buffer((32, 16), "float32"), B: T.Buffer((32,), "float32")):
        for t_init in T.thread_binding(0, 32, thread="threadIdx.x"):
            with T.block("B_init"):
                vi = T.axis.remap("S", [t_init])
                B[vi] = T.float32(0)
        for t in T.thread_binding(0, 32, thread="threadIdx.x"):
            for r in T.serial(16):
                with T.block("B"):
                    vi, vr = T.axis.remap("SR", [t, r])
                    B[vi] += A[vi, vr]


if __name__ == "__main__":
    tvm.testing.main()
