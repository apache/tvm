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
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@T.prim_func
def single_elementwise(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

# fmt: on
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def test_blockize_outer():
    @T.prim_func
    def after_blockize_outer(
        A: T.Buffer[(128, 128), "float32"],
        B: T.Buffer[(128, 128), "float32"],
    ) -> None:
        with T.block("blockized_B"):
            vio = T.axis.spatial(1, 0)
            vjo = T.axis.spatial(1, 0)
            for i, j in T.grid(128, 128):
                with T.block("B"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] * 2.0

    func = single_elementwise
    s = tir.Schedule(func, debug_mask="all")
    x, _ = s.get_loops(s.get_block("B"))
    s.blockize(x)
    tvm.ir.assert_structural_equal(s.mod["main"], after_blockize_outer)
    verify_trace_roundtrip(sch=s, mod=func)


def test_blockize_inner():
    @T.prim_func
    def after_blockize_inner(
        A: T.Buffer[(128, 128), "float32"],
        B: T.Buffer[(128, 128), "float32"],
    ) -> None:
        for i in T.serial(128):
            with T.block("blockized_B"):
                vi = T.axis.spatial(128, i)
                vjo = T.axis.spatial(1, 0)
                for j in T.serial(128):
                    with T.block("B"):
                        vj = T.axis.remap("S", [j])
                        B[vi, vj] = A[vi, vj] * 2.0

    func = single_elementwise
    s = tir.Schedule(func, debug_mask="all")
    _, y = s.get_loops(s.get_block("B"))
    s.blockize(y)
    tvm.ir.assert_structural_equal(s.mod["main"], after_blockize_inner)
    verify_trace_roundtrip(sch=s, mod=func)


def test_two_elementwise_blockize_reverse_compute_at():
    @T.prim_func
    def before_blockize_rca(
        A: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ) -> None:
        B = T.alloc_buffer([128, 128], dtype="float32")
        for i, j in T.grid(8, 8):
            with T.block("B_o"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                T.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [i_1, j_1])
                        T.reads(A[vi * 16 + vi_i, vj * 16 + vj_i])
                        T.writes(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        B[vi * 16 + vi_i, vj * 16 + vj_i] = A[vi * 16 + vi_i, vj * 16 + vj_i] * 2.0
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C"):
                    vi = T.axis.spatial(128, i * 16 + ax0)
                    vj = T.axis.spatial(128, j * 16 + ax1)
                    T.reads(B[vi, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = B[vi, vj] + 1.0

    @T.prim_func
    def after_blockize_rca(
        A: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ) -> None:
        B = T.alloc_buffer([128, 128], dtype="float32")
        for i, j in T.grid(8, 8):
            with T.block("B_o"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                T.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [i_1, j_1])
                        T.reads(A[vi * 16 + vi_i, vj * 16 + vj_i])
                        T.writes(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        B[vi * 16 + vi_i, vj * 16 + vj_i] = A[vi * 16 + vi_i, vj * 16 + vj_i] * 2.0
            with T.block("C_o"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                T.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("C"):
                        vi_i, vj_i = T.axis.remap("SS", [ax0, ax1])
                        T.reads(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        T.writes(C[vi * 16 + vi_i, vj * 16 + vj_i])
                        C[vi * 16 + vi_i, vj * 16 + vj_i] = B[vi * 16 + vi_i, vj * 16 + vj_i] + 1.0

    func = before_blockize_rca
    s = tir.Schedule(func, debug_mask="all")
    _, _, x, _ = s.get_loops(s.get_block("C"))
    s.blockize(x)
    tvm.ir.assert_structural_equal(s.mod["main"], after_blockize_rca)
    verify_trace_roundtrip(sch=s, mod=func)


def test_two_elementwise_blockize_compute_at():
    @T.prim_func
    def before_blockize_compute_at(
        A: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ) -> None:
        # body
        # with T.block("root")
        B = T.alloc_buffer([128, 128], dtype="float32")
        for i_0, j_0 in T.grid(8, 8):
            for ax0, ax1 in T.grid(16, 16):
                with T.block("B"):
                    vi = T.axis.spatial(128, i_0 * 16 + ax0)
                    vj = T.axis.spatial(128, j_0 * 16 + ax1)
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * 2.0
            with T.block("C_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                T.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("C"):
                        vi_i, vj_i = T.axis.remap("SS", [i_1, j_1])
                        T.reads(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        T.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + 1.0
                        )

    @T.prim_func
    def after_blockize_compute_at(
        A: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ) -> None:
        B = T.alloc_buffer([128, 128], dtype="float32")
        for i_0, j_0 in T.grid(8, 8):
            with T.block("B_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads(A[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                T.writes(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [ax0, ax1])
                        T.reads(A[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        T.writes(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            A[vi_o * 16 + vi_i, vj_o * 16 + vj_i] * 2.0
                        )
            with T.block("C_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                T.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("C"):
                        vi_i, vj_i = T.axis.remap("SS", [i_1, j_1])
                        T.reads(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        T.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + 1.0
                        )

    func = before_blockize_compute_at
    s = tir.Schedule(func, debug_mask="all")
    _, _, x, _ = s.get_loops(s.get_block("B"))
    s.blockize(x)
    tvm.ir.assert_structural_equal(s.mod["main"], after_blockize_compute_at)
    verify_trace_roundtrip(sch=s, mod=func)


def test_blockize_init_loops():
    @T.prim_func
    def rowsum(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128,), "float32"]) -> None:
        for k, i in T.grid(128, 128):
            with T.block("B"):
                vk, vi = T.axis.remap("RS", [k, i])
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]

    @T.prim_func
    def after_rowsum_blockize(
        A: T.Buffer[(128, 128), "float32"],
        B: T.Buffer[(128,), "float32"],
    ) -> None:
        with T.block("blockized_B"):
            vko = T.axis.R(1, 0)
            vio = T.axis.S(1, 0)
            with T.init():
                for i1 in T.serial(0, 128):
                    with T.block("B_init"):
                        vi_init = T.axis.S(128, i1)
                        B[vi_init] = T.float32(0)
            for i0, i1_1 in T.grid(128, 128):
                with T.block("B"):
                    vk, vi = T.axis.remap("RS", [i0, i1_1])
                    B[vi] = B[vi] + A[vi, vk]

    s = tir.Schedule(rowsum, debug_mask="all")
    k, _ = s.get_loops(s.get_block("B"))
    s.blockize(k)
    tvm.ir.assert_structural_equal(s.mod["main"], after_rowsum_blockize)
    verify_trace_roundtrip(sch=s, mod=rowsum)


def test_blockize_outer_int64_shape():
    @T.prim_func
    def single_elementwise_int64(
        A: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
        B: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
    ) -> None:
        for i0, j0, i1, j1 in T.grid(T.int64(1), T.int64(8), T.int64(16), T.int64(16)):
            with T.block("B"):
                vi = T.axis.S(T.int64(16), i0 * T.int64(16) + i1)
                vj = T.axis.S(T.int64(128), j0 * T.int64(16) + j1)
                B[vi, vj] = A[vi, vj] + 1.0

    @T.prim_func
    def after_single_elementwise_int64_blockize(
        A: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
        B: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
    ) -> None:
        for i0, j0 in T.grid(T.int64(1), T.int64(8)):
            with T.block("B_o"):
                vi_o = T.axis.spatial(T.int64(1), T.int64(0))
                vj_o = T.axis.spatial(T.int64(8), j0)
                for i1, j1 in T.grid(T.int64(16), T.int64(16)):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [i1, j1])
                        B[vi_i, vj_o * T.int64(16) + vj_i] = A[
                            vi_i, vj_o * T.int64(16) + vj_i
                        ] + T.float32(1)

    s = tir.Schedule(single_elementwise_int64, debug_mask="all")
    _, _, i1, _ = s.get_loops(s.get_block("B"))
    s.blockize(i1)
    tvm.ir.assert_structural_equal(s.mod["main"], after_single_elementwise_int64_blockize)
    verify_trace_roundtrip(sch=s, mod=single_elementwise_int64)


if __name__ == "__main__":
    tvm.testing.main()
