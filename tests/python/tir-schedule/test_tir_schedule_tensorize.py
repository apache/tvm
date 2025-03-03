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

import pytest
import tvm
import tvm.testing
from tvm import te, tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm.tir.tensor_intrin.arm_cpu import (
    DP4A_S8S8S32_INTRIN,
    DP4A_U8U8U32_INTRIN,
    DP4A_U8S8S32_INTRIN,
    DP4A_S8U8S32_INTRIN,
    ARM_DOT_4x4_i8_NEON_INTRIN,
    ARM_DOT_4x4_i8_SDOT_INTRIN,
)
from tvm.tir.tensor_intrin.rocm import AMDGPU_SDOT4_INTRIN
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN, AVX512_DOT_16x4_INTRIN
from tvm.tir.tensor_intrin.hexagon import VRMPY_u8u8i32_INTRIN, VDMPY_i16i16i32_INTRIN

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@T.prim_func
def mma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=64, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=64, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=64, offset_factor=1)

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def mma_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=64, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=64, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=64, offset_factor=1)

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(
            T.tvm_mma_sync(
                C.data,
                C.elem_offset // 256,
                A.data,
                A.elem_offset // 256,
                B.data,
                B.elem_offset // 256,
                C.data,
                C.elem_offset // 256,
                dtype="handle",
            )
        )


@T.prim_func
def dot_product_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,))
    B = T.match_buffer(b, (4,))
    C = T.match_buffer(c, ())

    with T.block("root"):
        T.reads(C[()], A[0 : 4], B[0 : 4])
        T.writes(C[()])
        for i in range(0, 4):
            with T.block("update"):
                vi = T.axis.remap("R", [i])
                C[()] = C[()] + A[vi] * B[vi]


@T.prim_func
def dot_product_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), offset_factor=1)
    B = T.match_buffer(b, (4,), offset_factor=1)
    C = T.match_buffer(c, (), offset_factor=1)

    with T.block("root"):
        T.reads(C[()], A[0 : 4], B[0 : 4])
        T.writes(C[()])
        T.evaluate(
            T.call_extern(
                "vec4add",
                C.data,
                C.elem_offset,
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int32",
            )
        )


@T.prim_func
def dot_product_intrin_annotated(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), offset_factor=1)
    B = T.match_buffer(b, (4,), offset_factor=1)
    C = T.match_buffer(c, (), offset_factor=1)

    with T.block("root"):
        T.reads(C[()], A[0 : 4], B[0 : 4])
        T.writes(C[()])
        T.block_attr({"test_annotation": True})
        T.evaluate(
            T.call_extern(
                "vec4add",
                C.data,
                C.elem_offset,
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int32",
            )
        )


@T.prim_func
def outer_product_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 1), offset_factor=1)
    B = T.match_buffer(b, (16, 1), offset_factor=1)
    C = T.match_buffer(c, (16, 16), offset_factor=1)

    with T.block("root"):
        T.reads(
            C[0 : 16, 0 : 16],
            A[0 : 16, 0 : 1],
            B[0 : 16, 0 : 1],
        )
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("update"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C[vii, vjj] + A[vii, 0] * B[vjj, 0]


@T.prim_func
def outer_product_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 1), offset_factor=1)
    B = T.match_buffer(b, (16, 1), offset_factor=1)
    C = T.match_buffer(c, (16, 16), offset_factor=1)

    with T.block("root"):
        T.reads(
            C[0 : 16, 0 : 16],
            A[0 : 16, 0 : 1],
            B[0 : 16, 0 : 1],
        )
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(
            T.call_extern(
                "outer_product",
                C.data,
                C.elem_offset,
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int32",
            )
        )


@T.prim_func
def matmul(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def tensorized_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=64, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=64, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=64, offset_factor=1)

    for i_outer, j_outer in T.grid(8, 8):
        for i_inner_init, j_inner_init in T.grid(16, 16):
            with T.block("init"):
                vi_init = T.axis.S(128, ((i_outer * 16) + i_inner_init))
                vj_init = T.axis.S(128, ((j_outer * 16) + j_inner_init))
                C[vi_init, vj_init] = T.float32(0)
        for k_outer in T.grid(8):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i_outer, j_outer, k_outer])
                T.reads(
                    [
                        C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                        A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                        B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    ]
                )
                T.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                A_elem_offset = T.int32()
                B_elem_offset = T.int32()
                C_elem_offset = T.int32()
                A_sub = T.match_buffer(
                    A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    elem_offset=A_elem_offset,
                )
                B_sub = T.match_buffer(
                    B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    elem_offset=B_elem_offset,
                )
                C_sub = T.match_buffer(
                    C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    [16, 16],
                    elem_offset=C_elem_offset,
                )
                T.evaluate(
                    T.tvm_mma_sync(
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        A_sub.data,
                        T.floordiv(A_sub.elem_offset, 256),
                        B_sub.data,
                        T.floordiv(B_sub.elem_offset, 256),
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        dtype="handle",
                    )
                )


@T.prim_func
def batch_matmul(
    A: T.Buffer((16, 128, 128), "float32"),
    B: T.Buffer((16, 128, 128), "float32"),
    C: T.Buffer((16, 128, 128), "float32"),
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            C[vn, vi, vj] = T.float32(0)

    for n, i, j, k in T.grid(16, 128, 128, 128):
        with T.block("update"):
            vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
            C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@T.prim_func
def tensorized_batch_matmul_mma(
    A: T.Buffer((16, 128, 128), "float32"),
    B: T.Buffer((16, 128, 128), "float32"),
    C: T.Buffer((16, 128, 128), "float32"),
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            T.reads()
            T.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
    for n in range(0, 16):
        for i, j, k in T.grid(8, 8, 8):
            with T.block("update"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                T.reads(
                    C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    A[vn : vn + 1, vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    B[vn : vn + 1, vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                )
                T.writes(C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                A_elem_offset = T.int32()
                B_elem_offset = T.int32()
                C_elem_offset = T.int32()
                A_sub = T.match_buffer(
                    A[vn : vn + 1, vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    (16, 16),
                    elem_offset=A_elem_offset,
                )
                B_sub = T.match_buffer(
                    B[vn : vn + 1, vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    (16, 16),
                    elem_offset=B_elem_offset,
                )
                C_sub = T.match_buffer(
                    C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    (16, 16),
                    elem_offset=C_elem_offset,
                )
                T.evaluate(
                    T.tvm_mma_sync(
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        A_sub.data,
                        T.floordiv(A_sub.elem_offset, 256),
                        B_sub.data,
                        T.floordiv(B_sub.elem_offset, 256),
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        dtype="handle",
                    )
                )


@T.prim_func
def tensorized_batch_matmul_dot_product(
    A: T.Buffer((16, 128, 128), "float32"),
    B: T.Buffer((16, 128, 128), "float32"),
    C: T.Buffer((16, 128, 128), "float32"),
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            T.reads()
            T.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
    for n, i, j, k_0 in T.grid(16, 128, 128, 32):
        with T.block("blockized_update"):
            vn, vi, vj, vko = T.axis.remap("SSSR", [n, i, j, k_0])
            T.reads(
                C[vn, vi, vj], A[vn, vi, vko * 4 : vko * 4 + 4], B[vn, vj, vko * 4 : vko * 4 + 4]
            )
            T.writes(C[vn, vi, vj])
            A_1 = T.match_buffer(
                A[vn, vi, vko * 4 : vko * 4 + 4], [4], dtype="float32", offset_factor=1
            )
            B_1 = T.match_buffer(
                B[vn, vj, vko * 4 : vko * 4 + 4], [4], dtype="float32", offset_factor=1
            )
            C_1 = T.match_buffer(C[vn, vi, vj], [], dtype="float32", offset_factor=1)
            T.evaluate(
                T.call_extern(
                    "vec4add",
                    C_1.data,
                    C_1.elem_offset,
                    A_1.data,
                    A_1.elem_offset,
                    B_1.data,
                    B_1.elem_offset,
                    dtype="int32",
                )
            )


@T.prim_func
def tensorized_batch_matmul_outer_product(
    A: T.Buffer((16, 128, 128), "float32"),
    B: T.Buffer((16, 128, 128), "float32"),
    C: T.Buffer((16, 128, 128), "float32"),
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            T.reads()
            T.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
    for n, i_0, j_0, k in T.grid(16, 8, 8, 128):
        with T.block("blockized_update"):
            vn, vio, vjo, vk = T.axis.remap("SSSR", [n, i_0, j_0, k])
            T.reads(
                C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                A[vn, vio * 16 : vio * 16 + 16, vk],
                B[vn, vjo * 16 : vjo * 16 + 16, vk],
            )
            T.writes(C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            A_1 = T.match_buffer(A[vn, vio * 16 : vio * 16 + 16, vk], [16, 1], dtype="float32", offset_factor=1)
            B_1 = T.match_buffer(B[vn, vjo * 16 : vjo * 16 + 16, vk], [16, 1], dtype="float32", offset_factor=1
            )
            C_1 = T.match_buffer(
                C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16], [16, 16], dtype="float32", offset_factor=1
            )
            T.evaluate(
                T.call_extern("outer_product", C_1.data, C_1.elem_offset, A_1.data, A_1.elem_offset,
                              B_1.data, B_1.elem_offset, dtype="int32"
                )
            )


@T.prim_func
def annotated_mma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=64, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=64, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=64, offset_factor=1)

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                T.block_attr({"test_annotation": True})
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def annotated_matmul(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.block_attr({"test_annotation": True})
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def annotated_tensorized_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=64, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=64, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=64, offset_factor=1)

    for i_outer, j_outer in T.grid(8, 8):
        for i_inner_init, j_inner_init in T.grid(16, 16):
            with T.block("init"):
                vi_init = T.axis.S(128, ((i_outer * 16) + i_inner_init))
                vj_init = T.axis.S(128, ((j_outer * 16) + j_inner_init))
                T.block_attr({"test_annotation": True})
                C[vi_init, vj_init] = T.float32(0)
        for k_outer in T.grid(8):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i_outer, j_outer, k_outer])
                T.reads(
                    [
                        C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                        A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                        B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    ]
                )
                T.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                A_elem_offset = T.int32()
                B_elem_offset = T.int32()
                C_elem_offset = T.int32()
                A_sub = T.match_buffer(
                    A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    elem_offset=A_elem_offset,
                )
                B_sub = T.match_buffer(
                    B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    elem_offset=B_elem_offset,
                )
                C_sub = T.match_buffer(
                    C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    [16, 16],
                    elem_offset=C_elem_offset,
                )
                T.evaluate(
                    T.tvm_mma_sync(
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        A_sub.data,
                        T.floordiv(A_sub.elem_offset, 256),
                        B_sub.data,
                        T.floordiv(B_sub.elem_offset, 256),
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        dtype="handle",
                    )
                )


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

tir.TensorIntrin.register("test_mma_intrin", mma_desc, mma_intrin)
tir.TensorIntrin.register("test_annotated_mma_intrin", annotated_mma_desc, mma_intrin)
tir.TensorIntrin.register("test_dot_product_intrin", dot_product_desc, dot_product_intrin)
tir.TensorIntrin.register("test_outer_product_intrin", outer_product_desc, outer_product_intrin)
tir.TensorIntrin.register("test_dot_product_intrin_annotated", dot_product_desc, dot_product_intrin_annotated)


def test_tensorize_matmul():
    func = matmul
    # schedule
    s = tir.Schedule(func, debug_mask="all")
    update = s.get_block("update")
    i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)
    s.tensorize(ii, "test_mma_intrin")
    assert_structural_equal_ignore_global_symbol(tensorized_matmul, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_tensorize_batch_matmul():
    func = batch_matmul
    s = tir.Schedule(func, debug_mask="all")
    update = s.get_block("update")
    _, i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.tensorize(ii, "test_mma_intrin")
    assert_structural_equal_ignore_global_symbol(tensorized_batch_matmul_mma, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=batch_matmul)


def test_tensorize_dot_product():
    func = batch_matmul
    s = tir.Schedule(func, debug_mask="all")
    C = s.get_block("update")
    _, _, _, k = s.get_loops(C)
    _, ki = s.split(k, factors=[None, 4])
    s.tensorize(ki, "test_dot_product_intrin")
    assert_structural_equal_ignore_global_symbol(tensorized_batch_matmul_dot_product, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_tensorize_outer_product():
    func = batch_matmul
    s = tir.Schedule(func, debug_mask="all")
    C = s.get_block("update")
    _, i, j, k = s.get_loops(C)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    s.reorder(io, jo, k, ii, ji)
    s.tensorize(ii, "test_outer_product_intrin")
    assert_structural_equal_ignore_global_symbol(tensorized_batch_matmul_outer_product, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_tensorize_with_annotation():
    func = annotated_matmul
    s = tir.Schedule(func, debug_mask="all")
    update = s.get_block("update")
    i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)
    s.tensorize(ii, "test_annotated_mma_intrin")
    assert_structural_equal_ignore_global_symbol(annotated_tensorized_matmul, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_tensorize_intrinsic_with_annotation():
    func = matmul
    s = tir.Schedule(func, debug_mask="all")
    update = s.get_block("update")
    _, _, k = s.get_loops(update)
    ko, ki = s.split(k, factors=[None, 4])
    s.decompose_reduction(update, ko)
    s.tensorize(ki, "test_dot_product_intrin_annotated")

    b = s.get(s.get_block("update_update_o"))
    assert b.annotations["test_annotation"] == T.bool(True)
    verify_trace_roundtrip(sch=s, mod=func)


def get_matmul_packed(m, n, k, lhs_type, rhs_dtype="int8"):
    X = te.placeholder((m, k), name="X", dtype=lhs_type)
    W = te.placeholder((n, k), name="W", dtype=rhs_dtype)

    ak = te.reduce_axis((0, k), name="k")
    matmul = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32") * W[j, ak].astype("int32"),
            axis=ak,
        ),
        name="compute",
    )

    return te.create_prim_func([X, W, matmul])


def tensorize_16x4_test(intrin=VNNI_DOT_16x4_INTRIN):
    m, n, k = 128, 128, 128

    func = get_matmul_packed(m, n, k, "uint8")

    sch = tir.Schedule(func, debug_mask="all")
    block = sch.get_block("compute")
    sch.transform_layout(block, "W", lambda i, j: [i//16, j//4, i%16, j%4])
    _, j, k = sch.get_loops(block)

    _, ji = sch.split(j, factors=[None, 16])
    ko, ki = sch.split(k, factors=[None, 4])
    sch.reorder(ko, ji, ki)

    sch.decompose_reduction(block, ko)
    sch.tensorize(ji, intrin)

    verify_trace_roundtrip(sch=sch, mod=func)


def test_tensorize_vnni():
    tensorize_16x4_test()


def test_tensorize_avx512():
    tensorize_16x4_test(AVX512_DOT_16x4_INTRIN)


def test_tensorize_arm_dot():
    m, n, k = 128, 128, 128

    func = get_matmul_packed(m, n, k, "int8")

    for intrin in [ARM_DOT_4x4_i8_SDOT_INTRIN, ARM_DOT_4x4_i8_NEON_INTRIN]:
        sch = tir.Schedule(func, debug_mask="all")
        block = sch.get_block("compute")
        sch.transform_layout(block, "W", lambda i, j: [i//4, j//4, i%4, j%4])
        _, j, k = sch.get_loops(block)

        _, ji = sch.split(j, factors=[None, 4])
        ko, ki = sch.split(k, factors=[None, 4])
        sch.reorder(ko, ji, ki)

        sch.decompose_reduction(block, ko)
        sch.tensorize(ji, intrin)

        verify_trace_roundtrip(sch=sch, mod=func)


def test_tensorize_vrmpy():
    m, n, k = 128, 128, 128

    func = get_matmul_packed(m, n, k, "uint8", "uint8")

    sch = tir.Schedule(func, debug_mask="all")
    block = sch.get_block("compute")
    sch.transform_layout(block, "W", lambda i, j: [i//32, j//4, i%32, j%4])
    _, j, k = sch.get_loops(block)

    _, ji = sch.split(j, factors=[None, 32])
    ko, ki = sch.split(k, factors=[None, 4])
    sch.reorder(ko, ji, ki)

    sch.decompose_reduction(block, ko)
    sch.tensorize(ji, VRMPY_u8u8i32_INTRIN)

    verify_trace_roundtrip(sch=sch, mod=func)


def test_tensorize_vdmpy():
    m, n, k = 128, 128, 128

    func = get_matmul_packed(m, n, k, "int16", "int16")

    sch = tir.Schedule(func, debug_mask="all")
    block = sch.get_block("compute")
    sch.transform_layout(block, "W", lambda i, j: [i//32, j//2, i%32, j%2])
    _, j, k = sch.get_loops(block)

    _, ji = sch.split(j, factors=[None, 32])
    ko, ki = sch.split(k, factors=[None, 2])
    sch.reorder(ko, ji, ki)

    sch.decompose_reduction(block, ko)
    sch.tensorize(ji, VDMPY_i16i16i32_INTRIN)

    verify_trace_roundtrip(sch=sch, mod=func)


def test_tensorize_dp4a():
    # pylint: disable=too-many-locals
    def _test_intrin(dtype_a, dtype_b, dtype_c, intrin):
        m, n, k = 128, 128, 128
        X = te.placeholder((m, k), name="X", dtype=dtype_a)
        W = te.placeholder((n, k), name="W", dtype=dtype_b)
        ak = te.reduce_axis((0, k), name="k")

        matmul = te.compute(
            (m, n),
            lambda i, j: te.sum(
                X[i, ak].astype(dtype_c) * W[j, ak].astype(dtype_c),
                axis=ak,
            ),
            name="compute",
        )

        func = te.create_prim_func([X, W, matmul])

        sch = tir.Schedule(func, debug_mask="all")
        block = sch.get_block("compute")
        i, j, k = sch.get_loops(block)

        by, ty, yi = sch.split(i, factors=sch.sample_perfect_tile(i, n=3))
        bx, tx, xi = sch.split(j, factors=sch.sample_perfect_tile(j, n=3))
        ko, ki = sch.split(k, [None, 4])
        ko, kt = sch.split(ko, factors=sch.sample_perfect_tile(ko, n=2))

        sch.reorder(by, bx, ty, tx, yi, xi)
        CC = sch.cache_write(block, 0, "local")
        sch.reverse_compute_at(CC, tx)

        def fetch_to_shared(block, idx):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, ko, True)
            return block_read

        fetch_to_shared(block, 0)
        fetch_to_shared(block, 1)

        sch.decompose_reduction(block, ko)
        sch.tensorize(ki, intrin)

        verify_trace_roundtrip(sch=sch, mod=func)

    for args in [
        ("int8", "int8", "int32", AMDGPU_SDOT4_INTRIN),
        ("int8", "int8", "int32", DP4A_S8S8S32_INTRIN),
        ("int8", "uint8", "int32", DP4A_S8U8S32_INTRIN),
        ("uint8", "int8", "int32", DP4A_U8S8S32_INTRIN),
        ("uint8", "uint8", "uint32", DP4A_U8U8U32_INTRIN),
    ]:
        _test_intrin(*args)


def test_tensor_intrin_look_up():
    intrin_name = 'non_existent_intrin'
    assert tir.TensorIntrin.get(intrin_name, allow_missing=True) is None
    with pytest.raises(ValueError):
        tir.TensorIntrin.get(intrin_name)


def test_tensorize_matmul_mixed_dtype():
    # fmt: off
    @T.prim_func
    def matmul_int64_shape(
        A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        C: T.Buffer((T.int64(128), T.int64(128)), "float32")
    ) -> None:
        for i_0, j_0 in T.grid(T.int64(8), T.int64(8)):
            for i_1_init, j_1_init in T.grid(T.int64(16), T.int64(16)):
                with T.block("init"):
                    vi = T.axis.spatial(T.int64(128), i_0 * T.int64(16) + i_1_init)
                    vj = T.axis.spatial(T.int64(128), j_0 * T.int64(16) + j_1_init)
                    C[vi, vj] = T.float32(0)
            for k_0, i_1, j_1, k_1 in T.grid(T.int64(8), T.int64(16), T.int64(16), T.int64(16)):
                with T.block("update"):
                    vi = T.axis.spatial(T.int64(128), i_0 * T.int64(16) + i_1)
                    vj = T.axis.spatial(T.int64(128), j_0 * T.int64(16) + j_1)
                    vk = T.axis.reduce(T.int64(128), k_0 * T.int64(16) + k_1)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    @T.prim_func
    def tensorized_matmul_int64_shape(
        A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        C: T.Buffer((T.int64(128), T.int64(128)), "float32")
    ) -> None:
        for i_outer, j_outer in T.grid(T.int64(8), T.int64(8)):
            for i_inner_init, j_inner_init in T.grid(T.int64(16), T.int64(16)):
                with T.block("init"):
                    vi = T.axis.spatial(T.int64(128), i_outer * T.int64(16) + i_inner_init)
                    vj = T.axis.spatial(T.int64(128), j_outer * T.int64(16) + j_inner_init)
                    C[vi, vj] = T.float32(0)
            for k_outer in T.grid(T.int64(8)):
                with T.block("update"):
                    vi, vj, vk = T.axis.remap("SSR", [i_outer, j_outer, k_outer])
                    T.reads(
                        [
                            C[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vj * T.int64(16) : vj * T.int64(16) + T.int64(16)],
                            A[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vk * T.int64(16) : vk * T.int64(16) + T.int64(16)],
                            B[vj * T.int64(16) : vj * T.int64(16) + T.int64(16), vk * T.int64(16) : vk * T.int64(16) + T.int64(16)],
                        ]
                    )
                    T.writes(C[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vj * T.int64(16) : vj * T.int64(16) + T.int64(16)])
                    A_elem_offset = T.int64()
                    B_elem_offset = T.int64()
                    C_elem_offset = T.int64()
                    A_sub = T.match_buffer(
                        A[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vk * T.int64(16) : vk * T.int64(16) + T.int64(16)],
                        [T.int64(16), T.int64(16)],
                        elem_offset=A_elem_offset,
                    )
                    B_sub = T.match_buffer(
                        B[vj * T.int64(16) : vj * T.int64(16) + T.int64(16), vk * T.int64(16) : vk * T.int64(16) + T.int64(16)],
                        [T.int64(16), T.int64(16)],
                        elem_offset=B_elem_offset,
                    )
                    C_sub = T.match_buffer(
                        C[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vj * T.int64(16) : vj * T.int64(16) + T.int64(16)],
                        [T.int64(16), T.int64(16)],
                        elem_offset=C_elem_offset,
                    )
                    T.evaluate(
                        T.tvm_mma_sync(
                            C_sub.data,
                            T.floordiv(C_sub.elem_offset, T.int64(256)),
                            A_sub.data,
                            T.floordiv(A_sub.elem_offset, T.int64(256)),
                            B_sub.data,
                            T.floordiv(B_sub.elem_offset, T.int64(256)),
                            C_sub.data,
                            T.floordiv(C_sub.elem_offset, T.int64(256)),
                            dtype="handle",
                        )
                    )
    # fmt: on

    s = tir.Schedule(matmul_int64_shape, debug_mask="all")
    update = s.get_block("update")
    ii = s.get_loops(update)[-3]
    s.tensorize(ii, "test_mma_intrin")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], tensorized_matmul_int64_shape)
    verify_trace_roundtrip(sch=s, mod=matmul_int64_shape)

def _tir_packed_int_to_int_to_float(storage_nbit: int):
    storage_dtype = "int" + str(storage_nbit)

    def f_convert(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype
        mask = tir.const((1 << nbit) - 1, "int32")
        unextended = (val >> (pos.astype("int32") * tir.const(nbit, "int32"))) & mask
        return tir.Cast(dtype, (unextended << tir.const(32 - nbit, "int32")) >> tir.const(32 - nbit, "int32"))

    return f_convert

@T.prim_func
def decode_i4s_to_f16_desc(compressed: T.handle, decompressed: T.handle) -> None:
    Compressed = T.match_buffer(
        compressed,
        [
            1,
        ],
        dtype="int32",
        scope="local",
    )
    Decompressed = T.match_buffer(
        decompressed,
        [
            8,
        ],
        dtype="float16",
        scope="local",
    )

    with T.block("root"):
        T.reads(Compressed[0:1])
        T.writes(Decompressed[0:8])
        for i in T.grid(8):
            with T.block("decode"):
                vi = T.axis.remap("S", [i])
                Decompressed[vi] = _tir_packed_int_to_int_to_float(32)(
                    4,
                    Compressed[vi // 8],
                    vi % 8,
                    dtype="float16",
                )

@T.prim_func
def decode_i4s_to_f16_impl(compressed: T.handle, decompressed: T.handle) -> None:
    Compressed = T.match_buffer(
        compressed,
        [
            1,
        ],
        dtype="int32",
        scope="local",
    )
    Decompressed = T.match_buffer(
        decompressed,
        [
            8,
        ],
        dtype="float16",
        scope="local",
    )

    with T.block("root"):
        T.reads(Compressed[0:1])
        T.writes(Decompressed[0:8])
        T.call_extern(
            "handle",
            "test_decode_i4s_to_f16",
            Compressed.data,
            Decompressed.data,
            8,
        )

tir.TensorIntrin.register("test_decode_i4s_to_f16_intrin", decode_i4s_to_f16_desc, decode_i4s_to_f16_impl)

def test_tensorize_arith_simplification():
    # fmt: off
    @T.prim_func
    def decode_i4s_to_int32_to_f16():
        B_decode_local = T.alloc_buffer((16384, 16384), "float16", scope="local")
        B_local = T.alloc_buffer((16384, 2048), "int32", scope="local")
        for ax0_0 in T.thread_binding(8192, thread="blockIdx.x"):
            for ax0_1 in T.thread_binding(2, thread="threadIdx.y"):
                for ax1_0 in range(32):
                    for ax1_1 in T.thread_binding(64, thread="threadIdx.x"):
                        for ax0, ax1 in T.grid(1, 8):
                            with T.block("B_decode_local"):
                                v0 = T.axis.spatial(16384, ax0_0 * 2 + ax0_1 + ax0)
                                v1 = T.axis.spatial(16384, ax1_0 * 512 + ax1_1 * 8 + ax1)
                                T.reads(B_local[v0, v1 // 8])
                                T.writes(B_decode_local[v0, v1])
                                B_decode_local[v0, v1] = T.Cast("float16", T.shift_right(T.shift_left(T.bitwise_and(T.shift_right(B_local[v0, v1 // 8], v1 % 8 * 4), 15), 28), 28))

    @T.prim_func
    def tensorized_decode_i4s_to_int32_to_f16():
        B_decode_local = T.alloc_buffer((16384, 16384), "float16", scope="local")
        B_local = T.alloc_buffer((16384, 2048), "int32", scope="local")
        for ax0_0 in T.thread_binding(8192, thread="blockIdx.x"):
            for ax0_1 in T.thread_binding(2, thread="threadIdx.y"):
                for ax1_0 in range(32):
                    for ax1_1 in T.thread_binding(64, thread="threadIdx.x"):
                        for ax0 in range(1):
                            with T.block("B_decode_local_o"):
                                v0_o = T.axis.spatial(16384, ax0_0 * 2 + ax0_1 + ax0)
                                v1_o = T.axis.spatial(2048, ax1_0 * 64 + ax1_1)
                                T.reads(B_local[v0_o, v1_o])
                                T.writes(B_decode_local[v0_o, v1_o * 8:v1_o * 8 + 8])
                                Compressed = T.match_buffer(B_local[v0_o, v1_o], (1,), "int32", scope="local")
                                Decompressed = T.match_buffer(B_decode_local[v0_o, v1_o * 8:v1_o * 8 + 8], (8,), "float16", scope="local")
                                T.call_extern("handle", "test_decode_i4s_to_f16", Compressed.data, Decompressed.data, 8)

    s = tir.Schedule(decode_i4s_to_int32_to_f16, debug_mask="all")
    update = s.get_block("B_decode_local")
    ii = s.get_loops(update)[-1]
    s.tensorize(ii, "test_decode_i4s_to_f16_intrin")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], tensorized_decode_i4s_to_int32_to_f16)
    verify_trace_roundtrip(sch=s, mod=decode_i4s_to_int32_to_f16)


if __name__ == "__main__":
    tvm.testing.main()
