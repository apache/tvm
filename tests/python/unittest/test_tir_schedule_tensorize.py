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
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@T.prim_func
def mma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def mma_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

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
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(128, 128), "float32"],
    C: T.Buffer[(128, 128), "float32"],
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def tensorized_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)

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
                A_elem_offset = T.var("int32")
                B_elem_offset = T.var("int32")
                C_elem_offset = T.var("int32")
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
    A: T.Buffer[(16, 128, 128), "float32"],
    B: T.Buffer[(16, 128, 128), "float32"],
    C: T.Buffer[(16, 128, 128), "float32"],
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
    A: T.Buffer[(16, 128, 128), "float32"],
    B: T.Buffer[(16, 128, 128), "float32"],
    C: T.Buffer[(16, 128, 128), "float32"],
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
                A_elem_offset = T.var("int32")
                B_elem_offset = T.var("int32")
                C_elem_offset = T.var("int32")
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
    A: T.Buffer[(16, 128, 128), "float32"],
    B: T.Buffer[(16, 128, 128), "float32"],
    C: T.Buffer[(16, 128, 128), "float32"],
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
    A: T.Buffer[(16, 128, 128), "float32"],
    B: T.Buffer[(16, 128, 128), "float32"],
    C: T.Buffer[(16, 128, 128), "float32"],
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


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

tir.TensorIntrin.register("test_mma_intrin", mma_desc, mma_intrin)
tir.TensorIntrin.register("test_dot_product_intrin", dot_product_desc, dot_product_intrin)
tir.TensorIntrin.register("test_outer_product_intrin", outer_product_desc, outer_product_intrin)


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
    tvm.ir.assert_structural_equal(tensorized_matmul, s.mod["main"])
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
    tvm.ir.assert_structural_equal(tensorized_batch_matmul_mma, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=batch_matmul)


def test_tensorize_dot_product():
    func = batch_matmul
    s = tir.Schedule(func, debug_mask="all")
    C = s.get_block("update")
    _, _, _, k = s.get_loops(C)
    _, ki = s.split(k, factors=[None, 4])
    s.tensorize(ki, "test_dot_product_intrin")
    tvm.ir.assert_structural_equal(tensorized_batch_matmul_dot_product, s.mod["main"])
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
    tvm.ir.assert_structural_equal(tensorized_batch_matmul_outer_product, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
