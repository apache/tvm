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
import tvm
from tvm import tir, te
from tvm.script import tir as T


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.CompactBufferAllocation()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    transformed = tvm.tir.transform.Simplify()(tvm.IRModule.from_expr(transformed))["main"]
    tvm.ir.assert_structural_equal(mod["main"], transformed)


@T.prim_func
def elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer((16, 16), "float32")
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(A[i, j])
                    T.writes(B[i, j])
                    B[i, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(B[i, j])
                    T.writes(C[i, j])
                    C[i, j] = B[i, j] * 2.0


@T.prim_func
def compacted_elementwise_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer((1, 16), "float32")
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(A[i, j])
                    T.writes(B[0, j])
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(B[0, j])
                    T.writes(C[i, j])
                    C[i, j] = B[0, j] * 2.0


@T.prim_func
def unschedulable_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer((16, 16), "float32")
            for j in range(0, 16):
                T.store(B.data, i * 16 + j, A[i, j] + 1.0)
            for j in range(0, 16):
                C[i, j] = B[i, j] * 2.0


@T.prim_func
def param_buffer_access_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (20, 20), "float32")
    B = T.match_buffer(c, (20, 20), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(B[i, 0:16])
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(A[i, j])
                    T.writes(B[i, j])
                    B[i, j] = A[i, j] + 1.0


@T.prim_func
def shared_mem_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in T.thread_binding(0, 2, thread="blockIdx.x"):
        for i1 in T.thread_binding(0, 2, thread="vthread"):
            for i2 in T.thread_binding(0, 4, thread="threadIdx.x"):
                with T.block():
                    T.reads(A[i0 * 8 + i1 * 4 + i2, 0:16])
                    T.writes(C[i0 * 8 + i1 * 4 + i2, 0:16])
                    B = T.alloc_buffer((16, 16), "float32", scope="shared")
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(A[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(B[i0 * 8 + i1 * 4 + i2, j])
                            B[i0 * 8 + i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(B[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(C[i0 * 8 + i1 * 4 + i2, j])
                            C[i0 * 8 + i1 * 4 + i2, j] = B[i0 * 8 + i1 * 4 + i2, j] * 2.0


@T.prim_func
def compacted_shared_mem_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in T.thread_binding(0, 2, thread="blockIdx.x"):
        for i1 in T.thread_binding(0, 2, thread="vthread"):
            for i2 in T.thread_binding(0, 4, thread="threadIdx.x"):
                with T.block():
                    T.reads(A[i0 * 8 + i1 * 4 + i2, 0:16])
                    T.writes(C[i0 * 8 + i1 * 4 + i2, 0:16])
                    B = T.alloc_buffer((8, 16), "float32", scope="shared")
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(A[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(B[i1 * 4 + i2, j])
                            B[i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(B[i1 * 4 + i2, j])
                            T.writes(C[i0 * 8 + i1 * 4 + i2, j])
                            C[i0 * 8 + i1 * 4 + i2, j] = B[i1 * 4 + i2, j] * 2.0


@T.prim_func
def warp_mem_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in T.thread_binding(0, 2, thread="blockIdx.x"):
        for i1 in T.thread_binding(0, 2, thread="vthread"):
            for i2 in T.thread_binding(0, 4, thread="threadIdx.x"):
                with T.block():
                    T.reads(A[i0 * 8 + i1 * 4 + i2, 0:16])
                    T.writes(C[i0 * 8 + i1 * 4 + i2, 0:16])
                    B = T.alloc_buffer((16, 16), "float32", scope="warp")
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(A[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(B[i0 * 8 + i1 * 4 + i2, j])
                            B[i0 * 8 + i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(B[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(C[i0 * 8 + i1 * 4 + i2, j])
                            C[i0 * 8 + i1 * 4 + i2, j] = B[i0 * 8 + i1 * 4 + i2, j] * 2.0


@T.prim_func
def compacted_warp_mem_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i0 in T.thread_binding(0, 2, thread="blockIdx.x"):
        for i1 in T.thread_binding(0, 2, thread="vthread"):
            for i2 in T.thread_binding(0, 4, thread="threadIdx.x"):
                with T.block():
                    T.reads(A[i0 * 8 + i1 * 4 + i2, 0:16])
                    T.writes(C[i0 * 8 + i1 * 4 + i2, 0:16])
                    B = T.alloc_buffer((4, 16), "float32", scope="warp")
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(A[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(B[i2, j])
                            B[i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block() as []:
                            T.reads(B[i2, j])
                            T.writes(C[i0 * 8 + i1 * 4 + i2, j])
                            C[i0 * 8 + i1 * 4 + i2, j] = B[i2, j] * 2.0


@T.prim_func
def symbolic_func(a: T.handle, c: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (n * 8,), "float32")
    C = T.match_buffer(c, (n * 8,), "float32")
    for i in range(0, n):
        with T.block():
            T.reads(A[i * 8 : i * 8 + 8])
            T.writes(C[i * 8 : i * 8 + 8])
            B = T.alloc_buffer((n * 8,), "float32")
            for j in range(0, 8):
                with T.block() as []:
                    T.reads(A[i * 8 + j])
                    T.writes(B[i * 8 + j])
                    B[i * 8 + j] = A[i * 8 + j] + 1.0
            for j in range(0, 8):
                with T.block() as []:
                    T.reads(B[i * 8 + j])
                    T.writes(C[i * 8 + j])
                    C[i * 8 + j] = B[i * 8 + j] * 2.0


@T.prim_func
def compacted_symbolic_func(a: T.handle, c: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (n * 8,), "float32")
    C = T.match_buffer(c, (n * 8,), "float32")
    for i in range(0, n):
        with T.block():
            T.reads(A[i * 8 : i * 8 + 8])
            T.writes(C[i * 8 : i * 8 + 8])
            B = T.alloc_buffer((8,), "float32")
            for j in range(0, 8):
                with T.block() as []:
                    T.reads(A[i * 8 + j])
                    T.writes(B[j])
                    B[j] = A[i * 8 + j] + 1.0
            for j in range(0, 8):
                with T.block() as []:
                    T.reads(B[j])
                    T.writes(C[i * 8 + j])
                    C[i * 8 + j] = B[j] * 2.0


@T.prim_func
def complex_func(a: T.handle, c: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (8, 8), "float32")
    C = T.match_buffer(c, (8, 8), "float32")
    for i in range(0, 8):
        with T.block():
            T.reads(A[0, 8])
            T.writes(C[0, 8])
            B = T.alloc_buffer((8, 8), "float32")
            for j in range(0, 4):
                with T.block() as []:
                    D = T.alloc_buffer((8, 8), "float32")
                    T.reads(A[i, j])
                    T.writes(B[i, j])
                    for k in range(4, 8):
                        D[k, j] = 1.0
                    for k in range(2, 4):
                        T.store(B.data, j, A[i, j] + D[k, j])
            for j in range(3, 5):
                with T.block() as []:
                    T.reads(B[i, j])
                    T.writes(C[i, j])
                    C[i, j] = B[i, j]
            for j in range(6, 8):
                with T.block() as []:
                    T.reads(B[i, j])
                    T.writes(C[i, j])
                    C[i, j] = B[i, j]


@T.prim_func
def compacted_complex_func(a: T.handle, c: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (8, 8), "float32")
    C = T.match_buffer(c, (8, 8), "float32")
    for i in range(0, 8):
        with T.block():
            T.reads(A[0, 8])
            T.writes(C[0, 8])
            B = T.alloc_buffer((1, 8), "float32")
            for j in range(0, 4):
                with T.block() as []:
                    D = T.alloc_buffer((6, 1), "float32")
                    T.reads(A[i, j])
                    T.writes(B[0, j])
                    for k in range(4, 8):
                        D[k - 2, 0] = 1.0
                    for k in range(2, 4):
                        T.store(B.data, j, A[i, j] + D[k - 2, 0])
            for j in range(3, 5):
                with T.block() as []:
                    T.reads(B[0, j])
                    T.writes(C[i, j])
                    C[i, j] = B[0, j]
            for j in range(6, 8):
                with T.block() as []:
                    T.reads(B[0, j])
                    T.writes(C[i, j])
                    C[i, j] = B[0, j]


@T.prim_func
def match_buffer_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16))
    C = T.match_buffer(c, (16, 16))
    for i in range(0, 16):
        with T.block():
            A0 = T.match_buffer(A[i, 0:16], (16))
            C0 = T.match_buffer(C[i, 0:16], (16))
            B = T.alloc_buffer((16, 16))
            with T.block():
                B0 = T.match_buffer(B[i, 0:16], (16))
                for j in range(0, 16):
                    with T.block() as []:
                        A1 = T.match_buffer(A0[j], ())
                        B1 = T.match_buffer(B0[j], ())
                        B1[()] = A1[()] + 1.0
            for j in range(0, 16):
                with T.block() as []:
                    C1 = T.match_buffer(C0[j], ())
                    B2 = T.match_buffer(B[i, j], ())
                    C1[()] = B2[()] * 2.0


@T.prim_func
def compacted_match_buffer_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16))
    C = T.match_buffer(c, (16, 16))
    for i in range(0, 16):
        with T.block():
            A0 = T.match_buffer(A[i, 0:16], (16))
            C0 = T.match_buffer(C[i, 0:16], (16))
            B = T.alloc_buffer((1, 16))
            with T.block():
                B0 = T.match_buffer(B[0, 0:16], (16))
                for j in range(0, 16):
                    with T.block() as []:
                        A1 = T.match_buffer(A0[j], ())
                        B1 = T.match_buffer(B0[j], ())
                        B1[()] = A1[()] + 1.0
            for j in range(0, 16):
                with T.block() as []:
                    C1 = T.match_buffer(C0[j], ())
                    B2 = T.match_buffer(B[0, j], ())
                    C1[()] = B2[()] * 2.0


@T.prim_func
def storage_align_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer((16, 16), "float32")
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(A[i, j])
                    T.writes(B[i, j])
                    T.block_attr({"buffer_dim_align": [[0, 0, 16, 15]]})
                    B[i, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(B[i, j])
                    T.writes(C[i, j])
                    C[i, j] = B[i, j] * 2.0


@T.prim_func
def compacted_storage_align_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with T.block():
            T.reads(A[i, 0:16])
            T.writes(C[i, 0:16])
            B = T.alloc_buffer((1, 16), strides=(31, 1), dtypes="float32")
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(A[i, j])
                    T.writes(B[0, j])
                    T.block_attr({"buffer_dim_align": [[0, 0, 16, 15]]})
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block() as []:
                    T.reads(B[0, j])
                    T.writes(C[i, j])
                    C[i, j] = B[0, j] * 2.0


@T.prim_func
def padding_pattern_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (20, 20), "float32")
    with T.block():
        B = T.alloc_buffer((20, 20), dtypes="float32")
        for i, j in T.grid(16, 16):
            with T.block():
                B[i, j] = A[i, j]
        for i, j in T.grid(20, 20):
            with T.block():
                C[i, j] = T.if_then_else(
                    2 <= i and i < 18 and 2 <= j and j < 18,
                    B[i - 2, j - 2],
                    0.0,
                    dtype="float32",
                )


@T.prim_func
def compacted_padding_pattern_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 16], dtype="float32")
    C = T.match_buffer(c, [20, 20], dtype="float32")
    with T.block():
        B = T.alloc_buffer([16, 16], dtype="float32")
        for i, j in T.grid(16, 16):
            with T.block():
                B[i, j] = A[i, j]
        for i, j in T.grid(20, 20):
            with T.block():
                C[i, j] = T.if_then_else(
                    2 <= i and i < 18 and 2 <= j and j < 18, B[i - 2, j - 2], 0.0, dtype="float32"
                )


@T.prim_func
def mem_access_in_branch_func(a: T.handle) -> None:
    A = T.match_buffer(a, (224, 224), "float32")
    with T.block():
        B1 = T.alloc_buffer((224, 224), dtypes="float32")
        B2 = T.alloc_buffer((224, 224), dtypes="float32")
        B3 = T.alloc_buffer((224, 224), dtypes="float32")
        B4 = T.alloc_buffer((224, 224), dtypes="float32")
        for i in range(0, 224):
            for j in range(0, 224):
                with T.block():
                    if i < 112 and j < 112:
                        B1[i, j] = A[i, j] * 2.0
                    else:
                        B2[i, j] = A[i, j] + 3.0
        for i in range(0, 224):
            for j in range(0, 224):
                with T.block():
                    if i < 112 or j < 112:
                        B3[i, j] = A[i, j] * 2.0
                    else:
                        B4[i, j] = A[i, j] + 3.0


@T.prim_func
def compacted_mem_access_in_branch_func(a: T.handle) -> None:
    A = T.match_buffer(a, [224, 224], dtype="float32")
    with T.block():
        B1 = T.alloc_buffer([112, 112], dtype="float32")
        B2 = T.alloc_buffer([224, 224], dtype="float32")
        B3 = T.alloc_buffer([224, 224], dtype="float32")
        B4 = T.alloc_buffer([112, 112], dtype="float32")
        for i, j in T.grid(224, 224):
            with T.block():
                if i < 112 and j < 112:
                    B1[i, j] = A[i, j] * 2.0
                else:
                    B2[i, j] = A[i, j] + 3.0
        for i, j in T.grid(224, 224):
            with T.block():
                if i < 112 or j < 112:
                    B3[i, j] = A[i, j] * 2.0
                else:
                    B4[i - 112, j - 112] = A[i, j] + 3.0


@T.prim_func
def opaque_access_annotated_func(a: T.handle) -> None:
    A = T.match_buffer(a, (1024,), "float32")
    with T.block():
        B = T.alloc_buffer((1024,), dtypes="float32")
        C = T.alloc_buffer((1024,), dtypes="float32")
        for i in range(0, 512):
            with T.block():
                # no annotation, opaque access will cover full region
                T.reads([])
                T.writes([])
                T.store(B.data, i, "float32", A[i])
            with T.block():
                # treat opaque access only access annotated regions, even if
                # they are not compatible with actual buffer accesses.
                T.reads([B[i]])
                T.writes([C[i : i + 9]])
                T.store(C.data, i, T.load("float32", B.data, i))


@T.prim_func
def compacted_opaque_access_annotated_func(a: T.handle) -> None:
    A = T.match_buffer(a, (1024,), "float32")
    with T.block():
        B = T.alloc_buffer((1024,), dtypes="float32")
        C = T.alloc_buffer((520,), dtypes="float32")
        for i in range(0, 512):
            with T.block():
                # no annotation, opaque access will cover full region
                T.reads([])
                T.writes([])
                T.store(B.data, i, "float32", A[i])
            with T.block():
                # treat opaque access only access annotated regions, even if
                # they are not compatible with actual buffer accesses.
                T.reads([B[i]])
                T.writes([C[i : i + 9]])
                T.store(C.data, i, T.load("float32", B.data, i))


@T.prim_func
def sparse_read_cache(
    A_data: T.Buffer[(819,), "float32"],
    B: T.Buffer[(128,), "float32"],
    A_indptr: T.Buffer[(129,), "int32"],
    A_indices: T.Buffer[(819,), "int32"],
) -> None:
    for i in T.serial(128):
        with T.block("rowsum_outer"):
            T.reads(
                A_indptr[i : i + 1],
                A_data[A_indptr[i] + 0 : A_indptr[i] + (A_indptr[i + 1] - A_indptr[i])],
            )
            T.writes(B[i])
            with T.block("rowsum_init"):
                T.reads()
                T.writes(B[i])
                B[i] = T.float32(0)
            for k in T.serial(A_indptr[i + 1] - A_indptr[i]):
                with T.block():
                    T.reads(A_indptr[i], A_data[A_indptr[i] + k], B[i])
                    T.writes(B[i])
                    A_data_local = T.alloc_buffer([819], dtype="float32", scope="local")
                    with T.block("A_data_cache_read"):
                        T.reads(A_indptr[i], A_data[A_indptr[i] + k])
                        T.writes(A_data_local[A_indptr[i] + k])
                        A_data_local[A_indptr[i] + k] = A_data[A_indptr[i] + k]
                    with T.block("rowsum_inner"):
                        T.reads(B[i], A_indptr[i], A_data[A_indptr[i] + k])
                        T.writes(B[i])
                        B[i] = B[i] + A_data_local[A_indptr[i] + k]


@T.prim_func
def compacted_sparse_read_cache(
    A_data: T.Buffer[(819,), "float32"],
    B: T.Buffer[(128,), "float32"],
    A_indptr: T.Buffer[(129,), "int32"],
    A_indices: T.Buffer[(819,), "int32"],
) -> None:
    for i in T.serial(128):
        with T.block("rowsum_outer"):
            T.reads(
                A_indptr[i : i + 1],
                A_data[A_indptr[i] + 0 : A_indptr[i] + 0 + (A_indptr[i + 1] - A_indptr[i])],
            )
            T.writes(B[i])
            with T.block("rowsum_init"):
                T.reads()
                T.writes(B[i])
                B[i] = T.float32(0)
            for k in T.serial(A_indptr[i + 1] - A_indptr[i]):
                with T.block():
                    T.reads(A_indptr[i], A_data[A_indptr[i] + k], B[i])
                    T.writes(B[i])
                    A_data_local = T.alloc_buffer([1], dtype="float32", scope="local")
                    with T.block("A_data_cache_read"):
                        T.reads(A_indptr[i], A_data[A_indptr[i] + k])
                        T.writes(A_data_local[A_indptr[i] + k - (A_indptr[i] + k)])
                        A_data_local[A_indptr[i] + k - (A_indptr[i] + k)] = A_data[A_indptr[i] + k]
                    with T.block("rowsum_inner"):
                        T.reads(B[i], A_indptr[i], A_data[A_indptr[i] + k])
                        T.writes(B[i])
                        B[i] = B[i] + A_data_local[A_indptr[i] + k - (A_indptr[i] + k)]


def test_elementwise():
    _check(elementwise_func, compacted_elementwise_func)


def test_unschedulable_block():
    _check(unschedulable_func, unschedulable_func)  # changes nothing


def test_param_access():
    _check(param_buffer_access_func, param_buffer_access_func)  # changes nothing


def test_shared_mem():
    _check(shared_mem_func, compacted_shared_mem_func)


def test_warp_mem():
    _check(warp_mem_func, compacted_warp_mem_func)


def test_symbolic():
    _check(symbolic_func, compacted_symbolic_func)


def test_complex():
    _check(complex_func, compacted_complex_func)


def test_match_buffer():
    _check(match_buffer_func, compacted_match_buffer_func)


def test_lower_te():
    x = te.placeholder((1,))
    y = te.compute((1,), lambda i: x[i] + 2)
    s = te.create_schedule(y.op)
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
    mod = tvm.tir.transform.CompactBufferAllocation()(orig_mod)
    tvm.ir.assert_structural_equal(mod, orig_mod)  # CompactBufferAllocation should do nothing on TE


def test_storage_align():
    _check(storage_align_func, compacted_storage_align_func)


def test_padding_pattern():
    _check(padding_pattern_func, compacted_padding_pattern_func)


def test_mem_access_in_branch_func():
    _check(mem_access_in_branch_func, compacted_mem_access_in_branch_func)


def test_opaque_access_annotated_func():
    _check(opaque_access_annotated_func, compacted_opaque_access_annotated_func)


def test_sparse_read_cache():
    _check(sparse_read_cache, compacted_sparse_read_cache)


if __name__ == "__main__":
    test_elementwise()
    test_unschedulable_block()
    test_param_access()
    test_shared_mem()
    test_warp_mem()
    test_symbolic()
    test_complex()
    test_match_buffer()
    test_storage_align()
    test_lower_te()
    test_padding_pattern()
    test_mem_access_in_branch_func()
    test_opaque_access_annotated_func()
    test_sparse_read_cache()
