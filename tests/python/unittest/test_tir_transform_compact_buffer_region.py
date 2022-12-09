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
import tvm.testing
from tvm import te
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
                with T.block():
                    T.reads(A[i, j])
                    T.writes(B[i, j])
                    B[i, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block():
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
                with T.block():
                    T.reads(A[i, j])
                    T.writes(B[0, j])
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block():
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
                T.evaluate(T.call_extern("dummy_extern_function", B.data, dtype="int32"))
                B[i, j] = A[i, j] + 1.0
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
                with T.block():
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
                        with T.block():
                            T.reads(A[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(B[i0 * 8 + i1 * 4 + i2, j])
                            B[i0 * 8 + i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block():
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
                        with T.block():
                            T.reads(A[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(B[i1 * 4 + i2, j])
                            B[i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block():
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
                        with T.block():
                            T.reads(A[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(B[i0 * 8 + i1 * 4 + i2, j])
                            B[i0 * 8 + i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block():
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
                        with T.block():
                            T.reads(A[i0 * 8 + i1 * 4 + i2, j])
                            T.writes(B[i2, j])
                            B[i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                    for j in range(0, 16):
                        with T.block():
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
                with T.block():
                    T.reads(A[i * 8 + j])
                    T.writes(B[i * 8 + j])
                    B[i * 8 + j] = A[i * 8 + j] + 1.0
            for j in range(0, 8):
                with T.block():
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
            B = T.alloc_buffer((T.min(n, 1) * 8,), "float32")
            for j in range(0, 8):
                with T.block():
                    T.reads(A[i * 8 + j])
                    T.writes(B[j])
                    B[j] = A[i * 8 + j] + 1.0
            for j in range(0, 8):
                with T.block():
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
                with T.block():
                    D = T.alloc_buffer((8, 8), "float32")
                    T.reads(A[i, j])
                    T.writes(B[i, j])
                    for k in range(4, 8):
                        D[k, j] = 1.0
                    for k in range(2, 4):
                        B[i, j] = A[i, j] + D[k, j]
            for j in range(3, 5):
                with T.block():
                    T.reads(B[i, j])
                    T.writes(C[i, j])
                    C[i, j] = B[i, j]
            for j in range(6, 8):
                with T.block():
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
                with T.block():
                    D = T.alloc_buffer((6, 1), "float32")
                    T.reads(A[i, j])
                    T.writes(B[0, j])
                    for k in range(4, 8):
                        D[k - 2, 0] = 1.0
                    for k in range(2, 4):
                        B[0, j] = A[i, j] + D[k - 2, 0]
            for j in range(3, 5):
                with T.block():
                    T.reads(B[0, j])
                    T.writes(C[i, j])
                    C[i, j] = B[0, j]
            for j in range(6, 8):
                with T.block():
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
                    with T.block():
                        A1 = T.match_buffer(A0[j], ())
                        B1 = T.match_buffer(B0[j], ())
                        B1[()] = A1[()] + 1.0
            for j in range(0, 16):
                with T.block():
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
                    with T.block():
                        A1 = T.match_buffer(A0[j], ())
                        B1 = T.match_buffer(B0[j], ())
                        B1[()] = A1[()] + 1.0
            for j in range(0, 16):
                with T.block():
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
                with T.block():
                    T.reads(A[i, j])
                    T.writes(B[i, j])
                    T.block_attr({"buffer_dim_align": [[0, 0, 16, 15]]})
                    B[i, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block():
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
            B = T.alloc_buffer((1, 16), strides=(31, 1), dtype="float32")
            for j in range(0, 16):
                with T.block():
                    T.reads(A[i, j])
                    T.writes(B[0, j])
                    T.block_attr({"buffer_dim_align": [[0, 0, 16, 15]]})
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with T.block():
                    T.reads(B[0, j])
                    T.writes(C[i, j])
                    C[i, j] = B[0, j] * 2.0


@T.prim_func
def padding_pattern_func(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    C = T.match_buffer(c, (20, 20), "float32")
    with T.block():
        B = T.alloc_buffer((20, 20), dtype="float32")
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
def padding_pattern_inlined(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [224, 224], dtype="float32")
    Y = T.match_buffer(b, [224, 224], dtype="float32")
    cache = T.alloc_buffer([224, 224], dtype="float32")
    for h, w in T.grid(224, 224):
        with T.block("cache"):
            cache[h, w] = X[h, w]
    for h, w, kh, kw in T.grid(224, 224, 3, 3):
        with T.block("compute"):
            Y[h, w] = T.max(
                Y[h, w],
                T.if_then_else(
                    T.likely(1 <= h + kh, dtype="bool")
                    and T.likely(h + kh < 225, dtype="bool")
                    and T.likely(1 <= w + kw, dtype="bool")
                    and T.likely(w + kw < 225, dtype="bool"),
                    cache[h + kh - 1, w + kw - 1],
                    0.0,
                    dtype="float32",
                ),
            )


@T.prim_func
def compacted_padding_pattern_inlined(
    X: T.Buffer[(224, 224), "float32"], Y: T.Buffer[(224, 224), "float32"]
) -> None:
    cache = T.alloc_buffer([224, 224], dtype="float32")
    for h, w in T.grid(224, 224):
        with T.block("cache"):
            cache[h, w] = X[h, w]
    for h, w, kh, kw in T.grid(224, 224, 3, 3):
        with T.block("compute"):
            Y[h, w] = T.max(
                Y[h, w],
                T.if_then_else(
                    T.likely(1 <= h + kh, dtype="bool")
                    and T.likely(h + kh < 225, dtype="bool")
                    and T.likely(1 <= w + kw, dtype="bool")
                    and T.likely(w + kw < 225, dtype="bool"),
                    cache[h + kh - 1, w + kw - 1],
                    0.0,
                    dtype="float32",
                ),
            )


@T.prim_func
def mem_access_in_branch_func(a: T.handle) -> None:
    A = T.match_buffer(a, (224, 224), "float32")
    with T.block():
        B1 = T.alloc_buffer((224, 224), dtype="float32")
        B2 = T.alloc_buffer((224, 224), dtype="float32")
        B3 = T.alloc_buffer((224, 224), dtype="float32")
        B4 = T.alloc_buffer((224, 224), dtype="float32")
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
        B = T.alloc_buffer((1024,), dtype="float32")
        C = T.alloc_buffer((1024,), dtype="float32")
        for i in range(0, 512):
            with T.block():
                # no annotation, opaque access will cover full region
                T.reads([])
                T.writes([])
                T.evaluate(T.call_extern("opaque_extern_function", A.data, B.data, dtype="int32"))
                B[i] = A[i]
            with T.block():
                # treat opaque access only access annotated regions, even if
                # they are not compatible with actual buffer accesses.
                T.reads([B[i]])
                T.writes([C[i : i + 9]])
                T.evaluate(T.call_extern("opaque_extern_function", B.data, C.data, dtype="int32"))
                C[i] = B[i]


@T.prim_func
def compacted_opaque_access_annotated_func(a: T.handle) -> None:
    A = T.match_buffer(a, (1024,), "float32")
    with T.block():
        B = T.alloc_buffer((1024,), dtype="float32")
        C = T.alloc_buffer((520,), dtype="float32")
        for i in range(0, 512):
            with T.block():
                # no annotation, opaque access will cover full region
                T.reads([])
                T.writes([])
                T.evaluate(T.call_extern("opaque_extern_function", A.data, B.data, dtype="int32"))
                B[i] = A[i]
            with T.block():
                # treat opaque access only access annotated regions, even if
                # they are not compatible with actual buffer accesses.
                T.reads([B[i]])
                T.writes([C[i : i + 9]])
                T.evaluate(T.call_extern("opaque_extern_function", B.data, C.data, dtype="int32"))
                C[i] = B[i]


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
                        T.writes(A_data_local[T.min(A_indptr[i] + k, 0)])
                        A_data_local[T.min(A_indptr[i] + k, 0)] = A_data[A_indptr[i] + k]
                    with T.block("rowsum_inner"):
                        T.reads(B[i], A_indptr[i], A_data[A_indptr[i] + k])
                        T.writes(B[i])
                        B[i] = B[i] + A_data_local[T.min(A_indptr[i] + k, 0)]


@T.prim_func
def narrow_shape(A: T.Buffer[(10,), "float32"], B: T.Buffer[(10,), "float32"]) -> None:
    B_cache = T.alloc_buffer(10, "float32")
    for j in T.serial(3):
        for k in T.serial(4):
            with T.block("B_cache"):
                T.where(j * 4 + k < 10)
                B_cache[j * 4 + k] = B[j]
    for i in T.serial(10):
        A[i] = B_cache[i] + T.float32(1)


@T.prim_func
def compacted_narrow_shape(A: T.Buffer[(10,), "float32"], B: T.Buffer[(10,), "float32"]) -> None:
    # body
    # with T.block("root")
    B_cache = T.alloc_buffer([10], dtype="float32")
    for j, k in T.grid(3, 4):
        with T.block("B_cache"):
            T.where(j * 4 + k < 10)
            T.reads(B[j])
            T.writes(B_cache[j * 4 + k])
            B_cache[j * 4 + k] = B[j]
    for i in T.serial(10):
        A[i] = B_cache[i] + T.float32(1)


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


def test_padding_pattern_inlined():
    _check(padding_pattern_inlined, compacted_padding_pattern_inlined)


def test_mem_access_in_branch_func():
    _check(mem_access_in_branch_func, compacted_mem_access_in_branch_func)


def test_opaque_access_annotated_func():
    _check(opaque_access_annotated_func, compacted_opaque_access_annotated_func)


def test_sparse_read_cache():
    _check(sparse_read_cache, compacted_sparse_read_cache)


def test_narrow_shape():
    _check(narrow_shape, compacted_narrow_shape)


def test_compact_with_let_binding():
    @T.prim_func
    def func_with_let_binding():
        A = T.alloc_buffer((64, 8), "float32")
        B = T.alloc_buffer((64, 8), "float32")
        C = T.alloc_buffer((8, 8), "float32")
        for rk in range(64):
            for rii, rjj in T.grid(8, 8):
                C[rii, rjj] = T.float32(0)
            for riijj in T.serial(8 * 8):
                rii: T.int32 = riijj // 8
                rjj: T.int32 = riijj % 8
                C[rii, rjj] += A[rk, rii] * B[rk, rjj]

    _check(func_with_let_binding, func_with_let_binding)

    @T.prim_func
    def func_with_non_index_let_binding():
        A = T.alloc_buffer((64), "float32")
        x1 = T.call_extern("get", dtype="float16")
        x2 = T.call_extern("get", dtype="float32")
        x3 = T.call_extern("get", dtype="float64")
        x4 = T.call_extern("get", dtype="uint8")
        x5 = T.call_extern("get", dtype="int32x16")
        x6 = T.call_extern("get", dtype="handle")
        x7 = T.call_extern("get", dtype="")
        for rk in range(64):
            A[rk] = T.call_extern("load_ptr", x1, x2, x3, x4, x5, x6, x7, dtype="float32")

    _check(func_with_non_index_let_binding, func_with_non_index_let_binding)


def test_compact_spatial_tiled_pad_and_pooling():
    @T.prim_func
    def spatial_tiled_pad_and_pooling(
        X: T.Buffer[(64, 112, 112), "int32"], Y: T.Buffer[(64, 56, 56), "int32"]
    ) -> None:
        for h_o, w_o in T.grid(14, 14):
            with T.block():
                X_cache = T.alloc_buffer([112, 112, 64], dtype="int32")
                for ax0, ax1, ax2 in T.grid(64, 9, 9):
                    with T.block("cache"):
                        T.where(1 <= h_o * 8 + ax1 and 1 <= w_o * 8 + ax2)
                        T.reads(X[ax0, h_o * 8 - 1 + ax1, w_o * 8 - 1 + ax2])
                        T.writes(X_cache[h_o * 8 - 1 + ax1, w_o * 8 - 1 + ax2, ax0])
                        X_cache[h_o * 8 - 1 + ax1, w_o * 8 - 1 + ax2, ax0] = X[
                            ax0, h_o * 8 - 1 + ax1, w_o * 8 - 1 + ax2
                        ]
                for h_i, w_i, kh, kw, c in T.grid(4, 4, 3, 3, 64):
                    with T.block("compute"):
                        T.reads(
                            X_cache[(h_o * 4 + h_i) * 2 + kh - 1, (w_o * 4 + w_i) * 2 + kw - 1, c]
                        )
                        T.writes(Y[h_o * 4 + h_i, w_o * 4 + w_i, c])
                        if kh == 0 and kw == 0:
                            Y[h_o * 4 + h_i, w_o * 4 + w_i, c] = 0
                        Y[h_o * 4 + h_i, w_o * 4 + w_i, c] = T.max(
                            Y[h_o * 4 + h_i, w_o * 4 + w_i, c],
                            T.if_then_else(
                                T.likely(1 <= (h_o * 4 + h_i) * 2 + kh, dtype="bool")
                                and T.likely((h_o * 4 + h_i) * 2 + kh < 113, dtype="bool")
                                and T.likely(1 <= (w_o * 4 + w_i) * 2 + kw, dtype="bool")
                                and T.likely((w_o * 4 + w_i) * 2 + kw < 113, dtype="bool"),
                                X_cache[
                                    (h_o * 4 + h_i) * 2 + kh - 1,
                                    (w_o * 4 + w_i) * 2 + kw - 1,
                                    c,
                                ],
                                0,
                                dtype="int32",
                            ),
                        )

    @T.prim_func
    def compacted_spatial_tiled_pad_and_pooling(
        X: T.Buffer[(64, 112, 112), "int32"], Y: T.Buffer[(64, 56, 56), "int32"]
    ) -> None:
        for h_o, w_o in T.grid(14, 14):
            with T.block():
                T.reads(X[0:64, h_o * 8 - 1 : h_o * 8 + 8, w_o * 8 - 1 : w_o * 8 + 8])
                T.writes(Y[h_o * 4 : h_o * 4 + 4, w_o * 4 : w_o * 4 + 4, 0:64])
                X_cache = T.alloc_buffer([9, 9, 64], dtype="int32")
                for ax0, ax1, ax2 in T.grid(64, 9, 9):
                    with T.block("cache"):
                        T.where(1 <= h_o * 8 + ax1 and 1 <= w_o * 8 + ax2)
                        T.reads(X[ax0, h_o * 8 + ax1 - 1, w_o * 8 + ax2 - 1])
                        T.writes(
                            X_cache[
                                h_o * 8 + ax1 - T.max(0, h_o * 8 - 1) - 1,
                                w_o * 8 + ax2 - T.max(0, w_o * 8 - 1) - 1,
                                ax0,
                            ]
                        )
                        X_cache[
                            h_o * 8 + ax1 - T.max(0, h_o * 8 - 1) - 1,
                            w_o * 8 + ax2 - T.max(0, w_o * 8 - 1) - 1,
                            ax0,
                        ] = X[ax0, h_o * 8 + ax1 - 1, w_o * 8 + ax2 - 1]
                for h_i, w_i, kh, kw, c in T.grid(4, 4, 3, 3, 64):
                    with T.block("compute"):
                        T.reads(
                            X_cache[
                                h_o * 8 + h_i * 2 + kh - T.max(0, h_o * 8 - 1) - 1,
                                w_o * 8 + w_i * 2 + kw - T.max(0, w_o * 8 - 1) - 1,
                                c,
                            ]
                        )
                        T.writes(Y[h_o * 4 + h_i, w_o * 4 + w_i, c])
                        if kh == 0 and kw == 0:
                            Y[h_o * 4 + h_i, w_o * 4 + w_i, c] = 0
                        Y[h_o * 4 + h_i, w_o * 4 + w_i, c] = T.max(
                            Y[h_o * 4 + h_i, w_o * 4 + w_i, c],
                            T.if_then_else(
                                T.likely(1 <= h_o * 8 + h_i * 2 + kh, dtype="bool")
                                and T.likely(1 <= w_o * 8 + w_i * 2 + kw, dtype="bool"),
                                X_cache[
                                    h_o * 8 + h_i * 2 + kh - T.max(0, h_o * 8 - 1) - 1,
                                    w_o * 8 + w_i * 2 + kw - T.max(0, w_o * 8 - 1) - 1,
                                    c,
                                ],
                                0,
                                dtype="int32",
                            ),
                        )

    _check(spatial_tiled_pad_and_pooling, compacted_spatial_tiled_pad_and_pooling)


def test_complex_case_1():
    """Meta-schedule matmul case for compact shared A, B matrix"""

    # fmt: off
    @T.prim_func
    def func(A: T.Buffer[(960, 770), "float32"], B: T.Buffer[(770, 2304), "float32"], C: T.Buffer[(960, 2304), "float32"]) -> None:
        for bx in T.thread_binding(144, thread="blockIdx.x"):
            for vx in T.thread_binding(2, thread="vthread.x"):
                for tx_p in T.thread_binding(256, thread="threadIdx.x"):
                    with T.block():
                        for k_0 in T.serial(193):
                            with T.block():
                                A_shared = T.alloc_buffer([960, 770], dtype="float32", scope="shared")
                                B_shared = T.alloc_buffer([770, 2304], dtype="float32", scope="shared")
                                for _u in T.serial(1):
                                    for tx in T.thread_binding(256, thread="threadIdx.x"):
                                        for vec in T.vectorized(3):
                                            with T.block("A_shared"):
                                                T.where(bx // 18 * 128 + ((_u * 256 + tx) * 3 + vec) // 4 < 960 and k_0 * 4 + ((_u * 256 + tx) * 3 + vec) % 4 < 770 and (_u * 256 + tx) * 3 + vec < 512)
                                                A_shared[bx // 18 * 128 + (_u * 768 + tx * 3 + vec) // 4, k_0 * 4 + (_u * 768 + tx * 3 + vec) % 4] = A[bx // 18 * 128 + (_u * 768 + tx * 3 + vec) // 4, k_0 * 4 + (_u * 768 + tx * 3 + vec) % 4]
                                for _u in T.serial(1):
                                    for tx in T.thread_binding(256, thread="threadIdx.x"):
                                        for vec in T.vectorized(4):
                                            with T.block("B_shared"):
                                                T.where(k_0 * 4 + ((_u * 256 + tx) * 4 + vec) // 128 < 770 and (_u * 256 + tx) * 4 + vec < 512)
                                                B_shared[k_0 * 4 + (_u * 1024 + tx * 4 + vec) // 128, bx % 18 * 128 + (_u * 1024 + tx * 4 + vec) % 128] = B[k_0 * 4 + (_u * 1024 + tx * 4 + vec) // 128, bx % 18 * 128 + (_u * 1024 + tx * 4 + vec) % 128]
                                for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(1, 8, 1, 4, 2, 2):
                                    with T.block("update_update"):
                                        C[(((bx // 18 + 0) * 8 + tx_p // 32) * 8 + i_3) * 2 + i_4, ((bx % 18 * 2 + vx % 2) * 32 + tx_p % 32 + j_3) * 2 + j_4] = C[(((bx // 18 + 0) * 8 + tx_p // 32) * 8 + i_3) * 2 + i_4, ((bx % 18 * 2 + vx % 2) * 32 + tx_p % 32 + j_3) * 2 + j_4] + A_shared[(((bx // 18 + 0) * 8 + tx_p // 32) * 8 + i_3) * 2 + i_4, (k_0 + k_1) * 4 + k_2] * B_shared[(k_0 + k_1) * 4 + k_2, ((bx % 18 * 2 + vx % 2) * 32 + tx_p % 32 + j_3) * 2 + j_4]

    @T.prim_func
    def compacted_func(A: T.Buffer[(960, 770), "float32"], B: T.Buffer[(770, 2304), "float32"], C: T.Buffer[(960, 2304), "float32"]) -> None:
        for bx in T.thread_binding(144, thread="blockIdx.x"):
            for vx in T.thread_binding(2, thread="vthread.x"):
                for tx_p in T.thread_binding(256, thread="threadIdx.x"):
                    with T.block():
                        for k_0 in T.serial(193):
                            with T.block():
                                A_shared = T.alloc_buffer([128, 4], dtype="float32", scope="shared")
                                B_shared = T.alloc_buffer([4, 128], dtype="float32", scope="shared")
                                for v_u in T.serial(1):
                                    for tx in T.thread_binding(256, thread="threadIdx.x"):
                                        for vec in T.vectorized(3):
                                            with T.block("A_shared"):
                                                T.where(bx // 18 * 128 + (tx * 3 + vec) // 4 < 960 and k_0 * 4 + (tx * 3 + vec) % 4 < 770 and tx * 3 + vec < 512)
                                                A_shared[(tx * 3 + vec) // 4, (tx * 3 + vec) % 4] = A[bx // 18 * 128 + (tx * 3 + vec) // 4, k_0 * 4 + (tx * 3 + vec) % 4]
                                for v_u in T.serial(1):
                                    for tx in T.thread_binding(256, thread="threadIdx.x"):
                                        for vec in T.vectorized(4):
                                            with T.block("B_shared"):
                                                T.where(k_0 * 4 + tx // 32 < 770 and tx * 4 + vec < 512)
                                                B_shared[tx // 32, tx % 32 * 4 + vec] = B[k_0 * 4 + tx // 32, bx % 18 * 128 + tx % 32 * 4 + vec]
                                for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(1, 8, 1, 4, 2, 2):
                                    with T.block("update_update"):
                                        C[bx // 18 * 128 + tx_p // 32 * 16 + i_3 * 2 + i_4, bx % 18 * 128 + vx * 64 + tx_p % 32 * 2 + j_4] = C[bx // 18 * 128 + tx_p // 32 * 16 + i_3 * 2 + i_4, bx % 18 * 128 + vx * 64 + tx_p % 32 * 2 + j_4] + A_shared[tx_p // 32 * 16 + i_3 * 2 + i_4, k_2] * B_shared[k_2, vx * 64 + tx_p % 32 * 2 + j_4]
    # fmt: on

    _check(func, compacted_func)


def test_compact_dependent_buffer_indices():
    """Check the upper bound on different indices could be independently estimated."""

    @T.prim_func
    def diagonal_access():
        for i in range(8):
            with T.block():
                A = T.alloc_buffer((256, 256), "float32")
                for j, k in T.grid(8, 8):
                    with T.block():
                        T.where(j * 8 + k < 60)
                        A[i * 64 + j * 8 + k, i * 64 + j * 8 + k] = 1.0

    @T.prim_func
    def diagonal_access_compacted() -> None:
        for i in T.serial(8):
            with T.block():
                A = T.alloc_buffer([60, 60], dtype="float32")
                for j, k in T.grid(8, 8):
                    with T.block():
                        T.where(j * 8 + k < 60)
                        A[j * 8 + k, j * 8 + k] = 1.0

    _check(diagonal_access, diagonal_access_compacted)


def test_compact_dependent_buffer_indices_of_packed_matmul():
    """Check the outer dimension of the packed M-dim should be compacted to 1 wrt split condition."""

    @T.prim_func
    def nonuniform_packed_matmul_write_cache(
        A: T.Buffer[(1020, 64), "float32"],
        B: T.Buffer[(1000, 64), "float32"],
        C: T.Buffer[(1020, 1000), "float32"],
    ):
        for i0, i1 in T.grid(4, 1):
            with T.block():
                C_local2 = T.alloc_buffer([4, 1, 16, 1000, 16], dtype="float32", scope="local")
                C_local1 = T.alloc_buffer([1020, 1000], dtype="float32", scope="local")
                for ax0, ax1, ax2 in T.grid(255, 1000, 64):
                    with T.block("matmul"):
                        if ax2 == 0:
                            C_local1[i0 * 255 + ax0, ax1] = 0
                        C_local1[i0 * 255 + ax0, ax1] = (
                            C_local1[i0 * 255 + ax0, ax1] + A[i0 * 255 + ax0, ax2] * B[ax1, ax2]
                        )
                for ax0, ax1 in T.grid(255, 1000):
                    with T.block("st1"):
                        C_local2[
                            (i0 * 255 + ax0) // 255,
                            0,
                            (i0 * 255 + ax0) % 255 // 16,
                            ax1,
                            (i0 * 255 + ax0) % 255 % 16,
                        ] = C_local1[i0 * 255 + ax0, ax1]
                for ax0, ax1, ax2 in T.grid(16, 16, 1000):
                    with T.block("st2"):
                        T.where(ax0 * 16 + ax1 < 255)
                        C[i0 * 255 + (ax0 * 16 + ax1), i1 * 1000 + ax2] = C_local2[
                            (i0 * 255 + ax0 * 16 + ax1) // 255,
                            0,
                            (i0 * 255 + ax0 * 16 + ax1) % 255 // 16,
                            i1 * 1000 + ax2,
                            (i0 * 255 + ax0 * 16 + ax1) % 255 % 16,
                        ]

    @T.prim_func
    def nonuniform_packed_matmul_write_cache_compacted(
        A: T.Buffer[(1020, 64), "float32"],
        B: T.Buffer[(1000, 64), "float32"],
        C: T.Buffer[(1020, 1000), "float32"],
    ) -> None:
        for i0, i1 in T.grid(4, 1):
            with T.block():
                C_local2 = T.alloc_buffer([1, 1, 15, 1000, 16], dtype="float32", scope="local")
                C_local1 = T.alloc_buffer([255, 1000], dtype="float32", scope="local")
                for ax0, ax1, ax2 in T.grid(255, 1000, 64):
                    with T.block("matmul"):
                        if ax2 == 0:
                            C_local1[ax0, ax1] = 0
                        C_local1[ax0, ax1] = (
                            C_local1[ax0, ax1] + A[i0 * 255 + ax0, ax2] * B[ax1, ax2]
                        )
                for ax0, ax1 in T.grid(255, 1000):
                    with T.block("st1"):
                        C_local2[0, 0, ax0 // 16, ax1, ax0 % 16] = C_local1[ax0, ax1]
                for ax0, ax1, ax2 in T.grid(16, 16, 1000):
                    with T.block("st2"):
                        T.where(ax0 * 16 + ax1 < 255)
                        C[i0 * 255 + ax0 * 16 + ax1, ax2] = C_local2[
                            (ax0 * 16 + ax1) // 255,
                            0,
                            (ax0 * 16 + ax1) % 255 // 16,
                            ax2,
                            (ax0 * 16 + ax1) % 255 % 16,
                        ]

    _check(nonuniform_packed_matmul_write_cache, nonuniform_packed_matmul_write_cache_compacted)


if __name__ == "__main__":
    tvm.testing.main()
