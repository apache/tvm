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
from tvm import tir
from tvm.script import tir as T


class BaseCompactTest:
    """Base testcase class. The inherit testcase should include:
    - `before` and `expected` primfunc used to check structural equality for the transformation.
    - `is_lower_order_free` tag, defaults to True, denotes that we would check
       (LowerOpaqueBlock . CompactBufferAllocation)(before) ==
       (CompactBufferAllocation . LowerOpaqueBlock)(before)
    - `is_strict` tag, defaults to True, controls the `is_strict` option of the compaction pass.
    """

    def test_compact(self):
        is_lower_order_free = getattr(self, "is_lower_order_free", True)
        is_strict = getattr(self, "is_strict_mode", True)

        before = tvm.IRModule.from_expr(self.before.with_attr("global_symbol", "main"))
        expected = tvm.IRModule.from_expr(self.expected.with_attr("global_symbol", "main"))
        simplify = tvm.transform.Sequential([tir.transform.Simplify(), tir.transform.RemoveNoOp()])
        after = simplify(tir.transform.CompactBufferAllocation(is_strict=is_strict)(before))
        expected = simplify(expected)
        try:
            tvm.ir.assert_structural_equal(after, expected)
        except ValueError as err:
            script = tvm.IRModule(
                {"expected": expected["main"], "after": after["main"], "before": before["main"]}
            ).script()
            raise ValueError(
                f"Function after simplification did not match expected:\n{script}"
            ) from err

        if not is_lower_order_free:
            return
        lower_before_compact = tir.transform.LowerOpaqueBlock()(before)
        lower_before_compact = tir.transform.CompactBufferAllocation(is_strict=is_strict)(
            lower_before_compact
        )
        lower_before_compact = simplify(lower_before_compact)
        lower_after_compact = tir.transform.LowerOpaqueBlock()(after)
        lower_after_compact = simplify(lower_after_compact)
        try:
            tvm.ir.assert_structural_equal(lower_before_compact, lower_after_compact)
        except ValueError as err:
            script = tvm.IRModule(
                {
                    "lower_before_compact": lower_before_compact["main"],
                    "lower_after_compact": lower_after_compact["main"],
                    "before": before["main"],
                }
            ).script()
            raise ValueError(
                f"Function after simplification did not match expected:\n{script}"
            ) from err


class TestElemwise(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle) -> None:
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
    def expected(a: T.handle, c: T.handle) -> None:
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


class TestUnschedulableFunc(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle) -> None:
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

    expected = before


class TestParamBufferAccess(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle) -> None:
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

    expected = before


class TestSharedMem(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle) -> None:
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
    def expected(a: T.handle, c: T.handle) -> None:
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


class TestWrapMem(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle) -> None:
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
    def expected(a: T.handle, c: T.handle) -> None:
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


class TestSymbolic(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle, n: T.int32) -> None:
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
    def expected(a: T.handle, c: T.handle, n: T.int32) -> None:
        A = T.match_buffer(a, (n * 8,), "float32")
        C = T.match_buffer(c, (n * 8,), "float32")
        for i in range(0, n):
            with T.block():
                T.reads(A[i * 8 : i * 8 + 8])
                T.writes(C[i * 8 : i * 8 + 8])
                B = T.alloc_buffer((8,), "float32")
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


class TestComplexFunc(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle, n: T.int32) -> None:
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
    def expected(a: T.handle, c: T.handle, n: T.int32) -> None:
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


class TestMatchBuffer(BaseCompactTest):
    is_lower_order_free = False

    @T.prim_func
    def before(a: T.handle, c: T.handle) -> None:
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
    def expected(a: T.handle, c: T.handle) -> None:
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


class TestStorageAlign(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle) -> None:
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
    def expected(a: T.handle, c: T.handle) -> None:
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


class TestPaddingPattern(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, c: T.handle) -> None:
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
    def expected(a: T.handle, c: T.handle) -> None:
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
                        2 <= i and i < 18 and 2 <= j and j < 18,
                        B[i - 2, j - 2],
                        0.0,
                        dtype="float32",
                    )


class TestPaddingPatternInlined(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle, b: T.handle) -> None:
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
    def expected(X: T.Buffer((224, 224), "float32"), Y: T.Buffer((224, 224), "float32")) -> None:
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


class TestMemAccessInBranch(BaseCompactTest):
    @T.prim_func
    def before(a: T.handle) -> None:
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
    def expected(a: T.handle) -> None:
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


class TestAnnotatedOpaqueAccess(BaseCompactTest):

    is_lower_order_free = False

    @T.prim_func
    def before(a: T.handle) -> None:
        A = T.match_buffer(a, (1024,), "float32")
        with T.block():
            B = T.alloc_buffer((1024,), dtype="float32")
            C = T.alloc_buffer((1024,), dtype="float32")
            for i in range(0, 512):
                with T.block():
                    # no annotation, opaque access will cover full region
                    T.reads([])
                    T.writes([])
                    T.evaluate(
                        T.call_extern("opaque_extern_function", A.data, B.data, dtype="int32")
                    )
                    B[i] = A[i]
                with T.block():
                    # treat opaque access only access annotated regions, even if
                    # they are not compatible with actual buffer accesses.
                    T.reads([B[i]])
                    T.writes([C[i : i + 9]])
                    T.evaluate(
                        T.call_extern("opaque_extern_function", B.data, C.data, dtype="int32")
                    )
                    C[i] = B[i]

    @T.prim_func
    def expected(a: T.handle) -> None:
        A = T.match_buffer(a, (1024,), "float32")
        with T.block():
            B = T.alloc_buffer((1024,), dtype="float32")
            C = T.alloc_buffer((520,), dtype="float32")
            for i in range(0, 512):
                with T.block():
                    # no annotation, opaque access will cover full region
                    T.reads([])
                    T.writes([])
                    T.evaluate(
                        T.call_extern("opaque_extern_function", A.data, B.data, dtype="int32")
                    )
                    B[i] = A[i]
                with T.block():
                    # treat opaque access only access annotated regions, even if
                    # they are not compatible with actual buffer accesses.
                    T.reads([B[i]])
                    T.writes([C[i : i + 9]])
                    T.evaluate(
                        T.call_extern("opaque_extern_function", B.data, C.data, dtype="int32")
                    )
                    C[i] = B[i]


class TestSparseReadCache(BaseCompactTest):
    @T.prim_func
    def before(
        A_data: T.Buffer((819,), "float32"),
        B: T.Buffer((128,), "float32"),
        A_indptr: T.Buffer((129,), "int32"),
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
    def expected(
        A_data: T.Buffer((819,), "float32"),
        B: T.Buffer((128,), "float32"),
        A_indptr: T.Buffer((129,), "int32"),
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


class TestDataDependentRegion(BaseCompactTest):
    """Partial code of NMS, the `argsort_nms_cpu`'s region depends on inner allocated buffer
    `nkeep`'s value, thus the buffer should not be compacted with data dependent region extent."""

    @T.prim_func
    def before(
        p0: T.Buffer((30,), "float32"),
        p1: T.Buffer((1,), "int32"),
        hybrid_nms: T.Buffer((30,), "float32"),
    ):
        argsort_nms_cpu = T.decl_buffer([5], "int32", scope="global")
        for i in range(1):
            nkeep = T.decl_buffer([1], "int32", scope="global")
            if 0 < p1[i]:
                nkeep[0] = p1[i]
                if 2 < nkeep[0]:
                    nkeep[0] = 2
                for j in T.parallel(nkeep[0]):
                    for k in range(6):
                        hybrid_nms[i * 30 + j * 6 + k] = p0[
                            i * 30 + argsort_nms_cpu[i * 5 + j] * 6 + k
                        ]
                    hybrid_nms[i * 5 + j] = argsort_nms_cpu[i * 5 + j]
                if 2 < p1[i]:
                    for j in T.parallel(p1[i] - nkeep[0]):
                        for k in range(6):
                            hybrid_nms[i * 30 + j * 6 + nkeep[0] * 6 + k] = T.float32(-1)
                        hybrid_nms[i * 5 + j + nkeep[0]] = -1

    expected = before


class TestNarrowShape(BaseCompactTest):
    @T.prim_func
    def before(A: T.Buffer((10,), "float32"), B: T.Buffer((10,), "float32")) -> None:
        B_cache = T.alloc_buffer(10, "float32")
        for j in T.serial(3):
            for k in T.serial(4):
                with T.block("B_cache"):
                    T.where(j * 4 + k < 10)
                    B_cache[j * 4 + k] = B[j]
        for i in T.serial(10):
            A[i] = B_cache[i] + T.float32(1)

    @T.prim_func
    def expected(A: T.Buffer((10,), "float32"), B: T.Buffer((10,), "float32")) -> None:
        B_cache = T.alloc_buffer([10], dtype="float32")
        for j, k in T.grid(3, 4):
            with T.block("B_cache"):
                T.where(j * 4 + k < 10)
                T.reads(B[j])
                T.writes(B_cache[j * 4 + k])
                B_cache[j * 4 + k] = B[j]
        for i in T.serial(10):
            A[i] = B_cache[i] + T.float32(1)


class TestLetBinding(BaseCompactTest):
    @T.prim_func
    def before():
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

    expected = before


class TestNonIndexLetBinding(BaseCompactTest):
    @T.prim_func
    def before():
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

    expected = before


class TestSpatialTiledPadPooling(BaseCompactTest):
    @T.prim_func
    def before(X: T.Buffer((64, 112, 112), "int32"), Y: T.Buffer((64, 56, 56), "int32")) -> None:
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
    def expected(X: T.Buffer((64, 112, 112), "int32"), Y: T.Buffer((64, 56, 56), "int32")) -> None:
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


class TestComplexCase1(BaseCompactTest):
    """Meta-schedule matmul case for compact shared A, B matrix"""

    # fmt: off
    @T.prim_func
    def before(A: T.Buffer((960, 770), "float32"), B: T.Buffer((770, 2304), "float32"), C: T.Buffer((960, 2304), "float32")) -> None:
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
    def expected(A: T.Buffer((960, 770), "float32"), B: T.Buffer((770, 2304), "float32"), C: T.Buffer((960, 2304), "float32")) -> None:
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


class TestDependentBufferIndices(BaseCompactTest):
    """Check the upper bound on different indices could be independently estimated."""

    @T.prim_func
    def before():
        """This is a diagnal buffer access pattern"""
        for i in range(8):
            with T.block():
                A = T.alloc_buffer((256, 256), "float32")
                for j, k in T.grid(8, 8):
                    with T.block():
                        T.where(j * 8 + k < 60)
                        A[i * 64 + j * 8 + k, i * 64 + j * 8 + k] = 1.0

    @T.prim_func
    def expected() -> None:
        for i in T.serial(8):
            with T.block():
                A = T.alloc_buffer([60, 60], dtype="float32")
                for j, k in T.grid(8, 8):
                    with T.block():
                        T.where(j * 8 + k < 60)
                        A[j * 8 + k, j * 8 + k] = 1.0


class TestDependentBufferIndicesOfPackedMatmul(BaseCompactTest):
    """Check the outer dimension of the packed M-dim should be compacted to 1 wrt split condition."""

    @T.prim_func
    def before(
        A: T.Buffer((1020, 64), "float32"),
        B: T.Buffer((1000, 64), "float32"),
        C: T.Buffer((1020, 1000), "float32"),
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
    def expected(
        A: T.Buffer((1020, 64), "float32"),
        B: T.Buffer((1000, 64), "float32"),
        C: T.Buffer((1020, 1000), "float32"),
    ) -> None:
        for i0, i1 in T.grid(4, 1):
            with T.block():
                C_local2 = T.alloc_buffer([1, 1, 16, 1000, 16], dtype="float32", scope="local")
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


class TestTileAwareCompaction(BaseCompactTest):
    """Each partitioned tile could be independently compacted."""

    # it is not an opaque block case intentionally
    is_lower_order_free = False

    @property
    def before(self):
        @T.prim_func
        def main(
            A: T.Buffer((128, 128), "float32"),
            B: T.Buffer((128, 128), "float32"),
            C: T.Buffer((128, 128), "float32"),
        ):
            for i_0 in range(5, annotations={"pragma_loop_partition_hint": 1}):
                for j_0 in range(5, annotations={"pragma_loop_partition_hint": 1}):
                    A_local = T.decl_buffer((26, 128), scope="local")
                    B_local = T.decl_buffer((128, 26), scope="local")
                    C_local = T.decl_buffer((26, 26), scope="local")
                    for ax0, ax1 in T.grid(26, 128):
                        if i_0 * 26 + ax0 < 128:
                            A_local[ax0, ax1] = A[i_0 * 26 + ax0, ax1]
                    for ax0, ax1 in T.grid(128, 26):
                        if j_0 * 26 + ax1 < 128:
                            B_local[ax0, ax1] = B[ax0, j_0 * 26 + ax1]
                    for i_1, j_1, k in T.grid(26, 26, 128):
                        if i_0 * 26 + i_1 < 128 and j_0 * 26 + j_1 < 128:
                            if k == 0:
                                C_local[i_1, j_1] = T.float32(0)
                            C_local[i_1, j_1] = (
                                C_local[i_1, j_1] + A_local[i_1, k] * B_local[k, j_1]
                            )
                    for ax0, ax1 in T.grid(26, 26):
                        if i_0 * 26 + ax0 < 128 and j_0 * 26 + ax1 < 128:
                            C[i_0 * 26 + ax0, j_0 * 26 + ax1] = C_local[ax0, ax1]

        # Get partitioned workload to compact
        mod = tvm.IRModule.from_expr(main)
        with tvm.transform.PassContext(
            config={"tir.LoopPartition": {"partition_const_loop": True}}
        ):
            mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
            mod = tvm.tir.transform.LoopPartition()(mod)

        return mod["main"]

    @T.prim_func
    def expected(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        for i_0 in range(4):
            for j_0 in range(4):
                A_local_tile0 = T.decl_buffer((26, 128), scope="local")
                B_local_tile0 = T.decl_buffer((128, 26), scope="local")
                C_local_tile0 = T.decl_buffer((26, 26), scope="local")
                for ax0, ax1 in T.grid(26, 128):
                    A_local_tile0[ax0, ax1] = A[i_0 * 26 + ax0, ax1]
                for ax0, ax1 in T.grid(128, 26):
                    B_local_tile0[ax0, ax1] = B[ax0, j_0 * 26 + ax1]
                for i_1, j_1, k in T.grid(26, 26, 128):
                    if k == 0:
                        C_local_tile0[i_1, j_1] = T.float32(0)
                    C_local_tile0[i_1, j_1] = (
                        C_local_tile0[i_1, j_1] + A_local_tile0[i_1, k] * B_local_tile0[k, j_1]
                    )
                for ax0, ax1 in T.grid(26, 26):
                    C[i_0 * 26 + ax0, j_0 * 26 + ax1] = C_local_tile0[ax0, ax1]

            A_local_tile1 = T.decl_buffer((26, 128), scope="local")
            B_local_tile1 = T.decl_buffer((128, 24), scope="local")
            C_local_tile1 = T.decl_buffer((26, 24), scope="local")
            for ax0, ax1 in T.grid(26, 128):
                A_local_tile1[ax0, ax1] = A[i_0 * 26 + ax0, ax1]
            for ax0, ax1 in T.grid(128, 26):
                if ax1 < 24:
                    B_local_tile1[ax0, ax1] = B[ax0, ax1 + 104]
            for i_1, j_1, k in T.grid(26, 26, 128):
                if j_1 < 24:
                    if k == 0:
                        C_local_tile1[i_1, j_1] = T.float32(0)
                    C_local_tile1[i_1, j_1] = (
                        C_local_tile1[i_1, j_1] + A_local_tile1[i_1, k] * B_local_tile1[k, j_1]
                    )
            for ax0, ax1 in T.grid(26, 26):
                if ax1 < 24:
                    C[i_0 * 26 + ax0, ax1 + 104] = C_local_tile1[ax0, ax1]

        for j_0 in range(4):
            A_local_tile2 = T.decl_buffer((24, 128), scope="local")
            B_local_tile2 = T.decl_buffer((128, 26), scope="local")
            C_local_tile2 = T.decl_buffer((24, 26), scope="local")
            for ax0, ax1 in T.grid(26, 128):
                if ax0 < 24:
                    A_local_tile2[ax0, ax1] = A[ax0 + 104, ax1]
            for ax0, ax1 in T.grid(128, 26):
                B_local_tile2[ax0, ax1] = B[ax0, j_0 * 26 + ax1]
            for i_1, j_1, k in T.grid(26, 26, 128):
                if i_1 < 24:
                    if k == 0:
                        C_local_tile2[i_1, j_1] = T.float32(0)
                    C_local_tile2[i_1, j_1] = (
                        C_local_tile2[i_1, j_1] + A_local_tile2[i_1, k] * B_local_tile2[k, j_1]
                    )
            for ax0, ax1 in T.grid(26, 26):
                if ax0 < 24:
                    C[ax0 + 104, j_0 * 26 + ax1] = C_local_tile2[ax0, ax1]

        A_local_tile3 = T.decl_buffer((24, 128), scope="local")
        B_local_tile3 = T.decl_buffer((128, 24), scope="local")
        C_local_tile3 = T.decl_buffer((24, 24), scope="local")
        for ax0, ax1 in T.grid(26, 128):
            if ax0 < 24:
                A_local_tile3[ax0, ax1] = A[ax0 + 104, ax1]
        for ax0, ax1 in T.grid(128, 26):
            if ax1 < 24:
                B_local_tile3[ax0, ax1] = B[ax0, ax1 + 104]
        for i_1, j_1, k in T.grid(26, 26, 128):
            if i_1 < 24 and j_1 < 24:
                if k == 0:
                    C_local_tile3[i_1, j_1] = T.float32(0)
                C_local_tile3[i_1, j_1] = (
                    C_local_tile3[i_1, j_1] + A_local_tile3[i_1, k] * B_local_tile3[k, j_1]
                )
        for ax0, ax1 in T.grid(26, 26):
            if ax0 < 24 and ax1 < 24:
                C[ax0 + 104, ax1 + 104] = C_local_tile3[ax0, ax1]


class TestNonStrictCompactionForPaddedMatmul(BaseCompactTest):

    is_strict_mode = False

    @T.prim_func
    def before(
        A: T.Buffer((127, 127), "float32"),
        B: T.Buffer((127, 127), "float32"),
        C: T.Buffer((127, 127), "float32"),
    ):
        """A mock workload where the intermediate buffer allocation is not enought originally"""
        for i_0, j_0 in T.grid(4, 4):
            with T.block(""):
                T.reads(A[i_0 * 32 : i_0 * 32 + 32, 0:128], B[0:128, j_0 * 32 : j_0 * 32 + 32])
                T.writes(C[i_0 * 32 : i_0 * 32 + 32, j_0 * 32 : j_0 * 32 + 32])
                A_local = T.alloc_buffer((127, 127), scope="local")
                B_local = T.alloc_buffer((127, 127), scope="local")
                C_local = T.alloc_buffer((127, 127), scope="local")
                for ax0, ax1 in T.grid(32, 128):
                    with T.block("A_local"):
                        A_local[i_0 * 32 + ax0, ax1] = T.if_then_else(
                            i_0 * 32 + ax0 < 127, A[i_0 * 32 + ax0, ax1], 0.0
                        )
                for ax0, ax1 in T.grid(128, 32):
                    with T.block("B_local"):
                        B_local[ax0, j_0 * 32 + ax1] = T.if_then_else(
                            j_0 * 32 + ax1 < 127, B[ax0, j_0 * 32 + ax1], 0.0
                        )
                for i_1, j_1, k in T.grid(32, 32, 128):
                    with T.block("compute"):
                        T.where(i_0 * 32 + i_1 < 127 and j_0 * 32 + j_1 < 127)
                        if k == 0:
                            C_local[i_0 * 32 + i_1, j_0 * 32 + j_1] = T.float32(0)
                        C_local[i_0 * 32 + i_1, j_0 * 32 + j_1] = (
                            C_local[i_0 * 32 + i_1, j_0 * 32 + j_1]
                            + A_local[i_0 * 32 + i_1, k] * B_local[k, j_0 * 32 + j_1]
                        )
                for ax0, ax1 in T.grid(32, 32):
                    with T.block("C_local"):
                        T.where(i_0 * 32 + ax0 < 127 and j_0 * 32 + ax1 < 127)
                        C[i_0 * 32 + ax0, j_0 * 32 + ax1] = C_local[i_0 * 32 + ax0, j_0 * 32 + ax1]

    @T.prim_func
    def expected(
        A: T.Buffer((127, 127), "float32"),
        B: T.Buffer((127, 127), "float32"),
        C: T.Buffer((127, 127), "float32"),
    ):
        for i_0, j_0 in T.grid(4, 4):
            with T.block(""):
                T.reads(A[i_0 * 32 : i_0 * 32 + 32, 0:128], B[0:128, j_0 * 32 : j_0 * 32 + 32])
                T.writes(C[i_0 * 32 : i_0 * 32 + 32, j_0 * 32 : j_0 * 32 + 32])
                A_local = T.alloc_buffer((32, 128), scope="local")
                B_local = T.alloc_buffer((128, 32), scope="local")
                C_local = T.alloc_buffer((32, 32), scope="local")
                for ax0, ax1 in T.grid(32, 128):
                    with T.block("A_local"):
                        A_local[ax0, ax1] = T.if_then_else(
                            i_0 * 32 + ax0 < 127, A[i_0 * 32 + ax0, ax1], T.float32(0)
                        )
                for ax0, ax1 in T.grid(128, 32):
                    with T.block("B_local"):
                        B_local[ax0, ax1] = T.if_then_else(
                            j_0 * 32 + ax1 < 127, B[ax0, j_0 * 32 + ax1], T.float32(0)
                        )
                for i_1, j_1, k in T.grid(32, 32, 128):
                    with T.block("compute"):
                        T.where(i_0 * 32 + i_1 < 127 and j_0 * 32 + j_1 < 127)
                        if k == 0:
                            C_local[i_1, j_1] = T.float32(0)
                        C_local[i_1, j_1] = C_local[i_1, j_1] + A_local[i_1, k] * B_local[k, j_1]
                for ax0, ax1 in T.grid(32, 32):
                    with T.block("C_local"):
                        T.where(i_0 * 32 + ax0 < 127 and j_0 * 32 + ax1 < 127)
                        C[i_0 * 32 + ax0, j_0 * 32 + ax1] = C_local[ax0, ax1]


class TestNotCompactAliasBuffer(BaseCompactTest):

    # it is not testcase on block form
    is_lower_order_free = False

    @T.prim_func
    def before():
        """Partially accessed buffer, but should not compact
        because existence of aliasing buffer B."""
        data = T.allocate([1024], "int8")
        A = T.decl_buffer([1024], "int8", data)
        B = T.decl_buffer([512], "float16", data)
        for i in range(10):
            A[i] = A[i] + T.int8(1)
        for i in range(10):
            B[i] = B[i] + T.float16(1)

    expected = before


class TestNotCompactBufferWithDifferentDtype(BaseCompactTest):

    # it is not testcase on block form
    is_lower_order_free = False

    @T.prim_func
    def before():
        """Partially accessed buffer, but should not compact
        because existence of aliasing buffer B."""
        data = T.allocate([1024], "int8")
        A = T.decl_buffer([256], "int32", data)
        for i in range(10):
            A[i] = A[i] + 1

    expected = before


class TestNonBoolCondition(BaseCompactTest):

    # it is not testcase on block form
    is_lower_order_free = False

    @T.prim_func
    def before():
        data = T.allocate([12], "int32")
        A = T.Buffer([12], "int32", data)
        for i in range(10):
            if i:
                A[i] = A[i] + 1

    @T.prim_func
    def expected():
        data = T.allocate([9], "int32")
        A = T.Buffer([9], "int32", data)
        for i in range(10):
            if i:
                A[i - 1] = A[i - 1] + 1


def test_lower_te():
    x = te.placeholder((1,))
    y = te.compute((1,), lambda i: x[i] + 2)
    s = te.create_schedule(y.op)
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
    mod = tvm.tir.transform.CompactBufferAllocation()(orig_mod)
    tvm.ir.assert_structural_equal(mod, orig_mod)  # CompactBufferAllocation should do nothing on TE


class TestCompactSymbolicBound0:
    """Test symbolic bound that get compacted to constant"""

    @T.prim_func
    def before(x: T.handle, y: T.handle, n: T.int64):
        X = T.match_buffer(x, (T.int64(8), n * T.int64(32)))
        Y = T.match_buffer(y, (T.int64(8), n * T.int64(32)))
        for i, k_0 in T.grid(T.int64(8), n):
            with T.block(""):
                X_global = T.alloc_buffer((T.int64(8), n * T.int64(32)))
                for ax0 in range(T.int64(32)):
                    with T.block("X_global"):
                        X_global[i, k_0 * T.int64(32) + ax0] = X[i, k_0 * T.int64(32) + ax0]
                for k_1 in range(T.int64(32)):
                    with T.block("Y"):
                        Y[i, k_0 * T.int64(32) + k_1] = X_global[i, k_0 * T.int64(32) + k_1]

    @T.prim_func
    def expected(x: T.handle, y: T.handle, n: T.int64):
        X = T.match_buffer(x, (T.int64(8), n * T.int64(32)))
        Y = T.match_buffer(y, (T.int64(8), n * T.int64(32)))
        for i, k_0 in T.grid(T.int64(8), n):
            with T.block(""):
                X_global = T.alloc_buffer((T.int64(1), T.int64(32)))
                for ax0 in range(T.int64(32)):
                    with T.block("X_global"):
                        X_global[T.int64(0), ax0] = X[i, k_0 * T.int64(32) + ax0]
                for k_1 in range(T.int64(32)):
                    with T.block("Y"):
                        Y[i, k_0 * T.int64(32) + k_1] = X_global[T.int64(0), k_1]


class TestCompactSymbolicBound1:
    """Test symbolic bound that get compacted to constant"""

    @T.prim_func
    def before(x: T.handle, y: T.handle, n: T.int64):
        X = T.match_buffer(x, (T.int64(8), n * T.int64(32)))
        Y = T.match_buffer(y, (T.int64(8), n * T.int64(32)))
        for i, k_0 in T.grid(T.int64(8), n):
            with T.block(""):
                X_global = T.alloc_buffer((T.int64(8), n * T.int64(32)))
                with T.block("X_global"):
                    for x0 in range(T.int64(32)):
                        X_global[i, k_0 * T.int64(32) + x0] = X[i, k_0 * T.int64(32) + x0]
                with T.block("Y"):
                    for x1 in range(T.int64(32)):
                        Y[i, k_0 * T.int64(32) + x1] = X_global[i, k_0 * T.int64(32) + x1]

    @T.prim_func
    def expected(x: T.handle, y: T.handle, n: T.int64):
        X = T.match_buffer(x, (T.int64(8), n * T.int64(32)))
        Y = T.match_buffer(y, (T.int64(8), n * T.int64(32)))
        # with T.block("root"):
        for i, k_0 in T.grid(T.int64(8), n):
            with T.block(""):
                X_global = T.alloc_buffer((T.int64(1), T.int64(32)))
                with T.block("X_global"):
                    for x0 in range(T.int64(32)):
                        X_global[T.int64(0), x0] = X[i, k_0 * T.int64(32) + x0]
                with T.block("Y"):
                    for x1 in range(T.int64(32)):
                        Y[i, k_0 * T.int64(32) + x1] = X_global[T.int64(0), x1]


class TestSymbolicDiagMaskCase:
    """Test symbolic allocation not too complex"""

    @T.prim_func
    def before(p_output0: T.handle, n: T.int32):
        A = T.match_buffer(p_output0, (1, 1, n, n))
        B = T.alloc_buffer((n, n))
        for i in T.thread_binding(256, thread="blockIdx.x"):
            for j in T.thread_binding(256, thread="threadIdx.x"):
                for k in range((n * n + 65535) // 65536):
                    with T.block("make_diag_mask_te"):
                        T.where((k * 256 + i) * 256 + j < n * n)
                        T.reads()
                        T.writes(B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n])
                        B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n] = T.Select(
                            (k * 65536 + i * 256 + j) // n < (k * 65536 + i * 256 + j) % n,
                            T.float32(-3.4028234663852886e38),
                            T.float32(3.4028234663852886e38),
                        )
        for i in T.thread_binding(256, thread="blockIdx.x"):
            for j in T.thread_binding(256, thread="threadIdx.x"):
                for k in range((n * n + 65535) // 65536):
                    with T.block("T_broadcast_to"):
                        T.where((k * 256 + i) * 256 + j < n * n)
                        T.reads(B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n])
                        T.writes(
                            A[0, 0, (k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n]
                        )
                        A[0, 0, (k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n] = B[
                            (k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n
                        ]

    @T.prim_func
    def expected(p_output0: T.handle, n: T.int32):
        A = T.match_buffer(p_output0, (1, 1, n, n))
        B = T.alloc_buffer((n, n))
        for i in T.thread_binding(256, thread="blockIdx.x"):
            for j in T.thread_binding(256, thread="threadIdx.x"):
                for k in range((n * n + 65535) // 65536):
                    with T.block("make_diag_mask_te"):
                        T.where(k * 65536 + i * 256 + j < n * n)
                        T.reads()
                        T.writes(B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n])
                        B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n] = T.Select(
                            (k * 65536 + i * 256 + j) // n < (k * 65536 + i * 256 + j) % n,
                            T.float32(-3.4028234663852886e38),
                            T.float32(3.4028234663852886e38),
                        )
        for i in T.thread_binding(256, thread="blockIdx.x"):
            for k in T.thread_binding(256, thread="threadIdx.x"):
                for k in range((n * n + 65535) // 65536):
                    with T.block("T_broadcast_to"):
                        T.where(k * 65536 + i * 256 + k < n * n)
                        T.reads(B[(k * 65536 + i * 256 + k) // n, (k * 65536 + i * 256 + k) % n])
                        T.writes(
                            A[0, 0, (k * 65536 + i * 256 + k) // n, (k * 65536 + i * 256 + k) % n]
                        )
                        A[0, 0, (k * 65536 + i * 256 + k) // n, (k * 65536 + i * 256 + k) % n] = B[
                            (k * 65536 + i * 256 + k) // n, (k * 65536 + i * 256 + k) % n
                        ]


if __name__ == "__main__":
    tvm.testing.main()
