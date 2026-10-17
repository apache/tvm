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
from __future__ import annotations

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.transform import FlattenTensor


def _transform():
    return tvm_tirx.ir.transform.Sequential(
        [
            tvm_tirx.tirx.transform.FlattenTensor(),
            tvm_tirx.tirx.transform.StmtSimplify(),
        ]
    )


def test_elementwise():
    """2-d tensors are flattened to 1-d"""

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
            for i in T.serial(0, 16):
                B_new = T.alloc_tensor(ty_args=[T.Tensor([1, 16], "float32")])
                for j in T.serial(0, 16):
                    B_new[0, j] = A[i, j] + 1.0
                for j in T.serial(0, 16):
                    C[i, j] = B_new[0, j] * 2.0

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
            A_1 = T.decl_tensor(
                A.data, ty_args=[T.Tensor(256, "float32", layout=None, data=A.data)]
            )
            C_1 = T.decl_tensor(
                C.data, ty_args=[T.Tensor(256, "float32", layout=None, data=C.data)]
            )
            for i in T.serial(0, 16):
                B_new = T.alloc_tensor(ty_args=[T.Tensor([16], "float32", layout=None)])
                for j in T.serial(0, 16):
                    B_new[j] = A_1[((i * 16) + j)] + 1.0
                for j in T.serial(0, 16):
                    C_1[((i * 16) + j)] = B_new[j] * 2.0

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_elementwise_without_decl_tensor():
    """2-d tensors are flattened to 1-d

    Like test_elementwise, but the TIR doesn't have the DeclTensor
    node.  The T.Tensor declaration applies only during the
    parsing the TVMScript, and doesn't occur in the TIR itself.  In
    this case, the allocation should be assumed to be targeting flat
    memory, and should be flattened to a 1-d allocation.
    """

    @I.ir_module(check_well_formed=False)
    class Before:
        @T.function
        def main(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
            for i in T.serial(0, 16):
                B_new_buf = T.alloc_tensor(ty_args=[T.Tensor((1, 16), "float32")])
                B_new = T.decl_tensor(
                    B_new_buf.data, ty_args=[T.Tensor([1, 16], "float32", data=B_new_buf.data)]
                )
                for j in T.serial(0, 16):
                    B_new[0, j] = A[i, j] + 1.0
                for j in T.serial(0, 16):
                    C[i, j] = B_new[0, j] * 2.0

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.function
        def main(input_A: T.Tensor((16, 16), "float32"), input_C: T.Tensor((16, 16), "float32")):
            A = T.decl_tensor(
                input_A.data, ty_args=[T.Tensor(256, "float32", layout=None, data=input_A.data)]
            )
            C = T.decl_tensor(
                input_C.data, ty_args=[T.Tensor(256, "float32", layout=None, data=input_C.data)]
            )
            for i in T.serial(0, 16):
                B_new_buf = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32", layout=None)])
                B_new = T.decl_tensor(
                    B_new_buf.data,
                    ty_args=[T.Tensor(16, "float32", layout=None, data=B_new_buf.data)],
                )
                for j in T.serial(0, 16):
                    B_new[j] = A[((i * 16) + j)] + 1.0
                for j in T.serial(0, 16):
                    C[((i * 16) + j)] = B_new[j] * 2.0

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_gpu():
    """Tensor flattening may have indices based on GPU thread vars"""

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
            i0 = T.launch_thread("blockIdx.x", 4)
            i1 = T.launch_thread("threadIdx.x", 2)
            i2 = T.launch_thread("vthread", 2)
            B = T.alloc_tensor(ty_args=[T.Tensor([1, 16], "float32", scope="local")])
            for j in range(0, 16):
                B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
            for j in range(0, 16):
                C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
            A_1 = T.decl_tensor(
                A.data, ty_args=[T.Tensor(256, "float32", layout=None, data=A.data)]
            )
            C_1 = T.decl_tensor(
                C.data, ty_args=[T.Tensor(256, "float32", layout=None, data=C.data)]
            )

            i0 = T.launch_thread("blockIdx.x", 4)
            i1 = T.launch_thread("threadIdx.x", 2)
            i2 = T.launch_thread("vthread", 2)
            B = T.alloc_tensor(ty_args=[T.Tensor([16], "float32", scope="local", layout=None)])
            for j in range(0, 16):
                B[j] = A_1[i0 * 64 + i1 * 32 + i2 * 16 + j] + 1.0
            for j in range(0, 16):
                C_1[i0 * 64 + i1 * 32 + i2 * 16 + j] = B[j] * 2.0

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_symbolic():
    """Dynamically-sized arrrays are flattened"""

    @I.ir_module
    class Before:
        @T.function
        def main(
            A: T.Tensor((n, m), "float32"),  # noqa: F821
            C: T.Tensor((n, m), "float32"),  # noqa: F821
            n: T.int32,
            m: T.int32,
        ) -> None:
            for i in range(0, n):
                B = T.alloc_tensor(ty_args=[T.Tensor([m], "float32")])
                for j in range(0, m):
                    B[j] = A[i, j] + 1.0
                for j in range(0, m):
                    C[i, j] = B[j] * 2.0

    @I.ir_module
    class Expected:
        @T.function
        def main(
            A: T.Tensor((n, m), "float32"),  # noqa: F821
            C: T.Tensor((n, m), "float32"),  # noqa: F821
            n: T.int32,
            m: T.int32,
        ) -> None:
            A_1 = T.decl_tensor(
                A.data, ty_args=[T.Tensor(n * m, "float32", layout=None, data=A.data)]
            )
            C_1 = T.decl_tensor(
                C.data, ty_args=[T.Tensor(n * m, "float32", layout=None, data=C.data)]
            )

            for i in range(0, n):
                B = T.alloc_tensor(ty_args=[T.Tensor([m], "float32", layout=None)])
                for j in range(0, m):
                    B[j] = A_1[i * m + j] + 1.0
                for j in range(0, m):
                    C_1[i * m + j] = B[j] * 2.0

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_fused_symbolic():
    """Dynamically-sized arrrays with fused iterator which can be flattened"""

    @I.ir_module
    class Before:
        @T.function
        def main(
            A: T.Tensor((32, n, n), "float32"),  # noqa: F821
            B: T.Tensor((32, n, n), "float32"),  # noqa: F821
            n: T.int32,
        ) -> None:
            for i in range(0, n * n * 32):
                B[i // (n * n), (i % (n * n)) // n, i % n] = A[
                    i // (n * n), (i % (n * n)) // n, i % n
                ]

    @I.ir_module
    class Expected:
        @T.function
        def main(
            input_A: T.Tensor((32, n, n), "float32"),  # noqa: F821
            input_B: T.Tensor((32, n, n), "float32"),  # noqa: F821
            n: T.int32,
        ) -> None:
            A = T.decl_tensor(
                input_A.data,
                ty_args=[T.Tensor(n * n * 32, "float32", layout=None, data=input_A.data)],
            )
            B = T.decl_tensor(
                input_B.data,
                ty_args=[T.Tensor(n * n * 32, "float32", layout=None, data=input_B.data)],
            )

            for i in range(0, n * n * 32):
                B[i] = A[i]

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_fused_symbolic_with_predicate():
    """Dynamically-sized arrrays with fused iterator which can be flattened with extra predicate"""

    @I.ir_module
    class Before:
        @T.function
        def main(
            A: T.Tensor((32, n, n), "float32"),  # noqa: F821
            B: T.Tensor((32, n, n), "float32"),  # noqa: F821
            n: T.int32,
        ) -> None:
            for bx, tx in T.grid((n * n + 1) // 2, 64):
                if bx * 64 + tx < n * n * 32:
                    B[
                        (bx * 64 + tx) // (n * n),
                        ((bx * 64 + tx) % (n * n)) // n,
                        (bx * 64 + tx) % n,
                    ] = A[
                        (bx * 64 + tx) // (n * n),
                        ((bx * 64 + tx) % (n * n)) // n,
                        (bx * 64 + tx) % n,
                    ]

    @I.ir_module
    class Expected:
        @T.function
        def main(
            input_A: T.Tensor((32, n, n), "float32"),  # noqa: F821
            input_B: T.Tensor((32, n, n), "float32"),  # noqa: F821
            n: T.int32,
        ) -> None:
            A = T.decl_tensor(
                input_A.data,
                ty_args=[T.Tensor(n * n * 32, "float32", layout=None, data=input_A.data)],
            )
            B = T.decl_tensor(
                input_B.data,
                ty_args=[T.Tensor(n * n * 32, "float32", layout=None, data=input_B.data)],
            )

            for bx, tx in T.grid((n * n + 1) // 2, 64):
                if bx * 64 + tx < n * n * 32:
                    B[bx * 64 + tx] = A[bx * 64 + tx]

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_multi_alloc():
    """If multiple allocations occur, all are flattened."""

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4, 32), "float32"), D: T.Tensor((4, 32), "float32")):
            for i, j in T.grid(4, 32):
                B = T.alloc_tensor(ty_args=[T.Tensor((4, 32), "float32", scope="global")])
                C = T.alloc_tensor(ty_args=[T.Tensor((4, 32), "float32", scope="global")])
                B[i, j] = A[i, j] + 1.0
                C[i, j] = A[i, j] + B[i, j]
                D[i, j] = C[i, j] * 2.0

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4, 32), "float32"), D: T.Tensor((4, 32), "float32")):
            A_1 = T.decl_tensor(
                A.data, ty_args=[T.Tensor(128, "float32", layout=None, data=A.data)]
            )
            D_1 = T.decl_tensor(
                D.data, ty_args=[T.Tensor(128, "float32", layout=None, data=D.data)]
            )

            for i, j in T.grid(4, 32):
                B = T.alloc_tensor(ty_args=[T.Tensor([128], "float32", layout=None)])
                C = T.alloc_tensor(ty_args=[T.Tensor([128], "float32", layout=None)])
                B[i * 32 + j] = A_1[i * 32 + j] + 1.0
                C[i * 32 + j] = A_1[i * 32 + j] + B[i * 32 + j]
                D_1[i * 32 + j] = C[i * 32 + j] * 2.0

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_strided():
    """Indices for flattened tensors use the specified striding."""

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
            for i0 in T.serial(4):
                B = T.alloc_tensor(ty_args=[T.Tensor([4, 17], "float32")])
                B_1 = T.decl_tensor(
                    B.data,
                    ty_args=[
                        T.Tensor(
                            [4, 16],
                            "float32",
                            layout=T.TileLayout(T.S[[4, 16] : [17, 1]]),
                            data=B.data,
                        )
                    ],
                )
                for i1, j in T.grid(4, 16):
                    B_1[i1, j] = A[i0 * 4 + i1, j] + 1.0
                for i1, j in T.grid(4, 16):
                    C[i0 * 4 + i1, j] = B_1[i1, j] * 2.0

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
            A_1 = T.decl_tensor(
                A.data, ty_args=[T.Tensor(256, "float32", layout=None, data=A.data)]
            )
            C_1 = T.decl_tensor(
                C.data, ty_args=[T.Tensor(256, "float32", layout=None, data=C.data)]
            )
            for i0 in T.serial(0, 4):
                B = T.alloc_tensor(ty_args=[T.Tensor([68], "float32", layout=None)])
                B_1 = T.decl_tensor(
                    B.data, ty_args=[T.Tensor([68], "float32", layout=None, data=B.data)]
                )
                for i1 in T.serial(0, 4):
                    for j in T.serial(0, 16):
                        B_1[i1 * 17 + j] = A_1[i0 * 64 + i1 * 16 + j] + 1.0
                for i1 in T.serial(0, 4):
                    for j in T.serial(0, 16):
                        C_1[i0 * 64 + i1 * 16 + j] = B_1[i1 * 17 + j] * 2.0

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_boolean():
    """Boolean tensors are flattened but kept as bool (no int8 backing array)"""

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor(10, "bool"), B: T.Tensor(10, "bool")) -> None:
            for i0 in T.serial(10):
                B[i0] = A[i0]

    @I.ir_module
    class Expected:
        @T.function
        def main(input_A: T.Tensor(10, "bool"), input_B: T.Tensor(10, "bool")) -> None:
            A = T.decl_tensor(
                input_A.data, ty_args=[T.Tensor(10, "bool", layout=None, data=input_A.data)]
            )
            B = T.decl_tensor(
                input_B.data, ty_args=[T.Tensor(10, "bool", layout=None, data=input_B.data)]
            )
            # body
            for i0 in T.serial(10):
                B[i0] = A[i0]

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_flatten_inside_block():
    """Flatten allocations and accesses after lowering a schedulable block."""

    @I.ir_module
    class Before:
        @T.function
        def main():
            A = T.alloc_tensor(ty_args=[T.Tensor([32, 32], "float32")])
            for i, j in T.grid(32, 32):
                T.evaluate(A[i, j])

    @I.ir_module
    class Expected:
        @T.function
        def main():
            A = T.alloc_tensor(ty_args=[T.Tensor([1024], "float32", layout=None)])
            for i, j in T.grid(32, 32):
                T.evaluate(A[i * 32 + j])

    After = _transform()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("value", [None, True, False])
def test_build_with_optional_unroll_explicit(value):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4, 5, 6), "int16"), B: T.Tensor((4, 5, 6), "int16")):
            for ax0 in T.serial(4, annotations={"tirx.unroll_explicit": value}):
                for ax1, ax2 in T.grid(5, 6):
                    B[ax0, ax1, ax2] = A[ax0, ax1, ax2]

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4, 5, 6), "int16"), B: T.Tensor((4, 5, 6), "int16")):
            A_1 = T.decl_tensor(
                A.data, ty_args=[T.Tensor((120,), "int16", layout=None, data=A.data)]
            )
            B_1 = T.decl_tensor(
                B.data, ty_args=[T.Tensor((120,), "int16", layout=None, data=B.data)]
            )
            for ax0 in T.serial(4, annotations={"tirx.unroll_explicit": value}):
                for ax1, ax2 in T.grid(5, 6):
                    B_1[ax0 * 30 + ax1 * 6 + ax2] = A_1[ax0 * 30 + ax1 * 6 + ax2]

    After = FlattenTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    tvm_tirx.tirx.compile(Before, target="llvm").mod


def test_flatten_remaps_loads_in_view_shape():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            n = T.alloc_local([1], "int32")
            n[0] = 8
            data = T.alloc_shared([64], "float16")
            view = T.decl_tensor(data.data, ty_args=[T.Tensor((n[0],), "float16", scope="shared")])
            view[0] = T.float16(0)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            n = T.alloc_local([1], "int32", layout=None)
            n[0] = 8
            data = T.alloc_shared([64], "float16", layout=None)
            view = T.decl_tensor(
                data.data, ty_args=[T.Tensor((n[0],), "float16", scope="shared", layout=None)]
            )
            view[0] = T.float16(0)

    After = FlattenTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    body = After["main"].body
    assert body[3].var.shape[0].source.same_as(body[0].var)
    assert not tvm_tirx.tirx.analysis.undefined_vars(body, [])


def test_flatten_remaps_loads_in_folded_elem_offset():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            n = T.alloc_local([1], "int32")
            n[0] = 4
            base = T.alloc_shared([128], "uint64")
            mbar = T.decl_tensor(
                base.data, ty_args=[T.Tensor((1,), "uint64", elem_offset=n[0], scope="shared")]
            )
            mbar[0] = T.uint64(1)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            n = T.alloc_local([1], "int32", layout=None)
            n[0] = 4
            base = T.alloc_shared([128], "uint64", layout=None)
            mbar = T.decl_tensor(
                base.data,
                ty_args=[
                    T.Tensor((1,), "uint64", elem_offset=T.int32(0), scope="shared", layout=None)
                ],
            )
            mbar[n[0]] = T.uint64(1)

    After = FlattenTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    body = After["main"].body
    assert body[-1].indices[0].source.same_as(body[0].var)
    assert not tvm_tirx.tirx.analysis.undefined_vars(body, [])


def test_flatten_keeps_identity_of_already_flat_tensors():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            flat = T.alloc_shared([32], "float32", layout=None)
            flat[0] = T.float32(0)

    Expected = Before
    After = FlattenTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    assert After["main"].body[0].var.same_as(Before["main"].body[0].var)


def test_tensor_flatten_offset():
    m = T.dynamic("m", "int32")
    n = T.dynamic("n", "int32")

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(m: T.int32, n: T.int32, A: T.Tensor((m, n), "float32", elem_offset=T.int32(100))):
            T.evaluate(A[2, 3])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(m: T.int32, n: T.int32, A: T.Tensor((m, n), "float32", elem_offset=T.int32(100))):
            flat = T.decl_tensor(
                A.data, ty_args=[T.Tensor((m * n,), "float32", elem_offset=T.int32(0), layout=None)]
            )
            T.evaluate(flat[n * 2 + 103])

    After = FlattenTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize(
    "shape,strides,span",
    [([16, 32], None, 512), ([16], None, 16), ([], None, 1), ([16, 32], [40, 1], 640)],
    ids=["matrix", "vector", "scalar", "strided"],
)
def test_tensor_flatten(shape, strides, span):
    layout = T.TileLayout(T.S[shape:strides]) if strides else None

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor(shape, "float32", layout=layout),
        ):
            T.evaluate(A[tuple(0 for _ in shape)])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(
            A: T.Tensor(shape, "float32", layout=layout),
        ):
            flat = T.decl_tensor(A.data, ty_args=[T.Tensor((span,), "float32", layout=None)])
            T.evaluate(flat[0])

    After = FlattenTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    source = After["main"].params[0]
    flat = After["main"].body[0].var
    assert not source.same_as(flat)
    assert flat.data.args[0].same_as(flat)
    assert flat.data.op.name == "tirx.tensor_data_ptr"


@pytest.mark.parametrize(
    "pattern", [1, 2, 3, 4], ids=["strided", "nested-mod", "split-sum", "unmerged"]
)
def test_tensor_index_merge_mult_mod(pattern):
    m = T.dynamic("m", "int32")
    n = T.dynamic("n", "int32")
    s = T.dynamic("s", "int32")

    def tensor_layout(m, n, s):
        return T.TileLayout(T.S[(m, n) : (s, 1)]) if pattern == 1 else None

    def storage_span(m, n, s):
        return T.max(s * m, (m - 1) * s + n) if pattern == 1 else m * n

    def indices(k0, k1, s, n):
        if pattern == 1:
            return ((k0 % k1) // s, (k0 % k1) % s + (k0 // k1) * k1)
        if pattern == 2:
            return ((k0 % (k1 // s)) // n, (k0 % (k1 // s)) % n + k0 % k1)
        if pattern == 3:
            q = (k0 // (k1 // s)) * (k1 // s)
            return (q // n + (k0 % (k1 // s)) // n, q % n + (k0 % (k1 // s)) % n)
        return ((k0 % (k1 // s)) // n, (k0 % (k1 // n)) % n + k0 % k1)

    def expected_index(k0, k1, s, n):
        if pattern in (1, 3):
            return k0
        if pattern == 2:
            return k0 % (k1 // s) + k0 % k1
        return ((k0 % (k1 // s)) // n) * n + (k0 % (k1 // n)) % n + k0 % k1

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            m: T.int32,
            n: T.int32,
            s: T.int32,
            k0: T.int32,
            k1: T.int32,
            A: T.Tensor(
                (m, n),
                "float32",
                layout=tensor_layout(m, n, s),
            ),
        ):
            T.evaluate(A[indices(k0, k1, s, n)])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(
            m: T.int32,
            n: T.int32,
            s: T.int32,
            k0: T.int32,
            k1: T.int32,
            A: T.Tensor(
                (m, n),
                "float32",
                layout=tensor_layout(m, n, s),
            ),
        ):
            flat = T.decl_tensor(
                A.data, ty_args=[T.Tensor((storage_span(m, n, s),), "float32", layout=None)]
            )
            T.evaluate(flat[expected_index(k0, k1, s, n)])

    After = FlattenTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("factored", [False, True], ids=["linear", "factored"])
def test_tensor_index_merge_rank_four(factored):
    def indices(i, j, k):
        if factored:
            q = i * 49 + j * 28 + k // 1024
            return (q // 14 // 14, (q // 14) % 14, (i * 7 + k // 1024) % 14, k % 1024)
        q = i * 50176 + j * 28672 + k
        return (q // 1024 // 14 // 14, (q // 1024 // 14) % 14, (q // 1024) % 14, q % 1024)

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(i: T.int32, j: T.int32, k: T.int32, A: T.Tensor((1, 14, 14, 1024), "float32")):
            T.evaluate(A[indices(i, j, k)])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(i: T.int32, j: T.int32, k: T.int32, A: T.Tensor((1, 14, 14, 1024), "float32")):
            flat = T.decl_tensor(A.data, ty_args=[T.Tensor((200704,), "float32", layout=None)])
            T.evaluate(flat[i * 50176 + j * 28672 + k])

    After = FlattenTensor()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
