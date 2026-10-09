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
# pylint: disable=invalid-name, missing-docstring

import tvm_ffi

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def _is_buffer_binding(node, *op_names):
    return (
        isinstance(node, tvm.tirx.Bind)
        and isinstance(node.value, tvm.ir.Call)
        and isinstance(node.value.op, tvm.ir.Op)
        and node.value.op.name in op_names
    )


def test_rewrite_to_shuffle_0():
    transform = tvm.tirx.transform.PointerValueTypeRewrite()

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "float32"), B: T.Tensor((4,), "float32")):
            A_local = T.alloc_tensor((16,), scope="local")
            for i in range(4):
                A_local[T.ramp(i * 4, 1, 4)] = A[T.ramp(i * 4, 1, 4)]
            for i in range(4):
                B[i] = A_local[i * 4] + A_local[i * 4 + 1] + A_local[i * 4 + 2] + A_local[i * 4 + 3]

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4,), "float32x4", layout=None), B: T.Tensor((4,), "float32")):
            A_local = T.alloc_tensor((4,), "float32x4", scope="local", layout=None)
            for i in range(4):
                A_local[T.Div(i * 4, 4)] = A[T.Div(i * 4, 4)]
            for i in range(4):
                B[i] = (
                    T.Shuffle([A_local[T.Div(i * 4, 4)]], [0])
                    + T.Shuffle([A_local[T.Div(i * 4 + 1, 4)]], [1])
                    + T.Shuffle([A_local[T.Div(i * 4 + 2, 4)]], [2])
                    + T.Shuffle([A_local[T.Div(i * 4 + 3, 4)]], [3])
                )

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_rewrite_to_shuffle_1():
    transform = tvm.tirx.transform.PointerValueTypeRewrite()

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((8,), "float32"), B: T.Tensor((1,), "float32")):
            A_local = T.alloc_tensor((8,), scope="local")
            A_local[T.ramp(0, 1, 4)] = A[T.ramp(0, 1, 4)]
            A_local[T.ramp(4, 1, 4)] = A[T.ramp(4, 1, 4)]
            B[0] = (
                A_local[0]
                + A_local[1]
                + A_local[2]
                + A_local[3]
                + A_local[4]
                + A_local[5]
                + A_local[6]
                + A_local[7]
            )

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((2,), "float32x4", layout=None), B: T.Tensor((1,), "float32")):
            A_local = T.alloc_tensor((2,), "float32x4", scope="local", layout=None)
            A_local[0] = A[0]
            A_local[1] = A[1]
            B[0] = (
                T.Shuffle([A_local[0]], [0])
                + T.Shuffle([A_local[0]], [1])
                + T.Shuffle([A_local[0]], [2])
                + T.Shuffle([A_local[0]], [3])
                + T.Shuffle([A_local[1]], [0])
                + T.Shuffle([A_local[1]], [1])
                + T.Shuffle([A_local[1]], [2])
                + T.Shuffle([A_local[1]], [3])
            )

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_address_of():
    transform = tvm.tirx.transform.PointerValueTypeRewrite()

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
            for i in range(4):
                T.evaluate(T.address_of(A[i * 4]))
                B[T.ramp(i * 4, 1, 4)] = A[T.ramp(i * 4, 1, 4)]

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((16,), "float32"), B: T.Tensor((4,), "float32x4", layout=None)):
            for i in range(4):
                T.evaluate(T.address_of(A[i * 4]))
                B[T.Div(i * 4, 4)] = A[T.ramp(i * 4, 1, 4)]

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_scalar_read_without_write():
    transform = tvm.tirx.transform.PointerValueTypeRewrite()

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            for i in range(4):
                T.evaluate(A[i * 4])

    # Expected is the same as Before - no transformation
    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            for i in range(4):
                T.evaluate(A[i * 4])

    After = transform(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_decl_buffer_alias_chain_uses_flat_root_map():
    transform = tvm.tirx.transform.PointerValueTypeRewrite()

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            A_view = T.decl_tensor((16,), "float32", data=A.data_ptr())
            A_view_2 = T.decl_tensor((16,), "float32", data=A_view.data_ptr())
            for i in range(4):
                A_view_2[T.ramp(i * 4, 1, 4)] = T.broadcast(T.float32(1), 4)

    After = transform(Before)
    assert tvm.tirx.analysis.verify_well_formed(After)
    func = After["main"]
    assert func.params[0].ty.dtype == tvm.ir.PrimType("float32x4")

    decl_buffers = []
    tensor_stores = []
    tvm_ffi.structural_walk(
        func.body,
        lambda node: (
            decl_buffers.append(node)
            if _is_buffer_binding(node, "tirx.decl_tensor")
            else tensor_stores.append(node)
            if isinstance(node, tvm.tirx.TensorStore)
            else None
        ),
    )
    assert len(decl_buffers) == 2
    assert all(decl.var.ty.dtype == tvm.ir.PrimType("float32x4") for decl in decl_buffers)
    assert len(tensor_stores) == 1
    assert tensor_stores[0].buffer.ty.dtype == tvm.ir.PrimType("float32x4")


if __name__ == "__main__":
    tvm.testing.main()
