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
# ruff: noqa: F401, F841
import sys
import uuid

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T

for _scope_name in ("local.L0A", "local.L0A2"):
    _scope = tvm.runtime.MemoryScope.register(_scope_name)
    tvm.runtime.MemoryScope.def_attr("runtime.memory_alloc_property")[_scope] = 1


def _is_tensor_binding(node, *op_names):
    return (
        isinstance(node, tvm.ir.Bind)
        and isinstance(node.value, tvm.ir.Call)
        and isinstance(node.value.op, tvm.ir.Op)
        and node.value.op.name in op_names
    )


def _scope_with_property(value):
    scope = tvm.runtime.MemoryScope.register("test.storage_rewrite." + uuid.uuid4().hex)
    tvm.runtime.MemoryScope.def_attr("runtime.memory_alloc_property")[scope] = value
    return scope


def _rewritten_allocations(func):
    result = tvm.tirx.transform.StorageRewrite()(tvm.IRModule.from_expr(func))
    allocations = []

    def collect(node):
        if _is_tensor_binding(node, "tirx.alloc_tensor"):
            allocations.append(node)

    tvm_ffi.structural_walk(result["func"].body, collect)
    return allocations


@pytest.mark.parametrize("property_value,expected_allocations", [(0, 1), (1, 1), (2, 2)])
def test_allocation_property_controls_reuse(property_value, expected_allocations):
    scope = _scope_with_property(property_value)

    @T.function
    def func(output: T.Tensor((2,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope)])
        for i in T.serial(128):
            A[i] = T.float32(i)
        output[0] = A[127]
        B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope)])
        for i in T.serial(128):
            B[i] = T.float32(i + 1)
        output[1] = B[127]

    assert len(_rewritten_allocations(func)) == expected_allocations


@pytest.mark.parametrize("property_value,expected_allocations", [(0, 2), (1, 1), (2, 2)])
def test_allocation_property_controls_packing(property_value, expected_allocations):
    scope = _scope_with_property(property_value)

    @T.function
    def func(output: T.Tensor((1,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope)])
        B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope)])
        for i in T.serial(128):
            A[i] = T.float32(i)
            B[i] = T.float32(i + 1)
        output[0] = A[127] + B[127]

    assert len(_rewritten_allocations(func)) == expected_allocations


@pytest.mark.parametrize("scope_name", [None, "shared.dyn"])
def test_single_allocation_symbolic_layout_is_deferred(scope_name):
    scope = tvm.runtime.MemoryScope(scope_name) if scope_name else _scope_with_property(1)

    @T.function
    def func(n: T.int32, output: T.Tensor((1,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32", scope)])
        B = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32", scope)])
        for i in T.serial(n):
            A[i] = T.float32(i)
            B[i] = T.float32(i + 1)
        output[0] = A[n - 1] + B[n - 1]

    allocations = _rewritten_allocations(func)
    assert len(allocations) == 2
    for allocation in allocations:
        tvm.ir.assert_structural_equal(allocation.var.shape[0], func.params[0])
        assert allocation.var.scope().same_as(scope)


def test_single_allocation_preserves_exact_metadata():
    scope = _scope_with_property(1)

    @T.function
    def func(output: T.Tensor((1,), "float32")):
        A = T.alloc_tensor(
            ty_args=[T.Tensor((128,), "float32", scope)], annotations={"test.constraint": 7}
        )
        B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope)])
        for i in T.serial(128):
            A[i] = T.float32(i)
            B[i] = T.float32(i + 1)
        output[0] = A[127] + B[127]

    allocations = _rewritten_allocations(func)
    assert len(allocations) == 2
    constrained = [alloc for alloc in allocations if "test.constraint" in alloc.value.attrs]
    assert len(constrained) == 1
    assert int(constrained[0].value.attrs["test.constraint"]) == 7
    assert int(constrained[0].var.shape[0]) == 128


@pytest.mark.parametrize("property_value", [0, 1])
@pytest.mark.parametrize("stride,offset", [(2, 0), (1, 4)])
def test_allocation_constraints_prevent_lifetime_reuse(property_value, stride, offset):
    scope = _scope_with_property(property_value)

    @T.function
    def func(output: T.Tensor((2,), "float32")):
        A = T.alloc_tensor(
            ty_args=[T.Tensor((128,), "float32", scope, strides=[stride], elem_offset=offset)]
        )
        for i in T.serial(128):
            A[i] = T.float32(i)
        output[0] = A[127]
        B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope)])
        for i in T.serial(128):
            B[i] = T.float32(i + 1)
        output[1] = B[127]

    allocations = _rewritten_allocations(func)
    assert len(allocations) == 2
    constrained = next(alloc for alloc in allocations if alloc.var.name == "A")
    assert int(constrained.var.ty.strides[0]) == stride
    assert int(constrained.var.ty.elem_offset) == offset


@pytest.mark.parametrize("property_value", [0, 1])
def test_lifetime_reuse_preserves_alignment(property_value):
    scope = _scope_with_property(property_value)

    @T.function
    def func(output: T.Tensor((2,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope, align=256)])
        for i in T.serial(128):
            A[i] = T.float32(i)
        output[0] = A[127]
        B = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32", scope, align=256)])
        for i in T.serial(256):
            B[i] = T.float32(i + 1)
        output[1] = B[255]

    allocations = _rewritten_allocations(func)
    assert len(allocations) == 1
    assert allocations[0].var.ty.data_alignment == 256


def test_packing_preserves_different_alignment_contracts():
    scope = _scope_with_property(1)

    @T.function
    def func(output: T.Tensor((2,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((17,), "float32", scope, align=64)])
        for i in T.serial(17):
            A[i] = T.float32(i)
        output[0] = A[16]
        B = T.alloc_tensor(
            ty_args=[T.Tensor((17,), "float32", scope, elem_offset=0, align=256, offset_factor=128)]
        )
        for i in T.serial(17):
            B[i] = T.float32(i + 1)
        output[1] = B[16]

    allocations = _rewritten_allocations(func)
    assert len(allocations) == 1
    assert allocations[0].var.ty.data_alignment == 512
    assert int(allocations[0].var.shape[0]) == 256


def test_packing_widens_large_combined_extent():
    scope = _scope_with_property(1)

    @T.function
    def func(output: T.Tensor((1,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((1073741824,), "float32", scope)])
        B = T.alloc_tensor(ty_args=[T.Tensor((1073741824,), "float32", scope)])
        A[0] = T.float32(1)
        B[0] = T.float32(2)
        output[0] = A[0] + B[0]

    allocations = _rewritten_allocations(func)
    assert len(allocations) == 1
    assert int(allocations[0].var.shape[0]) == 2147483648
    assert allocations[0].var.shape[0].ty == tvm.ir.PrimType("int64")


def test_alloc_seq():
    scope_tb = "local.L0A"

    @T.function
    def func(n: T.int32):
        for i in T.serial(n):
            for j in range(10):
                A = T.alloc_tensor(ty_args=[T.Tensor((200,), "float32", scope=scope_tb)])
                A[j] = T.float32(1.2)
            for j in range(10):
                B = T.alloc_tensor(ty_args=[T.Tensor((200,), "float32", scope=scope_tb)])
                B[j] = T.float32(1.3)

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body

    num_alloc = [0]

    def verify(n):
        if _is_tensor_binding(n, "tirx.alloc_tensor"):
            num_alloc[0] += 1
            assert n.var.ty.shape[0].value == 200

    tvm_ffi.structural_walk(body, verify)
    assert num_alloc[0] == 1


def test_alloc_different_dtypes():
    # Test cross-loop tensor access with tensors allocated in parent scope
    def make_mod(dtype_list, length):
        assert len(dtype_list) == 4

        @T.function
        def func():
            # Allocate all tensors in parent scope (before any loops)
            A = T.alloc_tensor(ty_args=[T.Tensor((length,), dtype_list[0], scope="local.L0A")])
            B = T.alloc_tensor(ty_args=[T.Tensor((length,), dtype_list[1], scope="local.L0A")])
            C = T.alloc_tensor(ty_args=[T.Tensor((length,), dtype_list[2], scope="local.L0A")])
            D = T.alloc_tensor(ty_args=[T.Tensor((length,), dtype_list[3], scope="local.L0A")])
            E = T.alloc_tensor(ty_args=[T.Tensor((length,), "int8", scope="local.L0A")])

            for j in range(length):
                A[j] = T.Cast(dtype_list[0], 1)
            for j in range(length):
                B[j] = T.Cast(dtype_list[1], 1)
            for j in range(length):
                C[j] = T.Cast(dtype_list[2], 1)
            for j in range(length):
                D[j] = T.Cast(dtype_list[3], 1)
            for j in range(length):
                E[j] = (
                    T.Cast("int8", A[j])
                    + T.Cast("int8", B[j])
                    + T.Cast("int8", C[j])
                    + T.Cast("int8", D[j])
                )

        return tvm.IRModule.from_expr(func)

    def dtype_bit_len(dtype):
        index = 0
        for i in dtype:
            if i.isdigit():
                break
            index += 1
        return int(dtype[index:])

    def offset_generater(dtype_list, length):
        dtype_len_list = [dtype_bit_len(i) for i in dtype_list]
        base_len = dtype_len_list[0]
        return sum([i * length / base_len for i in dtype_len_list])

    def dtype_test(dtype_list, length):
        def verify(n):
            if _is_tensor_binding(n, "tirx.alloc_tensor"):
                assert n.var.ty.shape[0].value == offset

        mod = make_mod(dtype_list, length)
        offset = offset_generater(dtype_list, length)

        body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body
        tvm_ffi.structural_walk(body, verify)

    length = 1024
    dtype_list = ["float16", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["float32", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["float64", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["int8", "int32", "uint16", "uint8"]
    dtype_test(dtype_list, length)


def test_address_of():
    # In this test, the storage rewrite pass is allowed to
    # combine tensors B and D, but not C
    @T.function
    def before(A: T.Tensor(8, "float32"), E: T.Tensor(8, "float32")):
        B = T.alloc_tensor(ty_args=[T.Tensor((8,), "float32")])
        for i in range(8):
            B[i] = (
                T.call_extern("deref", T.address_of(A[i]), ty="float32")
                + T.call_extern("deref", T.address_of(A[0]), ty="float32")
                + T.float32(1)
            )
        C = T.alloc_tensor(ty_args=[T.Tensor((8,), "float32")])
        for i in range(8):
            C[i] = (
                T.call_extern("deref", T.address_of(B[i]), ty="float32")
                + T.call_extern("deref", T.address_of(B[0]), ty="float32")
                + T.float32(2)
            )
        D = T.alloc_tensor(ty_args=[T.Tensor((8,), "float32")])
        for i in range(8):
            D[i] = (
                T.call_extern("deref", T.address_of(C[i]), ty="float32")
                + T.call_extern("deref", T.address_of(C[0]), ty="float32")
                + T.float32(2)
            )
        for i in range(8):
            E[i] = (
                T.call_extern("deref", T.address_of(D[i]), ty="float32")
                + T.call_extern("deref", T.address_of(D[0]), ty="float32")
                + T.float32(3)
            )

    def verify(n):
        if _is_tensor_binding(n, "tirx.alloc_tensor"):
            total_alloc[0] += n.var.ty.shape[0].value

    total_alloc = [0]
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    tvm_ffi.structural_walk(mod["main"].body, verify)
    assert total_alloc[0] == 24

    total_alloc[0] = 0
    mod = tvm.tirx.transform.StorageRewrite()(mod)
    tvm_ffi.structural_walk(mod["main"].body, verify)
    assert total_alloc[0] == 16


def test_parallel_alloc():
    @T.function
    def func1(n: T.int32):
        for i in T.parallel(n):
            for j in range(10):
                A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                A[j] = A[j] + T.float32(2)

    mod = tvm.IRModule.from_expr(func1)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func1"]

    # With flat AllocTensor, the for body is a SeqStmt; first element is AllocTensor
    assert _is_tensor_binding(body.body[0].body[0], "tirx.alloc_tensor")

    @T.function
    def func2(n: T.int32):
        for t in T.serial(n):
            with T.parallel_launch():
                for i in T.parallel(n):
                    for j in range(10):
                        A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                        A[j] = A[j] + T.float32(2)

    mod = tvm.IRModule.from_expr(func2)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func2"]

    # The launch body owns the allocation executed by each worker.
    assert _is_tensor_binding(body.body[0].body[0].body[0], "tirx.alloc_tensor")


def test_while_alloc():
    @T.function
    def func_parallel(n: T.int32):
        for i in T.parallel(n):
            j = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32")])
            j[0] = 0
            while j[0] < 10:
                A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                A[j[0]] = A[j[0]] + T.float32(2)
                j[0] = j[0] + j[0] + 1

    @T.function
    def func_serial(n: T.int32):
        for i in T.serial(n):
            j = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32")])
            j[0] = 0
            while j[0] < 10:
                A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                A[j[0]] = A[j[0]] + T.float32(2)
                j[0] = j[0] + j[0] + 1

    mod = tvm.IRModule.from_expr(func_parallel)
    # parallel (i, 0, n) {
    #   allocate j[int32 * 1]
    #   j[0] = 0
    #   while((j[0] < 10)){
    #     // attr [A] storage_scope = "global"
    #     allocate A[float32 * n]
    #     A[j[0]] = (A[j[0]] + 2f)
    #     j[0] = (j[0] + (j[0] + 1))
    #   }
    # }
    body = tvm.tirx.transform.StorageRewrite()(mod)["func_parallel"]
    # Navigate to inside the for loop, then check that allocations exist
    # The structure with DeclTensor is:
    #   parallel (i, 0, n) { DeclTensor(j, DeclTensor(A, ...)) }
    # or with Allocate+DeclTensor pairs
    inner = body.body[0].body  # inside For
    # Skip DeclTensor nodes to find Allocate
    num_alloc = [0]

    def count_alloc(n):
        if _is_tensor_binding(n, "tirx.alloc_tensor"):
            num_alloc[0] += 1

    tvm_ffi.structural_walk(inner, count_alloc)
    assert num_alloc[0] == 2  # j and A allocations

    mod = tvm.IRModule.from_expr(func_serial)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func_serial"]
    num_alloc[0] = 0
    tvm_ffi.structural_walk(body.body, count_alloc)
    assert num_alloc[0] == 2  # j and A allocations


@pytest.mark.parametrize("alignment,expected_size", [(4, 500), (64, 528)])
def test_alloc_seq_type(alignment, expected_size):
    @T.function
    def func(n: T.int32):
        for i in T.serial(n):
            for j in range(10):
                A = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "float32", scope="local.L0A", align=alignment)]
                )
                A1 = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "float32", scope="local.L0A", align=alignment)]
                )
                A[j] = T.float32(1.2)
                A1[j] = T.float32(1.3)
                B = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "int16", scope="local.L0A", align=alignment)]
                )
                B[j] = T.int16(1)
                C = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "int16", scope="local.L0A", align=alignment)]
                )
                C[j] = T.int16(1)
                D = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "int16", scope="local.L0A", align=alignment)]
                )
                D[j] = B[j] + C[j]
                A2 = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "float32", scope="local.L0A", align=alignment)]
                )
                A2[j] = A[j]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body

    num_alloc = [0]

    def verify(n):
        if _is_tensor_binding(n, "tirx.alloc_tensor"):
            num_alloc[0] += 1
            # Each packed region starts at a multiple of its declared byte alignment.
            assert n.var.ty.shape[0].value == expected_size
            assert n.var.ty.data_alignment == alignment

    tvm_ffi.structural_walk(body, verify)
    assert num_alloc[0] == 1


def test_alloc_seq_type2():
    scope_tb = "local.L0A2"

    @T.function
    def func(n: T.int32):
        for i in T.serial(n):
            for j in range(10):
                A = T.alloc_tensor(ty_args=[T.Tensor((200,), "float32", scope=scope_tb)])
                A[j] = T.float32(1.2)
            for j in range(20):
                B = T.alloc_tensor(ty_args=[T.Tensor((400,), "int16", scope=scope_tb)])
                B[j] = T.int16(1)
            for j in range(10):
                C = T.alloc_tensor(ty_args=[T.Tensor((200,), "float32", scope=scope_tb)])
                C[j] = T.float32(1.2)

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body

    num_alloc = [0]

    def verify(n):
        if _is_tensor_binding(n, "tirx.alloc_tensor"):
            num_alloc[0] += 1
            assert n.var.ty.shape[0].value == 200

    tvm_ffi.structural_walk(body, verify)
    assert num_alloc[0] == 1


@pytest.mark.parametrize("alignment,expected_size", [(4, 800), (64, 832)])
def test_reuse_small_tensor(alignment, expected_size):
    @T.function
    def func(n: T.int32):
        for i in T.serial(n):
            for j in range(10):
                A = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "int16", scope="local.L0A", align=alignment)]
                )
                A[j] = T.int16(1)
                B = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "int16", scope="local.L0A", align=alignment)]
                )
                B[j] = T.int16(1)
                B1 = T.alloc_tensor(
                    ty_args=[T.Tensor((200,), "int16", scope="local.L0A", align=alignment)]
                )
                B1[j] = A[j] + B[j]
                C = T.alloc_tensor(
                    ty_args=[T.Tensor((400,), "int16", scope="local.L0A", align=alignment)]
                )
                C[j] = T.int16(1)
                D = T.alloc_tensor(
                    ty_args=[T.Tensor((400,), "int16", scope="local.L0A", align=alignment)]
                )
                D[j] = T.int16(1)
                E = T.alloc_tensor(
                    ty_args=[T.Tensor((400,), "int16", scope="local.L0A", align=alignment)]
                )
                E[j] = C[j]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body

    num_alloc = [0]

    def verify(n):
        if _is_tensor_binding(n, "tirx.alloc_tensor"):
            num_alloc[0] += 1
            # Reused entries hold 400 int16 values each, padded to their alignment.
            assert n.var.ty.shape[0].value == expected_size
            assert n.var.ty.data_alignment == alignment

    tvm_ffi.structural_walk(body, verify)
    assert num_alloc[0] == 1


def test_access_in_let_value():
    @T.function
    def func(A: T.Tensor((8,), "float32")):
        for i in range(8):
            B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32")])
            B[0] = 3.14
            x: T.let[T.float32] = T.exp(B[0])
            A[i] = (x + 1.0) / (x - 1.0)

    @T.function
    def func_rewritten(A: T.Tensor((8,), "float32")) -> None:
        B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32")])
        for i in range(8):
            B[0] = 3.14
            x: T.let[T.float32] = T.exp(B[0])
            A[i] = (x + 1.0) / (x - 1.0)

    mod = tvm.tirx.transform.StorageRewrite()(
        tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    )
    tvm.ir.assert_structural_equal(mod["main"], func_rewritten.with_attr("global_symbol", "main"))


def test_decl_tensor_is_not_vectorized():
    """StorageRewrite leaves explicit DeclTensor views unchanged.

    Vectorization of DeclTensor views was dropped because the rewritten result
    violates the immutable TensorVar binding invariants.
    """

    @I.ir_module
    class Before:
        @T.function
        def main() -> None:
            A_data: T.let[T.handle("int32")] = T.call_extern("dummy_func", ty=T.handle("int32").ty)
            A = T.decl_tensor(A_data, ty_args=[T.Tensor([8], "int32", data=A_data)])
            A[T.ramp(0, 1, 8)] = T.broadcast(42, 8)

    After = tvm.tirx.transform.StorageRewrite()(Before)
    tvm.ir.assert_structural_equal(After, Before)


def test_rewrite_decl_tensor():
    """A DeclTensor node may appear in StorageRewrite's input"""

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor(16, "float32"), D: T.Tensor(16, "float32")):
            B = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
            C = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])

            for i in range(16):
                B[i] = A[i]

            for i in range(16):
                C[i] = 2.0 * B[i]

            for i in range(16):
                D[i] = C[i]

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor(16, "float32"), D: T.Tensor(16, "float32")):
            B = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
            C = T.decl_tensor(B.data, ty_args=[T.Tensor(16, "float32", data=B.data)])

            for i in range(16):
                B[i] = A[i]

            for i in range(16):
                C[i] = 2.0 * B[i]

            for i in range(16):
                D[i] = C[i]

    After = tvm.tirx.transform.StorageRewrite()(Before)
    assert tvm.tirx.analysis.verify_well_formed(After)
    tvm.ir.assert_structural_equal(After, Expected)


def test_decl_tensor_alias_chain_uses_flat_root():
    """StorageRewrite resolves every alias in a chain to the same flat root."""

    @I.ir_module
    class Before:
        @T.function
        def main(D: T.Tensor(1, "float32")):
            A = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
            B = T.decl_tensor(A.data, ty_args=[T.Tensor(16, "float32", data=A.data)])
            C = T.decl_tensor(B.data, ty_args=[T.Tensor(16, "float32", data=B.data)])
            A[0] = 1.0
            D[0] = C[0]

    @I.ir_module
    class Expected:
        @T.function
        def main(D: T.Tensor(1, "float32")):
            A = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
            B = T.decl_tensor(A.data, ty_args=[T.Tensor(16, "float32", data=A.data)])
            C = T.decl_tensor(A.data, ty_args=[T.Tensor(16, "float32", data=A.data)])
            A[0] = 1.0
            D[0] = C[0]

    After = tvm.tirx.transform.StorageRewrite()(Before)
    assert tvm.tirx.analysis.verify_well_formed(After)
    tvm.ir.assert_structural_equal(After, Expected)


def test_decl_tensor_alias_extends_source_lifetime():
    """An access through an alias prevents reuse of its source allocation."""

    @T.function
    def func(D: T.Tensor(1, "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
        B = T.decl_tensor(A.data, ty_args=[T.Tensor(16, "float32", data=A.data)])
        A[0] = 1.0

        C = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
        C[0] = 2.0
        D[0] = B[0] + C[0]

    after = tvm.tirx.transform.StorageRewrite()(tvm.IRModule.from_expr(func))["func"]
    allocations = []
    tvm_ffi.structural_walk(
        after.body,
        lambda node: (
            allocations.append(node) if _is_tensor_binding(node, "tirx.alloc_tensor") else None
        ),
    )
    assert len(allocations) == 2


def test_no_orphaned_decl_tensor():
    """A DeclTensor of an unused Allocate should be removed

    StorageRewrite removes any allocations that are unused.  When it
    does so, any DeclTensor that refers to that allocation should also
    be removed.
    """

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor(16, "float32"), D: T.Tensor(16, "float32")):
            B = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
            C = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
            Unused = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])

            for i in range(16):
                B[i] = A[i]

            for i in range(16):
                C[i] = 2.0 * B[i]

            for i in range(16):
                D[i] = C[i]

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor(16, "float32"), D: T.Tensor(16, "float32")):
            B = T.alloc_tensor(ty_args=[T.Tensor(16, "float32")])
            C = T.decl_tensor(B.data, ty_args=[T.Tensor(16, "float32", data=B.data)])

            for i in range(16):
                B[i] = A[i]

            for i in range(16):
                C[i] = 2.0 * B[i]

            for i in range(16):
                D[i] = C[i]

    After = tvm.tirx.transform.StorageRewrite()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
