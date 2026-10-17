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
import uuid

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

for _scope_name in ("local.L0A", "local.L0A2"):
    _scope = tvm_tirx.runtime.MemoryScope.register(_scope_name)
    tvm_tirx.runtime.MemoryScope.def_attr("runtime.memory_alloc_property")[_scope] = 1


def _scope_with_property(value):
    scope = tvm_tirx.runtime.MemoryScope.register("test.storage_rewrite." + uuid.uuid4().hex)
    if value != "absent":
        tvm_tirx.runtime.MemoryScope.def_attr("runtime.memory_alloc_property")[scope] = value
    return scope


def _rewrite(func):
    @I.ir_module(check_well_formed=False)
    class Before:
        main = func

    return tvm_tirx.tirx.transform.StorageRewrite()(Before)["main"]


@pytest.mark.parametrize("property_value", ["absent", None, False, True, 0, 1, 2, 99])
def test_allocation_property_controls_reuse(property_value):
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

    @I.ir_module
    class Expected:
        @T.function
        def func(output: T.Tensor((2,), T.float32)):
            A = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope=scope)])
            for i in range(128):
                A[i] = T.Cast("float32", i)
            output[0] = A[127]
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((128,), T.float32, scope=scope)])
            for i_1 in range(128):
                A_1[i_1] = T.Cast("float32", i_1 + 1)
            output[1] = A_1[127]

    if property_value is None:
        with pytest.raises(TypeError):
            _rewrite(func)
    elif property_value == 99:
        with pytest.raises(ValueError, match="Invalid memory allocation property"):
            _rewrite(func)
    else:
        expected = Expected["func"] if property_value != 2 else func
        tvm_tirx.ir.assert_structural_equal(_rewrite(func), expected)


@pytest.mark.parametrize("property_value", [0, 1, 2])
def test_allocation_property_controls_packing(property_value):
    scope = _scope_with_property(property_value)

    @T.function
    def func(output: T.Tensor((1,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope)])
        B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope)])
        for i in T.serial(128):
            A[i] = T.float32(i)
            B[i] = T.float32(i + 1)
        output[0] = A[127] + B[127]

    @I.ir_module
    class Expected:
        @T.function
        def func(output: T.Tensor((1,), T.float32)):
            A = T.alloc_tensor(ty_args=[T.Tensor((256,), T.float32, scope=scope)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((128,), T.float32, scope=scope)])
            A_2 = T.decl_tensor(A.data, ty_args=[T.Tensor((128,), T.float32, scope=scope)])
            for i in range(128):
                A_1[i] = T.Cast("float32", i)
                A_2[128 + i] = T.Cast("float32", i + 1)
            output[0] = A_1[127] + A_2[255]

    expected = Expected["func"] if property_value == 1 else func
    tvm_tirx.ir.assert_structural_equal(_rewrite(func), expected)


@pytest.mark.parametrize("scope_name", [None, "shared.dyn"])
def test_single_allocation_symbolic_layout_is_deferred(scope_name):
    scope = tvm_tirx.runtime.MemoryScope(scope_name) if scope_name else _scope_with_property(1)

    @T.function
    def func(n: T.int32, output: T.Tensor((1,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32", scope)])
        B = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32", scope)])
        for i in T.serial(n):
            A[i] = T.float32(i)
            B[i] = T.float32(i + 1)
        output[0] = A[n - 1] + B[n - 1]

    tvm_tirx.ir.assert_structural_equal(_rewrite(func), func)


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

    @I.ir_module
    class Expected:
        @T.function
        def func(output: T.Tensor((1,), T.float32)):
            B = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope=scope)])
            A = T.alloc_tensor(
                ty_args=[T.Tensor((128,), T.float32, scope=scope)],
                annotations={"test.constraint": 7},
            )
            for i in range(128):
                A[i] = T.Cast("float32", i)
                B[i] = T.Cast("float32", i + 1)
            output[0] = A[127] + B[127]

    after = _rewrite(func)
    tvm_tirx.ir.assert_structural_equal(after, Expected["func"])


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

    @I.ir_module
    class Expected:
        @T.function
        def func(output: T.Tensor((2,), T.float32)):
            B = T.alloc_tensor(ty_args=[T.Tensor((128,), T.float32, scope=scope)])
            A = T.alloc_tensor(
                ty_args=[
                    T.Tensor((128,), T.float32, scope=scope, strides=[stride], elem_offset=offset)
                ]
            )
            for i in range(128):
                A[i] = T.Cast("float32", i)
            output[0] = A[127]
            for i_1 in range(128):
                B[i_1] = T.Cast("float32", i_1 + 1)
            output[1] = B[127]

    tvm_tirx.ir.assert_structural_equal(_rewrite(func), Expected["func"])


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

    @I.ir_module
    class Expected:
        @T.function
        def func(output: T.Tensor((2,), T.float32)):
            A = T.alloc_tensor(
                ty_args=[T.Tensor((256,), T.float32, scope=scope, data_alignment=256)]
            )
            A_1 = T.decl_tensor(
                A.data, ty_args=[T.Tensor((128,), T.float32, scope=scope, data_alignment=256)]
            )
            for i in range(128):
                A_1[i] = T.Cast("float32", i)
            output[0] = A_1[127]
            A_2 = T.decl_tensor(
                A.data, ty_args=[T.Tensor((256,), T.float32, scope=scope, data_alignment=256)]
            )
            for i_1 in range(256):
                A_2[i_1] = T.Cast("float32", i_1 + 1)
            output[1] = A_2[255]

    after = _rewrite(func)
    tvm_tirx.ir.assert_structural_equal(after, Expected["func"])


def test_packing_preserves_different_alignment_contracts():
    scope = _scope_with_property(1)

    @T.function
    def func(output: T.Tensor((2,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((17,), "float32", scope, align=64)])
        for i in T.serial(17):
            A[i] = T.float32(i)
        output[0] = A[16]
        B = T.alloc_tensor(ty_args=[T.Tensor((17,), "float32", scope, elem_offset=0, align=512)])
        for i in T.serial(17):
            B[i] = T.float32(i + 1)
        output[1] = B[16]

    @I.ir_module
    class Expected:
        @T.function
        def func(output: T.Tensor((2,), T.float32)):
            A = T.alloc_tensor(
                ty_args=[T.Tensor((256,), T.float32, scope=scope, data_alignment=512)]
            )
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((17,), T.float32, scope=scope)])
            for i in range(17):
                A_1[i] = T.Cast("float32", i)
            output[0] = A_1[16]
            A_2 = T.decl_tensor(
                A.data,
                ty_args=[T.Tensor((17,), T.float32, scope=scope, data_alignment=512)],
            )
            for i_1 in range(17):
                A_2[128 + i_1] = T.Cast("float32", i_1 + 1)
            output[1] = A_2[144]

    after = _rewrite(func)
    tvm_tirx.ir.assert_structural_equal(after, Expected["func"])


def test_packing_widens_large_combined_extent():
    scope = _scope_with_property(1)

    @T.function
    def func(output: T.Tensor((1,), "float32")):
        A = T.alloc_tensor(ty_args=[T.Tensor((1073741824,), "float32", scope)])
        B = T.alloc_tensor(ty_args=[T.Tensor((1073741824,), "float32", scope)])
        A[0] = T.float32(1)
        B[0] = T.float32(2)
        output[0] = A[0] + B[0]

    @I.ir_module
    class Expected:
        @T.function
        def func(output: T.Tensor((1,), T.float32)):
            A = T.alloc_tensor(ty_args=[T.Tensor((T.int64(2147483648),), T.float32, scope=scope)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1073741824,), T.float32, scope=scope)])
            A_2 = T.decl_tensor(A.data, ty_args=[T.Tensor((1073741824,), T.float32, scope=scope)])
            A_1[0] = T.float32(1.0)
            A_2[1073741824] = T.float32(2.0)
            output[0] = A_1[0] + A_2[1073741824]

    after = _rewrite(func)
    tvm_tirx.ir.assert_structural_equal(after, Expected["func"])


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

    @I.ir_module
    class Expected:
        @T.function
        def func(n: T.int32):
            A = T.alloc_tensor(ty_args=[T.Tensor((200,), T.float32, scope="local.L0A")])
            for i in range(n):
                for j in range(10):
                    A[j] = T.float32(1.2)
                for j_1 in range(10):
                    A_1 = T.decl_tensor(
                        A.data, ty_args=[T.Tensor((200,), T.float32, scope="local.L0A")]
                    )
                    A_1[j_1] = T.float32(1.3)

    after = _rewrite(func)
    tvm_tirx.ir.assert_structural_equal(after, Expected["func"])


@pytest.mark.parametrize(
    "first_dtype,last_dtype,size,offsets",
    [
        ("float16", "int8", 4608, (512, 3072, 8192, 8192)),
        ("float32", "int8", 2816, (1024, 4096, 10240, 10240)),
        ("float64", "int8", 1920, (2048, 6144, 14336, 14336)),
        ("int8", "uint8", 8192, (256, 2560, 7168, 0)),
    ],
)
def test_alloc_different_dtypes(first_dtype, last_dtype, size, offsets):
    @T.function
    def before():
        A = T.alloc_tensor(ty_args=[T.Tensor((1024,), first_dtype, "local.L0A")])
        B = T.alloc_tensor(ty_args=[T.Tensor((1024,), "int32", "local.L0A")])
        C = T.alloc_tensor(ty_args=[T.Tensor((1024,), "uint16", "local.L0A")])
        D = T.alloc_tensor(ty_args=[T.Tensor((1024,), last_dtype, "local.L0A")])
        E = T.alloc_tensor(ty_args=[T.Tensor((1024,), "int8", "local.L0A")])
        for j in range(1024):
            A[j] = T.Cast(first_dtype, 1)
        for j in range(1024):
            B[j] = T.Cast("int32", 1)
        for j in range(1024):
            C[j] = T.Cast("uint16", 1)
        for j in range(1024):
            D[j] = T.Cast(last_dtype, 1)
        for j in range(1024):
            E[j] = (
                T.Cast("int8", A[j])
                + T.Cast("int8", B[j])
                + T.Cast("int8", C[j])
                + T.Cast("int8", D[j])
            )

    @I.ir_module
    class Expected:
        @T.function
        def before():
            storage = T.alloc_tensor(ty_args=[T.Tensor((size,), first_dtype, "local.L0A")])
            A = T.decl_tensor(storage.data, ty_args=[T.Tensor((1024,), first_dtype, "local.L0A")])
            B = T.decl_tensor(storage.data, ty_args=[T.Tensor((1024,), "int32", "local.L0A")])
            C = T.decl_tensor(storage.data, ty_args=[T.Tensor((1024,), "uint16", "local.L0A")])
            D = T.decl_tensor(storage.data, ty_args=[T.Tensor((1024,), last_dtype, "local.L0A")])
            E = T.decl_tensor(storage.data, ty_args=[T.Tensor((1024,), "int8", "local.L0A")])
            for j in range(1024):
                A[j] = T.Cast(first_dtype, 1)
            for j in range(1024):
                B[offsets[0] + j] = T.Cast("int32", 1)
            for j in range(1024):
                C[offsets[1] + j] = T.Cast("uint16", 1)
            for j in range(1024):
                D[offsets[2] + j] = T.Cast(last_dtype, 1)
            for j in range(1024):
                E[j if T.constexpr(offsets[3] == 0) else offsets[3] + j] = (
                    T.Cast("int8", A[j])
                    + T.Cast("int8", B[offsets[0] + j])
                    + T.Cast("int8", C[offsets[1] + j])
                    + T.Cast("int8", D[offsets[2] + j])
                )

    tvm_tirx.ir.assert_structural_equal(_rewrite(before), Expected["before"])


def test_address_of():
    # In this test, the storage rewrite pass is allowed to
    # combine tensors B and D, but not C
    @T.function
    def before(A: T.Tensor(8, "float32"), E: T.Tensor(8, "float32")):
        T.func_attr({"global_symbol": "main"})
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

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((8,), T.float32), E: T.Tensor((8,), T.float32)):
            B = T.alloc_tensor(ty_args=[T.Tensor((8,), T.float32)])
            C = T.alloc_tensor(ty_args=[T.Tensor((8,), T.float32)])
            for i in range(8):
                B[i] = (
                    T.call_extern("deref", T.address_of(A[i]), ty="float32")
                    + T.call_extern("deref", T.address_of(A[0]), ty="float32")
                    + T.float32(1.0)
                )
            for i_1 in range(8):
                C[i_1] = (
                    T.call_extern("deref", T.address_of(B[i_1]), ty="float32")
                    + T.call_extern("deref", T.address_of(B[0]), ty="float32")
                    + T.float32(2.0)
                )
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((8,), T.float32)])
            for i_2 in range(8):
                B_1[i_2] = (
                    T.call_extern("deref", T.address_of(C[i_2]), ty="float32")
                    + T.call_extern("deref", T.address_of(C[0]), ty="float32")
                    + T.float32(2.0)
                )
            for i_3 in range(8):
                E[i_3] = (
                    T.call_extern("deref", T.address_of(B_1[i_3]), ty="float32")
                    + T.call_extern("deref", T.address_of(B_1[0]), ty="float32")
                    + T.float32(3.0)
                )

    after = _rewrite(before)
    tvm_tirx.ir.assert_structural_equal(after, Expected["main"])


def test_parallel_alloc():
    @T.function
    def func1(n: T.int32):
        for i in T.parallel(n):
            for j in range(10):
                A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                A[j] = A[j] + T.float32(2)

    @T.function
    def func2(n: T.int32):
        for t in T.serial(n):
            with T.parallel_launch():
                for i in T.parallel(n):
                    for j in range(10):
                        A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                        A[j] = A[j] + T.float32(2)

    @I.ir_module
    class ExpectedWorker:
        @T.function
        def func1(n: T.int32):
            for i in T.parallel(0, n):
                A = T.alloc_tensor(ty_args=[T.Tensor((n,), T.float32)])
                for j in range(10):
                    A[j] = A[j] + T.float32(2.0)

    @I.ir_module
    class ExpectedLaunch:
        @T.function
        def func2(n: T.int32):
            for t in range(n):
                with T.parallel_launch():
                    A = T.alloc_tensor(ty_args=[T.Tensor((n,), T.float32)])
                    for i in T.parallel(0, n):
                        for j in range(10):
                            A[j] = A[j] + T.float32(2.0)

    tvm_tirx.ir.assert_structural_equal(_rewrite(func1), ExpectedWorker["func1"])
    tvm_tirx.ir.assert_structural_equal(_rewrite(func2), ExpectedLaunch["func2"])


@pytest.mark.parametrize("parallel", [False, True])
def test_while_alloc(parallel):
    loop = T.parallel if parallel else T.serial

    @T.function
    def before(n: T.int32):
        for i in loop(n):
            j = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32")])
            j[0] = 0
            while j[0] < 10:
                A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                A[j[0]] = A[j[0]] + T.float32(2)
                j[0] = j[0] + j[0] + 1

    @I.ir_module
    class Expected:
        @T.function
        def before(n: T.int32):
            if T.constexpr(not parallel):
                j = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32")])
                A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
            for i in loop(n):
                if T.constexpr(parallel):
                    j = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32")])
                    A = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
                j[0] = 0
                while j[0] < 10:
                    A[j[0]] = A[j[0]] + T.float32(2)
                    j[0] = j[0] + j[0] + 1

    tvm_tirx.ir.assert_structural_equal(_rewrite(before), Expected["before"])


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

    packed_offset = 200 if alignment == 4 else 208

    @I.ir_module
    class Expected:
        @T.function
        def func(n: T.int32):
            A = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        (expected_size,), T.float32, scope="local.L0A", data_alignment=alignment
                    )
                ]
            )
            for i in range(n):
                for j in range(10):
                    A_1 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.float32, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_2 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.float32, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_1[j] = T.float32(1.2)
                    A_2[packed_offset + j] = T.float32(1.3)
                    A_3 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_3[2 * packed_offset + j] = T.int16(1)
                    A_4 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_4[4 * packed_offset + j] = T.int16(1)
                    A_5 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_5[4 * packed_offset + j] = (
                        A_3[2 * packed_offset + j] + A_4[4 * packed_offset + j]
                    )
                    A_6 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.float32, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_6[j] = A_1[j]

    tvm_tirx.ir.assert_structural_equal(_rewrite(func), Expected["func"])


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

    @I.ir_module
    class Expected:
        @T.function
        def func(n: T.int32):
            A = T.alloc_tensor(ty_args=[T.Tensor((200,), T.float32, scope="local.L0A2")])
            for i in range(n):
                for j in range(10):
                    A_1 = T.decl_tensor(
                        A.data, ty_args=[T.Tensor((200,), T.float32, scope="local.L0A2")]
                    )
                    A_1[j] = T.float32(1.2)
                for j_1 in range(20):
                    A_2 = T.decl_tensor(
                        A.data, ty_args=[T.Tensor((400,), T.int16, scope="local.L0A2")]
                    )
                    A_2[j_1] = T.int16(1)
                for j_2 in range(10):
                    A_3 = T.decl_tensor(
                        A.data, ty_args=[T.Tensor((200,), T.float32, scope="local.L0A2")]
                    )
                    A_3[j_2] = T.float32(1.2)

    after = _rewrite(func)
    tvm_tirx.ir.assert_structural_equal(after, Expected["func"])


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

    packed_offset = expected_size // 2

    @I.ir_module
    class Expected:
        @T.function
        def func(n: T.int32):
            A = T.alloc_tensor(
                ty_args=[
                    T.Tensor((expected_size,), T.int16, scope="local.L0A", data_alignment=alignment)
                ]
            )
            for i in range(n):
                for j in range(10):
                    A_1 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_1[j] = T.int16(1)
                    A_2 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_2[packed_offset + j] = T.int16(1)
                    A_3 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((200,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_3[j] = A_1[j] + A_2[packed_offset + j]
                    A_4 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((400,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_4[j] = T.int16(1)
                    A_5 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((400,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_5[packed_offset + j] = T.int16(1)
                    A_6 = T.decl_tensor(
                        A.data,
                        ty_args=[
                            T.Tensor((400,), T.int16, scope="local.L0A", data_alignment=alignment)
                        ],
                    )
                    A_6[j] = A_4[j]

    tvm_tirx.ir.assert_structural_equal(_rewrite(func), Expected["func"])


def test_access_in_let_value():
    @T.function(private=True)
    def func(A: T.Tensor((8,), "float32")):
        for i in range(8):
            B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32")])
            B[0] = 3.14
            x: T.let[T.float32] = T.exp(B[0])
            A[i] = (x + 1.0) / (x - 1.0)

    @T.function(private=True)
    def func_rewritten(A: T.Tensor((8,), "float32")) -> None:
        B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32")])
        for i in range(8):
            B[0] = 3.14
            x: T.let[T.float32] = T.exp(B[0])
            A[i] = (x + 1.0) / (x - 1.0)

    tvm_tirx.ir.assert_structural_equal(_rewrite(func), func_rewritten)


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

    After = tvm_tirx.tirx.transform.StorageRewrite()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Before)


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

    After = tvm_tirx.tirx.transform.StorageRewrite()(Before)
    assert tvm_tirx.tirx.analysis.verify_well_formed(After)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


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

    After = tvm_tirx.tirx.transform.StorageRewrite()(Before)
    assert tvm_tirx.tirx.analysis.verify_well_formed(After)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


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

    @I.ir_module
    class Expected:
        @T.function
        def func(D: T.Tensor((1,), T.float32)):
            A = T.alloc_tensor(ty_args=[T.Tensor((16,), T.float32)])
            C = T.alloc_tensor(ty_args=[T.Tensor((16,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), T.float32)])
            A[0] = T.float32(1.0)
            C[0] = T.float32(2.0)
            D[0] = A_1[0] + C[0]

    after = _rewrite(func)
    tvm_tirx.ir.assert_structural_equal(after, Expected["func"])


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

    After = tvm_tirx.tirx.transform.StorageRewrite()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
