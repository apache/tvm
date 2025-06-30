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

from tvm.script import ir as I
from tvm.script import tir as T

import numpy as np

from tvm.tir.schedule.testing import assert_structural_equal_ignore_global_symbol


@tvm.register_func("tvm.test_matmul")
def my_matmul(a, b, c):
    c.copyfrom(np.dot(a.numpy(), b.numpy()))


def test_lower_call_packed():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(
            A: T.Buffer((64, 64), "float32"),
            B: T.Buffer((64, 64), "float32"),
            C: T.Buffer((64, 64), "float32"),
        ):
            T.func_attr({"target": tvm.target.Target("llvm")})
            T.attr("", "device_id", T.int32(0))
            T.call_packed("tvm.test_matmul", A, B, C)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(
            A: T.Buffer((64, 64), "float32"),
            B: T.Buffer((64, 64), "float32"),
            C: T.Buffer((64, 64), "float32"),
        ):
            T.func_attr({"target": tvm.target.Target("llvm")})
            stack_ffi_any: T.handle = T.tvm_stack_alloca("tvm_ffi_any", 4)
            stack_array: T.handle = T.tvm_stack_alloca("array", 3)
            stack_shape: T.handle("int64") = T.tvm_stack_alloca("shape", 6)
            stack_shape_1 = T.decl_buffer((T.int64(6),), "int64", data=stack_shape)
            stack_shape_1[0] = T.int64(64)
            stack_shape_1[1] = T.int64(64)
            T.tvm_struct_set(stack_array, 0, 1, A.data)
            stack_shape_2 = T.Buffer((1,), "int64", data=stack_shape)
            T.tvm_struct_set(stack_array, 0, 2, T.address_of(stack_shape_2[0]))
            T.tvm_struct_set(stack_array, 0, 3, T.reinterpret("handle", T.uint64(0)))
            T.tvm_struct_set(stack_array, 0, 4, 2)
            T.tvm_struct_set(stack_array, 0, 5, T.uint8(2))
            T.tvm_struct_set(stack_array, 0, 6, T.uint8(32))
            T.tvm_struct_set(stack_array, 0, 7, T.uint16(1))
            T.tvm_struct_set(stack_array, 0, 8, T.uint64(0))
            T.tvm_struct_set(stack_array, 0, 9, 0)
            T.tvm_struct_set(stack_array, 0, 10, 1)
            stack_shape_1[2] = T.int64(64)
            stack_shape_1[3] = T.int64(64)
            T.tvm_struct_set(stack_array, 1, 1, B.data)
            stack_shape_3 = T.Buffer((3,), "int64", data=stack_shape)
            T.tvm_struct_set(stack_array, 1, 2, T.address_of(stack_shape_3[2]))
            T.tvm_struct_set(stack_array, 1, 3, T.reinterpret("handle", T.uint64(0)))
            T.tvm_struct_set(stack_array, 1, 4, 2)
            T.tvm_struct_set(stack_array, 1, 5, T.uint8(2))
            T.tvm_struct_set(stack_array, 1, 6, T.uint8(32))
            T.tvm_struct_set(stack_array, 1, 7, T.uint16(1))
            T.tvm_struct_set(stack_array, 1, 8, T.uint64(0))
            T.tvm_struct_set(stack_array, 1, 9, 0)
            T.tvm_struct_set(stack_array, 1, 10, 1)
            stack_shape_1[4] = T.int64(64)
            stack_shape_1[5] = T.int64(64)
            T.tvm_struct_set(stack_array, 2, 1, C.data)
            stack_shape_4 = T.Buffer((5,), "int64", data=stack_shape)
            T.tvm_struct_set(stack_array, 2, 2, T.address_of(stack_shape_4[4]))
            T.tvm_struct_set(stack_array, 2, 3, T.reinterpret("handle", T.uint64(0)))
            T.tvm_struct_set(stack_array, 2, 4, 2)
            T.tvm_struct_set(stack_array, 2, 5, T.uint8(2))
            T.tvm_struct_set(stack_array, 2, 6, T.uint8(32))
            T.tvm_struct_set(stack_array, 2, 7, T.uint16(1))
            T.tvm_struct_set(stack_array, 2, 8, T.uint64(0))
            T.tvm_struct_set(stack_array, 2, 9, 0)
            T.tvm_struct_set(stack_array, 2, 10, 1)
            T.tvm_struct_set(stack_ffi_any, 0, 13, 7)
            T.tvm_struct_set(stack_ffi_any, 0, 14, T.tvm_struct_get(stack_array, 0, 0, "handle"))
            T.tvm_struct_set(stack_ffi_any, 1, 13, 7)
            T.tvm_struct_set(stack_ffi_any, 1, 14, T.tvm_struct_get(stack_array, 1, 0, "handle"))
            T.tvm_struct_set(stack_ffi_any, 2, 13, 7)
            T.tvm_struct_set(stack_ffi_any, 2, 14, T.tvm_struct_get(stack_array, 2, 0, "handle"))
            T.tvm_struct_set(stack_ffi_any, 3, 13, 0)
            T.tvm_struct_set(stack_ffi_any, 3, 14, T.int64(0))
            T.call_packed_lowered("tvm.test_matmul", stack_ffi_any, 0, 3)

    After = tvm.tir.transform.LowerTVMBuiltin()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


@tvm.testing.requires_llvm
def test_call_packed_return_non_i32():
    # This call packed that return non i32 types
    expected_value = np.array([1.2, 1.4], dtype="float32")

    def packed_echo(value):
        return tvm.tir.call_intrin(
            value.dtype, tvm.ir.Op.get("tir.tvm_call_packed"), "testing.echo", value
        )

    def build_tir():
        Ab = tvm.tir.decl_buffer((2,), "float32")
        ib = tvm.tir.ir_builder.create()
        Aptr = ib.buffer_ptr(Ab)
        # return f32
        # Aptr[0] = testing.echo(expected_value[0])
        Aptr[0] = packed_echo(tvm.tir.const(expected_value[0], "float32"))
        # return handle
        # let Aptr_var = testing.echo(Aptr) in Aptr_var[1] = expected_value[1]
        Aptr_var = ib.let("Aptr_dup", packed_echo(Aptr.asobject().data))
        ib.emit(tvm.tir.BufferStore(Aptr, tvm.tir.const(expected_value[1], "float32"), [1]))

        stmt = ib.get()
        return tvm.IRModule.from_expr(
            tvm.tir.PrimFunc([Ab], stmt).with_attr("global_symbol", "packed_test")
        )

    mod = build_tir()
    f = tvm.compile(mod, None)
    a = tvm.nd.array(np.zeros(2, dtype="float32"))
    f(a)
    tvm.testing.assert_allclose(a.numpy(), expected_value)


def test_lower_overflow_int32():
    @T.prim_func
    def variance4(rxplaceholder: T.Buffer((T.int64(1), T.int64(32), T.int64(25690112)), "float32")):
        T.func_attr({"global_symbol": "variance4", "tir.noalias": True})
        rxplaceholder_red = T.allocate([32], "float32", "global")
        T_subtract = T.allocate([822083584], "float32", "global")
        rxplaceholder_red_1 = T.Buffer((T.int64(32),), data=rxplaceholder_red)
        rxplaceholder_1 = T.Buffer((T.int64(822083584),), data=rxplaceholder.data)
        T_subtract_1 = T.Buffer((T.int64(822083584),), data=T_subtract)
        for ax1, ax2 in T.grid(32, 25690112):
            cse_v1: T.int32 = ax1 * 25690112 + ax2
            T_subtract_1[cse_v1] = rxplaceholder_1[cse_v1] - rxplaceholder_red_1[ax1]

    func = variance4
    tvm.compile(func, target="llvm")  # should not crash


class TestLowerDeviceAllocate(tvm.testing.CompareBeforeAfter):
    """Device allocations are lowered to TVMBackend* calls

    This test validates the current behavior of LowerTVMBuiltin.  This
    unit test may be improved in the future by addressing:

    - TVMScript always produces "handle" dtype for
      `T.tvm_throw_last_error`, while LowerTVMBuiltin outputs "int32"
      dtype.
    """

    transform = tvm.tir.transform.LowerTVMBuiltin()

    def before():
        T.func_attr({"target": T.target("llvm")})
        T.attr("dummy", "device_type", 2)  # kDLCuda
        T.attr("dummy", "device_id", 0)
        ptr = T.allocate([16], "float32")
        buf = T.decl_buffer(16, "float32", data=ptr)
        buf[0] = 0.0

    def expected():
        T.func_attr({"target": T.target("llvm")})
        ptr: T.handle("float32") = T.TVMBackendAllocWorkspace(2, 0, T.uint64(64), 2, 32)
        T.attr(ptr, "storage_alignment", 64)
        if T.isnullptr(ptr):
            T.Call("int32", "tir.tvm_throw_last_error", [])
        buf = T.decl_buffer((16,), data=ptr)
        buf[0] = T.float32(0)
        if T.TVMBackendFreeWorkspace(2, 0, ptr) != 0:
            T.Call("int32", "tir.tvm_throw_last_error", [])

    def test_compare(self, before, expected, transform):
        after = transform(before)
        assert_structural_equal_ignore_global_symbol(after, expected, map_free_vars=True)


class TestLowerCPUAllocation(tvm.testing.CompareBeforeAfter):
    """CPU allocations can be handled at codegen time"""

    transform = tvm.tir.transform.LowerTVMBuiltin()

    def before():
        T.func_attr({"target": T.target("llvm")})
        T.attr("dummy", "device_type", 1)  # kDLCPU
        T.attr("dummy", "device_id", 0)
        ptr = T.allocate([16], "float32")
        buf = T.decl_buffer(16, "float32", data=ptr)
        buf[0] = 0.0

    def expected():
        T.func_attr({"target": T.target("llvm")})
        ptr = T.allocate([16], "float32")
        buf = T.decl_buffer(16, "float32", data=ptr)
        buf[0] = 0.0


class TestLowerAllocateRequiresDeviceID(tvm.testing.CompareBeforeAfter):
    """If device id is missing, error."""

    transform = tvm.tir.transform.LowerTVMBuiltin()

    def before():
        T.func_attr({"target": T.target("llvm")})
        T.attr("dummy", "device_type", 2)  # kDLCuda
        ptr = T.allocate([16], "float32")
        buf = T.decl_buffer(16, "float32", data=ptr)
        buf[0] = 0.0

    expected = tvm.TVMError


class TestLowerAllocateRequiresDeviceType(tvm.testing.CompareBeforeAfter):
    """If device type is missing, error.

    The device type can be inferred either from the `"device_type"`
    statement attribute, or from the `"target"` function attribute.
    Here, we provide neither.  The `"tir.is_host_func"` attribute is
    provided as otherwise the function would be skipped altogether by
    LowerTVMBuiltin.
    """

    transform = tvm.tir.transform.LowerTVMBuiltin()

    def before():
        T.func_attr({"tir.is_host_func": True})
        T.attr("dummy", "device_id", 0)
        ptr = T.allocate([1024 * 1024], "float32")
        buf = T.decl_buffer(1024 * 1024, "float32", data=ptr)
        buf[0] = 0.0

    expected = tvm.TVMError


class TestLowerCPUAllocWithFunctionAttr(tvm.testing.CompareBeforeAfter):
    """CPU allocations can be handled at codegen time

    Like `TestLowerCPUAllocation`, but the device type is taken from
    the function attribute.  The `AttrStmt` can override the device
    type for allocations within its scope, but it defaults to the
    function's target.
    """

    transform = tvm.tir.transform.LowerTVMBuiltin()

    def before():
        T.func_attr({"target": T.target("llvm")})
        ptr = T.allocate([16], "float32")
        buf = T.decl_buffer(16, "float32", data=ptr)
        buf[0] = 0.0

    expected = before


if __name__ == "__main__":
    tvm.testing.main()
