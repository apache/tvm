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

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@tvm_ffi.register_global_func("tvm.test_matmul")
def my_matmul(a, b, c):
    c.copyfrom(np.dot(a.numpy(), b.numpy()))


def test_lower_call_packed():
    @I.ir_module
    class Before:
        @T.function
        def main(
            A: T.Tensor((64, 64), "float32"),
            B: T.Tensor((64, 64), "float32"),
            C: T.Tensor((64, 64), "float32"),
        ):
            T.func_attr({"target": tvm_tirx.target.Target("llvm")})
            with T.device_context(1, 0):
                T.call_packed("tvm.test_matmul", A, B, C)

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.function
        def main(
            A: T.Tensor((64, 64), "float32"),
            B: T.Tensor((64, 64), "float32"),
            C: T.Tensor((64, 64), "float32"),
        ):
            T.func_attr({"target": tvm_tirx.target.Target("llvm")})
            stack_ffi_any: T.let[T.handle] = T.stack_alloca("tvm_ffi_any", 4)
            stack_array: T.let[T.handle] = T.stack_alloca("array", 3)
            stack_shape = T.decl_tensor(
                T.stack_alloca("shape", 6),
                ty_args=[
                    T.Tensor((T.int64(6),), "int64", layout=None, data=T.stack_alloca("shape", 6))
                ],
            )
            stack_shape[0] = T.int64(64)
            stack_shape[1] = T.int64(64)
            T.abi_field_set(stack_array, 0, 1, A.data)
            T.abi_field_set(stack_array, 0, 2, T.address_of(stack_shape[0]))
            T.abi_field_set(stack_array, 0, 3, T.reinterpret(T.uint64(0), ty="handle"))
            T.abi_field_set(stack_array, 0, 4, 2)
            T.abi_field_set(stack_array, 0, 5, T.uint8(2))
            T.abi_field_set(stack_array, 0, 6, T.uint8(32))
            T.abi_field_set(stack_array, 0, 7, T.uint16(1))
            T.abi_field_set(stack_array, 0, 8, T.uint64(0))
            T.abi_field_set(stack_array, 0, 9, 0)
            T.abi_field_set(stack_array, 0, 10, 1)
            stack_shape[2] = T.int64(64)
            stack_shape[3] = T.int64(64)
            T.abi_field_set(stack_array, 1, 1, B.data)
            T.abi_field_set(stack_array, 1, 2, T.address_of(stack_shape[2]))
            T.abi_field_set(stack_array, 1, 3, T.reinterpret(T.uint64(0), ty="handle"))
            T.abi_field_set(stack_array, 1, 4, 2)
            T.abi_field_set(stack_array, 1, 5, T.uint8(2))
            T.abi_field_set(stack_array, 1, 6, T.uint8(32))
            T.abi_field_set(stack_array, 1, 7, T.uint16(1))
            T.abi_field_set(stack_array, 1, 8, T.uint64(0))
            T.abi_field_set(stack_array, 1, 9, 0)
            T.abi_field_set(stack_array, 1, 10, 1)
            stack_shape[4] = T.int64(64)
            stack_shape[5] = T.int64(64)
            T.abi_field_set(stack_array, 2, 1, C.data)
            T.abi_field_set(stack_array, 2, 2, T.address_of(stack_shape[4]))
            T.abi_field_set(stack_array, 2, 3, T.reinterpret(T.uint64(0), ty="handle"))
            T.abi_field_set(stack_array, 2, 4, 2)
            T.abi_field_set(stack_array, 2, 5, T.uint8(2))
            T.abi_field_set(stack_array, 2, 6, T.uint8(32))
            T.abi_field_set(stack_array, 2, 7, T.uint16(1))
            T.abi_field_set(stack_array, 2, 8, T.uint64(0))
            T.abi_field_set(stack_array, 2, 9, 0)
            T.abi_field_set(stack_array, 2, 10, 1)
            T.abi_field_set(stack_ffi_any, 0, 13, 7)
            T.abi_field_set(stack_ffi_any, 0, 14, 0)
            T.abi_field_set(stack_ffi_any, 0, 15, T.abi_field_get(stack_array, 0, 0, ty="handle"))
            T.abi_field_set(stack_ffi_any, 1, 13, 7)
            T.abi_field_set(stack_ffi_any, 1, 14, 0)
            T.abi_field_set(stack_ffi_any, 1, 15, T.abi_field_get(stack_array, 1, 0, ty="handle"))
            T.abi_field_set(stack_ffi_any, 2, 13, 7)
            T.abi_field_set(stack_ffi_any, 2, 14, 0)
            T.abi_field_set(stack_ffi_any, 2, 15, T.abi_field_get(stack_array, 2, 0, ty="handle"))
            T.abi_field_set(stack_ffi_any, 3, 13, 0)
            T.abi_field_set(stack_ffi_any, 3, 14, 0)
            T.abi_field_set(stack_ffi_any, 3, 15, T.int64(0))
            T.call_packed_lowered("tvm.test_matmul", stack_ffi_any, 0, 3)

    After = tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("call", [tvm_tirx.tirx.call_packed, tvm_tirx.tirx.call_ffi_kernel])
def test_lower_call_packed_raw_string(call):
    def invoke(*args):
        if call is tvm_tirx.tirx.call_ffi_kernel:
            return call(*args, launch_params=[])
        return call(*args)

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"target": tvm_tirx.target.Target("llvm")})
            T.evaluate(invoke("testing.echo", "payload"))

    @I.ir_module
    class Expected:
        @T.function
        def main():
            T.func_attr({"target": tvm_tirx.target.Target("llvm")})
            stack_ffi_any: T.let[T.handle] = T.stack_alloca("tvm_ffi_any", 2)
            T.abi_field_set(stack_ffi_any, 0, 13, 8)
            T.abi_field_set(stack_ffi_any, 0, 14, 0)
            T.abi_field_set(stack_ffi_any, 0, 15, T.reinterpret("payload", ty=T.handle().ty))
            T.abi_field_set(stack_ffi_any, 1, 13, 0)
            T.abi_field_set(stack_ffi_any, 1, 14, 0)
            T.abi_field_set(stack_ffi_any, 1, 15, T.int64(0))
            T.call_packed_lowered("testing.echo", stack_ffi_any, 0, 1)

    After = tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)

    # The typed pointer is required by the LLVM TVMFFIAny lowering.
    tvm_tirx.tirx.compile(Before, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_call_packed_return_non_i32():
    # This call packed that return non i32 types
    expected_value = np.array([1.2, 1.4], dtype="float32")

    @I.ir_module
    class Before:
        @T.function
        def packed_test(A: T.Tensor((2,), "float32")):
            T.func_attr({"target": T.target("llvm")})
            A[0] = T.call_intrin("tirx.call_packed", "testing.echo", T.float32(1.2), ty="float32")
            duplicate = T.call_intrin("tirx.call_packed", "testing.echo", A.data, ty=A.data.ty)
            A[1] = T.float32(1.4)

    @I.ir_module
    class Expected:
        @T.function
        def packed_test(A: T.Tensor((2,), T.float32)):
            T.func_attr({"target": T.target("llvm")})
            stack_ffi_any: T.let[T.Ptr(T.PrimType("void"))] = T.stack_alloca("tvm_ffi_any", 2)
            T.abi_field_set(stack_ffi_any, 0, 13, 3)
            T.abi_field_set(stack_ffi_any, 0, 14, 0)
            T.abi_field_set(stack_ffi_any, 0, 15, T.Cast("float64", T.float32(1.2)))
            T.abi_field_set(stack_ffi_any, 1, 13, 0)
            T.abi_field_set(stack_ffi_any, 1, 14, 0)
            T.abi_field_set(stack_ffi_any, 1, 15, T.int64(0))
            A[0] = T.call_packed_lowered("testing.echo", stack_ffi_any, 0, 1, ty="float32")
            if T.isnullptr(A.data):
                T.abi_field_set(stack_ffi_any, 0, 13, 0)
            else:
                T.abi_field_set(stack_ffi_any, 0, 13, 4)
            T.abi_field_set(stack_ffi_any, 0, 14, 0)
            T.abi_field_set(stack_ffi_any, 0, 15, A.data)
            T.abi_field_set(stack_ffi_any, 1, 13, 0)
            T.abi_field_set(stack_ffi_any, 1, 14, 0)
            T.abi_field_set(stack_ffi_any, 1, 15, T.int64(0))
            duplicate: T.let[T.Ptr(T.PrimType("float32"))] = T.call_packed_lowered(
                "testing.echo", stack_ffi_any, 0, 1, ty=A.data.ty
            )
            A[1] = T.float32(1.3999999999999999)

    After = tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    f = tvm_tirx.tirx.compile(Before, "llvm")
    a = tvm_tirx.runtime.tensor(np.zeros(2, dtype="float32"))
    f(a)
    tvm_tirx.testing.assert_allclose(a.numpy(), expected_value)


def test_lower_overflow_int32():
    @I.ir_module(check_well_formed=False)
    class Before:
        @T.function(check_well_formed=False)
        def variance4(
            rxplaceholder: T.Tensor((T.int64(1), T.int64(32), T.int64(25690112)), "float32"),
        ):
            T.func_attr(
                {"global_symbol": "variance4", "tirx.noalias": True, "target": T.target("llvm")}
            )
            with T.device_context(1, 0):
                rxplaceholder_red = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32")])
                T_subtract = T.alloc_tensor(ty_args=[T.Tensor((822083584,), "float32")])
                rxplaceholder_red_1 = T.decl_tensor(
                    rxplaceholder_red.data,
                    ty_args=[T.Tensor((T.int64(32),), "float32", data=rxplaceholder_red.data)],
                )
                rxplaceholder_1 = T.decl_tensor(
                    rxplaceholder.data,
                    ty_args=[T.Tensor((T.int64(822083584),), "float32", data=rxplaceholder.data)],
                )
                T_subtract_1 = T.decl_tensor(
                    T_subtract.data,
                    ty_args=[T.Tensor((T.int64(822083584),), "float32", data=T_subtract.data)],
                )
                for ax1, ax2 in T.grid(32, 25690112):
                    cse_v1: T.let[T.int32] = ax1 * 25690112 + ax2
                    T_subtract_1[cse_v1] = rxplaceholder_1[cse_v1] - rxplaceholder_red_1[ax1]

    @I.ir_module
    class Expected:
        @T.function
        def variance4(
            rxplaceholder: T.Tensor((T.int64(1), T.int64(32), T.int64(25690112)), T.float32),
        ):
            T.func_attr({"target": T.target("llvm"), "tirx.noalias": True})
            rxplaceholder_red = T.alloc_tensor(ty_args=[T.Tensor((32,), T.float32)])
            T_subtract = T.decl_tensor(
                T.alloc_workspace(1, 0, T.uint64(3288334336), 2, 32, ty=T.Ptr(T.float32)),
                ty_args=[T.Tensor((822083584,), T.float32)],
            )
            if T.isnullptr(T_subtract.data):
                T.throw_last_error(ty="int32")
            rxplaceholder_red_1 = T.decl_tensor(
                rxplaceholder_red.data, ty_args=[T.Tensor((T.int64(32),), T.float32)]
            )
            rxplaceholder_1 = T.decl_tensor(
                rxplaceholder.data, ty_args=[T.Tensor((T.int64(822083584),), T.float32)]
            )
            T_subtract_1 = T.decl_tensor(
                T_subtract.data, ty_args=[T.Tensor((T.int64(822083584),), T.float32)]
            )
            for ax1 in range(32):
                for ax2 in range(25690112):
                    cse_v1: T.let[T.int32] = ax1 * 25690112 + ax2
                    T_subtract_1[cse_v1] = rxplaceholder_1[cse_v1] - rxplaceholder_red_1[ax1]
            if T.free_workspace(1, 0, T_subtract.data) != 0:
                T.throw_last_error(ty="int32")

    After = tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    tvm_tirx.tirx.compile(Before, target="llvm")


def test_lower_device_allocate():
    """Device allocations are lowered to workspace operations."""

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"target": T.target("llvm")})
            with T.device_context(2, 0):  # kDLCuda
                ptr = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
                buf = T.decl_tensor(ptr.data, ty_args=[T.Tensor(16, "float32", data=ptr.data)])
                buf[0] = 0.0

    @I.ir_module
    class Expected:
        @T.function
        def main():
            T.func_attr({"target": T.target("llvm")})
            ptr = T.decl_tensor(
                T.alloc_workspace(2, 0, T.uint64(64), 2, 32, ty=T.Ptr(T.float32)),
                ty_args=[T.Tensor((16,), T.float32)],
            )
            if T.isnullptr(ptr.data):
                T.throw_last_error(ty="int32")
            buf = T.decl_tensor(ptr.data, ty_args=[T.Tensor((16,), T.float32)])
            buf[0] = T.float32(0.0)
            if T.free_workspace(2, 0, ptr.data) != 0:
                T.throw_last_error(ty="int32")

    After = tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_lower_cpu_allocation():
    """CPU allocations can be handled at codegen time"""

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"target": T.target("llvm")})
            with T.device_context(1, 0):  # kDLCPU
                ptr = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
                buf = T.decl_tensor(ptr.data, ty_args=[T.Tensor(16, "float32", data=ptr.data)])
                buf[0] = 0.0

    @I.ir_module
    class Expected:
        @T.function
        def main():
            T.func_attr({"target": T.target("llvm")})
            ptr = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            buf = T.decl_tensor(ptr.data, ty_args=[T.Tensor(16, "float32", data=ptr.data)])
            buf[0] = 0.0

    After = tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_lower_allocate_requires_device_id():
    """If device id is missing, error."""

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"target": T.target("cuda"), "tirx.is_host_func": True})
            ptr = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            buf = T.decl_tensor(ptr.data, ty_args=[T.Tensor(16, "float32", data=ptr.data)])
            buf[0] = 0.0

    with pytest.raises(RuntimeError):
        tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)


def test_lower_allocate_requires_device_type():
    """If device type is missing, error.

    The device type can be inferred either from a device_context
    region or from the `"target"` function attribute.
    Here, we provide neither.  The `"tirx.is_host_func"` attribute is
    provided as otherwise the function would be skipped altogether by
    LowerTVMBuiltin.
    """

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"tirx.is_host_func": True})
            ptr = T.alloc_tensor(ty_args=[T.Tensor((1024 * 1024,), "float32")])
            buf = T.decl_tensor(ptr.data, ty_args=[T.Tensor(1024 * 1024, "float32", data=ptr.data)])
            buf[0] = 0.0

    with pytest.raises(RuntimeError):
        tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)


def test_lower_cpu_alloc_with_function_attr():
    """CPU allocations can be handled at codegen time

    Like `test_lower_cpu_allocation`, but the device type is taken from
    the function attribute.  A device_context region can override the device
    type for allocations within its scope, but it defaults to the
    function's target.
    """

    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"target": T.target("llvm")})
            ptr = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            buf = T.decl_tensor(ptr.data, ty_args=[T.Tensor(16, "float32", data=ptr.data)])
            buf[0] = 0.0

    # Expected is same as before for this transform
    Expected = Before

    After = tvm_tirx.tirx.transform.LowerTVMBuiltin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
