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
"""Basic runtime enablement test."""

import math
import os
import pathlib
import tempfile

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import te

dtype = tvm.testing.parameter("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32")


def test_nd_create(target, dev, dtype):
    x = np.random.randint(0, 10, size=(3, 4))
    x = np.array(x, dtype=dtype)
    y = tvm.nd.array(x, device=dev)
    z = y.copyto(dev)
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert isinstance(y, tvm.nd.NDArray)
    np.testing.assert_equal(x, y.numpy())
    np.testing.assert_equal(x, z.numpy())

    # no need here, just to test usablity
    dev.sync()


def test_memory_usage(target, dev, dtype):
    available_memory_before = dev.available_global_memory
    if available_memory_before is None:
        pytest.skip(reason=f"Target '{target}' does not support queries of available memory")

    arr = tvm.nd.empty([1024, 1024], dtype=dtype, device=dev)
    available_memory_after = dev.available_global_memory

    num_elements = math.prod(arr.shape)
    element_nbytes = tvm.runtime.DataType(dtype).itemsize()
    expected_memory_after = available_memory_before - num_elements * element_nbytes

    # Allocations may be padded out to provide alignment, to match a
    # page boundary, due to additional device-side bookkeeping
    # required by the TVM backend or the driver, etc.  Therefore, the
    # available memory may decrease by more than the requested amount.
    assert available_memory_after <= expected_memory_after

    # TVM's NDArray type is a reference-counted handle to the
    # underlying reference.  After the last reference to an NDArray is
    # cleared, the backing allocation will be freed.
    del arr

    assert dev.available_global_memory == available_memory_before


# @pytest.mark.parametrize(
#     "src",
#     [
#         # "float32",
#         "float16",
#     ],
# )
# @pytest.mark.parametrize(
#     "dst",
#     [
#         "float32",
#         # "float16",
#     ],
# )
# @tvm.testing.parametrize_targets(
#     # "llvm",
#     "llvm -opt-level=0",
# )
# def test_fp16_conversion(src, dst, target, dev):
#     # DEBUG PRINT, REMOVE BEFORE MERGE
#     print("LLVM version:", tvm.support.libinfo()["LLVM_VERSION"])

#     n = 100

#     from tvm.script import ir as I, tir as T

#     @I.ir_module
#     class Module:
#         @T.prim_func
#         def main(
#             output_handle: T.handle,
#             input_handle: T.handle,
#         ):
#             Input = T.match_buffer(output_handle, 100, src)
#             Output = T.match_buffer(input_handle, 100, dst)

#             T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})

#             T.call_extern("void", "printf", "Start of function\n")
#             T.call_extern("void", "fflush", T.int64(0))

#             for i in range(100):
#                 T.call_extern("void", "printf", "Start of iteration %d\n", i)
#                 T.call_extern("void", "fflush", T.int64(0))

#                 input_value = Input[i]

#                 T.call_extern("void", "printf", "Read value in iteration %d\n", i)
#                 T.call_extern("void", "fflush", T.int64(0))

#                 output_value = T.Cast(dst, input_value)

#                 T.call_extern(
#                     "void", "printf", "Converted value to output type in iteration %d\n", i
#                 )
#                 T.call_extern("void", "fflush", T.int64(0))

#                 Output[i] = output_value

#                 T.call_extern("void", "printf", "Wrote value to output array in iteration %d\n", i)
#                 T.call_extern("void", "fflush", T.int64(0))

#                 T.call_extern("void", "printf", "End of iteration %d\n", i)
#                 T.call_extern("void", "fflush", T.int64(0))

#             T.call_extern("void", "printf", "End of function\n")
#             T.call_extern("void", "fflush", T.int64(0))

#     # A = te.placeholder((n,), dtype=src)
#     # B = te.compute((n,), lambda i: A[i].astype(dst))

#     # s = te.create_schedule([B.op])

#     # # DEBUG PRINT, REMOVE BEFORE MERGE
#     # tvm.lower(s, [A, B]).show()

#     # func = tvm.build(s, [A, B], target)

#     func = tvm.build(Module, target=target)

#     # DEBUG PRINT, REMOVE BEFORE MERGE
#     print(func.get_source(), flush=True)

#     x_tvm = tvm.nd.array(100 * np.random.randn(n).astype(src) - 50, dev)
#     y_tvm = tvm.nd.array(100 * np.random.randn(n).astype(dst) - 50, dev)

#     print(f"Input shape: {x_tvm.shape}, input dtype: {x_tvm.dtype}", flush=True)
#     print(f"Output shape: {y_tvm.shape}, output dtype: {y_tvm.dtype}", flush=True)

#     func(x_tvm, y_tvm)

#     expected = x_tvm.numpy().astype(dst)
#     real = y_tvm.numpy()

#     tvm.testing.assert_allclose(expected, real)

from tvm.script import ir as I, tir as T


@tvm.testing.parametrize_targets(
    # "llvm",
    "llvm -opt-level=0",
)
def test_fp16_conversion(target, dev):
    src = "float16"
    dst = "float32"
    # DEBUG PRINT, REMOVE BEFORE MERGE
    print("LLVM version:", tvm.support.libinfo()["LLVM_VERSION"])

    n = 100

    @I.ir_module
    class Module:
        I.module_attrs({"runtime": None})

        @T.prim_func
        def main(
            args: T.handle,
            arg_type_ids: T.handle("int32"),
            num_args: T.int32,
            out_ret_value: T.handle("void"),
            out_ret_tcode: T.handle("int32"),
            resource_handle: T.handle,
        ) -> T.int32:
            T.func_attr(
                {
                    "calling_conv": 1,
                    "from_legacy_te_schedule": T.bool(True),
                    "target": T.target(
                        {
                            "keys": ["cpu"],
                            "kind": "llvm",
                            "mtriple": "x86_64-pc-linux-gnu",
                            "opt-level": 0,
                            "tag": "",
                        }
                    ),
                    "tir.is_entry_func": T.bool(True),
                    "tir.noalias": T.bool(True),
                }
            )
            T.call_extern("void", "printf", "Start of TIR PrimFunc\n")
            T.call_extern("void", "fflush", T.int64(0))

            assert num_args == 2, "main: num_args should be 2"
            assert not T.isnullptr(args), "main: TVMValue* arg pointer was NULL"
            assert not T.isnullptr(arg_type_ids), "main: int* type_codes was NULL"

            arg_type_ids_1 = T.decl_buffer((2,), "int32", data=arg_type_ids)
            output_handle_code: T.int32 = arg_type_ids_1[0]
            assert (
                output_handle_code == 3
                or output_handle_code == 13
                or output_handle_code == 7
                or output_handle_code == 4
            ), "main: Expect arg[0] to be pointer"
            input_handle_code: T.int32 = arg_type_ids_1[1]
            assert (
                input_handle_code == 3
                or input_handle_code == 13
                or input_handle_code == 7
                or input_handle_code == 4
            ), "main: Expect arg[1] to be pointer"

            T.call_extern("void", "printf", "In TIR PrimFunc, after type-code asserts\n")
            T.call_extern("void", "fflush", T.int64(0))

            output_handle: T.handle = T.tvm_struct_get(args, 0, 12, "handle")
            input_handle: T.handle = T.tvm_struct_get(args, 1, 12, "handle")
            assert not T.isnullptr(
                output_handle
            ), "main.output_handle is expected to have non-NULL DLTensor* pointer"
            assert 1 == T.tvm_struct_get(
                output_handle, 0, 4, "int32"
            ), "main.output_handle.ndim is expected to equal 1"
            main_output_handle_shape: T.handle("int64") = T.tvm_struct_get(
                output_handle, 0, 2, "handle"
            )
            main_output_handle_shape_1 = T.decl_buffer((1,), "int64", data=main_output_handle_shape)
            main_output_handle_strides: T.handle("int64") = T.tvm_struct_get(
                output_handle, 0, 3, "handle"
            )
            main_output_handle_strides_1 = T.decl_buffer(
                (0,), "int64", data=main_output_handle_strides
            )
            dev_id: T.int32 = T.tvm_struct_get(output_handle, 0, 9, "int32")
            Input: T.handle("float16", "global") = T.tvm_struct_get(output_handle, 0, 1, "handle")
            T.attr(Input, "storage_alignment", 64)
            assert not T.isnullptr(
                input_handle
            ), "main.input_handle is expected to have non-NULL DLTensor* pointer"
            assert 1 == T.tvm_struct_get(
                input_handle, 0, 4, "int32"
            ), "main.input_handle.ndim is expected to equal 1"
            main_input_handle_shape: T.handle("int64") = T.tvm_struct_get(
                input_handle, 0, 2, "handle"
            )
            main_input_handle_shape_1 = T.decl_buffer((1,), "int64", data=main_input_handle_shape)
            main_input_handle_strides: T.handle("int64") = T.tvm_struct_get(
                input_handle, 0, 3, "handle"
            )
            main_input_handle_strides_1 = T.decl_buffer(
                (0,), "int64", data=main_input_handle_strides
            )
            Output: T.handle("float32", "global") = T.tvm_struct_get(input_handle, 0, 1, "handle")
            T.attr(Output, "storage_alignment", 64)
            T.attr("default", "device_id", dev_id)
            T.attr("default", "device_type", 1)
            assert (
                T.tvm_struct_get(output_handle, 0, 5, "uint8") == T.uint8(2)
                and T.tvm_struct_get(output_handle, 0, 6, "uint8") == T.uint8(16)
                and T.tvm_struct_get(output_handle, 0, 7, "uint16") == T.uint16(1)
            ), "main.output_handle.dtype is expected to be float16"
            assert (
                T.Cast("int32", main_output_handle_shape_1[0]) == 100
            ), 'Argument main.output_handle.shape[0] has an unsatisfied constraint: 100 == T.Cast("int32", main_output_handle_shape[0])'
            if not T.isnullptr(main_output_handle_strides):
                assert 1 == T.Cast(
                    "int32", main_output_handle_strides_1[0]
                ), "main.output_handle.strides: expected to be compact array"
                T.evaluate(0)
            assert T.uint64(0) == T.tvm_struct_get(
                output_handle, 0, 8, "uint64"
            ), 'Argument main.output_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(output_handle, 0, 8, "uint64")'
            assert (
                T.tvm_struct_get(output_handle, 0, 10, "int32") == 1
            ), 'Argument main.output_handle.device_type has an unsatisfied constraint: 1 == T.tvm_struct_get(output_handle, 0, 10, "int32")'
            assert not T.isnullptr(
                Input
            ), "main.output_handle is expected to have non-NULL data pointer"
            assert (
                T.tvm_struct_get(input_handle, 0, 5, "uint8") == T.uint8(2)
                and T.tvm_struct_get(input_handle, 0, 6, "uint8") == T.uint8(32)
                and T.tvm_struct_get(input_handle, 0, 7, "uint16") == T.uint16(1)
            ), "main.input_handle.dtype is expected to be float32"
            assert (
                T.Cast("int32", main_input_handle_shape_1[0]) == 100
            ), 'Argument main.input_handle.shape[0] has an unsatisfied constraint: 100 == T.Cast("int32", main_input_handle_shape[0])'
            if not T.isnullptr(main_input_handle_strides):
                assert 1 == T.Cast(
                    "int32", main_input_handle_strides_1[0]
                ), "main.input_handle.strides: expected to be compact array"
                T.evaluate(0)
            assert T.uint64(0) == T.tvm_struct_get(
                input_handle, 0, 8, "uint64"
            ), 'Argument main.input_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(input_handle, 0, 8, "uint64")'
            assert (
                T.tvm_struct_get(input_handle, 0, 10, "int32") == 1
            ), 'Argument main.input_handle.device_type has an unsatisfied constraint: 1 == T.tvm_struct_get(input_handle, 0, 10, "int32")'
            assert dev_id == T.tvm_struct_get(
                input_handle, 0, 9, "int32"
            ), 'Argument main.input_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(input_handle, 0, 9, "int32")'
            assert not T.isnullptr(
                Output
            ), "main.input_handle is expected to have non-NULL data pointer"

            T.call_extern(
                "void", "printf", "In TIR PrimFunc, after PackedFunc argument unpacking\n"
            )
            T.call_extern("void", "fflush", T.int64(0))

            Input_1 = T.decl_buffer((100,), "float16", data=Input)
            Output_1 = T.decl_buffer((100,), data=Output)

            T.call_extern("void", "printf", "Before entering compute_scope\n")
            T.call_extern("void", "fflush", T.int64(0))

            with T.attr(0, "compute_scope", "main_compute_"):
                T.call_extern("void", "printf", "Start of compute_scope\n")
                T.call_extern("void", "fflush", T.int64(0))
                for i in range(100):
                    T.call_extern("void", "printf", "Start of iteration %d\n", i)
                    T.call_extern("void", "fflush", T.int64(0))
                    input_value: T.float16 = Input_1[i]
                    T.call_extern("void", "printf", "Read value in iteration %d\n", i)
                    T.call_extern("void", "fflush", T.int64(0))
                    output_value: T.float32 = T.Cast("float32", input_value)
                    T.call_extern(
                        "void",
                        "printf",
                        "Converted value to output type in iteration %d\n",
                        i,
                    )
                    T.call_extern("void", "fflush", T.int64(0))
                    Output_1[i] = output_value
                    T.call_extern(
                        "void", "printf", "Wrote value to output array in iteration %d\n", i
                    )
                    T.call_extern("void", "fflush", T.int64(0))
                    T.call_extern("void", "printf", "End of iteration %d\n", i)
                    T.call_extern("void", "fflush", T.int64(0))
                T.call_extern("void", "printf", "End of function\n")
                T.call_extern("void", "fflush", T.int64(0))
            return 0

    # A = te.placeholder((n,), dtype=src)
    # B = te.compute((n,), lambda i: A[i].astype(dst))

    # s = te.create_schedule([B.op])

    # # DEBUG PRINT, REMOVE BEFORE MERGE
    # tvm.lower(s, [A, B]).show()

    # func = tvm.build(s, [A, B], target)

    with tvm.transform.PassContext(
        disabled_pass=[
            "tir.StorageFlatten",
        ],
    ):
        func = tvm.build(Module, target=target)

    # DEBUG PRINT, REMOVE BEFORE MERGE
    print(func.get_source(), flush=True)

    x_tvm = tvm.nd.array(100 * np.random.randn(n).astype(src) - 50, dev)
    y_tvm = tvm.nd.array(100 * np.random.randn(n).astype(dst) - 50, dev)

    print(f"Input shape: {x_tvm.shape}, input dtype: {x_tvm.dtype}", flush=True)
    print(f"Output shape: {y_tvm.shape}, output dtype: {y_tvm.dtype}", flush=True)

    with tempfile.TemporaryDirectory(prefix="tvm_ndarray_") as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        extension = "dll" if os.name == "nt" else "so"

        libname = temp_dir.joinpath(f"libexec.{extension}")
        func.export_library(libname)
        func = tvm.runtime.load_module(libname)

        func(x_tvm, y_tvm)

    expected = x_tvm.numpy().astype(dst)
    real = y_tvm.numpy()

    tvm.testing.assert_allclose(expected, real)


def test_dtype():
    dtype = tvm.DataType("handle")
    assert dtype.type_code == tvm.DataTypeCode.HANDLE


if __name__ == "__main__":
    tvm.testing.main()
