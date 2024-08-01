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
from tvm.script import ir as I, relax as R, tir as T

import numpy as np
import pytest


def test_infer_shape_of_1d_static_view():
    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor) -> R.Tensor([4096]):
        B: R.Tensor([4096]) = R.memory.view(A, R.shape([4096]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor):
        B = R.memory.view(A, R.shape([4096]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_infer_shape_of_2d_static_view():
    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor) -> R.Tensor([64, 64]):
        B: R.Tensor([64, 64]) = R.memory.view(A, R.shape([64, 64]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor):
        B = R.memory.view(A, R.shape([64, 64]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_error_if_shape_argument_is_not_shape():
    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor([16])):
            B = R.memory.view(A, R.prim_value(42))
            return B


def test_infer_shape_of_1d_static_view_smaller_than_1d_source():
    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor([4096])) -> R.Tensor([16]):
        B: R.Tensor([16]) = R.memory.view(A, R.shape([16]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([4096])):
        B = R.memory.view(A, R.shape([16]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_infer_shape_of_2d_static_view_smaller_than_1d_source():
    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor([4096])) -> R.Tensor([4, 4]):
        B: R.Tensor([4, 4]) = R.memory.view(A, R.shape([4, 4]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([4096])):
        B = R.memory.view(A, R.shape([4, 4]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_infer_shape_of_2d_static_view_same_size_as_2d_source():
    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor([64, 64])) -> R.Tensor([16, 256]):
        B: R.Tensor([16, 256]) = R.memory.view(A, R.shape([16, 256]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([64, 64])):
        B = R.memory.view(A, R.shape([16, 256]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_error_if_1d_static_view_larger_than_1d_source():
    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor([16])):
            B = R.memory.view(A, R.shape([17]))
            return B


def test_error_if_static_2d_view_larger_than_source():
    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor([16])):
            B = R.memory.view(A, R.shape([4, 5]))
            return B


def test_infer_shape_of_1d_dynamic_view():
    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor(["N"])) -> R.Tensor(["N // 2"]):
        N = T.int64()
        B: R.Tensor([N // 2]) = R.memory.view(A, R.shape([N // 2]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor(["N"])):
        N = T.int64()
        B = R.memory.view(A, R.shape([N // 2]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_infer_shape_of_2d_dynamic_view_of_1d_source():
    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor(["N"])) -> R.Tensor(["N // 8", 8]):
        N = T.int64()
        B: R.Tensor([N // 8, 8]) = R.memory.view(A, R.shape([N // 8, 8]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor(["N"])):
        N = T.int64()
        B = R.memory.view(A, R.shape([N // 8, 8]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_infer_shape_of_2d_dynamic_view():
    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor(["N"])) -> R.Tensor(["N // 2"]):
        N = T.int64()
        B: R.Tensor([N // 2]) = R.memory.view(A, R.shape([N // 2]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor(["N"])):
        N = T.int64()
        B = R.memory.view(A, R.shape([N // 2]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_error_if_1d_dynamic_view_larger_than_1d_source():
    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor(["N"])):
            N = T.int64()
            B = R.memory.view(A, R.shape([N + 1]))
            return B


@pytest.mark.xfail(reason="See https://github.com/apache/tvm/pull/16877")
def test_error_if_1d_dynamic_view_provably_larger_than_1d_source():
    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor(["N"])):
            N = T.int64()
            B = R.memory.view(A, R.shape([N + T.if_then_else(N < 0, -1, 1)]))
            return B


def test_error_if_2d_dynamic_view_provably_larger_than_1d_source():
    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor(["N"])):
            N = T.int64()
            B = R.memory.view(A, R.shape([N // 4 + 1, 4]))
            return B


def test_validity_of_dynamic_view_may_depend_on_runtime_value():
    """Validity checks may be delayed until runtime

    The runtime implementation of `R.memory.view` checks the validity of any
    dynamic shape.  A compile-time error should only be issued the
    runtime check would fail for *all* dynamic shapes.

    In this example, the output of `R.memory.view` contains `N` elements when
    `N` is evenly divisible by 4, and `N+4` elements otherwise.  The
    runtime check would pass whenever the argument's size is divisible
    by 4.  Even though the runtime check would fail when `N` isn't
    divisible by 4, no compile-time error should be emitted.

    """

    @R.function
    def func(A: R.Tensor(["N"])):
        N = T.int64()
        B = R.memory.view(A, R.shape([(N + 3) // 4, 4]))
        return B


def test_infer_dtype_of_float32_view():
    """R.memory.view can reinterpret the contents as another type

    For example, if the same backing allocation is used for multiple
    arrays with distinct datatypes.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor) -> R.Tensor("float32"):
        B: R.Tensor("float32") = R.memory.view(A, dtype=R.dtype("float32"))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor):
        B = R.memory.view(A, dtype=R.dtype("float32"))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_view_without_explicit_dtype_keeps_input_dtype():
    """If R.memory.view only specifies the shape, the dtype is unchanged"""

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor([16], "float32")) -> R.Tensor([4, 4], "float32"):
        B: R.Tensor([4, 4], "float32") = R.memory.view(A, R.shape([4, 4]))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([16], "float32")):
        B = R.memory.view(A, R.shape([4, 4]))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_infer_dtype_of_float32_view_from_relax_var():
    """R.memory.view can reinterpret the contents as another type

    Any relax object can be stored in a relax variable.  Even if the
    `R.dtype` argument is stored in a variable, struct inference may
    be applied.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor) -> R.Tensor("float32"):
        dtype = R.dtype("float32")
        B: R.Tensor("float32") = R.memory.view(A, dtype=dtype)
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor):
        dtype = R.dtype("float32")
        B = R.memory.view(A, dtype=dtype)
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_infer_dtype_of_view_with_unknown_dtype():
    """DType may be provided as argument

    Because we do not know the value provided in `dtype`, the element
    type of the array is unknown.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor("float32"), dtype: R.Object) -> R.Tensor:
        B: R.Tensor = R.memory.view(A, dtype=dtype)
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor("float32"), dtype: R.Object):
        B = R.memory.view(A, dtype=dtype)
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_view_dtype_may_be_smaller_than_input_dtype():
    """Viewing with a smaller dtype does not exceed original bounds

    This is not typically desired behavior, as the view would span
    fewer bytes than the original array.  However, this is legal, and
    may occur as the result of optimization passes.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor("uint32")) -> R.Tensor("float8"):
        B: R.Tensor("float8") = R.memory.view(A, dtype=R.dtype("float8"))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor("uint32")):
        B = R.memory.view(A, dtype=R.dtype("float8"))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_error_if_view_dtype_is_larger_than_input_dtype():
    """A view may not exceed the bounds of the viewed array"""
    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor([16], "uint8")):
            B = R.memory.view(A, dtype=R.dtype("float16"))
            return B


def test_increase_dtype_size_while_decreasing_number_of_elements():
    """R.memory.view may update both dtype and shape simultaneously

    Like `test_error_if_dtype_results_in_larger_view`, but the view
    contains fewer elements than the backing array.  This results in a
    view that is the same size as the backing array, and would not
    exceed the bounds of the original array.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor([16], "uint8")) -> R.Tensor([8], "float16"):
        B: R.Tensor([8], "float16") = R.memory.view(A, shape=R.shape([8]), dtype=R.dtype("float16"))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([16], "uint8")):
        B = R.memory.view(A, shape=R.shape([8]), dtype=R.dtype("float16"))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_decrease_dtype_size_while_increasing_number_of_elements():
    """R.memory.view may update both dtype and shape simultaneously"""

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor([8], "float16")) -> R.Tensor([16], "uint8"):
        B: R.Tensor([16], "uint8") = R.memory.view(A, shape=R.shape([16]), dtype=R.dtype("uint8"))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([8], "float16")):
        B = R.memory.view(A, shape=R.shape([16]), dtype=R.dtype("uint8"))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_error_if_number_of_bytes_of_view_is_larger_than_original():
    """R.memory.view may update both dtype and shape simultaneously

    In this test case, the source array is 16 bytes (8 elements * 2
    bytes/element), but the view is 32 bytes (32 elements * 1
    byte/element).

    """
    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor([8], "float16")):
            B = R.memory.view(A, shape=R.shape([32]), dtype=R.dtype("uint8"))
            return B


def test_error_for_non_zero_relative_byte_offset():
    """R.memory.view must not exceed bounds of the original array

    Providing a non-zero `relative_byte_offset`, without updating
    either the dtype or the shape of the array, would allow the view
    to overrun the end of the original array.

    """

    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor):
            B = R.memory.view(A, relative_byte_offset=16)
            return B


def test_applying_relative_byte_offset_of_zero_is_legal():
    """Using relative_byte_offset=0 is no-op

    Providing a `relative_byte_offset` of zero, without updating
    either the dtype or the shape of the array, is legal, though it is
    a no-op.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor) -> R.Tensor:
        B: R.Tensor = R.memory.view(A, relative_byte_offset=R.prim_value(0))
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor):
        B = R.memory.view(A, relative_byte_offset=R.prim_value(0))
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_applying_unknown_relative_byte_offset_is_legal():
    """Using an unknown relative_byte_offset is legal

    Since providing a `relative_byte_offset` of zero, without updating
    either the dtype or the shape of the array, is legal, we may not
    emit a compile-time error for an unknown `relative_byte_offset` in
    this case.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor, relative_byte_offset: R.Prim("int64")) -> R.Tensor:
        B: R.Tensor = R.memory.view(A, relative_byte_offset=relative_byte_offset)
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor, relative_byte_offset: R.Prim("int64")):
        B = R.memory.view(A, relative_byte_offset=relative_byte_offset)
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_legalize_is_no_op():
    """R.memory.view is not legalized until LowerRuntimeBuiltin"""

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.memory.view(A)
            return B

    Expected = Before

    After = tvm.relax.transform.LegalizeOps()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_lower_runtime_builtin_shape_change():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.memory.view(A, shape=R.shape([64, 64]))
            return B

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.ExternFunc(
                "runtime.TVMArrayCreateView",
                R.Callable(
                    derive_func="tvm.relax.struct_info.infer_view_sinfo",
                    purity=True,
                ),
            )(
                A,
                R.shape([64, 64]),
                R.dtype("float32"),
                R.prim_value(0),
            )
            return B

    After = tvm.relax.transform.LowerRuntimeBuiltin()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_lower_runtime_builtin_view_shape_from_unknown():
    """R.memory.view does not require the input tensor to have a known shape"""

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(dtype="float32")):
            B = R.memory.view(A, shape=R.shape([64, 64]))
            return B

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor(dtype="float32")):
            B = R.ExternFunc(
                "runtime.TVMArrayCreateView",
                R.Callable(
                    derive_func="tvm.relax.struct_info.infer_view_sinfo",
                    purity=True,
                ),
            )(
                A,
                R.shape([64, 64]),
                R.dtype("float32"),
                R.prim_value(0),
            )
            return B

    After = tvm.relax.transform.LowerRuntimeBuiltin()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_lower_runtime_builtin_dtype_change():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.memory.view(A, dtype=R.dtype("int32"))
            return B

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.ExternFunc(
                "runtime.TVMArrayCreateView",
                R.Callable(
                    derive_func="tvm.relax.struct_info.infer_view_sinfo",
                    purity=True,
                ),
            )(
                A,
                R.shape([4096]),
                R.dtype("int32"),
                R.prim_value(0),
            )
            return B

    After = tvm.relax.transform.LowerRuntimeBuiltin()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_lower_runtime_builtin_byte_offset():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.memory.view(A, relative_byte_offset=R.prim_value(0))
            return B

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.ExternFunc(
                "runtime.TVMArrayCreateView",
                R.Callable(
                    derive_func="tvm.relax.struct_info.infer_view_sinfo",
                    purity=True,
                ),
            )(
                A,
                R.shape([4096]),
                R.dtype("float32"),
                R.prim_value(0),
            )
            return B

    After = tvm.relax.transform.LowerRuntimeBuiltin()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_lower_runtime_builtin_view_with_multiple_updated_fields():
    """R.memory.view may update more than one field in the view

    In this test case, a 4-kilobyte buffer is provided.  The first
    2-kilobytes of the buffer are used as a 1-d array of 512 int32.
    The last 2-kilobytes of the buffer are used as a 2-d array of
    [16,64] float16 values.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([4096], "uint8")):
            B = R.memory.view(
                A,
                shape=R.shape([512]),
                dtype=R.dtype("int32"),
            )
            C = R.memory.view(
                A,
                shape=R.shape([16, 64]),
                dtype=R.dtype("float16"),
                relative_byte_offset=R.prim_value(2048),
            )
            return (B, C)

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([4096], "uint8")):
            B = R.ExternFunc(
                "runtime.TVMArrayCreateView",
                R.Callable(
                    derive_func="tvm.relax.struct_info.infer_view_sinfo",
                    purity=True,
                ),
            )(
                A,
                R.shape([512]),
                R.dtype("int32"),
                R.prim_value(0),
            )
            C = R.ExternFunc(
                "runtime.TVMArrayCreateView",
                R.Callable(
                    derive_func="tvm.relax.struct_info.infer_view_sinfo",
                    purity=True,
                ),
            )(
                A,
                R.shape([16, 64]),
                R.dtype("float16"),
                R.prim_value(2048),
            )
            return (B, C)

    After = tvm.relax.transform.LowerRuntimeBuiltin()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_execute_no_op_view(target, dev):
    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.memory.view(A)
            return B

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, device=dev)

    np_input = np.random.random([4096]).astype("float32")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_execute_view_with_new_shape(target, dev):
    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.memory.view(A, shape=R.shape([64, 64]))
            return B

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, device=dev)

    np_input = np.random.random([4096]).astype("float32")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.reshape(64, 64)

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_execute_view_with_new_byte_offset(target, dev):
    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.memory.view(
                A,
                shape=R.shape([16, 64]),
                relative_byte_offset=32 * 64 * 4,
            )
            return B

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, device=dev)

    np_input = np.random.random([4096]).astype("float32")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.reshape(64, 64)[32:48, :]

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_execute_view_with_new_dtype(target, dev):
    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([4096], "float32")):
            B = R.memory.view(A, dtype="uint32")
            return B

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, device=dev)

    np_input = np.random.random([4096]).astype("float32")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.view("uint32")

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_execute_view_with_multiple_updated_fields(target, dev):
    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([4096], "uint8")):
            B = R.memory.view(
                A,
                shape=R.shape([512]),
                dtype=R.dtype("int32"),
            )
            C = R.memory.view(
                A,
                shape=R.shape([16, 64]),
                dtype=R.dtype("float16"),
                relative_byte_offset=R.prim_value(2048),
            )
            return (B, C)

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, device=dev)

    np_input = np.random.randint(0, 255, size=[4096]).astype("uint8")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = [
        np_input[:2048].view("int32"),
        np_input[2048:].view("float16").reshape(16, 64),
    ]

    tvm.testing.assert_allclose(tvm_output[0].numpy(), np_expected[0])
    tvm.testing.assert_allclose(tvm_output[1].numpy(), np_expected[1])


if __name__ == "__main__":
    tvm.testing.main()
