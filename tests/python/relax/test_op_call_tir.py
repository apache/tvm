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

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op, VDevice
from tvm.script import ir as I, relax as R, tir as T

exec_mode = tvm.testing.parameter("bytecode", "compiled")


def test_basic(exec_mode):
    """Relax may call into TIR functions with R.call_tir"""

    target = "llvm"
    dev = tvm.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "int32")):
            B = R.call_tir(Module.tir_func, [A], out_sinfo=R.Tensor([16], "int32"))
            return B

        @T.prim_func
        def tir_func(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
            for i in range(16):
                B[i] = A[i] * 2

    built = tvm.relax.build(Module, target=target, exec_mode=exec_mode)
    vm = tvm.relax.VirtualMachine(built, dev)

    arg = np.arange(16, dtype="int32")
    expected = 2 * arg

    output = vm["main"](tvm.nd.array(arg)).numpy()

    np.testing.assert_equal(expected, output)


def test_tir_var_scalar(exec_mode):
    """R.call_tir may accept dynamic shape arguments

    R.call_tir may provide TIR variables with the `tir_vars` argument.
    These appear after the output tensor arguments.

    Because `tir_vars` are stored internally as a `R.Shape`, they only
    may only define int64 values.

    """

    target = "llvm"
    dev = tvm.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "int64"), _: R.Prim(value="scale")):
            scale = T.int64()
            B = R.call_tir(
                Module.tir_func, [A], out_sinfo=R.Tensor([16], "int64"), tir_vars=[scale]
            )
            return B

        @T.prim_func
        def tir_func(
            A: T.Buffer(16, "int64"),
            B: T.Buffer(16, "int64"),
            scale: T.int64,
        ):
            for i in range(16):
                B[i] = A[i] * scale

    built = tvm.relax.build(Module, target=target, exec_mode=exec_mode)
    vm = tvm.relax.VirtualMachine(built, dev)

    arg = np.arange(16, dtype="int64")
    expected = 4 * arg

    output = vm["main"](tvm.nd.array(arg), 4).numpy()

    np.testing.assert_equal(expected, output)


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32"])
def test_prim_value_scalar(exec_mode, dtype):
    """Relax may pass PrimValue arguments to TIR

    R.call_tir may provide TIR scalars as arguments of type `R.Prim`.
    These appear in-line with the tensor arguments, before the output
    arguments.

    Unlike the `tir_vars` parameter, any scalar dtype may be provided
    as a PrimValue argument.

    """

    target = "llvm"
    dev = tvm.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], dtype), scale: R.Prim(dtype)):
            B = R.call_tir(Module.tir_func, [A, scale], out_sinfo=R.Tensor([16], dtype))
            return B

        @T.prim_func
        def tir_func(
            A: T.Buffer(16, dtype),
            scale: T.var(dtype),
            B: T.Buffer(16, dtype),
        ):
            _ = T.meta_var(dtype)
            for i in range(16):
                B[i] = A[i] * scale

    built = tvm.relax.build(Module, target=target, exec_mode=exec_mode)
    vm = tvm.relax.VirtualMachine(built, dev)

    arg = np.arange(16, dtype=dtype)
    expected = 4 * arg

    output = vm["main"](tvm.nd.array(arg), np.int64(4).astype(dtype)).numpy()

    np.testing.assert_equal(expected, output)


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_prim_value_scalar_computed_in_relax(exec_mode, dtype):
    """Relax may compute new scalar values and then pass them to TIR"""

    if "float" in dtype:
        pytest.xfail(
            reason=(
                "Computing float values in Relax not yet supported.  "
                "To add support, ComputePrimValue should use the returned PrimValue in later expressions, "
                "and should strip the PrimStructInfo::value to prevent VMShapeLower "
                "from checking non-integer arguments."
            )
        )

    target = "llvm"
    dev = tvm.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], dtype), scale: R.Prim(value="tir_scale")):
            tir_scale = T.var(dtype)
            new_scale = R.prim_value(tir_scale * 2)
            B = R.call_tir(Module.tir_func, [A, new_scale], out_sinfo=R.Tensor([16], dtype))
            return B

        @T.prim_func
        def tir_func(
            A: T.Buffer(16, dtype),
            scale: T.var(dtype),
            B: T.Buffer(16, dtype),
        ):
            _ = T.meta_var(dtype)
            for i in range(16):
                B[i] = A[i] * scale

    built = tvm.relax.build(Module, target=target, exec_mode=exec_mode)
    vm = tvm.relax.VirtualMachine(built, dev)

    arg = np.arange(16, dtype=dtype)
    expected = 8 * arg

    output = vm["main"](tvm.nd.array(arg), np.int64(4).astype(dtype)).numpy()

    np.testing.assert_equal(expected, output)


def test_omit_out_sinfo(exec_mode):
    """The out_sinfo may be inferred from the PrimFunc signature"""

    target = "llvm"
    dev = tvm.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "int32")):
            B = R.call_tir(Module.tir_func, [A])
            return B

        @T.prim_func
        def tir_func(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
            for i in range(16):
                B[i] = A[i] * 2

    built = tvm.relax.build(Module, target=target, exec_mode=exec_mode)
    vm = tvm.relax.VirtualMachine(built, dev)

    arg = np.arange(16, dtype="int32")
    expected = 2 * arg

    output = vm["main"](tvm.nd.array(arg)).numpy()

    np.testing.assert_equal(expected, output)


def test_omit_out_sinfo_with_tir_var_scalar(exec_mode):
    """The out_sinfo may be inferred when tir_vars are present.

    Because `tir_vars` are stored internally as a `R.Shape`, they only
    may only define int64 values.  As a result, when `out_sinfo` is
    omitted, the output should be inferred based on the PrimFunc
    parameters after `len(call.args)`, but before
    `len(call.attrs.tir_vars)`.

    """

    target = "llvm"
    dev = tvm.cpu()

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "int64"), _: R.Prim(value="scale")):
            scale = T.int64()
            B = R.call_tir(Module.tir_func, [A], tir_vars=[scale])
            return B

        @T.prim_func
        def tir_func(
            A: T.Buffer(16, "int64"),
            B: T.Buffer(16, "int64"),
            scale: T.int64,
        ):
            for i in range(16):
                B[i] = A[i] * scale

    built = tvm.relax.build(Module, target=target, exec_mode=exec_mode)
    vm = tvm.relax.VirtualMachine(built, dev)

    arg = np.arange(16, dtype="int64")
    expected = 4 * arg

    output = vm["main"](tvm.nd.array(arg), 4).numpy()

    np.testing.assert_equal(expected, output)


def test_inferred_out_sinfo_with_dynamic_shapes():
    """The out_sinfo may contain dynamic shapes"""

    # The dynamic shapes for the TIR function are specified externally
    # rather than as part of a `T.match_buffer` statement, due to a
    # limitation in the TVMScript parsing.  When parsing an IRModule,
    # function signatures are parsed without inspecting the body, to
    # be used for StructInfo inference when calling a subroutine.
    # However, while Relax dynamic shapes can be fully expressed in
    # the function signature (e.g. `R.Tensor(["dynamic_var"])`), TIR
    # dynamic shapes are expressed in `T.match_buffer` statements that
    # appear inside the function body.
    #
    # Resolving this limitation will require either (1) fully parsing
    # TIR functions prior to Relax functions, or (2) inspecting the
    # prelude of a TIR function for `T.match_buffer` as part of the
    # signature parsing.
    A_len = T.var("int64", "A_len")
    B_len = T.var("int64", "B_len")

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "int32"), B: R.Tensor([32], "int32")):
            C = R.call_tir(Module.concat, [A, B])
            return C

        @T.prim_func
        def concat(
            A: T.Buffer(A_len, "int32"),
            B: T.Buffer(B_len, "int32"),
            C: T.Buffer(A_len + B_len, "int32"),
        ):
            for i in range(A_len):
                C[i] = A[i]
            for i in range(B_len):
                C[A_len + i] = B[i]

    assert list(Module["main"].struct_info.ret.shape) == [48]


@pytest.mark.parametrize("provide_explicit_out_sinfo", [True, False])
def test_exception_raised_if_shape_inference_requires_output(provide_explicit_out_sinfo, capfd):
    """Shape inference may not be possible for all PrimFuncs

    For some PrimFuncs, the shape of the output tensor may be the only
    place where dynamic shape parameters are defined.  If this occurs,
    then the output shape cannot be inferred.

    If the `out_sinfo` argument is omitted when calling such a
    PrimFunc, then the attempted inference should fail.  The error
    message provided to the user should indicate that the problem can
    be avoided by provided the `out_sinfo` argument.

    """

    def define_function():
        if provide_explicit_out_sinfo:
            out_sinfo = R.Tensor([1, 8], "float32")
        else:
            out_sinfo = None

        @I.ir_module
        class Module:
            @R.function
            def main(A: R.Tensor((2, 4), "float32")):
                out = R.call_tir(
                    Module.reshape,
                    [A],
                    out_sinfo=out_sinfo,
                )
                return out

            @T.prim_func
            def reshape(
                A: T.Buffer([T.int64(2), T.int64(4)], "float32"),
                B: T.Buffer([T.var("int64", "n"), T.var("int64", "m")], "float32"),
            ):
                n = T.meta_var(B.shape[0])
                m = T.meta_var(B.shape[1])
                for i, j in T.grid(n, m):
                    index = i * m + j
                    B[i, j] = A[index // 4, index % 4]

    if provide_explicit_out_sinfo:
        define_function()
    else:
        with pytest.raises(TVMError):
            define_function()

        # The diagnostic render for TVMScript prints the original
        # exception to stderr, so verifying the error message cannot
        # be done with the `match` argument to `pytest.raises`.
        _stdout, stderr = capfd.readouterr()
        assert "Please update the `R.call_tir` call with an explicit `out_sinfo` argument" in stderr


if __name__ == "__main__":
    tvm.testing.main()
