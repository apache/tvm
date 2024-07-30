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


@pytest.mark.xfail(reason="Known limitation in TVMScript.")
def test_inferred_out_sinfo_with_dynamic_shapes():
    """The out_sinfo may contain dynamic shapes

    Currently, this test fails due to limitations in the TVMScript
    parsing.  When parsing an IRModule, function signatures are parsed
    without inspecting the body, to be used for StructInfo inference
    when calling a subroutine.  However, while Relax dynamic shapes
    can be fully expressed in the function signature
    (e.g. `R.Tensor(["dynamic_var"])`), TIR dynamic shapes are
    expressed in `T.match_buffer` statements that appear inside the
    function body.

    Resolving this limitation will require either (1) fully parsing
    TIR functions prior to Relax functions, or (2) inspecting the
    prelude of a TIR function for `T.match_buffer` as part of the
    signature parsing.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "int32"), B: R.Tensor([32], "int32")):
            C = R.call_tir(Module.concat, [A, B])
            return C

        @T.prim_func
        def concat(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
            A_len = T.int64()
            B_len = T.int64()

            A = T.match_buffer(A_handle, A_len, "int32")
            B = T.match_buffer(B_handle, B_len, "int32")
            C = T.match_buffer(C_handle, A_len + B_len, "int32")
            for i in range(A_len):
                C[i] = A[i]
            for i in range(B_len):
                C[A_len + i] = B[i]

    assert Module["main"].struct_info.ret.shape == [48]


if __name__ == "__main__":
    tvm.testing.main()
