#!/usr/bin/env python3

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
# pylint: disable=missing-function-docstring,missing-module-docstring

import pytest
import numpy as np

import tvm
import tvm.testing

from tvm.script import tir as T, ir as I


@tvm.testing.parametrize_targets("llvm")
def test_call_noop(target, dev):
    """TIR functions on the CPU may call other functions

    The simplest test case, where the subroutine is a no-op.
    """

    @I.ir_module
    class module:
        @T.prim_func
        def subroutine():
            T.evaluate(0)

        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main"})
            module.subroutine()
            A[0] = 42.0

    built = tvm.build(module, target=target)

    arr = tvm.nd.empty([1], dtype="float32", device=dev)
    built(arr)

    assert arr.numpy()[0] == 42.0


@tvm.testing.parametrize_targets("llvm")
def test_call_noop_defined_below(target, dev):
    """Calling a subroutine does not depend on the definition order

    All GlobalVar instances are in-scope for subroutine calls.
    """

    @I.ir_module
    class module:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main"})
            module.subroutine()
            A[0] = 42.0

        @T.prim_func
        def subroutine():
            T.evaluate(0)

    built = tvm.build(module, target=target)

    arr = tvm.nd.empty([1], dtype="float32", device=dev)
    built(arr)

    assert arr.numpy()[0] == 42.0


@tvm.testing.parametrize_targets("llvm")
def test_subroutine_call_with_pointer_param(target, dev):
    """TIR functions on the CPU may call other functions

    Buffers may be exposed to subroutines through data pointers.
    """

    @I.ir_module
    class module:
        @T.prim_func
        def main(A: T.Buffer(2, "float32")):
            T.func_attr({"global_symbol": "main"})
            module.subroutine(A.data)
            module.subroutine(T.address_of(A[1]))

        @T.prim_func
        def subroutine(A_data: T.handle("float32")):
            A = T.decl_buffer(shape=[1], dtype="float32", data=A_data)
            A[0] = 42.0

    built = tvm.build(module, target=target)

    arr = tvm.nd.empty([2], dtype="float32", device=dev)
    built(arr)

    assert arr.numpy()[0] == 42.0
    assert arr.numpy()[1] == 42.0


@pytest.mark.xfail(reason="Depends on LLVM version")
@tvm.testing.parametrize_targets("llvm")
def test_failed_subroutine_call_for_incorrect_type(target, dev):
    """Calls into a subroutine must have correct argument types

    This currently relies on the `llvm::verifyModule` function during
    codegen.  In the future, this should be moved to a dedicated check
    of TIR validity.
    """

    @I.ir_module
    class module:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main"})
            module.subroutine(A.data)

        @T.prim_func
        def subroutine(A_data: T.handle("int32")):
            A = T.decl_buffer(shape=[1], dtype="int32", data=A_data)
            A[0] = -1

    lowered = tvm.lower(module)
    with pytest.raises(tvm.TVMError):
        tvm.build(lowered)


@tvm.testing.parametrize_targets("llvm")
def test_subroutine_call_with_scalar_param(target, dev):
    """Subroutines may also accept scalar parameters"""

    @I.ir_module
    class module:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main"})
            module.subroutine(A.data, 42.0)

        @T.prim_func
        def subroutine(A_data: T.handle("float32"), val: T.float32):
            A = T.decl_buffer([1], "float32", data=A_data)
            A[0] = 2 * val

    built = tvm.build(module, target=target)

    arr = tvm.nd.empty([1], dtype="float32", device=dev)
    built(arr)

    assert arr.numpy()[0] == 84.0


@tvm.testing.parametrize_targets("llvm")
def test_internal_subroutine_is_not_exposed_externally(target, dev):
    """An internal subroutine may not be called externally

    An internal subroutine is any subroutine without a "global_symbol"
    attribute.  These are not exposed in the runtime::Module and do
    not have an externally linkable symbol.
    """

    @I.ir_module
    class module:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main"})
            module.subroutine(A.data, 42.0)

        @T.prim_func
        def subroutine(A_data: T.handle("float32"), val: T.float32):
            A = T.decl_buffer([1], "float32", data=A_data)
            A[0] = 2 * val

    built = tvm.build(module, target=target)
    with pytest.raises(AttributeError):
        built["subroutine"]


@tvm.testing.parametrize_targets("llvm")
def test_call_to_externally_visible_subroutine(target, dev):
    """Subroutines may be exposed externally.

    A subroutine may be exposed externally.  Externally-exposed
    subroutines may be called by an external API, or may be called by
    other functions in the same IRModule.

    The current implementation lowers internal subroutine calls to
    `T.tvm_call_cpacked`.  This avoids the overhead of the global
    registry lookup used by `T.tvm_call_packed`, but still requires
    the overhead of packing/unpacking the `PackedFunc` interface, and
    is limited to callers whose target supports the `PackedFunc`
    interface.
    """

    @I.ir_module
    class module:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main"})
            module.subroutine(A.data, 42.0)

        @T.prim_func
        def subroutine(A_data: T.handle("float32"), val: T.float32):
            T.func_attr({"global_symbol": "subroutine"})
            A = T.Buffer([1], "float32", data=A_data)
            A[0] = 2 * val

    built = tvm.build(module, target=target)

    arr = tvm.nd.empty([1], dtype="float32", device=dev)
    built["main"](arr)
    assert arr.numpy()[0] == 84.0

    arr = np.zeros(shape=[1], dtype="float32")
    built["subroutine"](arr.ctypes._data, 100.0)
    assert arr[0] == 200.0


is_external_subroutine = tvm.testing.parameter(by_dict={"external": True, "internal": False})


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_call_to_device_subroutine(target, dev, is_external_subroutine):
    """Subroutines may be exposed externally.

    This feature is currently limited to host-side subroutine calls of
    externally-exposed subroutines.
    """
    is_gpu = "gpu" in tvm.target.Target(target).keys

    if is_gpu and not is_external_subroutine:
        pytest.xfail(reason="Not yet implemented.")

    if is_external_subroutine:
        func_attr = {"global_symbol": "subroutine"}
    else:
        func_attr = {}

    @I.ir_module
    class module:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main"})
            module.subroutine(A.data, 42.0)

        @T.prim_func
        def subroutine(A_data: T.handle("float32"), val: T.float32):
            T.func_attr(func_attr)
            A = T.Buffer([1], "float32", data=A_data)
            iterator = T.meta_var(
                T.thread_binding(0, 1, thread="threadIdx.x") if is_gpu else range(1)
            )
            for i in iterator:
                A[0] = 2 * val

    built = tvm.build(module, target=target)

    arr = tvm.nd.empty([1], dtype="float32", device=dev)
    built["main"](arr)
    assert arr.numpy()[0] == 84.0


if __name__ == "__main__":
    tvm.testing.main()
