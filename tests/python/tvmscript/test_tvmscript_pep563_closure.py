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
"""Test TVMScript with PEP 563 (from __future__ import annotations).

IMPORTANT: The `from __future__ import annotations` import below is the
test condition itself, because we need to test compatibility with it.
"""

from __future__ import annotations

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T


def _normalize(func):
    """Strip the global_symbol so function names do not affect structural equality."""
    return func.with_attr("global_symbol", "")


def test_prim_func_closure_shape():
    """Closure variable used in Buffer shape annotation."""

    def f(M=16):
        @T.prim_func
        def func(A: T.Buffer((M,), "float32")):
            T.evaluate(0)

        return func

    @T.prim_func
    def expected_16(A: T.Buffer((16,), "float32")):
        T.evaluate(0)

    @T.prim_func
    def expected_32(A: T.Buffer((32,), "float32")):
        T.evaluate(0)

    tvm.ir.assert_structural_equal(_normalize(f(16)), _normalize(expected_16))
    tvm.ir.assert_structural_equal(_normalize(f(32)), _normalize(expected_32))


def test_prim_func_closure_dtype():
    """Closure variable used as Buffer dtype."""

    def f(dtype="float32"):
        @T.prim_func
        def func(A: T.Buffer((16,), dtype)):
            T.evaluate(0)

        return func

    @T.prim_func
    def expected_f32(A: T.Buffer((16,), "float32")):
        T.evaluate(0)

    @T.prim_func
    def expected_f16(A: T.Buffer((16,), "float16")):
        T.evaluate(0)

    tvm.ir.assert_structural_equal(_normalize(f("float32")), _normalize(expected_f32))
    tvm.ir.assert_structural_equal(_normalize(f("float16")), _normalize(expected_f16))


def test_prim_func_nested_closure():
    """Variables from enclosing scope active on the call stack (grandparent frame fallback).

    With PEP 563, closure-only variables are missing from __closure__ unless they
    appear in the function body. The ChainMap fallback walks the live call stack,
    so this works when the enclosing frames are still active (outer calls middle
    which applies the decorator, keeping outer's frame alive on the stack).
    """

    def outer(M=16):
        def middle(N=8):
            @T.prim_func
            def func(A: T.Buffer((M, N), "float32")):
                T.evaluate(0)

            return func

        return middle()

    @T.prim_func
    def expected_16_8(A: T.Buffer((16, 8), "float32")):
        T.evaluate(0)

    @T.prim_func
    def expected_32_8(A: T.Buffer((32, 8), "float32")):
        T.evaluate(0)

    tvm.ir.assert_structural_equal(_normalize(outer(16)), _normalize(expected_16_8))
    tvm.ir.assert_structural_equal(_normalize(outer(32)), _normalize(expected_32_8))


def test_ir_module_closure():
    """Closure variable in @I.ir_module class method."""

    def f(M=16):
        @I.ir_module
        class Mod:
            @T.prim_func
            def main(A: T.Buffer((M,), "float32")):
                T.evaluate(0)

        return Mod

    @T.prim_func
    def expected_16(A: T.Buffer((16,), "float32")):
        T.evaluate(0)

    @T.prim_func
    def expected_32(A: T.Buffer((32,), "float32")):
        T.evaluate(0)

    tvm.ir.assert_structural_equal(_normalize(f(16)["main"]), _normalize(expected_16))
    tvm.ir.assert_structural_equal(_normalize(f(32)["main"]), _normalize(expected_32))


def test_mixed_closure_usage():
    """Closure var used in both annotation AND body -- regression check."""

    def f(M=16):
        @T.prim_func
        def func(A: T.Buffer((M,), "float32")):
            T.evaluate(M)

        return func

    @T.prim_func
    def expected_16(A: T.Buffer((16,), "float32")):
        T.evaluate(16)

    @T.prim_func
    def expected_32(A: T.Buffer((32,), "float32")):
        T.evaluate(32)

    tvm.ir.assert_structural_equal(_normalize(f(16)), _normalize(expected_16))
    tvm.ir.assert_structural_equal(_normalize(f(32)), _normalize(expected_32))


if __name__ == "__main__":
    tvm.testing.main()
