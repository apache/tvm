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

"""TIRx script meta programming."""

from __future__ import annotations

import pytest

import tvm
import tvm.script
import tvm.testing
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T


def test_meta_class_constructor_rejects_unowned_resource():
    @T.meta_class
    class Bad:
        def __init__(self):
            tmp = T.alloc_buffer((1,), "int32", scope="local")

    with pytest.raises(ValueError):

        @T.prim_func
        def test():
            T.device_entry()
            bad = Bad()


def test_meta_class_multiple_instances_preserve_owned_resources():
    instances = []

    @T.meta_class
    class Holder:
        def __init__(self, external):
            self.external = external
            self.buf = T.alloc_buffer((2,), "int32", scope="local")
            self.scalar = T.local_scalar("int32")
            instances.append(self)

    @T.prim_func(private=True)
    def test():
        T.device_entry()
        external = T.alloc_buffer((2,), "int32", scope="local")
        first = Holder(external)
        second = Holder(external)
        T.evaluate(
            first.buf[0]
            + second.buf[1]
            + first.scalar
            + second.scalar
            + first.external[0]
            + second.external[1]
        )

    @T.prim_func(private=True)
    def expected():
        T.device_entry()
        external = T.alloc_local((2,), "int32")
        first_buf = T.alloc_local((2,), "int32")
        first_scalar: T.int32
        second_buf = T.alloc_local((2,), "int32")
        second_scalar: T.int32
        T.evaluate(
            first_buf[0] + second_buf[1] + first_scalar + second_scalar + external[0] + external[1]  # noqa: F821
        )

    assert len(instances) == 2
    assert instances[0].external.same_as(instances[1].external)
    assert_structural_equal(test, expected)


def from_source(code):
    return tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})


def test_macro():
    # fmt: off
    @T.inline
    def mul(x, c):
        T.evaluate(x * c)

    @T.prim_func(private=True)
    def test():
        T.device_entry()
        for x in range(10):

            @T.inline
            def add(c):
                T.evaluate(x + c)

            @T.inline
            def two_add_and_mul(c):
                add(c)
                add(c + c)
                mul(x, c)

            two_add_and_mul(1)
            two_add_and_mul(2)

    @T.prim_func(private=True)
    def expected():
        T.device_entry()
        for x in range(10):
            T.evaluate(x + 1)
            T.evaluate(x + 2)
            T.evaluate(x)
            T.evaluate(x + 2)
            T.evaluate(x + 4)
            T.evaluate(x * 2)
        # fmt: on
    assert_structural_equal(test, from_source(test.script()))
    assert_structural_equal(test, expected)


def test_macro_recursive():
    # fmt: off
    @T.prim_func(private=True)
    def test():
        T.device_entry()
        for x in T.serial(10):

            @T.inline
            def add(x, c):
                if T.constexpr(c > 0):
                    add(x, c - 1)
                T.evaluate(x)

            add(x, 5)

    @T.prim_func(private=True)
    def expected():
        T.device_entry()
        for x in range(10):
            T.evaluate(x)
            T.evaluate(x)
            T.evaluate(x)
            T.evaluate(x)
            T.evaluate(x)
            T.evaluate(x)
        # fmt: on
    assert_structural_equal(test, from_source(test.script()))
    assert_structural_equal(test, expected)


def test_list_comprehension():
    # fmt: off
    @T.prim_func(private=True)
    def test():
        T.device_entry()
        acc = T.alloc_local([10], "bool")
        regs = T.meta_var([acc[_] for _ in range(10)])
        T.evaluate(regs[0])
        T.evaluate(tvm.tirx.all(*regs))
        T.evaluate(tvm.tirx.all(*[acc[_] for _ in range(10)]))
        T.evaluate(tvm.tirx.all(*([acc[_] for _ in range(2, 4)] + [acc[_] for _ in range(6, 8)])))

    @T.prim_func(private=True)
    def expected():
        T.device_entry()
        acc = T.alloc_local((10,), "bool")
        T.evaluate(acc[0])
        T.evaluate(
            acc[0] and acc[1] and acc[2] and acc[3] and acc[4]
            and acc[5] and acc[6] and acc[7] and acc[8] and acc[9]
        )
        T.evaluate(
            acc[0] and acc[1] and acc[2] and acc[3] and acc[4]
            and acc[5] and acc[6] and acc[7] and acc[8] and acc[9]
        )
        T.evaluate(acc[2] and acc[3] and acc[6] and acc[7])
        # fmt: on

    assert_structural_equal(test, from_source(test.script()))
    assert_structural_equal(test, expected)


def test_shared_meta_var_alias():
    assert I.meta_var is T.meta_var

    @T.prim_func(private=True)
    def via_ir_namespace():
        value = I.meta_var(T.int32(1))
        T.evaluate(value)

    @T.prim_func(private=True)
    def via_tirx_alias():
        value = T.meta_var(T.int32(1))
        T.evaluate(value)

    assert_structural_equal(via_ir_namespace, via_tirx_alias)
    assert_structural_equal(via_ir_namespace, from_source(via_ir_namespace.script()))


def test_scalar_assign_in_macro():
    """Regression: the parser's scalar-assignment sugar (scalar = Expr) must
    work in macro context via self.attr.

    The parser narrowed ``except Exception: pass`` around the scalar-detection
    path. This test verifies that Expr assignment to a scalar attribute in
    a macro still goes through buffer_store correctly.

    The full integration regression for the TypeError fallthrough path
    (meta_var assigned to a scalar variable) is covered by
    test_hgemm::test_hgemm (tile_scheduler.m_idx pattern)."""

    # fmt: off
    class State:
        def __init__(self, counter):
            self.counter = counter

        @T.inline
        def add_one(self):
            # Expr assigned to scalar via self.attr → buffer_store succeeds
            self.counter = self.counter + T.int32(1)

    @T.prim_func(private=True)
    def test():
        T.device_entry()
        counter: T.int32
        state = T.meta_var(State(counter))  # noqa: F821
        state.add_one()
        T.evaluate(state.counter)

    @T.prim_func(private=True)
    def expected():
        T.device_entry()
        counter: T.int32
        counter = counter + 1  # noqa: F821
        T.evaluate(counter)
        # fmt: on

    assert_structural_equal(test, from_source(test.script()))
    assert_structural_equal(test, expected)


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


def _normalize(func):
    """Strip the global_symbol so function names do not affect structural equality."""
    return func.with_attr("global_symbol", "")


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
