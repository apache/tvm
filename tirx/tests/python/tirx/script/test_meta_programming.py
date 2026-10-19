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

import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_meta_class_constructor_rejects_unowned_resource():
    @T.meta_class
    class Bad:
        def __init__(self):
            allocation = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
            tmp = tvm_tirx.ir.Var("tmp", allocation.ty)
            T.emit_(T.Bind(tmp, allocation))

    with pytest.raises(ValueError):

        @T.function
        def test():
            T.device_entry()
            bad = Bad()


def test_meta_class_multiple_instances_preserve_owned_resources():
    instances = []

    @T.meta_class
    class Holder:
        def __init__(self, external):
            self.external = external
            allocation = T.alloc_tensor(ty_args=[T.Tensor((2,), "int32", scope="local")])
            self.buf = tvm_tirx.ir.Var("buf", allocation.ty)
            T.emit_(T.Bind(self.buf, allocation))
            self.scalar = T.local_scalar("int32")
            instances.append(self)

    @T.function(private=True)
    def test():
        T.device_entry()
        external = T.alloc_tensor(ty_args=[T.Tensor((2,), "int32", scope="local")])
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

    @T.function(private=True)
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
    return tvm_tirx.script.from_source(
        code, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
    )


def test_macro():
    # fmt: off
    @T.inline
    def mul(x, c):
        T.evaluate(x * c)

    @T.function(private=True)
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

    @T.function(private=True)
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
    @T.function(private=True)
    def test():
        T.device_entry()
        for x in T.serial(10):

            @T.inline
            def add(x, c):
                if T.constexpr(c > 0):
                    add(x, c - 1)
                T.evaluate(x)

            add(x, 5)

    @T.function(private=True)
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
    @T.function(private=True)
    def test():
        T.device_entry()
        acc = T.alloc_local([10], "bool")
        regs = T.meta_var([acc[_] for _ in range(10)])
        T.evaluate(regs[0])
        T.evaluate(tvm_tirx.tirx.all(*regs))
        T.evaluate(tvm_tirx.tirx.all(*[acc[_] for _ in range(10)]))
        T.evaluate(
            tvm_tirx.tirx.all(*([acc[_] for _ in range(2, 4)] + [acc[_] for _ in range(6, 8)]))
        )

    @T.function(private=True)
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

    @T.function(private=True)
    def via_ir_namespace():
        value = I.meta_var(T.int32(1))
        T.evaluate(value)

    @T.function(private=True)
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
    a macro still goes through tensor_store correctly.

    The full integration regression for the TypeError fallthrough path
    (meta_var assigned to a scalar variable) is covered by
    test_hgemm::test_hgemm (tile_scheduler.m_idx pattern)."""

    # fmt: off
    class State:
        def __init__(self, counter):
            self.counter = counter

        @T.inline
        def add_one(self):
            # Expr assigned to scalar via self.attr → tensor_store succeeds
            self.counter = self.counter + T.int32(1)

    @T.function(private=True)
    def test():
        T.device_entry()
        counter: T.int32
        state = T.meta_var(State(counter))  # noqa: F821
        state.add_one()
        T.evaluate(state.counter)

    @T.function(private=True)
    def expected():
        T.device_entry()
        counter: T.int32
        counter = counter + 1  # noqa: F821
        T.evaluate(counter)
        # fmt: on

    assert_structural_equal(test, from_source(test.script()))
    assert_structural_equal(test, expected)


def test_function_closure_shape():
    """Closure variable used in Tensor shape annotation."""

    def f(M=16):
        @T.function
        def func(A: T.Tensor((M,), "float32")):
            T.evaluate(0)

        return func

    @T.function
    def expected_16(A: T.Tensor((16,), "float32")):
        T.evaluate(0)

    @T.function
    def expected_32(A: T.Tensor((32,), "float32")):
        T.evaluate(0)

    tvm_tirx.ir.assert_structural_equal(_normalize(f(16)), _normalize(expected_16))
    tvm_tirx.ir.assert_structural_equal(_normalize(f(32)), _normalize(expected_32))


def _normalize(func):
    """Strip the global_symbol so function names do not affect structural equality."""
    return func.with_attr("global_symbol", "")


def test_function_closure_dtype():
    """Closure variable used as Tensor dtype."""

    def f(dtype="float32"):
        @T.function
        def func(A: T.Tensor((16,), dtype)):
            T.evaluate(0)

        return func

    @T.function
    def expected_f32(A: T.Tensor((16,), "float32")):
        T.evaluate(0)

    @T.function
    def expected_f16(A: T.Tensor((16,), "float16")):
        T.evaluate(0)

    tvm_tirx.ir.assert_structural_equal(_normalize(f("float32")), _normalize(expected_f32))
    tvm_tirx.ir.assert_structural_equal(_normalize(f("float16")), _normalize(expected_f16))


def test_function_nested_closure():
    """Variables from enclosing scope active on the call stack (grandparent frame fallback).

    With PEP 563, closure-only variables are missing from __closure__ unless they
    appear in the function body. The ChainMap fallback walks the live call stack,
    so this works when the enclosing frames are still active (outer calls middle
    which applies the decorator, keeping outer's frame alive on the stack).
    """

    def outer(M=16):
        def middle(N=8):
            @T.function
            def func(A: T.Tensor((M, N), "float32")):
                T.evaluate(0)

            return func

        return middle()

    @T.function
    def expected_16_8(A: T.Tensor((16, 8), "float32")):
        T.evaluate(0)

    @T.function
    def expected_32_8(A: T.Tensor((32, 8), "float32")):
        T.evaluate(0)

    tvm_tirx.ir.assert_structural_equal(_normalize(outer(16)), _normalize(expected_16_8))
    tvm_tirx.ir.assert_structural_equal(_normalize(outer(32)), _normalize(expected_32_8))


def test_ir_module_closure():
    """Closure variable in @T.ir_module class method."""

    def f(M=16):
        @I.ir_module
        class Mod:
            @T.function
            def main(A: T.Tensor((M,), "float32")):
                T.evaluate(0)

        return Mod

    @T.function
    def expected_16(A: T.Tensor((16,), "float32")):
        T.evaluate(0)

    @T.function
    def expected_32(A: T.Tensor((32,), "float32")):
        T.evaluate(0)

    tvm_tirx.ir.assert_structural_equal(_normalize(f(16)["main"]), _normalize(expected_16))
    tvm_tirx.ir.assert_structural_equal(_normalize(f(32)["main"]), _normalize(expected_32))


def test_mixed_closure_usage():
    """Closure var used in both annotation AND body -- regression check."""

    def f(M=16):
        @T.function
        def func(A: T.Tensor((M,), "float32")):
            T.evaluate(M)

        return func

    @T.function
    def expected_16(A: T.Tensor((16,), "float32")):
        T.evaluate(16)

    @T.function
    def expected_32(A: T.Tensor((32,), "float32")):
        T.evaluate(32)

    tvm_tirx.ir.assert_structural_equal(_normalize(f(16)), _normalize(expected_16))
    tvm_tirx.ir.assert_structural_equal(_normalize(f(32)), _normalize(expected_32))


def test_tir_macro_decorator_signature():
    @T.function(private=True)
    def evaluate0():
        T.evaluate(0)

    # Ok, no parentheses
    @T.inline
    def func1():
        T.evaluate(0)

    @T.function(private=True)
    def use1():
        func1()

    tvm_tirx.ir.assert_structural_equal(use1, evaluate0)

    # Ok, empty parentheses
    @T.inline()
    def func2():
        T.evaluate(0)

    @T.function(private=True)
    def use2():
        func2()

    tvm_tirx.ir.assert_structural_equal(use1, evaluate0)

    with pytest.raises(ValueError):
        # Wrong: non-keyword argument
        @T.inline(True)
        def func3():
            T.evaluate()


def test_tir_macro_hygienic():
    x_value = 128

    @T.inline
    def static_capture(A, B):
        B[()] = A[x_value]

    @T.function(private=True)
    def use_hygienic(A: T.Tensor((1024,), "int32"), B: T.Tensor((), "int32")) -> None:
        for x_value in T.serial(10):
            static_capture(A, B)

    @T.function(private=True)
    def expected_hygienic(A: T.Tensor((1024,), "int32"), B: T.Tensor((), "int32")) -> None:
        for x_value in range(10):
            B[()] = A[128]

    tvm_tirx.ir.assert_structural_equal(use_hygienic, expected_hygienic)


def test_tir_inline_late_binding():
    """Inline defined inside function uses LEGB late binding:
    it sees the current value of variables from its enclosing scope at call time."""

    @T.function(private=True)
    def use_late_binding(A: T.Tensor((1024,), "int32"), B: T.Tensor((), "int32")) -> None:
        for x_value in T.serial(10):

            @T.inline
            def capture(A, B):
                B[()] = A[x_value]

            capture(A, B)

    @T.function(private=True)
    def expected(A: T.Tensor((1024,), "int32"), B: T.Tensor((), "int32")) -> None:
        for x_value in range(10):
            B[()] = A[x_value]

    tvm_tirx.ir.assert_structural_equal(use_late_binding, expected)


def test_tir_starred_expression():
    dims = (128, 128)

    @T.function(private=True)
    def starred(A: T.Tensor([128, *dims], "int32")) -> None:
        for i, j, k in T.grid(128, *dims):
            A[i, j, k] = T.int32(1)

    @T.function(private=True)
    def non_starred(A: T.Tensor([128, 128, 128], "int32")) -> None:
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm_tirx.ir.assert_structural_equal(starred, non_starred)


def test_tir_dynamic_for_loop():
    dims = (128, 128)

    @T.function(private=True)
    def starred(A: T.Tensor([128, *dims], "int32")) -> None:
        for (*iters,) in T.grid(*A.shape):
            A[iters] = T.int32(1)

    @T.function(private=True)
    def non_starred(A: T.Tensor([128, 128, 128], "int32")) -> None:
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm_tirx.ir.assert_structural_equal(starred, non_starred)


def test_tir_builtin_expression():
    dims = (128, 128)

    @T.function(private=True)
    def with_builtin(A: T.Tensor([len(dims), *dims], "int32")) -> None:
        for i, j, k in T.grid(*A.shape):
            A[i, j, k] = T.int32(1 + len(A.shape))

    @T.function(private=True)
    def evaluated(A: T.Tensor((2, 128, 128), "int32")):
        for i, j, k in T.grid(2, 128, 128):
            A[i, j, k] = 4

    tvm_tirx.ir.assert_structural_equal(with_builtin, evaluated)


def test_deterministic_branch():
    """Test deterministic branch"""

    def create_func(predicate: bool):
        @T.function(private=True)
        def func() -> None:
            if T.constexpr(predicate):
                T.evaluate(0)
            else:
                T.evaluate(1)

        return func

    def create_expected(value):
        @T.function(private=True)
        def expected() -> None:
            T.evaluate(value)

        return expected

    tvm_tirx.ir.assert_structural_equal(create_func(True), create_expected(0))
    tvm_tirx.ir.assert_structural_equal(create_func(False), create_expected(1))


def test_var_capturing_order():
    b = 2

    @T.function
    def test_case():
        k: T.let[T.int32] = b

    @T.function
    def func_ref():
        k: T.let[T.int32] = 2
        T.evaluate(0)

    tvm_tirx.ir.assert_structural_equal(
        test_case.with_attr("global_symbol", "main"), func_ref.with_attr("global_symbol", "main")
    )


def test_bind_with_constant():
    @T.function
    def constant_binds():
        x = T.meta_var(1)
        y = T.meta_var(42.0)
        T.evaluate(T.cast(x, ty="float32") + y)

    @T.function
    def constant_binds_wrapped():
        x = T.meta_var(T.int32(1))
        y = T.meta_var(T.float32(42.0))
        T.evaluate(T.cast(x, ty="float32") + y)

    tvm_tirx.ir.assert_structural_equal(
        _normalize(constant_binds), _normalize(constant_binds_wrapped)
    )
