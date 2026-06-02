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
"""Tests for T.inline / Tx.inline with Python LEGB scoping semantics."""

from tvm.ir import assert_structural_equal
from tvm.script import tirx as T
from tvm.script import tirx as Tx

# Module-level constant for testing global visibility
MODULE_CONST = 42


def test_local_shadows_enclosing():
    """A local parameter in the inline shadows a variable from the enclosing scope."""

    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int32")) -> None:
        T.int32(10)

        @T.inline
        def write(x):
            # x here is the parameter, not the enclosing x=10
            A[0] = x

        write(T.int32(20))

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "int32")) -> None:
        T.int32(10)
        A[0] = T.int32(20)

    assert_structural_equal(func, expected)


def test_enclosing_variable_capture():
    """Inline captures a variable from its enclosing scope (not a parameter)."""
    val = 64

    @T.inline
    def write_val(A):
        A[0] = val

    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int32")) -> None:
        write_val(A)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "int32")) -> None:
        A[0] = 64

    assert_structural_equal(func, expected)


def test_nested_inline():
    """Inner inline can call outer inline (inline-in-inline)."""

    @T.inline
    def add_one(A):
        A[0] = A[0] + 1

    @T.inline
    def add_two(A):
        add_one(A)
        add_one(A)

    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int32")) -> None:
        add_two(A)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "int32")) -> None:
        A[0] = A[0] + 1
        A[0] = A[0] + 1

    assert_structural_equal(func, expected)


def test_module_globals_visible():
    """Inline can see module-level globals."""

    @T.inline
    def write_const(A):
        A[0] = MODULE_CONST

    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int32")) -> None:
        write_const(A)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "int32")) -> None:
        A[0] = 42

    assert_structural_equal(func, expected)


def test_shadowing_in_inner_scope():
    """An inline defined inside a for-loop captures the loop variable."""

    @T.prim_func(private=True)
    def func(A: T.Buffer((10,), "int32")) -> None:
        for i in T.serial(10):

            @T.inline
            def write_i(A):
                A[i] = i

            write_i(A)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((10,), "int32")) -> None:
        for i in range(10):
            A[i] = i

    assert_structural_equal(func, expected)


def test_lexical_not_dynamic():
    """An inline defined outside prim_func does NOT see the caller's locals.
    Specifically, x_value captured at definition time (128) is used,
    not the loop variable x_value from the caller."""
    x_value = 128

    @T.inline
    def static_capture(A, B):
        B[()] = A[x_value]

    @T.prim_func(private=True)
    def func(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in T.serial(10):
            static_capture(A, B)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in range(10):
            B[()] = A[128]

    assert_structural_equal(func, expected)


def test_callback_pattern():
    """Inline passed as an argument to another inline."""

    @T.inline
    def apply_fn(fn, A):
        fn(A)

    @T.inline
    def inc(A):
        A[0] = A[0] + 1

    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int32")) -> None:
        apply_fn(inc, A)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "int32")) -> None:
        A[0] = A[0] + 1

    assert_structural_equal(func, expected)


def test_sibling_calls():
    """Two independent inlines called in sequence."""

    @T.inline
    def write_a(A):
        A[0] = 1

    @T.inline
    def write_b(A):
        A[1] = 2

    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int32")) -> None:
        write_a(A)
        write_b(A)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "int32")) -> None:
        A[0] = 1
        A[1] = 2

    assert_structural_equal(func, expected)


def test_recursive_inline():
    """Recursive inline (defined inside prim_func)."""

    # fmt: off
    @Tx.prim_func(private=True)
    def func():
        Tx.device_entry()
        for x in Tx.serial(10):

            @Tx.inline
            def add(x, c):
                if c > 0:
                    add(x, c - 1)
                Tx.evaluate(x)

            add(x, 3)

    @Tx.prim_func(private=True)
    def expected():
        Tx.device_entry()
        for x in range(10):
            Tx.evaluate(x)
            Tx.evaluate(x)
            Tx.evaluate(x)
            Tx.evaluate(x)
        # fmt: on

    assert_structural_equal(func, expected)


def test_late_binding():
    """Variable defined after inline but before call (inside prim_func)."""

    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int32")) -> None:
        @T.inline
        def write(A):
            A[0] = val

        val = T.int32(99)
        write(A)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "int32")) -> None:
        val = T.int32(99)
        A[0] = val

    assert_structural_equal(func, expected)


if __name__ == "__main__":
    test_local_shadows_enclosing()
    test_enclosing_variable_capture()
    test_nested_inline()
    test_module_globals_visible()
    test_shadowing_in_inner_scope()
    test_lexical_not_dynamic()
    test_callback_pattern()
    test_sibling_calls()
    test_recursive_inline()
    test_late_binding()
    print("All tests passed!")
