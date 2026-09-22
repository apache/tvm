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
"""Explicit host specialization constructs and executes canonical functions."""

from __future__ import annotations

import pytest

import tvm
from tvm.script import parser

parser._initialize()
T = parser.T
I = parser.I  # noqa: E741


def test_optional_handle_present_and_absent():
    @T.jit(private=True)
    def kernel(a: T.Optional(T.handle), out_h: T.handle):
        out = T.match_buffer(out_h, (1,), "int32")
        if I.constexpr(a is not None):
            A = T.match_buffer(a, (1,), "int32")
            out[0] = A[0]
        else:
            out[0] = 0

    present = kernel.specialize()
    absent = kernel.specialize(a=None)
    expected_present = parser.parse("""
@T.prim_func(private=True)
def expected(a: T.handle, out_h: T.handle):
    A = T.match_buffer(a, (1,), "int32")
    out = T.match_buffer(out_h, (1,), "int32")
    out[0] = A[0]
""")
    expected_absent = parser.parse("""
@T.prim_func(private=True)
def expected(out_h: T.handle):
    out = T.match_buffer(out_h, (1,), "int32")
    out[0] = 0
""")
    tvm.ir.assert_structural_equal(present, expected_present, map_free_vars=True)
    tvm.ir.assert_structural_equal(absent, expected_absent, map_free_vars=True)
    assert len(present.params) == 2
    assert len(absent.params) == 1
    assert kernel.specialize() is present
    assert kernel.specialize(a=None) is absent


def test_constexpr_defaults_dependent_shapes_and_parent_scope():
    @T.jit(private=True)
    def kernel(
        a: T.Optional(T.handle),
        out: T.Buffer((N,), "int32"),  # noqa: F821
        *,
        N: T.constexpr = 4,
    ):
        if T.constexpr(a is not None):
            A = T.match_buffer(a, (N,), "int32")
        if I.constexpr(a is not None):
            out[0] = A[0]
        else:
            out[0] = N

    present = kernel.specialize()
    absent = kernel.specialize(a=None, N=8)
    assert len(present.params) == 2
    assert len(absent.params) == 1
    assert int(absent.params[0].ty.shape[0]) == 8
    assert kernel.specialize(N=4) is present


def test_jit_argument_validation():
    @T.jit
    def kernel(a: T.Optional(T.handle), *, N: T.constexpr):
        T.evaluate(N)

    with pytest.raises(TypeError, match="missing constexpr"):
        kernel.specialize()
    with pytest.raises(TypeError, match="unexpected"):
        kernel.specialize(N=1, unknown=2)
    with pytest.raises(TypeError, match="only accept None"):
        kernel.specialize(N=1, a=1)
    with pytest.raises(TypeError, match="hashable"):
        kernel.specialize(N=[])


def test_absent_annotation_is_not_evaluated():
    calls = []

    def annotation():
        calls.append(True)
        return T.handle

    @T.jit(private=True)
    def kernel(a: T.Optional(annotation()), *, N: T.constexpr = 1):
        T.evaluate(N)

    calls.clear()  # JIT annotation classification is distinct from construction.
    result = kernel.specialize(a=None)
    assert len(result.params) == 0
    assert calls == []


def test_specialization_context_restored_after_failure():
    @T.jit(private=True)
    def broken(*, N: T.constexpr):
        if I.constexpr(N == 1):
            missing_call()  # noqa: F821
        T.evaluate(N)

    with pytest.raises(tvm.error.DiagnosticError):
        broken.specialize(N=1)
    assert len(broken.specialize(N=2).params) == 0
    result = parser.parse("@T.prim_func\ndef normal(N: T.int32):\n    T.evaluate(N)\n")
    assert len(result.params) == 1


def test_optional_specializations_execute_on_llvm():
    import numpy as np

    @T.jit
    def kernel(a: T.Optional(T.handle), out_h: T.handle):
        out = T.match_buffer(out_h, (1,), "int32")
        if I.constexpr(a is not None):
            A = T.match_buffer(a, (1,), "int32")
            out[0] = A[0]
        else:
            out[0] = 0

    present = tvm.compile(kernel.specialize(), target="llvm", tir_pipeline="tirx")
    absent = tvm.compile(kernel.specialize(a=None), target="llvm", tir_pipeline="tirx")
    values = tvm.runtime.tensor(np.array([17], dtype="int32"))
    output = tvm.runtime.tensor(np.array([-1], dtype="int32"))
    present(values, output)
    np.testing.assert_array_equal(output.numpy(), [17])
    absent(output)
    np.testing.assert_array_equal(output.numpy(), [0])


def test_constexpr_cache_distinguishes_bool_and_int():
    @T.jit(private=True)
    def kernel(*, value: T.constexpr):
        T.evaluate(value)

    integer = kernel.specialize(value=1)
    boolean = kernel.specialize(value=True)
    assert integer is not boolean
    assert str(integer.body.value.ty.dtype) == "int32"
    assert str(boolean.body.value.ty.dtype) == "bool"


def test_captured_constexpr_bindings_match_specialized_function():
    source = """
@T.jit(private=True)
def scaled_copy(
    A: T.Buffer((N,), "int32"),
    B: T.Buffer((N,), "int32"),
    *,
    N: T.constexpr,
    SCALE: T.constexpr,
):
    for i in range(N):
        B[i] = A[i] * SCALE
"""
    expected = parser.parse("""
@T.prim_func(private=True)
def scaled_copy(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "int32")):
    for i in range(16):
        B[i] = A[i] * 3
""")
    captures = {"N": 16, "SCALE": 3}
    result = parser.parse(source, extra_vars=captures, absent_params={})
    tvm.ir.assert_structural_equal(expected, result)
    assert len(result.params) == 2


@pytest.mark.parametrize("absent", [False, True])
def test_explicit_absence_uses_parent_builder_scope(absent):
    source = """
@T.jit(private=True)
def kernel(a: T.Optional(T.handle), out_h: T.handle):
    out = T.match_buffer(out_h, (1,), "int32")
    if I.constexpr(a is not None):
        A = T.match_buffer(a, (1,), "int32")
        out[0] = A[0]
    else:
        out[0] = 0
"""
    expected_source = (
        """
@T.prim_func(private=True)
def expected(out_h: T.handle):
    out = T.match_buffer(out_h, (1,), "int32")
    out[0] = 0
"""
        if absent
        else """
@T.prim_func(private=True)
def expected(a: T.handle, out_h: T.handle):
    out = T.match_buffer(out_h, (1,), "int32")
    A = T.match_buffer(a, (1,), "int32")
    out[0] = A[0]
"""
    )
    options = {"absent_params": {"a": None} if absent else {}}
    result = parser.parse(source, **options)
    tvm.ir.assert_structural_equal(parser.parse(expected_source), result)
    assert len(result.params) == (1 if absent else 2)


def test_capture_transport_does_not_specialize_ordinary_captured_names():
    result = parser.parse(
        "@T.prim_func\ndef normal(N: T.int32):\n    T.evaluate(N)\n", extra_vars={"N": 16}
    )
    assert len(result.params) == 1
    assert result.body.value.same_as(result.params[0])


def test_missing_constexpr_binding_has_explicit_diagnostic():
    with pytest.raises(tvm.error.DiagnosticError, match="requires a specialization binding"):
        parser.parse("@T.jit\ndef kernel(N: T.constexpr):\n    T.evaluate(N)\n")


def test_explicit_absence_is_lazy_and_restored_after_failure():
    calls = []

    def annotation():
        calls.append(True)
        return T.handle

    source = """
@T.jit(private=True)
def kernel(a: T.Optional(annotation())):
    if I.constexpr(a is None):
        missing_call()
    T.evaluate(1)
"""
    with pytest.raises(tvm.error.DiagnosticError, match="missing_call"):
        parser.parse(source, extra_vars={"annotation": annotation}, absent_params={"a": None})
    assert calls == []
    result = parser.parse("@T.prim_func\ndef kernel(a: T.handle):\n    T.evaluate(1)\n")
    assert len(result.params) == 1


def test_explicit_absence_rejects_non_absent_values():
    with pytest.raises(tvm.error.DiagnosticError, match="absent_params values must be None"):
        parser.parse(
            "@T.prim_func\ndef kernel(a: T.handle):\n    T.evaluate(1)\n", absent_params={"a": 1}
        )


def test_reentrant_same_name_parse_cannot_inherit_jit_bindings():
    nested_counts = []

    def construct_nested():
        nested = parser.parse("@T.prim_func\ndef kernel(N: T.int32):\n    T.evaluate(N)\n")
        nested_counts.append(len(nested.params))

    @T.jit(private=True)
    def kernel(*, N: T.constexpr):
        construct_nested()
        T.evaluate(N)

    specialized = kernel.specialize(N=4)
    assert len(specialized.params) == 0
    assert nested_counts == [1]


def test_reentrant_same_name_parse_cannot_inherit_explicit_absence():
    nested_counts = []

    def construct_nested():
        nested = parser.parse("@T.prim_func\ndef kernel(a: T.handle):\n    T.evaluate(1)\n")
        nested_counts.append(len(nested.params))

    source = """
@T.jit(private=True)
def kernel(a: T.Optional(T.handle)):
    construct_nested()
    T.evaluate(1)
"""
    result = parser.parse(
        source, extra_vars={"construct_nested": construct_nested}, absent_params={"a": None}
    )
    assert len(result.params) == 0
    assert nested_counts == [1]
