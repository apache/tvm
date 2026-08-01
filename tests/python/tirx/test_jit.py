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
# ruff: noqa: F821
"""Tests for ``@T.jit`` compile-time specialization."""

from __future__ import annotations

import typing

import pytest

import tvm
from tvm.ir import assert_structural_equal
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


def test_int_constexpr_specializes_loop_bound():
    @T.jit(private=True)
    def add(
        A: T.Buffer((N,), "int32"),
        B: T.Buffer((N,), "int32"),
        C: T.Buffer((N,), "int32"),
        *,
        N: T.constexpr,
    ):
        for i in range(N):
            C[i] = A[i] + B[i]

    @T.prim_func(private=True)
    def expected(
        A: T.Buffer((128,), "int32"),
        B: T.Buffer((128,), "int32"),
        C: T.Buffer((128,), "int32"),
    ):
        for i in range(128):
            C[i] = A[i] + B[i]

    assert_structural_equal(add.specialize(N=128), expected, map_free_vars=True)


def test_constexpr_in_2d_buffer_shape():
    @T.jit(private=True)
    def matadd(
        A: T.Buffer((M, K), "int32"),
        B: T.Buffer((M, K), "int32"),
        C: T.Buffer((M, K), "int32"),
        *,
        M: T.constexpr,
        K: T.constexpr,
    ):
        for m in range(M):
            for k in range(K):
                C[m, k] = A[m, k] + B[m, k]

    @T.prim_func(private=True)
    def expected(
        A: T.Buffer((4, 8), "int32"),
        B: T.Buffer((4, 8), "int32"),
        C: T.Buffer((4, 8), "int32"),
    ):
        for m in range(4):
            for k in range(8):
                C[m, k] = A[m, k] + B[m, k]

    assert_structural_equal(matadd.specialize(M=4, K=8), expected, map_free_vars=True)


def test_constexpr_in_body_expression():
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

    @T.prim_func(private=True)
    def expected(
        A: T.Buffer((16,), "int32"),
        B: T.Buffer((16,), "int32"),
    ):
        for i in range(16):
            B[i] = A[i] * 3

    assert_structural_equal(scaled_copy.specialize(N=16, SCALE=3), expected, map_free_vars=True)


def test_specialize_cache_returns_same_instance():
    @T.jit(private=True)
    def k(
        A: T.Buffer((N,), "int32"),
        *,
        N: T.constexpr,
    ):
        for i in range(N):
            A[i] = 0

    a = k.specialize(N=8)
    b = k.specialize(N=8)
    assert a is b


def test_specialize_different_args_produce_different_funcs():
    @T.jit(private=True)
    def k(
        A: T.Buffer((N,), "int32"),
        *,
        N: T.constexpr,
    ):
        for i in range(N):
            A[i] = 0

    assert k.specialize(N=8) is not k.specialize(N=16)


def test_specialize_missing_constexpr_raises():
    @T.jit(private=True)
    def k(
        A: T.Buffer((N,), "int32"),
        *,
        N: T.constexpr,
        SCALE: T.constexpr,
    ):
        for i in range(N):
            A[i] = SCALE

    with pytest.raises(TypeError, match="missing"):
        k.specialize(N=8)


def test_specialize_extra_kwarg_raises():
    @T.jit(private=True)
    def k(
        A: T.Buffer((N,), "int32"),
        *,
        N: T.constexpr,
    ):
        for i in range(N):
            A[i] = 0

    with pytest.raises(TypeError, match="unexpected"):
        k.specialize(N=8, BOGUS=42)


def test_jit_kernel_with_nested_inline_helper():
    @T.jit(private=True)
    def k(
        A: T.Buffer((N,), "int32"),
        *,
        N: T.constexpr,
    ):
        @T.inline
        def double(x):
            A[x] = A[x] * 2

        for i in range(N):
            double(i)

    @T.prim_func(private=True)
    def expected(
        A: T.Buffer((4,), "int32"),
    ):
        for i in range(4):
            A[i] = A[i] * 2

    assert_structural_equal(k.specialize(N=4), expected, map_free_vars=True)


def test_constexpr_default_value():
    @T.jit(private=True)
    def k(
        A: T.Buffer((N,), "int32"),
        *,
        N: T.constexpr,
        SCALE: T.constexpr = 7,
    ):
        for i in range(N):
            A[i] = SCALE

    @T.prim_func(private=True)
    def expected(
        A: T.Buffer((8,), "int32"),
    ):
        for i in range(8):
            A[i] = 7

    assert_structural_equal(k.specialize(N=8), expected, map_free_vars=True)
    # Override the default
    overridden = k.specialize(N=8, SCALE=99)
    assert k.specialize(N=8) is not overridden


def test_specialize_returns_primfunc():
    @T.jit(private=True)
    def k(
        A: T.Buffer((N,), "int32"),
        *,
        N: T.constexpr,
    ):
        for i in range(N):
            A[i] = 0

    spec = k.specialize(N=8)
    assert isinstance(spec, tvm.tirx.PrimFunc)
    # Specialized PrimFunc has only the runtime params (constexpr stripped).
    assert len(spec.params) == 1


def test_constexpr_specializes_nested_selector_condition():
    @T.jit(private=True)
    def k(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        C: T.Buffer((8,), "float16"),
        flag: T.int32,
        *,
        LIMIT: T.constexpr,
    ):
        Tx.copy_async(
            C[:],
            A[:],
            dispatch="tma_explicit",
            mbar=C.data,
            src_selector=[(flag < LIMIT, B)],
        )

    specialized = k.specialize(LIMIT=4)
    op_call = specialized.body
    condition, candidate = op_call.config["src_selector"][0]
    assert isinstance(condition, tvm.tirx.LT)
    assert int(condition.b) == 4
    assert candidate.same_as(specialized.buffer_map[specialized.params[1]])


def test_optional_param_present_and_absent_ir():
    @T.jit(private=True)
    def kernel(a: T.Optional(T.handle), out_h: T.handle):
        out = T.match_buffer(out_h, (1,), "int32")
        if a is not None:
            A = T.match_buffer(a, (1,), "int32")
            out[0] = A[0]
        else:
            out[0] = -1

    @T.prim_func(private=True)
    def expected_present(a: T.handle, out_h: T.handle):
        A = T.match_buffer(a, (1,), "int32")
        out = T.match_buffer(out_h, (1,), "int32")
        out[0] = A[0]

    @T.prim_func(private=True)
    def expected_absent(out_h: T.handle):
        out = T.match_buffer(out_h, (1,), "int32")
        out[0] = -1

    present = kernel.specialize()
    absent = kernel.specialize(a=None)
    assert_structural_equal(present, expected_present, map_free_vars=True)
    assert_structural_equal(absent, expected_absent, map_free_vars=True)
    assert [param.name for param in present.params] == ["a", "out_h"]
    assert [param.name for param in absent.params] == ["out_h"]
    assert {param.name for param in present.buffer_map} == {"a", "out_h"}
    assert {param.name for param in absent.buffer_map} == {"out_h"}


def test_optional_specialization_cache_includes_presence():
    @T.jit(private=True)
    def kernel(a: T.Optional(T.handle), out_h: T.handle):
        out = T.match_buffer(out_h, (1,), "int32")
        if a is not None:
            A = T.match_buffer(a, (1,), "int32")
            out[0] = A[0]
        else:
            out[0] = 0

    present = kernel.specialize()
    absent = kernel.specialize(a=None)
    assert present is kernel.specialize()
    assert absent is kernel.specialize(a=None)
    assert present is not absent


def test_multiple_optional_params_preserve_runtime_order():
    @T.jit(private=True)
    def kernel(
        first_h: T.handle,
        a: T.Optional(T.handle),
        scale: T.int32,
        b: T.Optional(T.handle),
        out_h: T.handle,
    ):
        first = T.match_buffer(first_h, (1,), "int32")
        out = T.match_buffer(out_h, (1,), "int32")
        out[0] = first[0] * scale
        if a is not None:
            A = T.match_buffer(a, (1,), "int32")
            out[0] = out[0] + A[0]
        if b is not None:
            B = T.match_buffer(b, (1,), "int32")
            out[0] = out[0] + B[0]

    assert [param.name for param in kernel.specialize().params] == [
        "first_h",
        "a",
        "scale",
        "b",
        "out_h",
    ]
    assert [param.name for param in kernel.specialize(a=None).params] == [
        "first_h",
        "scale",
        "b",
        "out_h",
    ]
    assert [param.name for param in kernel.specialize(a=None, b=None).params] == [
        "first_h",
        "scale",
        "out_h",
    ]


def test_optional_only_accepts_none_at_specialization_time():
    @T.jit(private=True)
    def kernel(a: T.Optional(T.handle), out_h: T.handle):
        if a is not None:
            T.match_buffer(a, (1,), "int32")
        T.match_buffer(out_h, (1,), "int32")

    with pytest.raises(TypeError, match="only accept None"):
        kernel.specialize(a=object())


def test_only_explicit_t_optional_is_specializable():
    @T.jit(private=True)
    def typing_optional(a: typing.Optional[T.handle], out_h: T.handle):  # noqa: UP045
        T.evaluate(0)

    @T.jit(private=True)
    def union_optional(a: T.handle | None, out_h: T.handle):
        T.evaluate(0)

    with pytest.raises(TypeError, match="unexpected"):
        typing_optional.specialize(a=None)
    with pytest.raises(TypeError, match="unexpected"):
        union_optional.specialize(a=None)


def test_t_optional_is_restricted_to_jit():
    with pytest.raises(tvm.error.DiagnosticError, match="only supported by @T.jit"):

        @T.prim_func(private=True)
        def invalid(a: T.Optional(T.handle)):
            T.evaluate(0)


def test_compile_time_if_binding_uses_python_scope():
    @T.jit(private=True)
    def kernel(a: T.Optional(T.handle), out_h: T.handle):
        if a is None:
            selected = T.match_buffer(out_h, (1,), "int32")
        else:
            selected = T.match_buffer(a, (1,), "int32")
        selected[0] = 1

    present = kernel.specialize()
    absent = kernel.specialize(a=None)
    assert len(present.params) == 2
    assert len(absent.params) == 1
    assert len(present.buffer_map) == 1
    assert len(absent.buffer_map) == 1


def test_compile_time_bool_ops_and_if_expression_short_circuit():
    def fail_if_evaluated():
        raise RuntimeError("dead expression was evaluated")

    @T.jit(private=True)
    def kernel(a: T.Optional(T.handle), out_h: T.handle):
        out = T.match_buffer(out_h, (1,), "int32")
        if a is None or fail_if_evaluated():
            out[0] = 1
        if a is not None and fail_if_evaluated():
            out[0] = 2
        out[0] = 3 if a is None else fail_if_evaluated()

    absent = kernel.specialize(a=None)
    assert [param.name for param in absent.params] == ["out_h"]


def test_runtime_tir_if_cannot_guard_absent_optional_param():
    @T.jit(private=True)
    def kernel(a: T.Optional(T.handle), flag: T.int32):
        if flag != 0:
            T.match_buffer(a, (1,), "int32")

    with pytest.raises(tvm.error.DiagnosticError, match="match_buffer"):
        kernel.specialize(a=None)


@pytest.mark.parametrize(
    ("operation", "source_text"),
    [
        ("subscript", "a[10]"),
        ("attribute", "a.ptr_to"),
        ("match_buffer", "T.match_buffer"),
    ],
)
def test_unguarded_absent_optional_param_reports_source(operation, source_text):
    @T.jit(private=True)
    def kernel(a: T.Optional(T.handle)):
        if operation == "subscript":
            a[10]
        elif operation == "attribute":
            a.ptr_to([0])
        else:
            T.match_buffer(a, (1,), "int32")

    with pytest.raises(tvm.error.DiagnosticError) as exc_info:
        kernel.specialize(a=None)
    assert source_text in str(exc_info.value)


def test_present_optional_param_still_rejects_ffi_none():
    @T.jit
    def kernel(a: T.Optional(T.handle)):
        A = T.match_buffer(a, (1,), "int32")
        A[0] = 0

    executable = tvm.compile(kernel.specialize(), target="llvm", tir_pipeline="tirx")
    with pytest.raises(TypeError, match="expected Tensor"):
        executable(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
