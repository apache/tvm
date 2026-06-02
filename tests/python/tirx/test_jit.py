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
"""Tests for ``@Tx.jit`` + ``Tx.constexpr``."""

from __future__ import annotations

import pytest

import tvm
from tvm.ir import assert_structural_equal
from tvm.script import tirx as Tx


def test_int_constexpr_specializes_loop_bound():
    @Tx.jit(private=True)
    def add(
        A: Tx.Buffer((N,), "int32"),
        B: Tx.Buffer((N,), "int32"),
        C: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
    ):
        for i in range(N):
            C[i] = A[i] + B[i]

    @Tx.prim_func(private=True)
    def expected(
        A: Tx.Buffer((128,), "int32"),
        B: Tx.Buffer((128,), "int32"),
        C: Tx.Buffer((128,), "int32"),
    ):
        for i in range(128):
            C[i] = A[i] + B[i]

    assert_structural_equal(add.specialize(N=128), expected, map_free_vars=True)


def test_constexpr_in_2d_buffer_shape():
    @Tx.jit(private=True)
    def matadd(
        A: Tx.Buffer((M, K), "int32"),
        B: Tx.Buffer((M, K), "int32"),
        C: Tx.Buffer((M, K), "int32"),
        *,
        M: Tx.constexpr,
        K: Tx.constexpr,
    ):
        for m in range(M):
            for k in range(K):
                C[m, k] = A[m, k] + B[m, k]

    @Tx.prim_func(private=True)
    def expected(
        A: Tx.Buffer((4, 8), "int32"),
        B: Tx.Buffer((4, 8), "int32"),
        C: Tx.Buffer((4, 8), "int32"),
    ):
        for m in range(4):
            for k in range(8):
                C[m, k] = A[m, k] + B[m, k]

    assert_structural_equal(matadd.specialize(M=4, K=8), expected, map_free_vars=True)


def test_constexpr_in_body_expression():
    @Tx.jit(private=True)
    def scaled_copy(
        A: Tx.Buffer((N,), "int32"),
        B: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
        SCALE: Tx.constexpr,
    ):
        for i in range(N):
            B[i] = A[i] * SCALE

    @Tx.prim_func(private=True)
    def expected(
        A: Tx.Buffer((16,), "int32"),
        B: Tx.Buffer((16,), "int32"),
    ):
        for i in range(16):
            B[i] = A[i] * 3

    assert_structural_equal(scaled_copy.specialize(N=16, SCALE=3), expected, map_free_vars=True)


def test_specialize_cache_returns_same_instance():
    @Tx.jit(private=True)
    def k(
        A: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
    ):
        for i in range(N):
            A[i] = 0

    a = k.specialize(N=8)
    b = k.specialize(N=8)
    assert a is b


def test_specialize_different_args_produce_different_funcs():
    @Tx.jit(private=True)
    def k(
        A: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
    ):
        for i in range(N):
            A[i] = 0

    assert k.specialize(N=8) is not k.specialize(N=16)


def test_specialize_missing_constexpr_raises():
    @Tx.jit(private=True)
    def k(
        A: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
        SCALE: Tx.constexpr,
    ):
        for i in range(N):
            A[i] = SCALE

    with pytest.raises(TypeError, match="missing"):
        k.specialize(N=8)


def test_specialize_extra_kwarg_raises():
    @Tx.jit(private=True)
    def k(
        A: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
    ):
        for i in range(N):
            A[i] = 0

    with pytest.raises(TypeError, match="unexpected"):
        k.specialize(N=8, BOGUS=42)


def test_jit_kernel_with_nested_inline_helper():
    @Tx.jit(private=True)
    def k(
        A: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
    ):
        @Tx.inline
        def double(x):
            A[x] = A[x] * 2

        for i in range(N):
            double(i)

    @Tx.prim_func(private=True)
    def expected(
        A: Tx.Buffer((4,), "int32"),
    ):
        for i in range(4):
            A[i] = A[i] * 2

    assert_structural_equal(k.specialize(N=4), expected, map_free_vars=True)


def test_constexpr_default_value():
    @Tx.jit(private=True)
    def k(
        A: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
        SCALE: Tx.constexpr = 7,
    ):
        for i in range(N):
            A[i] = SCALE

    @Tx.prim_func(private=True)
    def expected(
        A: Tx.Buffer((8,), "int32"),
    ):
        for i in range(8):
            A[i] = 7

    assert_structural_equal(k.specialize(N=8), expected, map_free_vars=True)
    # Override the default
    overridden = k.specialize(N=8, SCALE=99)
    assert k.specialize(N=8) is not overridden


def test_specialize_returns_primfunc():
    @Tx.jit(private=True)
    def k(
        A: Tx.Buffer((N,), "int32"),
        *,
        N: Tx.constexpr,
    ):
        for i in range(N):
            A[i] = 0

    spec = k.specialize(N=8)
    assert isinstance(spec, tvm.tirx.PrimFunc)
    # Specialized PrimFunc has only the runtime params (constexpr stripped).
    assert len(spec.params) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
