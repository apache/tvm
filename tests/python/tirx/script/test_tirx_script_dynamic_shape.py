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
"""TIRX parser integration for symbolic shape."""

from __future__ import annotations

import pytest

from tvm.script import ir as I
from tvm.script import tirx as T


def test_captured_shape_requires_concrete_symbols():
    # Native shape construction preserves concrete symbols and rejects strings.
    def build(shape):
        @T.prim_func
        def main(x: T.Buffer(shape, "float32")):
            T.evaluate(0)

        return main

    n = T.dynamic("n")
    function = build((n, 16))
    assert function.params[0].ty.shape[0].same_as(n)
    with pytest.raises(AssertionError, match="data must be int or Expr, but got n"):
        build(("n", 16))


def test_dynamic_symbols_are_fresh_and_scope_independent():
    assert T.dynamic is I.dynamic
    n = T.dynamic("n")
    same_name = I.dynamic("n")
    k = I.dynamic("k", "int32")
    assert n.ty.dtype == "int64"
    assert k.ty.dtype == "int32"
    assert not n.same_as(same_name)

    @I.ir_module
    class Module:
        @T.prim_func
        def first(x: T.Buffer((n,), "float32")):
            T.evaluate(n)

    assert Module["first"].params[0].ty.shape[0].same_as(n)
    assert Module["first"].body.value.same_as(n)
