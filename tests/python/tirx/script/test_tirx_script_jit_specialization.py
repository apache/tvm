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
"""Native TIRx JIT specialization results."""

from __future__ import annotations

from tvm.ir import assert_structural_equal
from tvm.script import tirx as T


def test_jit_buffer_annotation():
    @T.jit(private=True)
    def kernel(output: T.Buffer((5,), "int32")):
        output[0] = 7

    @T.prim_func(private=True)
    def expected(output: T.Buffer((5,), "int32")):
        output[0] = 7

    assert_structural_equal(kernel.specialize(), expected, map_free_vars=True)


def test_jit_optional_buffer():
    @T.jit(private=True)
    def kernel(value: T.Optional(T.Buffer((5,), "int32"))):
        if T.constexpr(value is not None):
            value[0] = 3
        else:
            T.evaluate(0)

    @T.prim_func(private=True)
    def absent():
        T.evaluate(0)

    @T.prim_func(private=True)
    def present(value: T.Buffer((5,), "int32")):
        value[0] = 3

    assert_structural_equal(kernel.specialize(value=None), absent, map_free_vars=True)
    assert_structural_equal(kernel.specialize(), present, map_free_vars=True)
