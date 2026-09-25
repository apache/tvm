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
"""TIRX parser integration for basic usage."""

from __future__ import annotations

# Script-local bindings are inspected through the constructed IR.
import pytest

from tvm.script import ir as I
from tvm.script import tirx as T


def test_native_concise_scopes_unwind_with_their_parent():
    # Nested concise thread scopes must preserve the original variables in the constructed IR.
    from tvm import tirx

    variables = []

    def observe(*items):
        variables.extend(items)

    @T.prim_func
    def main():
        bx = T.launch_thread("blockIdx.x", 2)
        tx = T.launch_thread("threadIdx.x", 32)
        observe(bx, tx)
        T.evaluate(bx + tx)

    bx, tx = variables
    body = main.body
    assert isinstance(body, tirx.AttrStmt) and isinstance(body.body, tirx.AttrStmt)
    assert body.node.var.same_as(bx) and body.body.node.var.same_as(tx)
    assert body.body.body.value.a.same_as(bx) and body.body.body.value.b.same_as(tx)


def test_loop_control_validation_preserves_valid_and_unchecked_ir():
    # Invalid loop placement must be rejected, while disabled checks preserve the original IR.
    from tvm import ir, tirx

    @T.prim_func(check_well_formed=False)
    def invalid():
        T.evaluate(tirx.break_loop())

    # This is exactly the native node emitted by source `break`; constructing
    # the intrinsic explicitly keeps the surrounding Python definition valid.
    ir.assert_structural_equal(invalid.body, tirx.Evaluate(tirx.break_loop()))
    with pytest.raises(ValueError, match="requires an enclosing loop"):

        @T.prim_func
        def rejected():
            T.evaluate(tirx.break_loop())

    @T.prim_func
    def valid():
        for i in range(2):
            break

    assert isinstance(valid.body, tirx.For)
    ir.assert_structural_equal(valid.body.body, invalid.body)

    @I.ir_module(check_well_formed=False, extra_vars={"invalid": invalid})
    class Unchecked:
        bad = invalid

    assert Unchecked["bad"].same_as(invalid)
    with pytest.raises(ValueError, match="requires an enclosing loop"):

        @I.ir_module(extra_vars={"invalid": invalid})
        class Rejected:
            bad = invalid
