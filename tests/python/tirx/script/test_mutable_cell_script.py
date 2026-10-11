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
"""Mutable declarations reconstruct dedicated cells, independent of tensors."""

import tvm_ffi

import tvm
from tvm.script import tirx as T


def _calls(function):
    calls = []
    tvm_ffi.structural_walk(
        function, lambda node: calls.append(node) if isinstance(node, tvm.ir.Call) else None
    )
    return calls


def test_mutable_cell_script_roundtrip():
    @T.function(private=True)
    def function(n: T.int32) -> T.int32:
        value: T.int32
        enabled: T.bool = True
        captured: T.let = value
        for i in range(n):
            if enabled:
                value += i
            else:
                value = captured
            enabled = not enabled
        explicit = T.alloc_cell("int32", 7, scope="local")
        vector = T.alloc_cell("int32x2", T.broadcast(2, 2), scope="local")
        T.evaluate(vector)
        return value + explicit

    code = function.script()
    assert "value: T.int32 = 0" in code
    assert "enabled: T.bool = T.bool(True)" in code
    assert "explicit: T.int32 = 7" in code
    assert 'vector = T.alloc_cell("int32x2"' in code
    assert "alloc_tensor" not in code
    assert ".source" not in code
    parsed = tvm.script.from_source(code, extra_vars={"T": T})
    tvm.ir.assert_structural_equal(function, parsed)
    tvm.ir.assert_structural_equal(function, tvm.ir.load_json(tvm.ir.save_json(function)))
    calls = _calls(function)
    assert any(call.op.name == "tirx.mutable_cell_alloc" for call in calls)
    assert any(call.op.name == "tirx.mutable_cell_store" for call in calls)
    assert all(
        call.ty.scope == "local" for call in calls if call.op.name == "tirx.mutable_cell_alloc"
    )


def test_free_cell_variable_type_roundtrip():
    cell = tvm.ir.Var("cell", tvm.tirx.MutableCellType("int32", "local"))
    source = cell.script(verbose_expr=True)
    assert "MutableCellType" in source
    assert "ExecScope" not in source
    namespace = {"I": tvm.script.ir, "T": T}
    exec(source, namespace)
    tvm.ir.assert_structural_equal(cell, namespace["cell"], map_free_vars=True)


def test_single_element_tensor_keeps_tensor_syntax():
    @T.function(private=True)
    def function():
        tensor = T.alloc_local((1,), "int32")
        tensor[0] = 3
        T.evaluate(tensor[0])

    code = function.script()
    assert "alloc_local" in code
    assert "tensor[0] = 3" in code
    tvm.ir.assert_structural_equal(function, tvm.script.from_source(code, extra_vars={"T": T}))
