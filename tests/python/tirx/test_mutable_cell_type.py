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
"""Construction and raw-IR legality of initialized mutable local cells."""

import pytest
import tvm_ffi

import tvm
from tvm import ir, tirx


def _alloc(cell, value=None):
    if value is None:
        value = ir.const(0, cell.ty.element_type.dtype)
    return ir.Call("tirx.mutable_cell_alloc", [value], ty=cell.ty)


def _load(cell, ty=None):
    return ir.Call("tirx.mutable_cell_load", [cell], ty=ty)


def _store(cell, value=1):
    return ir.Call("tirx.mutable_cell_store", [cell, ir.const(value, cell.ty.element_type.dtype)])


def _function(*body, params=(), ret_type=None):
    return tirx.Function(list(params), ir.SeqStmt(list(body)), ret_type=ret_type)


def _valid(function):
    return tirx.analysis.verify_mutable_cells(function, assert_mode=False)


@pytest.mark.parametrize("dtype", ["bool", "int32", "uint128", "float32", "float32x4"])
@pytest.mark.parametrize("scope_kwargs", [{}, {"scope": "local"}])
def test_cell_type_roundtrip(dtype, scope_kwargs):
    ty = tirx.MutableCellType(dtype, **scope_kwargs)
    assert ty.element_type.dtype == dtype
    assert ty.scope == "local"
    ir.assert_structural_equal(ty, tirx.MutableCellType(dtype, "local"))
    ir.assert_structural_equal(ty, ir.load_json(ir.save_json(ty)))
    visited = []
    tvm_ffi.structural_walk(ty, lambda node: visited.append(node))
    assert any(isinstance(node, ir.PrimType) for node in visited)
    mapped = tvm_ffi.structural_mutate(
        ty, (ir.PrimType, lambda _node, _mutator: ir.PrimType("float64"))
    )
    assert mapped.element_type.dtype == "float64"
    assert mapped.scope == "local"
    assert ty.element_type.dtype == dtype


@pytest.mark.parametrize("dtype", ["void", "int32xvscalex4"])
def test_cell_rejects_unsupported_elements(dtype):
    with pytest.raises((TypeError, ValueError)):
        tirx.MutableCellType(dtype)


def test_cell_rejects_nonprimitive_elements():
    with pytest.raises(TypeError):
        tirx.MutableCellType(ir.PtrType(ir.PrimType("int32")))
    with pytest.raises(TypeError):
        tirx.MutableCellType(tirx.MutableCellType("int32"))


@pytest.mark.parametrize(
    "scope", ["", "global", "shared", "shared.dyn", "local.tag", "thread", "warp"]
)
def test_cell_rejects_unsupported_storage_scopes(scope):
    with pytest.raises(ValueError, match="local storage scope"):
        tirx.MutableCellType("int32", scope)
    raw_type = ir.make_node("tirx.MutableCellType", element_type=ir.PrimType("int32"), scope=scope)
    cell = ir.Var("cell", raw_type)
    with pytest.raises(ValueError, match="local storage scope"):
        _alloc(cell).validate()
    assert not _valid(_function(ir.Bind(cell, _alloc(cell))))


@pytest.mark.parametrize("scope", [None, 0, tirx.ExecScope("thread")])
def test_cell_requires_string_storage_scope(scope):
    with pytest.raises(TypeError):
        tirx.MutableCellType("int32", scope)


def test_cell_op_types_effects_and_validation():
    cell = ir.Var("cell", tirx.MutableCellType("int32"))
    operations = [_alloc(cell), _load(cell), _store(cell)]
    assert operations[1].ty.dtype == "int32"
    ir.assert_structural_equal(operations[2].ty, ir.PrimType("void"))
    for call, effect in zip(operations, [3, 2, 3]):
        call.validate()
        assert call.op.get_attr("TCallEffectKind") == effect
    malformed = [
        ir.Call("tirx.mutable_cell_alloc", [], ty=cell.ty),
        _alloc(cell, ir.const(0, "float32")),
        _load(cell, "float32"),
        ir.Call("tirx.mutable_cell_load", [ir.const(1, "int32")], ty="int32"),
        ir.Call("tirx.mutable_cell_load", [cell], attrs={}, ty="int32"),
        ir.Call("tirx.mutable_cell_load", [cell], ty_args=[ir.PrimType("int32")], ty="int32"),
        ir.Call("tirx.mutable_cell_store", [cell, ir.const(1, "float32")], ty="void"),
        ir.Call("tirx.mutable_cell_store", [cell, ir.const(1, "int32")], ty="int32"),
    ]
    for call in malformed:
        with pytest.raises((TypeError, ValueError)):
            call.validate()


def test_raw_cells_require_local_allocation_and_direct_operations():
    cell = ir.Var("cell", tirx.MutableCellType("int32"))
    bind = ir.Bind(cell, _alloc(cell))
    assert _valid(_function(bind, ir.Evaluate(_store(cell)), ir.Return(_load(cell))))
    invalid = [
        _function(ir.Evaluate(_alloc(cell))),
        _function(ir.Evaluate(_load(cell))),
        _function(bind, ir.Bind(ir.Var("alias", cell.ty), cell)),
        _function(bind, ir.Return(cell)),
        _function(ir.Evaluate(_load(cell)), params=[cell]),
        _function(bind, ret_type=cell.ty),
        _function(
            bind,
            ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("escape"), cell], ty="void")),
        ),
        _function(bind, ir.Bind(ir.Var("bad", ir.PrimType("void")), _store(cell))),
        _function(
            bind,
            ir.Evaluate(
                ir.Call("tirx.address_of", [_load(cell)], ty=ir.PtrType(ir.PrimType("int32")))
            ),
        ),
        _function(ir.If(ir.const(True, "bool"), [bind], None), ir.Evaluate(_load(cell))),
    ]
    for function in invalid:
        assert not _valid(function)


def test_parallel_cells_are_fresh_and_vectorized_cells_are_rejected():
    cell = ir.Var("cell", tirx.MutableCellType("int32"))
    index = ir.Var("i", ir.PrimType("int32"))
    bind = ir.Bind(cell, _alloc(cell))

    def loop(kind, body):
        return ir.For(index, ir.const(0, "int32"), ir.const(4, "int32"), kind, body)

    assert _valid(_function(loop(ir.ForKind.PARALLEL, [bind, ir.Evaluate(_store(cell))])))
    capture = _function(bind, loop(ir.ForKind.PARALLEL, [ir.Evaluate(_store(cell))]))
    vectorized = _function(loop(ir.ForKind.VECTORIZED, [bind, ir.Evaluate(_store(cell))]))
    assert not _valid(capture)
    assert not _valid(vectorized)
    for transform in [tirx.transform.VectorizeLoop(), tirx.transform.LowerTIRx()]:
        with tvm.target.Target("llvm"), pytest.raises(tvm.error.InternalError, match="vectorized"):
            transform(ir.IRModule({"main": vectorized}))
    with pytest.raises(tvm.error.InternalError, match="parallel"):
        tirx.transform.SplitHostDevice()(ir.IRModule({"main": capture}))
