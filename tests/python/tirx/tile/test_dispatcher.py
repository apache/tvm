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

from types import SimpleNamespace

import pytest

import tvm
from tvm.ir import Call, Evaluate, Op, PrimType, Return, const, make_node, register_op_attr
from tvm.script import tirx as T
from tvm.tirx import Function, decl_tensor
from tvm.tirx.tile import DispatchContext
from tvm.tirx.tile.dispatch import run_dispatch
from tvm.tirx.tile.instruction import SPECS


def test_selected_lowerer_reports_context_and_cause(monkeypatch):
    a = decl_tensor((16,), "float32", scope="local", layout="default")
    call = T.cuda.tile.sqrt(a, a)

    def failing(view, context):
        raise ValueError("unsupported layout")

    monkeypatch.setattr(SPECS[call.op.name], "lower", failing)
    context = SimpleNamespace(target=tvm.target.Target("cuda"), scope_kind="thread")
    with pytest.raises(
        RuntimeError, match="op=tirx.cuda.tile.sqrt target=cuda scope=thread"
    ) as error:
        run_dispatch(call, context)
    assert "unsupported layout" in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)


def test_backend_mismatch_is_rejected():
    a = decl_tensor((16,), "float32", scope="local", layout="default")
    context = SimpleNamespace(
        target=SimpleNamespace(kind=SimpleNamespace(name="trn")), scope_kind="thread"
    )
    with pytest.raises(ValueError, match="does not match cuda"):
        run_dispatch(T.cuda.tile.sqrt(a, a), context)


@pytest.mark.parametrize("instruction", ["cp_async_bulk_tensor_load", "tcgen05.cp"])
def test_single_thread_instructions_reject_collective_scope(instruction):
    a = decl_tensor((16, 16), "float32", scope="global")
    b = decl_tensor((16, 16), "float32", scope="shared", layout="default")
    with pytest.raises(ValueError, match="unsupported scope 'warp'"):
        if instruction == "cp_async_bulk_tensor_load":
            T.cuda.tile.cp_async_bulk_tensor_load(b, a, 0, scope="warp")
        else:
            T.cuda.tile.tcgen05.cp(b, a, scope="warp")


@pytest.mark.parametrize("backend", ["cuda", "trn"])
@pytest.mark.parametrize("scope", ["kernel", "unknown"])
def test_instruction_scope_contract(backend, scope):
    tensor = decl_tensor((16,), "float32", scope="local", layout="default")
    instruction = T.cuda.tile.add if backend == "cuda" else T.trn.tile.tensortensor
    with pytest.raises(ValueError, match=f"unsupported scope '{scope}'"):
        instruction(tensor, tensor, tensor, scope=scope)

    # Raw Calls obey the same instruction contract through Op validation.
    call = instruction(tensor, tensor, tensor)
    attrs = tvm.ir.make_node("tirx.tensor." + SPECS[call.op.name].schema, scope=scope)
    raw = tvm.ir.Call(call.op, call.args, attrs=attrs, ty=call.ty)
    with pytest.raises(ValueError, match=f"unsupported scope '{scope}'"):
        raw.validate()


def _custom_call(name):
    op_name = f"test.tile_dispatch.{name}"
    register_op_attr(op_name, "TFixedReturnType", PrimType("void"))
    register_op_attr(op_name, "TCallEffectKind", 3)
    op = Op.get(op_name)
    op.set_signature(args=["token"])
    op.set_attrs_type_key("tirx.tensor.ScopeAttrs")
    return Call(op, [const(7)], attrs=make_node("tirx.tensor.ScopeAttrs"))


def _lower(body):
    func = Function([], body).with_attr("target", tvm.target.Target("cuda"))
    return tvm.tirx.transform.TileDispatch()(tvm.IRModule({"main": func}))["main"]


def test_op_attribute_dispatches_without_python_instruction_spec():
    # Initialize the pass's cached attribute view before adding a new Op.
    _lower(Evaluate(0))
    call = _custom_call("custom")
    assert call.op.name not in SPECS
    seen = []

    @register_op_attr(call.op.name, "FTileOpDispatch")
    def dispatch(actual_call, context):
        assert actual_call.same_as(call)
        assert isinstance(context, DispatchContext)
        assert context.target.kind.name == "cuda"
        assert context.exec_scope.backend == "cuda"
        assert context.scope_kind == "thread"
        seen.append(actual_call)
        return Function([], Evaluate(23))

    lowered = _lower(Evaluate(call))
    assert len(seen) == 1
    assert lowered.body.seq[0].value == 23

    # The live native attribute map observes a replacement on the same Op.
    register_op_attr(
        call.op.name,
        "FTileOpDispatch",
        lambda actual_call, context: Function([], Evaluate(41)),
        override=True,
    )
    assert _lower(Evaluate(call)).body.seq[0].value == 41


def test_dispatch_recursively_expands_tile_implementations():
    outer = _custom_call("outer")
    inner = _custom_call("inner")
    register_op_attr(
        outer.op.name, "FTileOpDispatch", lambda call, context: Function([], Evaluate(inner))
    )
    register_op_attr(
        inner.op.name, "FTileOpDispatch", lambda call, context: Function([], Evaluate(31))
    )
    assert _lower(Evaluate(outer)).body.seq[0].value == 31


def test_op_without_dispatch_attribute_is_preserved():
    call = _custom_call("ordinary")
    assert call.op.get_attr("FTileOpDispatch") is None
    assert _lower(Evaluate(call)).body.seq[0].value.same_as(call)


def test_residual_tile_call_is_rejected():
    call = _custom_call("residual")
    register_op_attr(
        call.op.name, "FTileOpDispatch", lambda call, context: Function([], Evaluate(0))
    )
    # Dispatch expands statement-boundary Calls; other positions must be rejected.
    with pytest.raises(tvm.error.InternalError, match="Failed to lower the TIRx program"):
        _lower(Return(call))


def test_dispatch_requires_function_result():
    call = _custom_call("invalid_result")
    register_op_attr(call.op.name, "FTileOpDispatch", lambda call, context: Evaluate(0))
    with pytest.raises(TypeError, match="tirx.Function"):
        _lower(Evaluate(call))
