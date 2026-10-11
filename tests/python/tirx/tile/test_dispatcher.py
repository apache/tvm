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
from tvm.script import tirx as T
from tvm.tirx import decl_tensor
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
