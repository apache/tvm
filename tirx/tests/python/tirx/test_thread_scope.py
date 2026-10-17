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
import os
import pickle
import subprocess
import sys

import pytest
import tvm_ffi
from tvm_ffi.dataclasses import Enum

import tvm_tirx
from tvm_tirx.ir import Call, Evaluate, make_node
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import Function, TensorType, Var
from tvm_tirx.tirx.thread_scope import ThreadScope
from tvm_tirx.tirx.tile import DispatchContext
from tvm_tirx.tirx.tile.instruction import TensorCall


@pytest.mark.parametrize(
    "backend,names",
    [
        ("cuda", ("kernel", "cluster", "cta", "warpgroup", "warp", "thread")),
        ("trn", ("kernel", "thread")),
    ],
)
def test_enum_identity_and_explicit_ranks(backend, names):
    for rank, name in enumerate(names, start=-1):
        scope = ThreadScope(f"{backend}.{name}")
        assert isinstance(scope, Enum)
        assert scope.same_as(ThreadScope.get(scope.name))
        assert scope.same_as(getattr(T, backend).ThreadScope(name))
        assert scope.name == f"{backend}.{name}"
        assert scope.local_name == name
        assert scope.backend == backend
        assert scope.rank == rank
        assert ThreadScope.def_attr("backend")[scope] == backend
        assert ThreadScope.def_attr("rank")[scope] == rank
        assert hash(scope) == hash(ThreadScope(scope.name))
    assert not ThreadScope("cuda.thread").same_as(ThreadScope("trn.thread"))


def test_hierarchy_comparison_and_lookup_errors():
    higher = tvm_ffi.get_global_func("tirx.ScopeHigher")
    assert higher(ThreadScope("cuda.kernel"), ThreadScope("cuda.thread"))
    assert not higher(ThreadScope("cuda.thread"), ThreadScope("cuda.kernel"))
    assert not higher(ThreadScope("cuda.warp"), ThreadScope("cuda.warp"))
    with pytest.raises(ValueError, match="different backends"):
        higher(ThreadScope("cuda.thread"), ThreadScope("trn.thread"))
    for name in ("warp", "cuda.unknown", "trn.warp", ""):
        with pytest.raises(ValueError):
            ThreadScope(name)


@pytest.mark.parametrize("name", ["cuda.kernel", "cuda.warp", "trn.thread"])
def test_scope_roundtrips_keep_singleton(name):
    scope = ThreadScope(name)
    assert tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(scope)).same_as(scope)
    assert pickle.loads(pickle.dumps(scope)).same_as(scope)
    text = scope.script()
    assert f'T.ThreadScope("{name}")' in text
    assert eval(text, {"T": T}).same_as(scope)


@pytest.mark.parametrize("scope", ["warp", "cuda.warp", T.cuda.ThreadScope("warp")])
def test_instruction_normalizes_scope_before_constructing_call(scope):
    tensor = Var("tensor", TensorType((16,), "float32", scope="local", layout="default"))
    call = T.cuda.tile.apply(T.ptx["sqrt.rn.f32"], tensor, tensor, scope=scope)
    tag = ThreadScope("cuda.warp")
    assert call.attrs.scope.same_as(tag)
    assert TensorCall.decode(call).scope.same_as(tag)
    default = T.cuda.tile.apply(T.ptx["sqrt.rn.f32"], tensor, tensor)
    assert default.attrs.scope.same_as(ThreadScope("cuda.thread"))
    context = DispatchContext(tvm_tirx.target.Target("cuda"), tag, {}, {})
    assert context.thread_scope.same_as(tag)
    assert context.scope_kind == "warp"
    with pytest.raises(ValueError, match="scope_kind"):
        DispatchContext(tvm_tirx.target.Target("cuda"), tag, {}, {}, scope_kind="thread")


def test_raw_scope_attrs_are_required_and_typed():
    for schema in ("ScopeAttrs", "TrnAttrs"):
        with pytest.raises(TypeError, match="scope"):
            make_node(f"tirx.tensor.{schema}")
        for invalid in (None, "thread", 1):
            with pytest.raises((TypeError, ValueError)):
                make_node(f"tirx.tensor.{schema}", scope=invalid)
        attrs = make_node(f"tirx.tensor.{schema}", scope="cuda.thread")
        assert attrs.scope.same_as(ThreadScope("cuda.thread"))
        assert tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(attrs)).scope.same_as(attrs.scope)


def test_cross_backend_scopes_are_rejected_at_each_boundary():
    tensor = Var("tensor", TensorType((16,), "float32", scope="local", layout="default"))
    trn_scope = ThreadScope("trn.thread")
    for scope in (trn_scope, "trn.thread"):
        with pytest.raises(ValueError, match="belongs to trn"):
            T.cuda.tile.apply(T.ptx["sqrt.rn.f32"], tensor, tensor, scope=scope)
    call = T.cuda.tile.apply(T.ptx["sqrt.rn.f32"], tensor, tensor)
    attrs = make_node("tirx.tensor.ScopeAttrs", scope=trn_scope)
    raw = Call(
        call.op,
        call.args,
        attrs=attrs,
        ty=call.ty,
    )
    with pytest.raises(ValueError, match="belongs to trn"):
        raw.validate()
    with pytest.raises(ValueError, match="does not match target"):
        DispatchContext(tvm_tirx.target.Target("cuda"), trn_scope, {}, {})
    trn_tensor = Var("tensor", TensorType((16,), "float32", scope="trn.sbuf", layout="default"))
    trn_call = T.trn.tile.tensortensor(trn_tensor, trn_tensor, trn_tensor)
    func = Function([], Evaluate(trn_call)).with_attr("target", tvm_tirx.target.Target("cuda"))
    with pytest.raises(ValueError, match="does not match target"):
        tvm_tirx.tirx.transform.TileDispatch()(tvm_tirx.IRModule({"main": func}))


def test_runtime_only_import_and_removed_interfaces():
    env = os.environ.copy()
    env["TVM_USE_RUNTIME_LIB"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import tvm_tirx; from tvm_tirx.tirx.thread_scope import ThreadScope; "
            "assert tvm_tirx.base._RUNTIME_ONLY",
        ],
        env=env,
        check=True,
    )
    assert not hasattr(tvm_tirx.tirx, "ExecScope")
    assert not hasattr(T.cuda, "ExecScope")
    assert tvm_ffi.get_global_func("tirx.RegisterExecScope", allow_missing=True) is None
