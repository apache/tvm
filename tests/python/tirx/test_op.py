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
import pickle

import pytest
import tvm_ffi

import tvm_tirx
from tvm_tirx.ir import Call, Evaluate, assert_structural_equal, load_json, save_json
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import TensorType, Var
from tvm_tirx.tirx.analysis import undefined_vars


def test_instruction_is_a_void_call():
    src = Var("tensor", TensorType((16, 4), "float32", scope="global"))
    dst = Var("tensor", TensorType((16, 4), "float32", scope="local"))
    call = T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], dst, src, scope="warp")
    assert isinstance(call, Call)
    assert call.ty == tvm_tirx.ir.PrimType("void")
    assert call.op.get_attr("TCallEffectKind") == tvm_tirx.ir.CallEffectKind.UpdateState
    assert callable(call.op.get_attr("FTileOpDispatch"))
    assert call.attrs.scope.same_as(T.cuda.ThreadScope("warp"))
    assert call.args[0].value == "ld.global.v4.b32"
    assert not hasattr(call, "config")
    assert not hasattr(call, "workspace")


@pytest.mark.parametrize("composite", [False, True])
def test_roundtrip_and_exactly_one_evaluate(composite):
    @T.function(private=True)
    def exp_scaled(x: T.float32) -> T.float32:
        return T.exp(x) * T.float32(2)

    @T.function
    def func(A: T.Tensor((16, 1), "float32", scope="local")):
        if T.constexpr(composite):
            T.cuda.tile.ewise(exp_scaled, A, A)
        else:
            T.cuda.tile.apply(T.ptx["sqrt.rn.f32"], A, A)

    nodes = []
    tvm_ffi.structural_walk(func.body, (Evaluate, lambda n: nodes.append(n)))
    assert len(nodes) == 1
    call = nodes[0].value
    assert callable(call.op.get_attr("FTileOpDispatch"))
    assert_structural_equal(
        func, tvm_tirx.script.from_source(func.script(), extra_vars={"T": T, "I": I})
    )
    assert_structural_equal(func, load_json(save_json(func)))
    assert_structural_equal(call, pickle.loads(pickle.dumps(call)), map_free_vars=True)


def test_runtime_operands_are_seen_and_replaced():
    src = Var("tensor", TensorType((16, 16), "float32", scope="global"))
    dst = Var("tensor", TensorType((4, 16), "float32", scope="shared"))
    mbar = Var("mbar", "handle")
    row = Var("row", "int32")
    mask = Var("mask", "uint16")
    policy = Var("policy", "uint64")
    call = T.cuda.tile.apply(
        T.ptx[
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        ],
        dst,
        src,
        box=(1, 16),
        coords=(0, 0),
        gather4=(row, row + 1, row + 2, row + 3),
        mbar=mbar,
        cta_mask=mask,
        cache_policy=policy,
    )
    free = undefined_vars(Evaluate(call))
    for value in (src, dst, mbar, row, mask, policy):
        assert any(v.same_as(value) for v in free)
    new_row = Var("new_row", "int32")
    replacement = tvm_ffi.structural_map(
        Evaluate(call), (Var, lambda var: new_row if var.same_as(row) else var), order="post"
    )
    new_free = undefined_vars(replacement)
    assert not any(v.same_as(row) for v in new_free)
    assert any(v.same_as(new_row) for v in new_free)
    assert_structural_equal(call.attrs, replacement.value.attrs)


def test_workspace_is_an_optional_call_operand():
    tvm_tirx.backend.load("trn")
    a = Var("tensor", TensorType((16, 16), "float32", scope="shared"))
    acc = Var("tensor", TensorType((16, 16), "float32", scope="trn.psum"))
    call = T.trn.tile.matmul(a, a, a, acc_psum=acc)
    assert call.args[-1].same_as(acc)
    assert any(v.same_as(acc) for v in undefined_vars(Evaluate(call)))
    assert_structural_equal(call, load_json(save_json(call)), map_free_vars=True)


def test_direct_call_validation_cannot_bypass_signature():
    a = Var("tensor", TensorType((16, 1), "float32", scope="local"))
    valid = T.cuda.tile.apply(T.ptx["mov.b32"], a, T.float32(0))
    for call in [
        Call(
            valid.op,
            valid.args[:-1],
            attrs=valid.attrs,
            ty="void",
        ),
        Call(
            valid.op,
            valid.args,
            attrs=valid.attrs,
            ty="int32",
        ),
        Call(
            valid.op,
            valid.args,
            ty="void",
        ),
    ]:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            call.validate()
