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

import tvm
from tvm.ir import Call, Evaluate, assert_structural_equal, load_json, save_json
from tvm.script import tirx as T
from tvm.tirx import Var, decl_tensor
from tvm.tirx.analysis import undefined_vars
from tvm.tirx.transform.common import BufferReplacer


def test_instruction_is_a_void_call():
    src = decl_tensor((64,), "float32", scope="global")
    dst = decl_tensor((64,), "float32", scope="local")
    call = T.cuda.tile.ld(dst, src, scope="warp", vec_bits=128)
    assert isinstance(call, Call)
    assert call.ty == tvm.ir.PrimType("void")
    assert call.op.get_attr("TCallEffectKind") == 3
    assert call.op.get_attr("TIRxOpCategory") == "tile_primitive"
    assert call.attrs.scope == "warp"
    assert call.attrs.vec_bits == 128
    assert not hasattr(call, "config")
    assert not hasattr(call, "workspace")


@pytest.mark.parametrize("composite", [False, True])
def test_roundtrip_and_exactly_one_evaluate(composite):
    operation = T.cuda.tile.compose.exp if composite else T.cuda.tile.sqrt

    @T.function
    def func(A: T.Tensor((16,), "float32", scope="local")):
        operation(A[:], A[:], scope="warp")

    nodes = []
    tvm_ffi.structural_walk(func.body, (Evaluate, lambda n: nodes.append(n)))
    assert len(nodes) == 1
    call = nodes[0].value
    assert isinstance(call, Call)
    expected = "tile_composite" if composite else "tile_primitive"
    assert call.op.get_attr("TIRxOpCategory") == expected
    parsed = tvm.script.from_source(func.script(), extra_vars={"T": T})
    assert_structural_equal(func, parsed)
    assert_structural_equal(func, load_json(save_json(func)))
    assert_structural_equal(call, pickle.loads(pickle.dumps(call)), map_free_vars=True)


def test_runtime_operands_are_seen_and_replaced():
    src = decl_tensor((16, 16), "float32", scope="global")
    dst = decl_tensor((4, 16), "float32", scope="shared")
    alternative = decl_tensor((16, 16), "float32", scope="global")
    mbar = Var("mbar", "handle")
    row = Var("row", "int32")
    mask = Var("mask", "uint16")
    policy = Var("policy", "uint64")
    call = T.cuda.tile.cp_async_bulk_tensor_load(
        dst[:, :],
        src[0:1, :],
        mbar,
        mask,
        False,
        [row, row + 1, row + 2, row + 3],
        [(row == 0, alternative[:, :])],
        policy,
        descriptor_mode="explicit",
    )
    free = undefined_vars(Evaluate(call))
    for value in [src, dst, alternative, mbar, row, mask, policy]:
        assert any(v.same_as(value) for v in free)
    new_row = Var("new_row", "int32")
    replacement = BufferReplacer(var_map={row: new_row})(Evaluate(call))
    new_free = undefined_vars(replacement)
    assert not any(v.same_as(row) for v in new_free)
    assert any(v.same_as(new_row) for v in new_free)
    assert_structural_equal(call.attrs, replacement.value.attrs)


def test_workspace_is_an_optional_call_operand():
    tvm.backend.load("trn")
    a = decl_tensor((16, 16), "float32", scope="shared")
    acc = decl_tensor((16, 16), "float32", scope="psum")
    call = T.trn.tile.matmul(a, a, a, acc_psum=acc)
    assert call.args[-1].same_as(acc)
    assert any(v.same_as(acc) for v in undefined_vars(Evaluate(call)))
    assert_structural_equal(call, load_json(save_json(call)), map_free_vars=True)


def test_selector_region_cannot_silently_discard_coordinates():
    src = decl_tensor((16, 16), "float32", scope="global")
    dst = decl_tensor((4, 16), "float32", scope="shared")
    with pytest.raises(ValueError, match="cover their full tensor view"):
        T.cuda.tile.cp_async_bulk_tensor_load(
            dst,
            src[0:4, :],
            0,
            src_selector=[(True, src[4:8, :])],
            descriptor_mode="explicit",
        )


@pytest.mark.parametrize(
    "kwargs", [{"dispatch": "reg"}, {"config": {}}, {"workspace": {}}, {"hint": "fast"}]
)
def test_removed_bags_are_rejected(kwargs):
    a = decl_tensor((16,), "float32", scope="local")
    with pytest.raises(TypeError, match="unknown qualifier"):
        T.cuda.tile.mov(a, 0.0, **kwargs)


def test_dynamic_qualifier_is_rejected():
    a = decl_tensor((16,), "float32", scope="local")
    with pytest.raises(TypeError, match="expressions belong in operands"):
        T.cuda.tile.ld(a, a, vec_bits=Var("width", "int32"))


def test_direct_call_validation_cannot_bypass_signature():
    a = decl_tensor((16,), "float32", scope="local")
    valid = T.cuda.tile.mov(a, 0.0)
    for call in [
        Call(valid.op, valid.args[:-1], attrs=valid.attrs, ty="void"),
        Call(valid.op, valid.args, attrs=valid.attrs, ty="int32"),
        Call(valid.op, valid.args, ty="void"),
    ]:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            call.validate()


def test_block_scale_requires_both_scales():
    a = decl_tensor((16, 16), "float16", scope="shared")
    with pytest.raises(TypeError, match="missing operand SFB"):
        T.cuda.tile.tcgen05.mma_block_scale(a, a, a, SFA=a)
    with pytest.raises(TypeError, match="unknown qualifier"):
        T.cuda.tile.tcgen05.mma(a, a, a, SFA=a)


def test_buffer_replacer_no_shared_default():
    r1, r2 = BufferReplacer(), BufferReplacer()
    a, b = decl_tensor((64,), "float32"), decl_tensor((64,), "float32")
    r1.buffer_map[a] = b
    assert not r2.buffer_map


def test_buffer_replacer_replaces_strides_and_elem_offset():
    n, m = Var("n", "int32"), Var("m", "int32")
    a = decl_tensor((64,), "float32", strides=[n], elem_offset=n)
    new = BufferReplacer(var_map={n: m})(tvm.ir.TensorStore(a, [0], 1.0))
    assert new.dest.strides[0].same_as(m)
    assert new.dest.elem_offset.same_as(m)
