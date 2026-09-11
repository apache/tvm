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
from tvm import tirx as tir
from tvm.ir import Op, assert_structural_equal
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx import Var
from tvm.tirx.buffer import decl_buffer
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.tile_primitive import TilePrimitiveCall


def _test(op: str, *args):
    return TilePrimitiveCall(*args, op=Op.get("tirx.tile." + op), workspace={}, config={})


def test_copy():
    A = decl_buffer((64, 64), "float32", scope="global")
    A_sm = decl_buffer((64, 64), "float32", scope="shared")
    _test("copy", A[0:64, 0:64], A_sm[0:64, 0:64])


def test_fill():
    A = decl_buffer((64, 64), "float32", scope="global")
    _test("fill", A[0:64, 0:64], 1.0)


def test_gemm():
    A = decl_buffer((64, 64), "float32", scope="global")
    B = decl_buffer((64, 64), "float32", scope="global")
    C = decl_buffer((64, 64), "float32", scope="global")
    D = decl_buffer((64, 64), "float32", scope="global")
    _test("gemm", D[:, :], A[:, :], B[:, :], C[:, :], True, False, 1.0, 0.0)


def test_tile_primitive_call_pickle_roundtrip():
    """TilePrimitiveCall reflection must provide a deserialization creator."""
    A = decl_buffer((64,), "float32", scope="local")
    workspace = decl_buffer((16,), "float32", scope="shared")
    call = TilePrimitiveCall(
        A[:],
        1.0,
        op=Op.get("tirx.tile.fill"),
        workspace={"scratch": workspace},
        config={"hint": "roundtrip", "stages": 2},
        dispatch="reg",
        scope=ExecScope("warpgroup"),
    )
    restored = pickle.loads(pickle.dumps(call))

    # Buffer values are ordinary free Vars.  Pickle reconstructs their
    # identities, so compare them under the standard free-Var mapping.
    assert_structural_equal(restored, call, map_free_vars=True)
    assert restored.op.same_as(call.op)
    assert_structural_equal(restored.args, call.args, map_free_vars=True)
    assert_structural_equal(restored.workspace, call.workspace, map_free_vars=True)
    assert_structural_equal(restored.config, call.config)
    assert restored.dispatch == call.dispatch
    assert_structural_equal(restored.scope, call.scope)


def test_buffer_replacer_no_shared_default():
    """Regression test for F4: BufferReplacer default dicts must not be shared."""
    from tvm.tirx.transform.common import BufferReplacer

    r1 = BufferReplacer()
    r2 = BufferReplacer()
    A = decl_buffer((64,), "float32")
    B = decl_buffer((64,), "float32")
    r1.buffer_map[A] = B
    # r2 must not see r1's mutation
    assert len(r2.buffer_map) == 0


def test_buffer_replacer_replaces_strides_and_elem_offset():
    """Vars in buffer strides/elem_offset must be replaced, not passed through."""
    from tvm.tirx import BufferStore, Var
    from tvm.tirx.transform.common import BufferReplacer

    n = Var("n", "int32")
    m = Var("m", "int32")
    A = decl_buffer((64,), "float32", strides=[n], elem_offset=n)
    store = BufferStore(A, 1.0, [0])

    new = BufferReplacer(var_map={n: m})(store)
    assert new.buffer.strides[0].same_as(m)
    assert new.buffer.elem_offset.same_as(m)


def test_gemm_async_partial_scale_factor():
    """Regression test for F7: gemm_async must reject partial scale factors."""
    from tvm.tirx.script.builder.tirx import gemm_async

    A = decl_buffer((64, 64), "float16", scope="shared")
    B = decl_buffer((64, 64), "float16", scope="shared")
    C = decl_buffer((64, 64), "float16", scope="shared")
    SF = decl_buffer((64,), "float16", scope="shared")

    with pytest.raises(ValueError, match="SFA and SFB must both be provided or both be None"):
        gemm_async(C[:, :], A[:, :], B[:, :], SFA=SF[:])

    with pytest.raises(ValueError, match="SFA and SFB must both be provided or both be None"):
        gemm_async(C[:, :], A[:, :], B[:, :], SFB=SF[:])


def test_op_call_config_mutated():
    """Test that StructuralMap updates Expr values inside TilePrimitiveCall.config.

    Regression test for B00004: lower_tirx_scope_ids creates new let-vars for
    scope IDs and uses StructuralMap to replace them in the body. Without visiting
    TilePrimitiveCall.config, the config retains stale var references.
    """

    @T.prim_func
    def op_call_with_config(A: T.Buffer((10,), "int32"), B: T.Buffer((10,), "int32")):
        T.device_entry()
        Tx.add(A, B, 1.0)

    op_call_stmt = op_call_with_config.body.body
    assert isinstance(op_call_stmt, tir.TilePrimitiveCall)

    # Create TilePrimitiveCall with a Var in the config
    old_var = Var("old_scope_id", "int32")
    new_var = Var("new_let_var", "int32")
    new_config = dict(op_call_stmt.config)
    new_config["cta_mask"] = old_var + tir.IntImm("int32", 5)
    op_call_with_var = tir.TilePrimitiveCall(
        *op_call_stmt.args, op=op_call_stmt.op, config=new_config
    )

    # StructuralMap old_var -> new_var
    def replace_var(var):
        return new_var if var.same_as(old_var) else var

    result = tvm_ffi.structural_map(op_call_with_var, (Var, replace_var), order="pre")
    assert isinstance(result, tir.TilePrimitiveCall)

    # The config value should now reference new_var, not old_var
    cta_mask_expr = result.config["cta_mask"]
    assert isinstance(cta_mask_expr, tir.Add)
    assert isinstance(cta_mask_expr.a, tir.Var)
    assert cta_mask_expr.a.name == "new_let_var", (
        f"Expected 'new_let_var' after substitution, got '{cta_mask_expr.a.name}'. "
        "StructuralMap should visit Expr values in TilePrimitiveCall.config."
    )


def test_op_call_nested_config_visited_and_substituted():
    """Nested selector arrays participate in the core visitor and mutator."""

    @T.prim_func
    def selector(
        A: T.Buffer((8,), "float16"),
        B: T.Buffer((8,), "float16"),
        C: T.Buffer((8,), "float16"),
        flag: T.int32,
    ):
        Tx.copy_async(
            C[:],
            A[:],
            dispatch="tma_explicit",
            mbar=C.data,
            src_selector=[(flag != 0, B)],
        )

    op_call = selector.body
    assert isinstance(op_call, tir.TilePrimitiveCall)
    original_b = selector.params[1]
    seen = []
    tvm_ffi.structural_walk(selector.body, seen.append)
    assert any(isinstance(node, Var) and node.same_as(selector.params[-1]) for node in seen)
    undefined = tvm.tirx.analysis.undefined_vars(op_call)
    assert any(var.same_as(original_b) for var in undefined)

    replacement = Var("replacement", "int32")

    def replace_var(var):
        return replacement if var.same_as(selector.params[-1]) else var

    updated = tvm_ffi.structural_map(op_call, (Var, replace_var), order="pre")
    condition, candidate = updated.config["src_selector"][0]
    assert isinstance(condition, tir.NE)
    assert condition.a.same_as(replacement)
    assert candidate.same_as(original_b)


if __name__ == "__main__":
    test_copy()
    test_fill()
    test_gemm()
    test_buffer_replacer_no_shared_default()
    test_gemm_async_partial_scale_factor()
