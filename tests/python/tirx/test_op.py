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

from tvm.ir import Op, assert_structural_equal
from tvm.tirx.buffer import decl_tensor
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.tile_primitive import TilePrimitiveCall


def _test(op: str, *args):
    return TilePrimitiveCall(*args, op=Op.get("tirx.tile." + op), workspace={}, config={})


def test_copy():
    A = decl_tensor((64, 64), "float32", scope="global")
    A_sm = decl_tensor((64, 64), "float32", scope="shared")
    _test("copy", A[0:64, 0:64], A_sm[0:64, 0:64])


def test_fill():
    A = decl_tensor((64, 64), "float32", scope="global")
    _test("fill", A[0:64, 0:64], 1.0)


def test_gemm():
    A = decl_tensor((64, 64), "float32", scope="global")
    B = decl_tensor((64, 64), "float32", scope="global")
    C = decl_tensor((64, 64), "float32", scope="global")
    D = decl_tensor((64, 64), "float32", scope="global")
    _test("gemm", D[:, :], A[:, :], B[:, :], C[:, :], True, False, 1.0, 0.0)


def test_tile_primitive_call_pickle_roundtrip():
    """TilePrimitiveCall reflection must provide a deserialization creator."""
    A = decl_tensor((64,), "float32", scope="local")
    workspace = decl_tensor((16,), "float32", scope="shared")
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


@pytest.mark.parametrize("direct_ffi", [False, True])
def test_tile_expression_fields_preserve_optional_slots(direct_ffi):
    from tvm.ir import Expr, StringImm, Tuple
    from tvm.tirx import IntImm, _ffi_api

    A = decl_tensor((8,), "float32")
    call = TilePrimitiveCall(
        A[:],
        A[:],
        None,
        None,
        op=Op.get("tirx.tile.sqrt"),
        config={"hint": "typed", "optional": None, "gather4": [0, 1, 2, 3]},
    )
    if direct_ffi:
        call = _ffi_api.TilePrimitiveCall(
            call.op, call.args, call.workspace, call.config, call.dispatch, call.scope
        )
    assert len(call.args) == 4
    assert call.args[2] is None and call.args[3] is None
    assert isinstance(call.args[0], Expr)
    assert isinstance(call.config["hint"], StringImm)
    assert "optional" in call.config and call.config["optional"] is None
    assert isinstance(call.config["gather4"], Tuple)
    assert all(isinstance(value, IntImm) for value in call.config["gather4"])
    assert_structural_equal(pickle.loads(pickle.dumps(call)), call, map_free_vars=True)


def test_tile_expression_axes_stay_in_place():
    from tvm.ir import Tuple
    from tvm.tirx.operator.tile_primitive.ops import Sum

    A = decl_tensor((8, 8), "float32")
    B = decl_tensor((8,), "float32")
    call = Sum(B[:], A[:, :], [1], False)
    assert len(call.args) == 4
    assert isinstance(call.reduce_axes, Tuple)
    assert int(call.reduce_axes[0]) == 1
    assert call.accum.ty.dtype == "bool" and not call.accum.value


def test_tile_expression_fields_reject_nonexpressions():
    from tvm.tirx import Evaluate

    with pytest.raises((TypeError, ValueError)):
        _test("fill", Evaluate(0))


@pytest.mark.parametrize("mma_m", [True, False, 128.0])
def test_typed_mma_config_rejects_noninteger_dimensions(mma_m):
    from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import _get_explicit_mma_tile

    call = TilePrimitiveCall(
        op=Op.get("tirx.tile.gemm_async"), config={"mma_m": mma_m, "mma_n": 64}
    )
    with pytest.raises(ValueError, match="positive integer"):
        _get_explicit_mma_tile(call.config)


def test_typed_literal_and_sequence_config_consumers():
    from tvm.backend.cuda.tile_primitive.copy.vec_forced import _ld_cache_config
    from tvm.backend.cuda.tile_primitive.copy_async.tcgen05_cp import _resolve_cp_shape
    from tvm.backend.cuda.tile_primitive.copy_async.tma import (
        _normalize_cache_hint,
        _normalize_gather4,
        _normalize_l2_promotion,
        _normalize_src_selector,
    )
    from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import _get_explicit_mma_tile
    from tvm.tirx import Var

    A = decl_tensor((8,), "float32", scope="global")
    condition = Var("condition", "bool")
    call = TilePrimitiveCall(
        op=Op.get("tirx.tile.copy_async"),
        config={
            "cache": "nc",
            "l1_evict": "L1::no_allocate",
            "empty": "",
            "l2": "L2::256B",
            "gather4": [0, 1, 2, 3],
            "src_selector": [(condition, A)],
            "mma_m": 128,
            "mma_n": 64,
        },
    )
    assert _ld_cache_config(call) == ("nc", {"l1_evict": "L1::no_allocate"})
    assert _normalize_cache_hint(call.config["cache"]) == ("nc", None)
    assert _normalize_cache_hint(call.config["empty"]) == ("", None)
    assert _normalize_l2_promotion(call.config["l2"]) == 3
    assert tuple(int(v) for v in _normalize_gather4(call.config["gather4"])) == (0, 1, 2, 3)
    selectors = _normalize_src_selector(call.config["src_selector"])
    assert selectors[0][0].same_as(condition) and selectors[0][1].same_as(A)
    assert _get_explicit_mma_tile(call.config) == (128, 64)
    shape_call = call.replace(config={"shape": "128x256b", "multicast": ""})
    assert _resolve_cp_shape(shape_call) == ("128x256b", "")


@pytest.mark.parametrize("axes", [(), (0,), (-1,)])
def test_typed_reduction_axes_script_roundtrip(axes):
    import tvm
    from tvm.script import tirx as T

    @T.prim_func
    def func(A: T.Tensor((8, 8), "float32"), B: T.Tensor((8,), "float32")):
        T.tile.sum(B[:], A[:, :], axes=axes)

    assert_structural_equal(func, tvm.script.from_source(func.script(), extra_vars={"T": T}))


def test_tile_raw_ffi_scalar_and_null_conversion():
    from tvm.ir import StringImm
    from tvm.tirx import IntImm, _ffi_api

    call = _ffi_api.TilePrimitiveCall(
        Op.get("tirx.tile.fill"),
        [1, False, None],
        {},
        {"hint": "raw", "flag": True, "optional": None},
        None,
        ExecScope("thread"),
    )
    assert isinstance(call.args[0], IntImm) and call.args[0].value == 1
    assert call.args[1].ty.dtype == "bool" and not call.args[1].value
    assert call.args[2] is None
    assert isinstance(call.config["hint"], StringImm) and call.config["hint"].value == "raw"
    assert call.config["flag"].ty.dtype == "bool" and call.config["flag"].value
    assert call.config["optional"] is None


def test_buffer_replacer_no_shared_default():
    """Regression test for F4: BufferReplacer default dicts must not be shared."""
    from tvm.tirx.transform.common import BufferReplacer

    r1 = BufferReplacer()
    r2 = BufferReplacer()
    A = decl_tensor((64,), "float32")
    B = decl_tensor((64,), "float32")
    r1.buffer_map[A] = B
    # r2 must not see r1's mutation
    assert len(r2.buffer_map) == 0


def test_buffer_replacer_replaces_strides_and_elem_offset():
    """Vars in buffer strides/elem_offset must be replaced, not passed through."""
    from tvm.tirx import BufferStore, Var
    from tvm.tirx.transform.common import BufferReplacer

    n = Var("n", "int32")
    m = Var("m", "int32")
    A = decl_tensor((64,), "float32", strides=[n], elem_offset=n)
    store = BufferStore(A, 1.0, [0])

    new = BufferReplacer(var_map={n: m})(store)
    assert new.buffer.strides[0].same_as(m)
    assert new.buffer.elem_offset.same_as(m)


def test_gemm_async_partial_scale_factor():
    """Regression test for F7: gemm_async must reject partial scale factors."""
    from tvm.tirx.script.ir_builder.tirx import gemm_async

    A = decl_tensor((64, 64), "float16", scope="shared")
    B = decl_tensor((64, 64), "float16", scope="shared")
    C = decl_tensor((64, 64), "float16", scope="shared")
    SF = decl_tensor((64,), "float16", scope="shared")

    with pytest.raises(ValueError, match="SFA and SFB must both be provided or both be None"):
        gemm_async(C[:, :], A[:, :], B[:, :], SFA=SF[:])

    with pytest.raises(ValueError, match="SFA and SFB must both be provided or both be None"):
        gemm_async(C[:, :], A[:, :], B[:, :], SFB=SF[:])


if __name__ == "__main__":
    test_copy()
    test_fill()
    test_gemm()
    test_buffer_replacer_no_shared_default()
    test_gemm_async_partial_scale_factor()
