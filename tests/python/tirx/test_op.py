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
import pytest

import tvm
from tvm.ir import Op
from tvm.script import tirx as T
from tvm.script import tirx as Tx
from tvm.tirx.buffer import decl_buffer
from tvm.tirx.stmt import TilePrimitiveCall


def _test(op: str, *args):
    return TilePrimitiveCall(*args, op=Op.get("tirx." + op), workspace={}, config={})


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


def test_generic_op_creates_op():
    """GenericOp auto-registers unknown ops."""
    from tvm.tirx.operator.tile_primitive.ops import GenericOp

    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="global")

    op_call = GenericOp(B[0:64], A[0:64], op_name="my_custom_op_1")
    assert op_call.op == Op.get("tirx.my_custom_op_1")
    assert len(op_call.args) == 2


def test_generic_op_reuses_registered_op():
    """GenericOp reuses already-registered ops without error."""
    from tvm.tirx.operator.tile_primitive.ops import GenericOp

    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="global")

    # Create twice with same name — should not error
    op1 = GenericOp(B[0:64], A[0:64], op_name="my_custom_op_2")
    op2 = GenericOp(B[0:64], A[0:64], op_name="my_custom_op_2")
    assert op1.op == op2.op


def test_generic_op_with_existing_tirx_op():
    """GenericOp works with already-registered tirx ops (e.g., tirx.copy)."""
    from tvm.tirx.operator.tile_primitive.ops import GenericOp

    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="global")

    op_call = GenericOp(B[0:64], A[0:64], op_name="copy")
    assert op_call.op == Op.get("tirx.copy")


def test_tx_dynamic_op_module_getattr():
    """Tx.some_undefined_op resolves via module __getattr__."""
    fn = Tx.my_dynamic_test_op
    assert callable(fn)
    assert fn.__name__ == "my_dynamic_test_op"


def test_tx_dynamic_op_in_prim_func():
    """Tx.copy_and_cast(...) works inside a prim_func without pre-registration."""

    @T.prim_func
    def func(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [64], "float32", scope="global")
        B = T.match_buffer(B_ptr, [64], "float16", scope="global")
        with T.kernel():
            Tx.copy_and_cast(B, A)

    # Walk IR to find TilePrimitiveCall with op="tirx.copy_and_cast"
    found = [False]

    def visit(stmt):
        if isinstance(stmt, TilePrimitiveCall) and stmt.op == Op.get("tirx.copy_and_cast"):
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected TilePrimitiveCall with tirx.copy_and_cast not found"


def test_tx_dynamic_op_with_workspace():
    """Tx.some_op(..., workspace={...}) passes workspace to TilePrimitiveCall."""

    @T.prim_func
    def func(A_ptr: T.handle, B_ptr: T.handle, W_ptr: T.handle):
        A = T.match_buffer(A_ptr, [64], "float32", scope="global")
        B = T.match_buffer(B_ptr, [64], "float32", scope="global")
        W = T.match_buffer(W_ptr, [64], "float32", scope="shared")
        with T.kernel():
            Tx.custom_with_ws(B, A, workspace={"tmp": W})

    found = [False]

    def visit(stmt):
        if isinstance(stmt, TilePrimitiveCall) and stmt.op == Op.get("tirx.custom_with_ws"):
            assert "tmp" in stmt.workspace
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected TilePrimitiveCall with workspace not found"


def test_tx_existing_op_not_overridden():
    """Existing Tx.copy still dispatches to the registered copy op, not __getattr__."""

    @T.prim_func
    def func(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [64], "float32", scope="global")
        B = T.match_buffer(B_ptr, [64], "float32", scope="global")
        with T.kernel():
            Tx.copy(B, A)

    found = [False]

    def visit(stmt):
        if isinstance(stmt, TilePrimitiveCall) and stmt.op == Op.get("tirx.copy"):
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected TilePrimitiveCall with tirx.copy not found"


def test_opcall_downcast_tolerant():
    """TilePrimitiveCall.downcast returns instance as-is for unknown ops."""
    from tvm.tirx.operator.tile_primitive.ops import GenericOp

    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="global")

    op_call = GenericOp(B[0:64], A[0:64], op_name="totally_unknown_op")
    # downcast should not raise
    result = TilePrimitiveCall.downcast(op_call)
    assert result is not None


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


def test_permute_dims_buffer_property():
    """Regression test for F2: PermuteDims.buffer should return args[0], not recurse."""
    from tvm.tirx.operator.tile_primitive.ops import PermuteDims

    A = decl_buffer((64, 64), "float32", scope="global")
    pd = PermuteDims(A[0:64, 0:64], [1, 0])
    # This would stack overflow before the fix
    buf = pd.buffer
    assert buf is not None


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


if __name__ == "__main__":
    test_copy()
    test_fill()
    test_gemm()
    test_generic_op_creates_op()
    test_generic_op_reuses_registered_op()
    test_generic_op_with_existing_tirx_op()
    test_tx_dynamic_op_module_getattr()
    test_tx_dynamic_op_in_prim_func()
    test_tx_dynamic_op_with_workspace()
    test_tx_existing_op_not_overridden()
    test_opcall_downcast_tolerant()
    test_buffer_replacer_no_shared_default()
    test_permute_dims_buffer_property()
    test_gemm_async_partial_scale_factor()
