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
# pylint: disable=missing-docstring
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import tirx
from tvm_tirx.backend.cuda import op as _cuda_op


def test_tir_op_tvm_struct_get():
    x = tirx.Var("x", ty="handle")
    expr = tirx.abi_field_get(x, 1, 2, ty="int32")
    assert expr.op.name == "tirx.abi_field_get"


def test_scalar_integer_signature():
    ptr = tirx.Var("ptr", ty="handle")
    for dtype in ("int8", "uint32", "int64", "uint64"):
        index = tirx.Var("index", dtype)
        assert tirx.abi_field_get(ptr, index, 2, ty="int32").args[1].same_as(index)
    assert tirx.abi_field_get(ptr, 1 << 40, 2, ty="int32").args[1].ty.dtype == "int64"
    for dtype in ("float32", "bool", "int32x4", "int32xvscalex4"):
        with pytest.raises(TypeError, match=rf"index.*expected `ir.IntExpr`.*\[ty={dtype}\]"):
            tirx.abi_field_get(ptr, tirx.Var("index", dtype), 2, ty="int32").validate()


def test_tir_op_tvm_struct_set():
    x = tirx.Var("x", ty="handle")
    expr = tirx.abi_field_set(x, 1, 2, 3)
    assert expr.op.name == "tirx.abi_field_set"


def test_tir_op_address_of():
    tensor = tirx.decl_tensor((128), "float32")
    expr = tirx.address_of(tensor[0])
    assert expr.op.name == "tirx.address_of"
    storage = tirx.Var("storage", tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("uint8"), "shared.dyn"))
    pooled_tensor = tirx.decl_tensor((128), "float32", data=storage, scope="shared.dyn")
    expected_ty = tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32"), "shared.dyn")
    assert tirx.address_of(pooled_tensor).ty == expected_ty
    assert tirx.address_of(pooled_tensor[0]).ty == expected_ty
    scalar_address = tirx.address_of(tirx.Var("value", "uint32"))
    assert scalar_address.ty == tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("uint32"))


def test_tir_op_reinterpret():
    x = tirx.Var("x", ty="int32")
    expr = tirx.reinterpret(x, ty="float32")
    assert expr.op.name == "tirx.reinterpret"
    with pytest.raises(TypeError, match="scalar 64-bit integer source"):
        tirx.reinterpret(x, ty="handle")
    pointer = tirx.reinterpret(tirx.Var("address", ty="uint64"), ty="handle")
    assert pointer.ty == tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("void"))


@pytest.mark.parametrize("op_name", ["call_extern", "call_pure_extern", "call_intrin"])
def test_result_type_keyword_matches_script_builder(op_name):
    from tvm_tirx.script import tirx as T

    x = tirx.Var("x", ty="float32")
    ty = tvm_tirx.ir.PrimType("float32")
    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("op.py"), 1, 0, 1, 8)
    name = "prim.exp" if op_name == "call_intrin" else "external_func"
    direct = getattr(tirx, op_name)(name, x, ty=ty, loc=loc)
    scripted = getattr(T, op_name)(name, x, ty=ty, loc=loc)
    assert direct.ty == ty
    assert direct.loc.same_as(loc)
    tvm_tirx.ir.assert_structural_equal(direct, scripted)
    with pytest.raises(TypeError):
        getattr(tirx, op_name)("float32", name, x)


@pytest.mark.parametrize("op_name", ["min_value", "max_value", "infinity"])
def test_primitive_result_type_keyword(op_name):
    from tvm_tirx.script import tirx as T

    ty = tvm_tirx.ir.PrimType("float32")
    direct = getattr(tirx, op_name)(ty=ty)
    scripted = getattr(T, op_name)(ty=ty)
    assert direct.ty == ty
    tvm_tirx.ir.assert_structural_equal(direct, scripted)
    with pytest.raises(TypeError):
        getattr(tirx, op_name)("float32")


def test_masked_memory_wrapper_parity_and_inference():
    from tvm_tirx.script import tirx as T

    tensor = tirx.decl_tensor((16,), "float32")
    index = tvm_tirx.ir.prim.Ramp(0, 1, 4)
    mask = tvm_tirx.ir.prim.Broadcast(True, 4)
    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("op.py"), 2, 0, 2, 8)
    loaded = tirx.op.masked_load(tensor, index, mask, loc=loc)
    assert loaded.ty == tvm_tirx.ir.PrimType("float32x4")
    assert loaded.loc.same_as(loc)
    tvm_tirx.ir.assert_structural_equal(loaded, T.masked_load(tensor, index, mask, loc=loc))
    explicit = tirx.op.masked_load(tensor, index, mask, ty=loaded.ty, loc=loc)
    tvm_tirx.ir.assert_structural_equal(loaded, explicit)
    stored = tirx.op.masked_store(tensor, loaded, index, mask, loc=loc)
    assert stored.ty == tvm_tirx.ir.PrimType("void")
    assert stored.loc.same_as(loc)
    tvm_tirx.ir.assert_structural_equal(
        stored, T.masked_store(tensor, loaded, index, mask, loc=loc)
    )


@pytest.mark.parametrize("ty", ["float32", tvm_tirx.ir.PrimType("float32")])
def test_cast_positional_and_keyword_result_type(ty):
    from tvm_tirx.script import tirx as T

    value = tirx.Var("x", ty="int32")
    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("op.py"), 3, 0, 3, 8)
    positional = T.cast(value, ty, loc=loc)
    keyword = T.cast(value, ty=ty, loc=loc)
    assert positional.ty == tvm_tirx.ir.PrimType("float32")
    assert positional.loc.same_as(loc)
    assert keyword.loc.same_as(loc)
    tvm_tirx.ir.assert_structural_equal(positional, keyword)
    assert T.cast(value, ty).loc.same_as(tvm_tirx.ir.UNKNOWN_LOC)
    assert tirx.reinterpret(value, ty=ty).ty == positional.ty
    with pytest.raises(TypeError):
        T.cast(value, ty, loc)
    with pytest.raises(TypeError):
        T.cast(value, ty, ty=ty)
    with pytest.raises(TypeError):
        T.cast(value)
    with pytest.raises(TypeError):
        tirx.reinterpret("float32", value)


def test_tir_op_isnullptr():
    x = tirx.Var("x", ty="int32")
    expr = tirx.isnullptr(x)
    assert expr.op.name == "tirx.isnullptr"


def test_tir_op_call_assume():
    x = tirx.Var("x", ty="int32")
    expr = tirx.assume(cond=x)
    assert expr.op.name == "prim.assume"


def test_tir_op_call_undef():
    expr = tirx.undef()
    assert expr.op.name == "tirx.undef"


def test_tir_op_call_likely():
    x = tirx.Var("x", ty="int32")
    expr = tirx.likely(cond=x)
    assert expr.op.name == "prim.likely"


def test_tir_op_tvm_thread_allreduce():
    tensor = tirx.decl_tensor((128,), "float32")
    result = tirx.decl_tensor((1,), "float32")
    axis = tirx.Var("thread", "int32")
    combine = tvm_tirx.ir.LambdaExpr(["float32", "float32"], lambda lhs, rhs: (lhs + rhs,))
    expr = tirx.gpu_thread_allreduce(
        combine, (tirx.const(0, "float32"),), (tensor[0],), True, (result[0],), (axis,)
    )
    assert expr.op.name == "tirx.gpu_thread_allreduce"


def test_tir_op_ptr_byte_offset():
    tensor = tirx.decl_tensor((128,), "float32", scope="shared")
    ty = tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("uint8"), "shared")
    offset_expr = tirx.ptr_byte_offset(tensor.data, 16, ty=ty)
    assert offset_expr.op.name == "tirx.ptr_byte_offset"
    assert offset_expr.ty == ty
    assert offset_expr.args[1] == 16


def test_tir_op_tvm_throw_last_error():
    expr = tirx.throw_last_error()
    assert expr.op.name == "tirx.throw_last_error"


def test_tir_op_tvm_load_matrix_sync():
    tensor = tirx.decl_tensor((16, 16), "float32")
    x = tirx.Var("x", "handle")
    expr = tvm_tirx.tirx.gpu_load_matrix_sync(tensor.data, 16, 16, 16, 0, x, 128, "row_major")
    assert expr.op.name == "tirx.gpu_load_matrix_sync"


def test_tir_op_tvm_store_matrix_sync():
    tensor = tirx.decl_tensor((16, 16), "float32")
    x = tirx.Var("x", "handle")
    expr = tvm_tirx.tirx.gpu_store_matrix_sync(tensor.data, 16, 16, 16, 0, x, 128, "row_major")
    assert expr.op.name == "tirx.gpu_store_matrix_sync"


def test_tir_op_tvm_mma_sync():
    tensor_0 = tirx.decl_tensor((16, 16), "float32")
    tensor_1 = tirx.decl_tensor((16, 16), "float32")
    tensor_2 = tirx.decl_tensor((16, 16), "float32")
    tensor_3 = tirx.decl_tensor((16, 16), "float32")
    expr = tvm_tirx.tirx.gpu_mma_sync(
        tensor_0.data, 0, tensor_1.data, 0, tensor_2.data, 0, tensor_3.data, 0
    )
    assert expr.op.name == "tirx.gpu_mma_sync"


def test_tir_op_tvm_bmma_sync():
    tensor_0 = tirx.decl_tensor((16, 16), "float32")
    tensor_1 = tirx.decl_tensor((16, 16), "float32")
    tensor_2 = tirx.decl_tensor((16, 16), "float32")
    tensor_3 = tirx.decl_tensor((16, 16), "float32")
    expr = tvm_tirx.backend.cuda.op.bmma_sync(
        tensor_0.data, 0, tensor_1.data, 0, tensor_2.data, 0, tensor_3.data, 0
    )
    assert expr.op.name == "tirx.cuda.bmma_sync"


def test_tir_op_tvm_fill_fragment():
    tensor = tirx.decl_tensor((16, 16), "float32")
    expr = tvm_tirx.tirx.gpu_fill_fragment(tensor.data, 16, 16, 16, 0, 0)
    assert expr.op.name == "tirx.gpu_fill_fragment"


def test_tir_op_ptx_mma():
    tensor_a = tirx.decl_tensor([32], "int4", scope="local")
    tensor_b = tirx.decl_tensor([16], "uint4", scope="local")
    tensor_c = tirx.decl_tensor([4], "int32", scope="local")
    expr = _cuda_op.ptx_legacy_mma(
        "m8n8k32",
        "row",
        "col",
        "int4",
        "uint4",
        "int32",
        tensor_a.data,
        0,
        tensor_b.data,
        0,
        tensor_c.data,
        0,
        False,
    )
    assert expr.op.name == "tirx.ptx_legacy.mma"


def test_tir_op_mma_store():
    x = tirx.Var("x", ty="int32")
    y = tirx.Var("y", ty="int32")
    tensor_w = tirx.decl_tensor([16, 8], dtype="int32", scope="warp", offset_factor=1)
    tensor = tirx.decl_tensor(
        [16, 16], dtype="int32", scope="global", offset_factor=1, strides=[x, y]
    )
    expr = _cuda_op.mma_store(
        16, 16, tensor.ptr_to([0, 0]), tensor_w.data, tensor_w.ty.elem_offset, x, ty="int32"
    )
    assert expr.op.name == "tirx.cuda.mma_store"


def test_tir_op_mma_fill():
    tensor_w = tirx.decl_tensor([16, 8], dtype="int32", scope="warp", offset_factor=1)
    expr = _cuda_op.mma_fill(8, tensor_w.data, tensor_w.ty.elem_offset, ty="int32")
    assert expr.op.name == "tirx.cuda.mma_fill"


def test_op_ptx_cp_async():
    tensor_shared = tirx.decl_tensor([16, 16], "float16", scope="shared")
    tensor_local = tirx.decl_tensor([8], "float16", scope="local")
    expr = _cuda_op.ptx_cp_async_legacy(tensor_shared.data, 0, tensor_local.data, 0, 16)
    assert expr.op.name == "tirx.s_tir.cp_async_raw"

    inner_dst = tirx.ptr_byte_offset(tensor_shared.data, 4, ty=tensor_shared.data.ty)
    inner_src = tirx.ptr_byte_offset(tensor_local.data, 8, ty=tensor_local.data.ty)
    expr = _cuda_op.ptx_cp_async_legacy(inner_dst, 3, inner_src, 5, 16, elem_dtype="float16")
    # Element offsets are converted to bytes while preserving each pointer's scope.
    for pointer, inner, expected_offset in zip(
        (expr.args[0], expr.args[2]), (inner_dst, inner_src), (6, 10)
    ):
        assert pointer.op.name == "tirx.ptr_byte_offset"
        tvm_tirx.ir.assert_structural_equal(pointer.args[0], inner)
        assert pointer.ty == inner.ty
        assert int(tvm_tirx.sym.Analyzer().simplify(pointer.args[1])) == expected_offset


def test_tir_op_vector_low():
    tensor = tirx.decl_tensor((4, 4), "int8", offset_factor=1)
    vec = tensor[0, tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 16)]
    expr = tirx.vector_low(vec, ty="int8x8")
    assert expr.op.name == "tirx.vector_low"


def test_tir_op_vector_high():
    tensor = tirx.decl_tensor((4, 4), "int8", offset_factor=1)
    vec = tensor[0, tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 16)]
    expr = tirx.vector_high(vec, ty="int8x8")
    assert expr.op.name == "tirx.vector_high"


def test_tir_op_dp4a():
    vec1 = tirx.Var("vec1", ty="int8x4")
    vec2 = tirx.Var("vec2", ty="int8x4")
    acc = tirx.Var("acc", ty="int32")
    expr = tirx.gpu_dp4a(vec1, vec2, acc)
    assert expr.op.name == "tirx.gpu_dp4a"


def test_tir_op_vector_combine():
    tensor = tirx.decl_tensor((4, 4), "int8", offset_factor=1)
    vec = tensor[0, tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 16)]
    expr = tirx.vector_combine(vec, vec, ty="int8x8")
    assert expr.op.name == "tirx.vector_combine"


def test_tir_op_shift_left():
    x = tirx.Var("x", ty="int32")
    y = tirx.Var("x", ty="int32")
    expr = tirx.shift_left(x, y)
    assert isinstance(expr, tirx.LShift)


def test_tir_op_shift_right():
    x = tirx.Var("x", ty="int32")
    y = tirx.Var("x", ty="int32")
    expr = tirx.shift_right(x, y)
    assert isinstance(expr, tirx.RShift)


def test_tir_op_bitwise():
    x = tirx.Var("x", ty="int32")
    y = tirx.Var("y", ty="int32")
    expr = tirx.bitwise_and(x, y)
    assert isinstance(expr, tirx.BitwiseAnd)
    expr = tirx.bitwise_or(x, y)
    assert isinstance(expr, tirx.BitwiseOr)
    expr = tirx.bitwise_not(x)
    assert isinstance(expr, tirx.BitwiseNot)
    expr = tirx.bitwise_xor(x, y)
    assert isinstance(expr, tirx.BitwiseXor)


def test_tir_op_TVMBackendAllocWorkspace():
    expr = tirx.alloc_workspace(0, 1, 2, 3, 4)
    assert expr.op.name == "tirx.alloc_workspace"


def test_tir_op_TVMBackendFreeWorkspace():
    tensor = tirx.decl_tensor((128), "float32")
    expr = tirx.free_workspace(0, 1, tensor.data)
    assert expr.op.name == "tirx.free_workspace"


def test_ptx_ops_have_variadic_call_signature():
    from tvm_tirx.backend.cuda.ptx import TABLE

    op = tvm_tirx.ir.Op.get(next(iter(TABLE.values())).op_name)
    assert not op.args_info
    assert op.var_args_info.name == "operands"
    tvm_tirx.ir.Call(None, op, []).validate()
    tvm_tirx.ir.Call(None, op, [tvm_tirx.ir.Var("x", tvm_tirx.ir.AnyType())]).validate()


if __name__ == "__main__":
    tvm_tirx.testing.main()
