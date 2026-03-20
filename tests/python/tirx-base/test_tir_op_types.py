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
import tvm
import tvm.testing
from tvm import tirx


def test_tir_op_tvm_tuple():
    x = tirx.Var("x", dtype="float32")
    y = tirx.Var("y", dtype="float32")
    z = tirx.Var("z", dtype="float32")
    expr = tirx.tvm_tuple(x, y, z, 1, 2, 3)
    assert expr.op.name == "tirx.tvm_tuple"


def test_tir_op_tvm_struct_get():
    x = tirx.Var("x", dtype="handle")
    expr = tirx.tvm_struct_get(x, 1, 2, dtype="int32")
    assert expr.op.name == "tirx.tvm_struct_get"


def test_tir_op_tvm_struct_set():
    x = tirx.Var("x", dtype="handle")
    expr = tirx.tvm_struct_set(x, 1, 2, 3)
    assert expr.op.name == "tirx.tvm_struct_set"


def test_tir_op_address_of():
    buffer = tirx.decl_buffer((128), "float32")
    expr = tirx.address_of(buffer[0])
    assert expr.op.name == "tirx.address_of"


def test_tir_op_lookup_param():
    expr = tirx.lookup_param("p0")
    assert expr.op.name == "tirx.lookup_param"


def test_tir_op_reinterpret():
    x = tirx.Var("x", dtype="int32")
    expr = tirx.reinterpret("float32", x)
    assert expr.op.name == "tirx.reinterpret"


def test_tir_op_isnullptr():
    x = tirx.Var("x", dtype="int32")
    expr = tirx.isnullptr(x)
    assert expr.op.name == "tirx.isnullptr"


def test_tir_op_call_assume():
    x = tirx.Var("x", dtype="int32")
    expr = tirx.assume(cond=x)
    assert expr.op.name == "tirx.assume"


def test_tir_op_call_undef():
    expr = tirx.undef()
    assert expr.op.name == "tirx.undef"


def test_tir_op_call_likely():
    x = tirx.Var("x", dtype="int32")
    expr = tirx.likely(cond=x)
    assert expr.op.name == "tirx.likely"


def test_tir_op_tvm_thread_allreduce():
    x = tirx.Var("x", "int32")
    buffer = tirx.decl_buffer((128), "float32")
    y = tirx.Var("y", "handle")
    z = tirx.Var("z", "int32")
    expr = tirx.tvm_thread_allreduce(x, buffer[0], True, y, z)
    assert expr.op.name == "tirx.tvm_thread_allreduce"


def test_tir_op_type_annotation():
    expr = tirx.type_annotation("int32")
    assert expr.op.name == "tirx.type_annotation"


def test_tir_op_tvm_access_ptr():
    buffer = tirx.decl_buffer((128), "float32")
    expr = tirx.tvm_access_ptr("float32", buffer.data, 0, 1, 2)
    assert expr.op.name == "tirx.tvm_access_ptr"


def test_tir_op_tvm_throw_last_error():
    expr = tirx.tvm_throw_last_error()
    assert expr.op.name == "tirx.tvm_throw_last_error"


def test_tir_op_tvm_load_matrix_sync():
    buffer = tirx.decl_buffer((16, 16), "float32")
    x = tirx.Var("x", "handle")
    expr = tirx.tvm_load_matrix_sync(buffer.data, 16, 16, 16, 0, x, 128, "row_major")
    assert expr.op.name == "tirx.tvm_load_matrix_sync"


def test_tir_op_tvm_store_matrix_sync():
    buffer = tirx.decl_buffer((16, 16), "float32")
    x = tirx.Var("x", "handle")
    expr = tirx.tvm_store_matrix_sync(buffer.data, 16, 16, 16, 0, x, 128, "row_major")
    assert expr.op.name == "tirx.tvm_store_matrix_sync"


def test_tir_op_tvm_mma_sync():
    buffer_0 = tirx.decl_buffer((16, 16), "float32")
    buffer_1 = tirx.decl_buffer((16, 16), "float32")
    buffer_2 = tirx.decl_buffer((16, 16), "float32")
    buffer_3 = tirx.decl_buffer((16, 16), "float32")
    expr = tirx.tvm_mma_sync(buffer_0.data, 0, buffer_1.data, 0, buffer_2.data, 0, buffer_3.data, 0)
    assert expr.op.name == "tirx.tvm_mma_sync"


def test_tir_op_tvm_bmma_sync():
    buffer_0 = tirx.decl_buffer((16, 16), "float32")
    buffer_1 = tirx.decl_buffer((16, 16), "float32")
    buffer_2 = tirx.decl_buffer((16, 16), "float32")
    buffer_3 = tirx.decl_buffer((16, 16), "float32")
    expr = tirx.tvm_bmma_sync(
        buffer_0.data, 0, buffer_1.data, 0, buffer_2.data, 0, buffer_3.data, 0
    )
    assert expr.op.name == "tirx.tvm_bmma_sync"


def test_tir_op_tvm_fill_fragment():
    buffer = tirx.decl_buffer((16, 16), "float32")
    expr = tirx.tvm_fill_fragment(buffer.data, 16, 16, 16, 0, 0)
    assert expr.op.name == "tirx.tvm_fill_fragment"


def test_tir_op_ptx_mma():
    buffer_a = tirx.decl_buffer([32], "int4", scope="local")
    buffer_b = tirx.decl_buffer([16], "uint4", scope="local")
    buffer_c = tirx.decl_buffer([4], "int32", scope="local")
    expr = tirx.ptx_mma(
        "int32",
        "m8n8k32",
        "row",
        "col",
        "int4",
        "uint4",
        "int32",
        buffer_a.data,
        0,
        buffer_b.data,
        0,
        buffer_c.data,
        0,
        False,
    )
    assert expr.op.name == "tirx.ptx_mma"


def test_tir_op_ptx_mma_sp():
    buffer_a = tirx.decl_buffer([32], "int4", scope="local")
    buffer_b = tirx.decl_buffer([16], "uint4", scope="local")
    buffer_c = tirx.decl_buffer([4], "int32", scope="local")
    buffer_d = tirx.decl_buffer([1], "uint32", scope="local")
    expr = tirx.ptx_mma_sp(
        "int32",
        "m8n8k32",
        "row",
        "col",
        "int4",
        "uint4",
        "int32",
        buffer_a.data,
        0,
        buffer_b.data,
        0,
        buffer_c.data,
        0,
        buffer_d.data,
        0,
        0,
        False,
    )
    assert expr.op.name == "tirx.ptx_mma_sp"


def test_tir_op_mma_store():
    x = tirx.Var("x", dtype="int32")
    y = tirx.Var("y", dtype="int32")
    buffer_w = tirx.decl_buffer([16, 8], dtype="int32", scope="warp", offset_factor=1)
    buffer = tirx.decl_buffer(
        [16, 16], dtype="int32", scope="global", offset_factor=1, strides=[x, y]
    )
    expr = tirx.mma_store(
        "int32",
        16,
        16,
        buffer.access_ptr("w"),
        buffer_w.data,
        buffer_w.elem_offset,
        x,
    )
    assert expr.op.name == "tirx.mma_store"


def test_tir_op_mma_fill():
    buffer_w = tirx.decl_buffer([16, 8], dtype="int32", scope="warp", offset_factor=1)
    expr = tirx.mma_fill("int32", 8, buffer_w.data, buffer_w.elem_offset)
    assert expr.op.name == "tirx.mma_fill"


def test_op_ptx_ldmatrix():
    buffer_shared = tirx.decl_buffer([16, 16], "float16", scope="shared")
    buffer_local = tirx.decl_buffer([8], "float16", scope="local")
    expr = tirx.ptx_ldmatrix(
        "float16", False, 4, ".b16", buffer_local.data, 0, buffer_shared.data, 0
    )
    assert expr.op.name == "tirx.ptx_ldmatrix"


def test_op_ptx_cp_async():
    buffer_shared = tirx.decl_buffer([16, 16], "float16", scope="shared")
    buffer_local = tirx.decl_buffer([8], "float16", scope="local")
    expr = tirx.ptx_cp_async("float16", buffer_shared.data, 0, buffer_local.data, 0, 16)
    assert expr.op.name == "tirx.ptx_cp_async"


def test_op_ptx_cp_async_bulk():
    buffer_shared = tirx.decl_buffer([16, 16], "float16", scope="shared")
    buffer_local = tirx.decl_buffer([8], "float16", scope="local")
    expr = tirx.ptx_cp_async_bulk("float16", buffer_shared.data, 0, buffer_local.data, 0, 16, 0)
    assert expr.op.name == "tirx.ptx_cp_async_bulk"


def test_op_ptx_commit_group():
    expr = tirx.ptx_commit_group()
    assert expr.op.name == "tirx.ptx_commit_group"


def test_op_ptx_wait_group():
    expr = tirx.ptx_wait_group(8)
    assert expr.op.name == "tirx.ptx_wait_group"


def test_op_ptx_cp_async_barrier():
    expr = tirx.ptx_cp_async_barrier(0)
    assert expr.op.name == "tirx.ptx_cp_async_barrier"


def test_op_ptx_init_barrier_thread_count():
    expr = tirx.ptx_init_barrier_thread_count(0, 32)
    assert expr.op.name == "tirx.ptx_init_barrier_thread_count"


def test_op_ptx_arrive_barrier():
    expr = tirx.ptx_arrive_barrier(0)
    assert expr.op.name == "tirx.ptx_arrive_barrier"


def test_op_ptx_arrive_barrier_expect_tx():
    expr = tirx.ptx_arrive_barrier_expect_tx(0, 32)
    assert expr.op.name == "tirx.ptx_arrive_barrier_expect_tx"


def test_op_ptx_wait_barrier():
    expr = tirx.ptx_wait_barrier(0)
    assert expr.op.name == "tirx.ptx_wait_barrier"


def test_op_create_barriers():
    expr = tirx.create_barriers(16)
    assert expr.op.name == "tirx.create_barriers"


def test_tir_op_vectorlow():
    buffer = tirx.decl_buffer((4, 4), "int8", offset_factor=1)
    vec = buffer.vload([0, 0], dtype="int8x16")
    expr = tirx.vectorlow("int8x8", vec)
    assert expr.op.name == "tirx.vectorlow"


def test_tir_op_vectorhigh():
    buffer = tirx.decl_buffer((4, 4), "int8", offset_factor=1)
    vec = buffer.vload([0, 0], dtype="int8x16")
    expr = tirx.vectorhigh("int8x8", vec)
    assert expr.op.name == "tirx.vectorhigh"


def test_tir_op_dp4a():
    vec1 = tirx.Var("vec1", dtype="int8x4")
    vec2 = tirx.Var("vec2", dtype="int8x4")
    acc = tirx.Var("acc", dtype="int32")
    expr = tirx.dp4a(vec1, vec2, acc)
    assert expr.op.name == "tirx.dp4a"


def test_tir_op_vectorcombine():
    buffer = tirx.decl_buffer((4, 4), "int8", offset_factor=1)
    vec = buffer.vload([0, 0], dtype="int8x16")
    expr = tirx.vectorcombine("int8x8", vec, vec)
    assert expr.op.name == "tirx.vectorcombine"


def test_tir_op_shift_left():
    x = tirx.Var("x", dtype="int32")
    y = tirx.Var("x", dtype="int32")
    expr = tirx.shift_left(x, y)
    assert expr.op.name == "tirx.shift_left"


def test_tir_op_shift_right():
    x = tirx.Var("x", dtype="int32")
    y = tirx.Var("x", dtype="int32")
    expr = tirx.shift_right(x, y)
    assert expr.op.name == "tirx.shift_right"


def test_tir_op_bitwise():
    x = tirx.Var("x", dtype="int32")
    y = tirx.Var("y", dtype="int32")
    expr = tirx.bitwise_and(x, y)
    assert expr.op.name == "tirx.bitwise_and"
    expr = tirx.bitwise_or(x, y)
    assert expr.op.name == "tirx.bitwise_or"
    expr = tirx.bitwise_not(x)
    assert expr.op.name == "tirx.bitwise_not"
    expr = tirx.bitwise_xor(x, y)
    assert expr.op.name == "tirx.bitwise_xor"


def test_tir_op_TVMBackendAllocWorkspace():
    expr = tirx.TVMBackendAllocWorkspace(0, 1, 2, 3, 4)
    assert expr.op.name == "tirx.TVMBackendAllocWorkspace"


def test_tir_op_TVMBackendFreeWorkspace():
    buffer = tirx.decl_buffer((128), "float32")
    expr = tirx.TVMBackendFreeWorkspace(0, 1, buffer.data)
    assert expr.op.name == "tirx.TVMBackendFreeWorkspace"


if __name__ == "__main__":
    tvm.testing.main()
