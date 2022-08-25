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
from tvm import tir


def test_tir_op_tvm_tuple():
    x = tir.Var("x", dtype="float32")
    y = tir.Var("y", dtype="float32")
    z = tir.Var("z", dtype="float32")
    expr = tir.tvm_tuple(x, y, z, 1, 2, 3)
    assert expr.op.name == "tir.tvm_tuple"


def test_tir_op_tvm_struct_get():
    x = tir.Var("x", dtype="handle")
    expr = tir.tvm_struct_get(x, 1, 2, dtype="int32")
    assert expr.op.name == "tir.tvm_struct_get"


def test_tir_op_tvm_struct_set():
    x = tir.Var("x", dtype="handle")
    expr = tir.tvm_struct_set(x, 1, 2, 3)
    assert expr.op.name == "tir.tvm_struct_set"


def test_tir_op_address_of():
    buffer = tir.decl_buffer((128), "float32")
    expr = tir.address_of(buffer[0])
    assert expr.op.name == "tir.address_of"


def test_tir_op_lookup_param():
    expr = tir.lookup_param("p0")
    assert expr.op.name == "tir.lookup_param"


def test_tir_op_reinterpret():
    x = tir.Var("x", dtype="int32")
    expr = tir.reinterpret("float32", x)
    assert expr.op.name == "tir.reinterpret"


def test_tir_op_isnullptr():
    x = tir.Var("x", dtype="int32")
    expr = tir.isnullptr(x)
    assert expr.op.name == "tir.isnullptr"


def test_tir_op_call_assume():
    x = tir.Var("x", dtype="int32")
    expr = tir.assume(cond=x)
    assert expr.op.name == "tir.assume"


def test_tir_op_call_undef():
    expr = tir.undef()
    assert expr.op.name == "tir.undef"


def test_tir_op_call_likely():
    x = tir.Var("x", dtype="int32")
    expr = tir.likely(cond=x)
    assert expr.op.name == "tir.likely"


def test_tir_op_tvm_thread_allreduce():
    x = tir.Var("x", "int32")
    buffer = tir.decl_buffer((128), "float32")
    y = tir.Var("y", "handle")
    z = tir.Var("z", "int32")
    expr = tir.tvm_thread_allreduce(x, buffer[0], True, y, z)
    assert expr.op.name == "tir.tvm_thread_allreduce"


def test_tir_op_type_annotation():
    expr = tir.type_annotation("int32")
    assert expr.op.name == "tir.type_annotation"


def test_tir_op_tvm_access_ptr():
    buffer = tir.decl_buffer((128), "float32")
    expr = tir.tvm_access_ptr("float32", buffer.data, 0, 1, 2)
    assert expr.op.name == "tir.tvm_access_ptr"


def test_tir_op_tvm_throw_last_error():
    expr = tir.tvm_throw_last_error()
    assert expr.op.name == "tir.tvm_throw_last_error"


def test_tir_op_tvm_load_matrix_sync():
    buffer = tir.decl_buffer((16, 16), "float32")
    x = tir.Var("x", "handle")
    expr = tir.tvm_load_matrix_sync(buffer.data, 16, 16, 16, 0, x, 128, "row_major")
    assert expr.op.name == "tir.tvm_load_matrix_sync"


def test_tir_op_tvm_store_matrix_sync():
    buffer = tir.decl_buffer((16, 16), "float32")
    x = tir.Var("x", "handle")
    expr = tir.tvm_store_matrix_sync(buffer.data, 16, 16, 16, 0, x, 128, "row_major")
    assert expr.op.name == "tir.tvm_store_matrix_sync"


def test_tir_op_tvm_mma_sync():
    buffer_0 = tir.decl_buffer((16, 16), "float32")
    buffer_1 = tir.decl_buffer((16, 16), "float32")
    buffer_2 = tir.decl_buffer((16, 16), "float32")
    buffer_3 = tir.decl_buffer((16, 16), "float32")
    expr = tir.tvm_mma_sync(buffer_0.data, 0, buffer_1.data, 0, buffer_2.data, 0, buffer_3.data, 0)
    assert expr.op.name == "tir.tvm_mma_sync"


def test_tir_op_tvm_bmma_sync():
    buffer_0 = tir.decl_buffer((16, 16), "float32")
    buffer_1 = tir.decl_buffer((16, 16), "float32")
    buffer_2 = tir.decl_buffer((16, 16), "float32")
    buffer_3 = tir.decl_buffer((16, 16), "float32")
    expr = tir.tvm_bmma_sync(buffer_0.data, 0, buffer_1.data, 0, buffer_2.data, 0, buffer_3.data, 0)
    assert expr.op.name == "tir.tvm_bmma_sync"


def test_tir_op_tvm_fill_fragment():
    buffer = tir.decl_buffer((16, 16), "float32")
    expr = tir.tvm_fill_fragment(buffer.data, 16, 16, 16, 0, 0)
    assert expr.op.name == "tir.tvm_fill_fragment"


def test_tir_op_vectorlow():
    buffer = tir.decl_buffer((4, 4), "int8", offset_factor=1)
    vec = buffer.vload([0, 0], dtype="int8x16")
    expr = tir.vectorlow("int8x8", vec)
    assert expr.op.name == "tir.vectorlow"


def test_tir_op_vectorhigh():
    buffer = tir.decl_buffer((4, 4), "int8", offset_factor=1)
    vec = buffer.vload([0, 0], dtype="int8x16")
    expr = tir.vectorhigh("int8x8", vec)
    assert expr.op.name == "tir.vectorhigh"


def test_tir_op_vectorcombine():
    buffer = tir.decl_buffer((4, 4), "int8", offset_factor=1)
    vec = buffer.vload([0, 0], dtype="int8x16")
    expr = tir.vectorcombine("int8x8", vec, vec)
    assert expr.op.name == "tir.vectorcombine"


def test_tir_op_shift_left():
    x = tir.Var("x", dtype="int32")
    y = tir.Var("x", dtype="int32")
    expr = tir.shift_left(x, y)
    assert expr.op.name == "tir.shift_left"


def test_tir_op_shift_right():
    x = tir.Var("x", dtype="int32")
    y = tir.Var("x", dtype="int32")
    expr = tir.shift_right(x, y)
    assert expr.op.name == "tir.shift_right"


def test_tir_op_TVMBackendAllocWorkspace():
    expr = tir.TVMBackendAllocWorkspace(0, 1, 2, 3, 4)
    assert expr.op.name == "tir.TVMBackendAllocWorkspace"


def test_tir_op_TVMBackendFreeWorkspace():
    buffer = tir.decl_buffer((128), "float32")
    expr = tir.TVMBackendFreeWorkspace(0, 1, buffer.data)
    assert expr.op.name == "tir.TVMBackendFreeWorkspace"


if __name__ == "__main__":
    test_tir_op_tvm_tuple()
    test_tir_op_tvm_struct_get()
    test_tir_op_tvm_struct_set()
    test_tir_op_address_of()
    test_tir_op_lookup_param()
    test_tir_op_reinterpret()
    test_tir_op_isnullptr()
    test_tir_op_call_assume()
    test_tir_op_call_undef()
    test_tir_op_call_likely()
    test_tir_op_tvm_thread_allreduce()
    test_tir_op_type_annotation()
    test_tir_op_tvm_access_ptr()
    test_tir_op_tvm_throw_last_error()
    test_tir_op_tvm_load_matrix_sync(),
    test_tir_op_tvm_store_matrix_sync(),
    test_tir_op_tvm_mma_sync(),
    test_tir_op_tvm_bmma_sync(),
    test_tir_op_tvm_fill_fragment(),
    test_tir_op_vectorlow()
    test_tir_op_vectorhigh()
    test_tir_op_vectorcombine()
    test_tir_op_shift_left()
    test_tir_op_shift_right()
    test_tir_op_TVMBackendAllocWorkspace()
    test_tir_op_TVMBackendFreeWorkspace()
