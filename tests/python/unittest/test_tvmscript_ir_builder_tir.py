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
# pylint: disable=invalid-name, missing-docstring
"""Unittests for tvm.script.ir_builder.tir"""
import pytest
import tvm.testing
import tvm
from tvm import tir
from tvm.script.ir_builder import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.ir.base import assert_structural_equal


def test_ir_builder_tir_primfunc_base():
    with IRBuilder() as ib:
        with T.prim_func():
            T.evaluate(0)
    # the prim_func generated by IRBuilder
    prim_func_actual = ib.get()

    # the expected prim_func
    prim_func_expected = tir.PrimFunc(
        params=[],
        body=tir.Evaluate(0),
        ret_type=None,
        buffer_map=None,
        preflattened_buffer_map=None,
        attrs=None,
    )
    # Check if the generated ir is expected
    assert_structural_equal(prim_func_actual, prim_func_expected, map_free_vars=True)


def test_ir_builder_tir_primfunc_complete():
    with IRBuilder() as ib:
        with T.prim_func():
            T.arg("a", T.handle())
            T.arg("b", T.var("int64"))
            T.arg("c", T.buffer_decl((128, 128), "float32"))
            d = T.arg("d", T.handle())
            e = T.arg("e", T.buffer_decl((1024,), "int8"))
            T.func_attr({"key": "value"})
            T.func_ret(tvm.ir.PrimType("int64"))
            buffer_d = T.match_buffer(d, (64, 64), "int64")
            T.preflattened_buffer(e, (32, 32), "int8", data=e.data)
            T.evaluate(0)
    # the prim_func generated by IRBuilder
    prim_func_actual = ib.get()

    # the expected prim_func
    c_handle, c_buffer = tir.Var("c_handle", "handle"), tir.decl_buffer(
        (128, 128), "float32", name="c"
    )
    d_handle, d_buffer = tir.Var("d", "handle"), tir.decl_buffer((64, 64), "int64", name="d")
    e_handle, e_buffer = tir.Var("e_handle", "handle"), tir.decl_buffer((1024,), "int8", name="e")
    prim_func_expected = tir.PrimFunc(
        params=[
            tir.Var("a", "handle"),
            tir.Var("b", "int64"),
            c_handle,
            d_handle,
            e_handle,
        ],
        body=tir.Evaluate(0),
        ret_type=tvm.ir.PrimType("int64"),
        buffer_map={c_handle: c_buffer, d_handle: d_buffer, e_handle: e_buffer},
        preflattened_buffer_map={
            e_handle: tir.decl_buffer((32, 32), "int8", name="e_preflatten", data=e_buffer.data)
        },
        attrs=tvm.ir.make_node("DictAttrs", key="value"),
    )
    # Check if the generated ir is expected
    assert_structural_equal(prim_func_actual, prim_func_expected, map_free_vars=True)


def test_ir_builder_tir_block():
    with IRBuilder() as ib:
        with T.block("block"):
            T.evaluate(0)
    # the block generated by IRBuilder
    block_realize_actual = ib.get()

    # the expected block
    block_expected = tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="block",
        body=tir.Evaluate(0),
        alloc_buffers=None,
        match_buffers=None,
        annotations={"tir.script_parsing_detect_access": tir.IntImm("int64", 3)},
    )
    block_realize_expected = tir.BlockRealize(
        iter_values=[],
        predicate=True,
        block=block_expected,
    )
    # Check if the generated ir is expected
    assert_structural_equal(block_realize_actual, block_realize_expected, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
