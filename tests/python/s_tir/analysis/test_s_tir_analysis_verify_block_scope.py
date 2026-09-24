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
from tvm.script import ir as I
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_buffer_region_bounds_are_visited():
    data = tvm.tirx.Var(
        "data", tvm.ir.PointerType(tvm.ir.PrimType("int32"), storage_scope="global")
    )
    buffer = tvm.tirx.decl_buffer([4], "int32", data=data)
    undefined = tvm.tirx.Var("undefined", "int32")
    region = tvm.tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(undefined, 4)])
    block = tvm.s_tir.SBlock([], [region], [], "region", tvm.tirx.Evaluate(0))
    func = tvm.tirx.PrimFunc([buffer], block)
    assert not tvm.s_tir.analysis.verify_well_formed(func, assert_mode=False)


def test_fail_use_out_loop_var():
    @Ts.prim_func(check_well_formed=False)
    def element_wise(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
    ):
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                # we cannot use `i` since it's defined outside the block
                B[vi, vj] = A[i, vj] * 2.0

    assert not tvm.s_tir.analysis.verify_well_formed(element_wise, assert_mode=False)


def test_block_match_buffer_defines_buffer_obj():
    """In a block, T.match_buffer defines a buffer view"""

    @I.ir_module
    class mod:
        @Ts.prim_func
        def func(A: T.Buffer([256, 256], "float32")):
            for (*iters,) in T.grid(16, 16, 16, 16):
                with Ts.sblock("compute"):
                    tile_i, tile_j, i, j = Ts.axis.remap("SSSS", iters)
                    B = T.match_buffer(
                        A[tile_i * 16 : (tile_i + 1) * 16, tile_j * 16 : (tile_j + 1) * 16],
                        dtype="float32",
                    )
                    B[i, j] = 0.0

    tvm.s_tir.analysis.verify_well_formed(mod)


def test_block_match_buffer_defines_symbolic_variables():
    """In a block, T.match_buffer may define symbolic variables"""

    @I.ir_module
    class mod:
        @Ts.prim_func
        def func(A: T.Buffer([256, 256], "int32")):
            for (*iters,) in T.grid(16, 16, 16, 16):
                with Ts.sblock("compute"):
                    tile_i, tile_j, i, j = Ts.axis.remap("SSSS", iters)

                    elem_offset = T.int32()
                    B = T.match_buffer(
                        A[tile_i * 16 : (tile_i + 1) * 16, tile_j * 16 : (tile_j + 1) * 16],
                        dtype="float32",
                        elem_offset=elem_offset,
                    )

                    B[i, j] = elem_offset

    tvm.s_tir.analysis.verify_well_formed(mod)


def test_match_buffer_in_block_is_well_formed():
    """SBlock::match_buffers introduces a buffer into scope for the block body."""

    @I.ir_module
    class mod:
        @Ts.prim_func
        def func(A: T.Buffer((128, 128), "float32")):
            for (*iters,) in T.grid(8, 8, 16, 16):
                with Ts.sblock("compute"):
                    ti, tj, i, j = Ts.axis.remap("SSSS", iters)
                    A_tile = T.match_buffer(
                        A[ti * 16 : (ti + 1) * 16, tj * 16 : (tj + 1) * 16],
                        dtype="float32",
                    )
                    A_tile[i, j] = A_tile[i, j] * 2.0

    tvm.s_tir.analysis.verify_well_formed(mod)


def test_error_undeclared_buffer_in_schedulable_tir():
    """In schedule-level TIR (with SBlock nodes), all buffers must be declared."""
    # Manually construct a BufferStore that uses a buffer without any declaration
    # inside a block context.
    n = tvm.tirx.Var("n", "int32")
    A = tvm.tirx.decl_buffer([n], "float32", name="A")
    i = tvm.tirx.Var("i", "int32")

    # Create an undeclared buffer using an explicit data pointer that is NOT
    # a function parameter and NOT wrapped with DeclBuffer.
    B_data = tvm.tirx.Var("B_data", tvm.ir.PointerType(tvm.ir.PrimType("float32")))
    B = tvm.tirx.decl_buffer([n], "float32", name="B", data=B_data)

    # Build a block that writes to B without any declaration of B.
    bi = tvm.tirx.Var("bi", "int32")
    block = tvm.s_tir.SBlock(
        iter_vars=[tvm.tirx.IterVar(tvm.ir.Range(0, n), bi, 0)],  # 0 = kDataPar
        reads=[tvm.tirx.BufferRegion(A, [tvm.ir.Range(bi, bi + 1)])],
        writes=[tvm.tirx.BufferRegion(B, [tvm.ir.Range(bi, bi + 1)])],
        body=tvm.tirx.BufferStore(B, tvm.tirx.BufferLoad(A, [bi]), [bi]),
        name_hint="write_B",
    )
    block_realize = tvm.s_tir.SBlockRealize(
        iter_values=[i],
        predicate=tvm.tirx.const(True),
        block=block,
    )

    prim_func = tvm.tirx.PrimFunc(
        params=[A, B_data],
        body=tvm.tirx.For(i, 0, n, tvm.tirx.ForKind.SERIAL, block_realize),
        # Note: B is NOT a function parameter, so its declaration scope is only
        # within a DeclBuffer node (which we intentionally omit here).
    )

    # B is used in the block but was never declared — should fail.
    with pytest.raises(
        (ValueError, tvm.error.InternalError), match="buffer B.*without a prior DeclBuffer"
    ):
        tvm.s_tir.analysis.verify_well_formed(prim_func)
