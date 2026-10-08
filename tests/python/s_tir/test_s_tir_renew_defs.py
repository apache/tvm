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

from __future__ import annotations

import tvm
import tvm.testing
from tvm.ir import Var
from tvm.s_tir import SBlock
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.tirx.function import Function


def _check_func_signature_remap(lhs: Function, rhs: Function):
    assert lhs != rhs
    for x, y in zip(lhs.params, rhs.params):
        assert x != y
        assert tvm.tirx.is_tensor_var(x) == tvm.tirx.is_tensor_var(y)


def _check_buffer_decl(lhs: Var, rhs: Var):
    assert lhs != rhs
    assert lhs.data != rhs.data


def _check_block_signature_remap(lhs: SBlock, rhs: SBlock):
    assert lhs != rhs
    for x, y in zip(lhs.iter_vars, rhs.iter_vars):
        assert x != y
        assert x.var != y.var
    for x, y in zip(lhs.alloc_buffers, rhs.alloc_buffers):
        _check_buffer_decl(x, y)
    for x, y in zip(lhs.match_buffers, rhs.match_buffers):
        assert x != y
        _check_buffer_decl(x.buffer, y.buffer)


def test_simple():
    @Ts.function
    # Var A should be remapped
    def elementwise(A: T.Tensor((128, 128), "float32")):
        # Var B should be remapped
        B = Ts.sblock_alloc_buffer((128, 128), "float32")
        # i, j should be remapped
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                # vi, vj should be remapped
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi, vj])
                Ts.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * 2.0

    f1 = elementwise
    f2 = tvm.tirx.renew_def(f1)
    tvm.ir.assert_structural_equal(f1, f2)

    _check_func_signature_remap(f1, f2)
    # check root block
    _check_block_signature_remap(f1.body[0].block, f2.body[0].block)
    # check remap of i
    assert f1.body[0].block.body[0].loop_var != f2.body[0].block.body[0].loop_var
    # check remap of j
    assert f1.body[0].block.body[0].body[0].loop_var != f2.body[0].block.body[0].body[0].loop_var

    # check inner block
    def _get_sblock(f):
        return f.body[0].block.body[0].body[0].body[0].block

    _check_block_signature_remap(_get_sblock(f1), _get_sblock(f2))


def test_match_buffer():
    # well-formed checker complains about multiple definitions for variable A0_s1,
    # likely stemming from strides=[s, s]
    s = T.dynamic("s", "int32")
    e = T.dynamic("e", "int32")

    @Ts.function(check_well_formed=False)
    # A and B should be remapped
    def func_match_buffer(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        with Ts.sblock("root"):
            # A0 should be remapped
            A0 = Ts.match_buffer(
                A[0:128, 0:128],
                shape=(128, 128),
                dtype="float32",
                # s and e should be remapped
                strides=[s, s],
                elem_offset=e,
            )
            for i, j in T.grid(128, 128):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A0[vi, vj] * 2.0

    f1 = func_match_buffer
    f2 = tvm.tirx.renew_def(f1)
    tvm.ir.assert_structural_equal(f1, f2)

    _check_func_signature_remap(f1, f2)
    _check_block_signature_remap(f1.body[0].block, f2.body[0].block)
    assert f1.body[0].block.body[0].loop_var != f2.body[0].block.body[0].loop_var

    def _get_sblock(f):
        return f.body[0].block

    block1 = _get_sblock(f1)
    block2 = _get_sblock(f2)
    _check_block_signature_remap(block1, block2)

    matched_buffer1 = block1.match_buffers[0].buffer
    matched_buffer2 = block2.match_buffers[0].buffer
    # Stride var s should be remapped
    assert matched_buffer1.strides[0] != matched_buffer2.strides[0]
    assert matched_buffer1.strides[1] != matched_buffer2.strides[1]
    # s should be only remapped once
    assert matched_buffer1.strides[0] == matched_buffer1.strides[1]
    assert matched_buffer2.strides[0] == matched_buffer2.strides[1]
    # Element-offset var e should be remapped
    assert matched_buffer1.elem_offset != matched_buffer2.elem_offset


def test_undefined_buffer():
    @Ts.function
    def access_alloc():
        # Var A should be remapped
        A = T.alloc_tensor((128,), "float16")
        T.evaluate(A.data)
        for i in range(128):
            A[i] = A[i] + T.float16(1.0)

    f1 = access_alloc
    f2 = tvm.tirx.renew_def(f1)
    tvm.ir.assert_structural_equal(f1, f2)

    # AllocTensor is now a flat statement in SeqStmt
    assert f1.body.seq[0].var.data != f2.body.seq[0].var.data

    def _get_tensor_store_buffer(f):
        # SeqStmt: [AllocTensor, Evaluate, For]; For body has the TensorStore
        return f.body.seq[2].body[0].buffer

    _check_buffer_decl(_get_tensor_store_buffer(f1), _get_tensor_store_buffer(f2))


def test_symbolic_func():
    m = T.dynamic("m", "int32")

    @Ts.function
    def symbolic_func(A: T.Tensor((n, m)), B: T.Tensor((n, m * 2)), n: T.int32):  # noqa: F821
        for i, j in T.grid(n, m):
            B[i, j * 2] = A[i, j]
            B[i, j * 2 + 1] = A[i, j]

    f1 = symbolic_func
    f2 = tvm.tirx.renew_def(f1)
    tvm.ir.assert_structural_equal(f1, f2)


def test_buffer_params():
    m = T.dynamic("m")

    @Ts.function
    def main(A: T.Tensor((m * 2,)), B: T.Tensor((m, 2))):
        for i, j in T.grid(m, 2):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi * 2 + vj]

    f1 = main
    f2 = tvm.tirx.renew_def(main)
    tvm.ir.assert_structural_equal(f1, f2)
    assert f1.params[1].shape[0] != f2.params[1].shape[0]


def test_compound_buffer_param_shape_var():
    n = tvm.tirx.Var("n", "int32")
    A = tvm.tirx.decl_tensor((tvm.tirx.max(n, 1),), layout=None)
    f1 = tvm.tirx.Function([A], tvm.tirx.Evaluate(n))
    f2 = tvm.tirx.renew_def(f1)

    tvm.ir.assert_structural_equal(f1, f2)
    assert not f1.body[0].value.same_as(f2.body[0].value)
    assert f2.params[0].shape[0].a.same_as(f2.body[0].value)


def test_gather():
    @Ts.function(private=True)
    def take(
        A: T.Tensor((4096, 4096), "float16"),
        B: T.Tensor((1,), "int32"),
        T_take: T.Tensor((1, 4096), "float16"),
    ):
        for ax0, ax1 in T.grid(1, 4096):
            with Ts.sblock("T_take"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                Ts.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    f1 = take
    f2 = tvm.tirx.renew_def(take)
    tvm.ir.assert_structural_equal(f1, f2)


if __name__ == "__main__":
    tvm.testing.main()
