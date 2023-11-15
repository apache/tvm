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
# pylint: disable=missing-function-docstring,missing-module-docstring
import gc
import sys

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.ir import IRModule
from tvm.script import tir as T

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j in T.grid(128, 128):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.float32(0)
        for k in range(0, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def block_in_opaque_block(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.match_buffer(b, (128, 128), "float32")
    for i in range(128):
        with T.block("B"):
            vi = T.axis.S(128, i)
            T.reads([A[0:128, 0:128]])
            T.writes([B[0:128, 0:128]])
            B[vi, 0] = A[vi, 0]
            if A[vi, 0] == 0.0:
                with T.block("C"):
                    T.reads([A[0:128, 0:128]])
                    T.writes([B[0:128, 0:128]])
                    for j in range(128):
                        with T.block("D"):
                            vj = T.axis.S(128, j)
                            B[vi, vj] = A[vi, vj] * 3.0
            else:
                with T.block("E"):
                    T.reads([A[0:128, 0:128]])
                    T.writes([B[0:128, 0:128]])
                    for j in range(128):
                        with T.block("F"):
                            vj = T.axis.S(128, j)
                            B[vi, vj] = A[vi, vj] * 2.0


# pylint: enable=no-member,invalid-name,unused-variable


def replace_ir_builder(deep_copy=False, realize=False):
    new_func = tvm.script.from_source(elementwise.script())
    s = tir.ScheduleState(new_func, debug_mask="all")
    target = tvm.tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="target",
        body=s.mod["main"].body.block.body[1],
        init=None,
        alloc_buffers=None,
        match_buffers=None,
        annotations=None,
    )
    if realize:
        target = tvm.tir.BlockRealize(
            iter_values=[],
            predicate=True,
            block=target,
        )
    if deep_copy:
        target.__setstate__(target.__getstate__())
    gc.collect()
    return s, target


def replace_ir_builder_module(deep_copy=False, realize=False):
    new_func = tvm.script.from_source(elementwise.script())
    other_func = tvm.script.from_source(elementwise.script())
    mod = IRModule(functions={"main": new_func, "other": other_func})
    s = tir.ScheduleState(mod, debug_mask="all")
    target = tvm.tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="target",
        body=s.mod["main"].body.block.body[1],
        init=None,
        alloc_buffers=None,
        match_buffers=None,
        annotations=None,
    )
    if realize:
        target = tvm.tir.BlockRealize(
            iter_values=[],
            predicate=True,
            block=target,
        )
    if deep_copy:
        target.__setstate__(target.__getstate__())
    gc.collect()
    return s, target


def replace_ir_builder_with_opaque():
    func = tvm.script.from_source(block_in_opaque_block.script())
    s = tir.ScheduleState(func, debug_mask="all")
    gc.collect()
    return s


def test_replace_direct_write0():
    s, target = replace_ir_builder(realize=True)
    old_hash = s.mod["main"].__hash__()
    sref = s.get_sref(s.mod["main"].body.block.body[1])
    s.replace(sref, target)
    # There is no other reference so the AST node can be written directly
    assert old_hash == s.mod["main"].__hash__()
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.mod["main"].body.block.body[1], target)
    # The target reuse the stmt of the sref, so the sref won't be None
    assert sref.stmt is not None


def test_replace_direct_write1():
    s, target = replace_ir_builder(realize=True)
    old_hash = s.mod["main"].body.block.body.__hash__()
    hold_ref = s.mod["main"].body.block.body[1]
    sref = s.get_sref(s.mod["main"].body.block.body[1])
    s.replace(sref, target)
    # There is no other reference so the AST node can be written directly
    assert old_hash == s.mod["main"].body.block.body.__hash__()
    assert not tvm.ir.structural_equal(hold_ref.body, target)
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.mod["main"].body.block.body[1], target)
    # The target reuse `sref.stmt`, so the sref won't be None
    assert sref.stmt is not None


def test_replace_copy():
    s, target = replace_ir_builder(deep_copy=True, realize=True)
    old_hash = s.mod["main"].__hash__()
    # We hold another reference of func
    old_func = s.mod["main"]
    sref = s.get_sref(s.mod["main"].body.block.body[0])
    s.replace(sref, target)
    # We need to copy the whole func to remain the old_func unchanged
    assert old_hash != s.mod["main"].__hash__()
    assert not tvm.ir.structural_equal(old_func.body, s.mod["main"].body)
    assert old_hash == old_func.__hash__()
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.mod["main"].body.block.body[0], target)
    # The replaced AST node will be deleted, so the ref will be None
    assert sref.stmt is None


def test_replace_partial_copy0():
    s, target = replace_ir_builder(deep_copy=True, realize=True)
    func_old_hash = s.mod["main"].__hash__()
    hold_ref = s.mod["main"].body.block.body[0]
    ref_old_hash = hold_ref.__hash__()
    sref = s.get_sref(s.mod["main"].body.block.body[0].body)
    other_part_hash = s.mod["main"].body.block.body[1].__hash__()
    s.replace(sref, target)
    # The stmt is held by `hold_sref`, so it will be coped in copy-on-write
    # because the ref count is not unique
    assert ref_old_hash != s.mod["main"].body.block.body[0].__hash__()
    assert not tvm.ir.structural_equal(hold_ref.body, target)
    # The function and the other part stmt can be directly written
    assert func_old_hash == s.mod["main"].__hash__()
    assert other_part_hash == s.mod["main"].body.block.body[1].__hash__()
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.mod["main"].body.block.body[0].body, target)
    # The replaced AST node will be deleted, so the ref will be None
    assert sref.stmt is None


def test_replace_partial_copy1():
    s, target = replace_ir_builder(deep_copy=True)
    func_old_hash = s.mod["main"].__hash__()
    hold_ref = s.mod["main"].body.block.body[0].body
    stmt_old_hash = s.mod["main"].body.block.body[0].__hash__()
    sref = s.get_sref(s.mod["main"].body.block.body[0].body.body.block)
    other_part_hash = s.mod["main"].body.block.body[1].__hash__()
    s.replace(sref, target)
    # The parent stmt will change since there is only one reference
    assert stmt_old_hash == s.mod["main"].body.block.body[0].__hash__()
    assert not tvm.ir.structural_equal(hold_ref.body, target)
    # The function and the other part stmt can be directly written
    assert func_old_hash == s.mod["main"].__hash__()
    assert other_part_hash == s.mod["main"].body.block.body[1].__hash__()
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.mod["main"].body.block.body[0].body.body.block, target)
    # The replaced AST node will be deleted, so the ref will be None
    assert sref.stmt is None


def test_replace_root_write():
    s, target = replace_ir_builder()
    old_hash = s.mod["main"].__hash__()
    sref = s.get_sref(s.mod["main"].body.block)
    s.replace(sref, target)
    # Check no copy and the new body equals to target
    assert old_hash == s.mod["main"].__hash__()
    tvm.ir.assert_structural_equal(s.mod["main"].body.block, target)


def test_replace_root_copy0():
    s, target = replace_ir_builder(deep_copy=True)
    old_hash = s.mod["main"].__hash__()
    func_ref = s.mod["main"]
    sref = s.get_sref(s.mod["main"].body.block)
    s.replace(sref, target)
    # Check the new body equals to target
    assert old_hash != s.mod["main"].__hash__()
    tvm.ir.assert_structural_equal(s.mod["main"].body.block, target)
    # Check the original func remains unchanged
    assert old_hash == func_ref.__hash__()
    assert not tvm.ir.structural_equal(func_ref.body, target)


def test_replace_root_copy1():
    s, target = replace_ir_builder(deep_copy=True, realize=True)
    old_hash = s.mod["main"].body.block.__hash__()
    func_ref = s.mod["main"].body.block
    sref = s.get_sref(s.mod["main"].body.block.body[0])
    s.replace(sref, target)
    # Check the new body equals to target
    assert old_hash != s.mod["main"].body.block.__hash__()
    tvm.ir.assert_structural_equal(s.mod["main"].body.block.body[0], target)
    # Check the original func remains unchanged
    assert old_hash == func_ref.__hash__()
    assert not tvm.ir.structural_equal(func_ref.body, target)


def test_replace_root_copy2():
    s, target = replace_ir_builder(deep_copy=True)
    old_hash = s.mod.functions.__hash__()
    func_ref = s.mod.functions
    sref = s.get_sref(s.mod["main"].body.block)
    s.replace(sref, target)
    # Check the new body equals to target
    assert old_hash != s.mod.functions.__hash__()
    tvm.ir.assert_structural_equal(s.mod["main"].body.block, target)
    # Check the original func remains unchanged
    assert old_hash == func_ref.__hash__()
    for _, v in func_ref.items():
        assert not tvm.ir.structural_equal(v.body.block, target)


def test_replace_root_copy3():
    s, target = replace_ir_builder(deep_copy=True)
    old_hash = s.mod.__hash__()
    func_ref = s.mod
    sref = s.get_sref(s.mod["main"].body.block)
    s.replace(sref, target)
    # Check the new body equals to target
    assert old_hash != s.mod.__hash__()
    tvm.ir.assert_structural_equal(s.mod["main"].body.block, target)
    # Check the original func remains unchanged
    assert old_hash == func_ref.__hash__()
    assert not tvm.ir.structural_equal(func_ref["main"].body.block, target)


def test_replace_block_remap():
    func = elementwise
    s = tir.ScheduleState(func, debug_mask="all")
    # The target stmt
    target = matmul.body.block.body.body.body[0].block
    sref = s.get_sref(s.mod["main"].body.block.body[0].body.body.block)
    s.replace(sref, target, {sref.stmt: target})
    sref_new = s.get_sref(s.mod["main"].body.block.body[0].body.body.block)
    # Check the original sref has been remapped
    assert sref.__hash__() == sref_new.__hash__()
    tvm.ir.assert_structural_equal(sref.stmt, target)


def test_replace_block_in_opaque_block():
    s = replace_ir_builder_with_opaque()
    root_hash = s.mod["main"].__hash__()
    for_loop = s.mod["main"].body.block.body.body.block.body[1].then_case.block.body
    sref = s.get_sref(for_loop)
    new_for_loop = tir.For(
        loop_var=for_loop.loop_var,
        min=0,
        extent=128,
        kind=tir.ForKind.SERIAL,
        body=tir.Evaluate(0),
        thread_binding=None,
        annotations=None,
    )
    s.replace(sref, new_for_loop)
    assert root_hash == s.mod["main"].__hash__()
    tvm.ir.assert_structural_equal(sref.stmt, new_for_loop)


def test_replace_ir_module():
    s, target = replace_ir_builder_module(deep_copy=True)
    old_hash = s.mod["main"].__hash__()
    other_func_hash = s.mod["other"].__hash__()
    func_ref = s.mod["main"]
    sref = s.get_sref(s.mod["main"].body.block)
    s.replace(sref, target)
    # Check the new body equals to target
    assert old_hash != s.mod["main"].__hash__()
    tvm.ir.assert_structural_equal(s.mod["main"].body.block, target)
    # Check the original func remains unchanged
    assert old_hash == func_ref.__hash__()
    assert not tvm.ir.structural_equal(func_ref.body, target)
    assert other_func_hash == s.mod["other"].__hash__()


if __name__ == "__main__":
    tvm.testing.main()
