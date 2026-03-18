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
# ruff: noqa: E741, F401
import numpy as np
import pytest

import tvm
from tvm import s_tir
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import enabled_targets

var_list = []


def verify_structure(stmt, expected_struct):
    node_dict = {}
    struct = {}

    def _extract_vars(op):
        global var_list
        if isinstance(op, tvm.tirx.Var):
            var_list.append(op.name)

    def _visit(op):
        key = op
        if isinstance(op, tvm.tirx.IfThenElse):
            global var_list
            tvm.tirx.stmt_functor.post_order_visit(op.condition, _extract_vars)
            val = [(op.then_case, op.else_case), ("tirx.IfThenElse", tuple(var_list))]
            var_list.clear()
        elif isinstance(op, tvm.tirx.For):
            val = [(op.body,), ("tirx.For", op.loop_var.name)]
        elif isinstance(op, tvm.tirx.AttrStmt):
            val = [(op.body,), ("tirx.AttrStmt", op.attr_key, int(op.value))]
        else:
            return
        node_dict[key] = val

    tvm.tirx.stmt_functor.post_order_visit(stmt, _visit)
    for key, val in node_dict.items():
        struct[val[1]] = tuple(
            node_dict[child][1] if child in node_dict else None for child in val[0]
        )

    assert struct == expected_struct, (
        f"Structure mismatch: expect {expected_struct} but got {struct}"
    )
    var_list.clear()


def _opaque_eval(var):
    return tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "dummy", var))


def test_hoist_top_for():
    @T.prim_func(private=True)
    def func(l: T.int32, m: T.int32, n: T.int32):
        for i in T.serial(l):
            for j in T.serial(m):
                for k in T.serial(n):
                    if T.likely(i < 2):
                        T.evaluate(T.call_extern("int32", "dummy", m))
                    else:
                        T.evaluate(T.call_extern("int32", "dummy", n))

    mod = tvm.IRModule.from_expr(func)
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tirx.For", "k"): (None,),
        ("tirx.For", "j"): (("tirx.For", "k"),),
        ("tirx.IfThenElse", ("i",)): (("tirx.For", "j"), ("tirx.For", "j")),
        ("tirx.For", "i"): (("tirx.IfThenElse", ("i",)),),
    }
    verify_structure(new_stmt, expected_struct)


def test_hoist_multi_var_if():
    @T.prim_func(private=True)
    def func(l: T.int32, m: T.int32, n: T.int32):
        for i in T.serial(l):
            for j in T.serial(m):
                for k in T.serial(n):
                    if T.likely(i + j < 2):
                        T.evaluate(T.call_extern("int32", "dummy", m))
                    else:
                        T.evaluate(T.call_extern("int32", "dummy", n))

    mod = tvm.IRModule.from_expr(func)
    new_mod = tvm.s_tir.transform.HoistIfThenElse()(mod)
    new_stmt = new_mod["main"].body
    expected_struct = {
        ("tirx.For", "k"): (None,),
        ("tirx.IfThenElse", ("i", "j")): (("tirx.For", "k"), ("tirx.For", "k")),
        ("tirx.For", "j"): (("tirx.IfThenElse", ("i", "j")),),
        ("tirx.For", "i"): (("tirx.For", "j"),),
    }
    verify_structure(new_stmt, expected_struct)


def test_hoist_no_match_for():
    @T.prim_func(private=True)
    def func(data: T.handle("float32"), l: T.int32, m: T.int32, n: T.int32):
        data_ptr = T.decl_buffer(1, "float32", data=data)
        for i in T.serial(l):
            for j in T.serial(m):
                data_ptr[i * 3 + j] = data_ptr[i * 3 + j] + T.float32(0.5)
                for k in T.serial(n):
                    if T.likely(i < 2):
                        T.evaluate(T.call_extern("int32", "dummy", m))
                    else:
                        T.evaluate(T.call_extern("int32", "dummy", n))

    mod = tvm.IRModule.from_expr(func)
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tirx.For", "k"): (None,),
        ("tirx.IfThenElse", ("i",)): (("tirx.For", "k"), ("tirx.For", "k")),
        ("tirx.For", "j"): (None,),
        ("tirx.For", "i"): (("tirx.For", "j"),),
    }
    verify_structure(new_stmt, expected_struct)


def test_no_else():
    @T.prim_func(private=True)
    def func(l: T.int32, m: T.int32, n: T.int32):
        for i in T.serial(l):
            for j in T.serial(m):
                for k in T.serial(n):
                    if T.likely(i < 2):
                        T.evaluate(T.call_extern("int32", "dummy", m))

    mod = tvm.IRModule.from_expr(func)
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tirx.For", "k"): (None,),
        ("tirx.For", "j"): (("tirx.For", "k"),),
        ("tirx.IfThenElse", ("i",)): (("tirx.For", "j"), None),
        ("tirx.For", "i"): (("tirx.IfThenElse", ("i",)),),
    }
    verify_structure(new_stmt, expected_struct)


def test_attr_stmt():
    dshape = (32, 64)

    @T.prim_func(private=True)
    def func(data: T.handle("float32"), l: T.int32, m: T.int32, n: T.int32):
        data_ptr = T.decl_buffer(1, "float32", data=data)
        tx = T.launch_thread("threadIdx.x", dshape[0])
        bx = T.launch_thread("blockIdx.x", dshape[1])
        for i in T.serial(l):
            for j in T.serial(m):
                for k in T.serial(n):
                    if i < 4 or j >= 8:
                        data_ptr[bx * j + tx * j * k] = data_ptr[bx * j + tx * j * k] + T.float32(
                            0.5
                        )
                    else:
                        data_ptr[bx * j + tx * j * k] = data_ptr[bx * j + tx * j * k] + T.float32(
                            1.0
                        )

    mod = tvm.IRModule.from_expr(func)
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tirx.For", "k"): (None,),
        ("tirx.IfThenElse", ("i", "j")): (("tirx.For", "k"), ("tirx.For", "k")),
        ("tirx.For", "j"): (("tirx.IfThenElse", ("i", "j")),),
        ("tirx.For", "i"): (("tirx.For", "j"),),
        ("tirx.AttrStmt", "thread_extent", 64): (("tirx.For", "i"),),
        ("tirx.AttrStmt", "thread_extent", 32): (("tirx.AttrStmt", "thread_extent", 64),),
    }
    verify_structure(new_stmt, expected_struct)


def test_nested_for():
    @T.prim_func(private=True)
    def func(data: T.handle("float32")):
        data_ptr = T.decl_buffer(1, "float32", data=data)
        for i in range(5):
            for j in range(10):
                if i >= 3:
                    data_ptr[i * 3 + j] = data_ptr[i * 3 + j] + T.float32(0.5)
                    for k in range(15):
                        for l in range(20):
                            if i < 4 or j >= 8:
                                data_ptr[i * 3 + j + k + l] = data_ptr[
                                    i * 3 + j + k + l
                                ] * T.float32(2)
                            else:
                                data_ptr[i * 3 + j + k + l] = data_ptr[
                                    i * 3 + j + k + l
                                ] * T.float32(1.5)

    mod = tvm.IRModule.from_expr(func)
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tirx.For", "l"): (None,),
        ("tirx.For", "k"): (("tirx.For", "l"),),
        ("tirx.IfThenElse", ("i", "j")): (("tirx.For", "k"), ("tirx.For", "k")),
        ("tirx.For", "j"): (None,),
        ("tirx.IfThenElse", ("i",)): (("tirx.For", "j"), None),
        ("tirx.For", "i"): (("tirx.IfThenElse", ("i",)),),
    }
    verify_structure(new_stmt, expected_struct)


def test_if_block():
    # Use different variable names for second loop nest to avoid dict key collision
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(data: T.Buffer((1,), "float32"), n: T.int32):
            # First loop nest: i, j, k, l
            for i in T.serial(5):
                for j in T.serial(10):
                    if i >= 3:
                        data[i * 3 + j] = data[i * 3 + j] + T.float32(0.5)
                        for k in T.serial(15):
                            for l in T.serial(20):
                                if i < 4 or j >= 8:
                                    data[i * 3 + j + k + l] = data[i * 3 + j + k + l] * T.float32(2)
                                else:
                                    data[i * 3 + j + k + l] = data[i * 3 + j + k + l] * T.float32(
                                        1.5
                                    )
                                if j < 5:
                                    data[i * 3 + j + k + l] = data[i * 3 + j + k + l] - T.float32(1)

            # Second loop nest: i2, j2, k2 (different names)
            for i2 in T.serial(5):
                for j2 in T.serial(10):
                    for k2 in T.serial(15):
                        if n >= 3:
                            data[i2 * 3 + j2 + k2] = data[i2 * 3 + j2 + k2] + T.float32(0.6)

    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    # Updated expected_struct with renamed second nest variables
    expected_struct = {
        ("tirx.IfThenElse", ("i", "j")): (None, None),
        ("tirx.IfThenElse", ("j",)): (None, None),
        ("tirx.For", "l"): (None,),
        ("tirx.For", "k"): (("tirx.For", "l"),),
        ("tirx.For", "j"): (None,),
        ("tirx.IfThenElse", ("i",)): (("tirx.For", "j"), None),
        ("tirx.For", "i"): (("tirx.IfThenElse", ("i",)),),
        ("tirx.For", "k2"): (None,),
        ("tirx.For", "j2"): (("tirx.For", "k2"),),
        ("tirx.For", "i2"): (("tirx.For", "j2"),),
        ("tirx.IfThenElse", ("n",)): (("tirx.For", "i2"), None),
    }
    verify_structure(new_stmt, expected_struct)


def test_multi_if():
    @T.prim_func(private=True)
    def func(data: T.handle("float32")):
        data_ptr = T.decl_buffer(1, "float32", data=data)
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if 3 <= i:
                        if 3 <= j:
                            data_ptr[i * 100 + j * 10 + k] = data_ptr[
                                i * 100 + j * 10 + k
                            ] + T.float32(0.5)

    mod = tvm.IRModule.from_expr(func)
    new_mod = tvm.s_tir.transform.HoistIfThenElse()(mod)
    new_stmt = new_mod["main"].body
    expected_struct = {
        ("tirx.For", "k"): (None,),
        ("tirx.IfThenElse", ("j",)): (("tirx.For", "k"), None),
        ("tirx.For", "j"): (("tirx.IfThenElse", ("j",)),),
        ("tirx.IfThenElse", ("i",)): (("tirx.For", "j"), None),
        ("tirx.For", "i"): (("tirx.IfThenElse", ("i",)),),
    }
    verify_structure(new_stmt, expected_struct)


def test_no_hoisting_1():
    @T.prim_func(private=True)
    def func(data: T.handle("float32")):
        data_ptr = T.decl_buffer(1, "float32", data=data)
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if k <= 3:
                        data_ptr[i * 100 + j * 10 + k] = data_ptr[i * 100 + j * 10 + k] + T.float32(
                            0.5
                        )

    mod = tvm.IRModule.from_expr(func)
    stmt = mod["main"].body
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_2():
    @T.prim_func(private=True)
    def func(data: T.handle("float32")):
        data_ptr = T.decl_buffer(1, "float32", data=data)
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if i <= 3:
                        data_ptr[i * 100 + j * 10 + k] = data_ptr[i * 100 + j * 10 + k] + T.float32(
                            0.3
                        )
                    data_ptr[i * 100 + j * 10 + k] = data_ptr[i * 100 + j * 10 + k] + T.float32(0.5)

    mod = tvm.IRModule.from_expr(func)
    stmt = mod["main"].body
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_4():
    dshape = (32, 64)
    dshape_inner = (33, 63)

    # Create iter_var for tx (used inside loop with T.attr)
    tx_var = tvm.tirx.Var("threadIdx.x", "int32")
    tx_iter = tvm.tirx.IterVar(
        tvm.ir.Range(0, dshape_inner[0]), tx_var, tvm.tirx.IterVar.ThreadIndex, "threadIdx.x"
    )

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(data: T.Buffer((1,), "float32"), l: T.int32, m: T.int32, n: T.int32):
            bx = T.launch_thread("blockIdx.x", dshape[1])
            for i in T.serial(l):
                for j in T.serial(m):
                    for k in T.serial(n):
                        T.attr(tx_iter, "thread_extent", dshape_inner[0])
                        if tx_var < 3:
                            data[bx * j + tx_var * j * k] = data[
                                bx * j + tx_var * j * k
                            ] + T.float32(0.3)
                        else:
                            data[bx * j + tx_var * j * k] = data[
                                bx * j + tx_var * j * k
                            ] + T.float32(1.3)

    stmt = Module["main"].body
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_6():
    dshape = (32, 64)

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(data: T.Buffer((1,), "float32"), l: T.int32, m: T.int32, n: T.int32):
            tx = T.launch_thread("threadIdx.x", dshape[0])
            bx = T.launch_thread("blockIdx.x", dshape[1])
            for i in T.serial(l):
                for j in T.serial(m):
                    for k in T.serial(n):
                        if tx + k < 3:
                            data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + T.float32(0.3)
                        else:
                            data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + T.float32(1.3)

    stmt = Module["main"].body
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_7():
    dshape = (32, 64)

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(data: T.Buffer((1,), "float32"), l: T.int32, m: T.int32, n: T.int32):
            tx = T.launch_thread("threadIdx.x", dshape[0])
            bx = T.launch_thread("blockIdx.x", dshape[1])
            for i in T.serial(l):
                for j in T.serial(m):
                    if tx + j < 9:
                        for k in T.serial(n):
                            if tx + k < 3:
                                data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + T.float32(
                                    0.3
                                )

    stmt = Module["main"].body
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_2():
    dshape = (32, 64)

    # Create iter_var for bx (used inside loop with T.attr)
    bx_var = tvm.tirx.Var("blockIdx.x", "int32")
    bx_iter = tvm.tirx.IterVar(
        tvm.ir.Range(0, dshape[1]), bx_var, tvm.tirx.IterVar.ThreadIndex, "blockIdx.x"
    )

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(data: T.Buffer((1,), "float32"), l: T.int32, m: T.int32, n: T.int32):
            tx = T.launch_thread("threadIdx.x", dshape[0])
            for i in T.serial(l):
                for j in T.serial(m):
                    for k in T.serial(n):
                        T.attr(bx_iter, "thread_extent", dshape[1])
                        if tx < 3:
                            data[bx_var * j + tx * j * k] = data[
                                bx_var * j + tx * j * k
                            ] + T.float32(0.3)
                        else:
                            data[bx_var * j + tx * j * k] = data[
                                bx_var * j + tx * j * k
                            ] + T.float32(1.3)

    mod = Module
    mod = tvm.tirx.transform.Simplify()(mod)
    mod = tvm.tirx.transform.RemoveNoOp()(mod)
    stmt = mod["main"].body

    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_5():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(data: T.Buffer((1,), "float32"), l: T.int32, m: T.int32, n: T.int32, g: T.int32):
            for i in T.serial(l):
                for j in T.serial(m):
                    for k in T.serial(n):
                        if data[g] < T.float32(3):
                            data[9 * j + 3 * j * k] = data[9 * j + 3 * j * k] + T.float32(0.3)
                        else:
                            data[9 * j + 3 * j * k] = data[9 * j + 3 * j * k] + T.float32(1.3)

    stmt = Module["main"].body
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)

    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([], new_stmt))
    stmt = new_stmt

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_6():
    dshape = (32, 64)

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(data: T.Buffer((1,), "float32"), l: T.int32, m: T.int32, n: T.int32):
            tx = T.launch_thread("threadIdx.x", dshape[0])
            bx = T.launch_thread("blockIdx.x", dshape[1])
            for i in T.serial(l):
                for j in T.serial(m):
                    for k in T.serial(n):
                        if tx + n < 3:
                            data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + T.float32(0.3)
                        else:
                            data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + T.float32(1.3)

    stmt = Module["main"].body
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_7():
    dshape = (32, 64)

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(data: T.Buffer((1,), "float32"), l: T.int32, m: T.int32, n: T.int32):
            tx = T.launch_thread("threadIdx.x", dshape[0])
            bx = T.launch_thread("blockIdx.x", dshape[1])
            for i in T.serial(l):
                for j in T.serial(m):
                    for k in T.serial(n):
                        if tx + i < 3:
                            data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + T.float32(0.3)
                        else:
                            data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + T.float32(1.3)

    stmt = Module["main"].body
    new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"s_tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.s_tir.transform.HoistIfThenElse()(Module)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)


if __name__ == "__main__":
    tvm.testing.main()
