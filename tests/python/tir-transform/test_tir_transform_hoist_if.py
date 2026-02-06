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
import tvm
from tvm import te
from tvm.script import tir as T, ir as I
import numpy as np
import pytest
from tvm.testing import enabled_targets

var_list = []


def verify_structure(stmt, expected_struct):
    node_dict = {}
    struct = {}

    def _extract_vars(op):
        global var_list
        if isinstance(op, tvm.tir.Var):
            var_list.append(op.name)

    def _visit(op):
        key = op
        if isinstance(op, tvm.tir.IfThenElse):
            global var_list
            tvm.tir.stmt_functor.post_order_visit(op.condition, _extract_vars)
            val = [(op.then_case, op.else_case), ("tir.IfThenElse", tuple(var_list))]
            var_list.clear()
        elif isinstance(op, tvm.tir.For):
            val = [(op.body,), ("tir.For", op.loop_var.name)]
        elif isinstance(op, tvm.tir.AttrStmt):
            val = [(op.body,), ("tir.AttrStmt", op.attr_key, int(op.value))]
        else:
            return
        node_dict[key] = val

    tvm.tir.stmt_functor.post_order_visit(stmt, _visit)
    for key, val in node_dict.items():
        struct[val[1]] = tuple(
            node_dict[child][1] if child in node_dict else None for child in val[0]
        )

    assert struct == expected_struct, "Structure mismatch: expect %s but got %s" % (
        expected_struct,
        struct,
    )
    var_list.clear()


def _opaque_eval(var):
    return tvm.tir.Evaluate(tvm.tir.call_extern("int32", "dummy", var))


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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.For", "j"): (("tir.For", "k"),),
        ("tir.IfThenElse", ("i",)): (("tir.For", "j"), ("tir.For", "j")),
        ("tir.For", "i"): (("tir.IfThenElse", ("i",)),),
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
    new_mod = tvm.tir.transform.HoistIfThenElse()(mod)
    new_stmt = new_mod["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.IfThenElse", ("i", "j")): (("tir.For", "k"), ("tir.For", "k")),
        ("tir.For", "j"): (("tir.IfThenElse", ("i", "j")),),
        ("tir.For", "i"): (("tir.For", "j"),),
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.IfThenElse", ("i",)): (("tir.For", "k"), ("tir.For", "k")),
        ("tir.For", "j"): (None,),
        ("tir.For", "i"): (("tir.For", "j"),),
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.For", "j"): (("tir.For", "k"),),
        ("tir.IfThenElse", ("i",)): (("tir.For", "j"), None),
        ("tir.For", "i"): (("tir.IfThenElse", ("i",)),),
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.IfThenElse", ("i", "j")): (("tir.For", "k"), ("tir.For", "k")),
        ("tir.For", "j"): (("tir.IfThenElse", ("i", "j")),),
        ("tir.For", "i"): (("tir.For", "j"),),
        ("tir.AttrStmt", "thread_extent", 64): (("tir.For", "i"),),
        ("tir.AttrStmt", "thread_extent", 32): (("tir.AttrStmt", "thread_extent", 64),),
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.For", "l"): (None,),
        ("tir.For", "k"): (("tir.For", "l"),),
        ("tir.IfThenElse", ("i", "j")): (("tir.For", "k"), ("tir.For", "k")),
        ("tir.For", "j"): (None,),
        ("tir.IfThenElse", ("i",)): (("tir.For", "j"), None),
        ("tir.For", "i"): (("tir.IfThenElse", ("i",)),),
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

    new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    # Updated expected_struct with renamed second nest variables
    expected_struct = {
        ("tir.IfThenElse", ("i", "j")): (None, None),
        ("tir.IfThenElse", ("j",)): (None, None),
        ("tir.For", "l"): (None,),
        ("tir.For", "k"): (("tir.For", "l"),),
        ("tir.For", "j"): (None,),
        ("tir.IfThenElse", ("i",)): (("tir.For", "j"), None),
        ("tir.For", "i"): (("tir.IfThenElse", ("i",)),),
        ("tir.For", "k2"): (None,),
        ("tir.For", "j2"): (("tir.For", "k2"),),
        ("tir.For", "i2"): (("tir.For", "j2"),),
        ("tir.IfThenElse", ("n",)): (("tir.For", "i2"), None),
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
    new_mod = tvm.tir.transform.HoistIfThenElse()(mod)
    new_stmt = new_mod["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.IfThenElse", ("j",)): (("tir.For", "k"), None),
        ("tir.For", "j"): (("tir.IfThenElse", ("j",)),),
        ("tir.IfThenElse", ("i",)): (("tir.For", "j"), None),
        ("tir.For", "i"): (("tir.IfThenElse", ("i",)),),
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_4():
    dshape = (32, 64)
    dshape_inner = (33, 63)

    # Create iter_var for tx (used inside loop with T.attr)
    tx_var = tvm.tir.Var("threadIdx.x", "int32")
    tx_iter = tvm.tir.IterVar(
        tvm.ir.Range(0, dshape_inner[0]), tx_var, tvm.tir.IterVar.ThreadIndex, "threadIdx.x"
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_2():
    dshape = (32, 64)

    # Create iter_var for bx (used inside loop with T.attr)
    bx_var = tvm.tir.Var("blockIdx.x", "int32")
    bx_iter = tvm.tir.IterVar(
        tvm.ir.Range(0, dshape[1]), bx_var, tvm.tir.IterVar.ThreadIndex, "blockIdx.x"
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
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.RemoveNoOp()(mod)
    stmt = mod["main"].body

    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], new_stmt))
    stmt = new_stmt

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
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
    new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hoisting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(Module)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)


if __name__ == "__main__":
    tvm.testing.main()
