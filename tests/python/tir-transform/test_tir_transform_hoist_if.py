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
    ib = tvm.tir.ir_builder.create()
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")
    data = ib.pointer("float32", name="data")

    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(ib.likely(i < 2)):
                    ib.emit(_opaque_eval(m))
                with ib.else_scope():
                    ib.emit(_opaque_eval(n))

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.For", "j"): (("tir.For", "k"),),
        ("tir.IfThenElse", ("i",)): (("tir.For", "j"), ("tir.For", "j")),
        ("tir.For", "i"): (("tir.IfThenElse", ("i",)),),
    }
    verify_structure(new_stmt, expected_struct)


def test_hoist_multi_var_if():
    ib = tvm.tir.ir_builder.create()
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")
    data = ib.pointer("float32", name="data")

    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(ib.likely(i + j < 2)):
                    ib.emit(_opaque_eval(m))
                with ib.else_scope():
                    ib.emit(_opaque_eval(n))

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
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
    ib = tvm.tir.ir_builder.create()
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")
    data = ib.pointer("float32", name="data")

    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            data[i * 3 + j] = data[i * 3 + j] + 0.5
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(ib.likely(i < 2)):
                    ib.emit(_opaque_eval(m))
                with ib.else_scope():
                    ib.emit(_opaque_eval(n))

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.IfThenElse", ("i",)): (("tir.For", "k"), ("tir.For", "k")),
        ("tir.For", "j"): (None,),
        ("tir.For", "i"): (("tir.For", "j"),),
    }
    verify_structure(new_stmt, expected_struct)


def test_no_else():
    ib = tvm.tir.ir_builder.create()
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(ib.likely(i < 2)):
                    ib.emit(_opaque_eval(m))

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.For", "k"): (None,),
        ("tir.For", "j"): (("tir.For", "k"),),
        ("tir.IfThenElse", ("i",)): (("tir.For", "j"), None),
        ("tir.For", "i"): (("tir.IfThenElse", ("i",)),),
    }
    verify_structure(new_stmt, expected_struct)


def test_attr_stmt():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(tvm.tir.any(i < 4, j >= 8)):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.5
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.0

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
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
    ib = tvm.tir.ir_builder.create()
    data = ib.pointer("float32", name="data")

    with ib.for_range(0, 5, "i") as i:
        with ib.for_range(0, 10, "j") as j:
            with ib.if_scope(i >= 3):
                data[i * 3 + j] = data[i * 3 + j] + 0.5
                with ib.for_range(0, 15, "k") as k:
                    with ib.for_range(0, 20, "l") as l:
                        with ib.if_scope(tvm.tir.any(i < 4, j >= 8)):
                            data[i * 3 + j + k + l] = data[i * 3 + j + k + l] * 2
                        with ib.else_scope():
                            data[i * 3 + j + k + l] = data[i * 3 + j + k + l] * 1.5

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
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
    ib = tvm.tir.ir_builder.create()
    data = ib.pointer("float32", name="data")
    n = te.var("n")

    with ib.for_range(0, 5, "i") as i:
        with ib.for_range(0, 10, "j") as j:
            with ib.if_scope(i >= 3):
                data[i * 3 + j] = data[i * 3 + j] + 0.5
                with ib.for_range(0, 15, "k") as k:
                    with ib.for_range(0, 20, "l") as l:
                        with ib.if_scope(tvm.tir.any(i < 4, j >= 8)):
                            data[i * 3 + j + k + l] = data[i * 3 + j + k + l] * 2
                        with ib.else_scope():
                            data[i * 3 + j + k + l] = data[i * 3 + j + k + l] * 1.5
                        with ib.if_scope(j < 5):
                            data[i * 3 + j + k + l] = data[i * 3 + j + k + l] - 1

    with ib.for_range(0, 5, "i") as i:
        with ib.for_range(0, 10, "j") as j:
            with ib.for_range(0, 15, "k") as k:
                with ib.if_scope(n >= 3):
                    data[i * 3 + j + k] = data[i * 3 + j + k] + 0.6

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    expected_struct = {
        ("tir.IfThenElse", ("i", "j")): (None, None),
        ("tir.IfThenElse", ("j",)): (None, None),
        ("tir.For", "l"): (None,),
        ("tir.For", "k"): (None,),
        ("tir.For", "j"): (("tir.For", "j"),),
        ("tir.IfThenElse", ("i",)): (("tir.For", "j"), None),
        ("tir.For", "i"): (("tir.IfThenElse", ("i",)),),
        ("tir.IfThenElse", ("n",)): (("tir.For", "j"), None),
    }
    verify_structure(new_stmt, expected_struct)


def test_multi_if():
    ib = tvm.tir.ir_builder.create()
    data = ib.pointer("float32", name="data")

    with ib.for_range(0, 10, "i") as i:
        with ib.for_range(0, 10, "j") as j:
            with ib.for_range(0, 10, "k") as k:
                with ib.if_scope(3 <= i):
                    with ib.if_scope(3 <= j):
                        data[i * 100 + j * 10 + k] = data[i * 100 + j * 10 + k] + 0.5

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
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
    ib = tvm.tir.ir_builder.create()
    data = ib.pointer("float32", name="data")
    n = te.var("n")

    with ib.for_range(0, 10, "i") as i:
        with ib.for_range(0, 10, "j") as j:
            with ib.for_range(0, 10, "k") as k:
                with ib.if_scope(k <= 3):
                    data[i * 100 + j * 10 + k] = data[i * 100 + j * 10 + k] + 0.5

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_2():
    ib = tvm.tir.ir_builder.create()
    data = ib.pointer("float32", name="data")
    n = te.var("n")
    x = te.var("x")

    with ib.for_range(0, 10, "i") as i:
        with ib.for_range(0, 10, "j") as j:
            with ib.for_range(0, 10, "k") as k:
                with ib.if_scope(i <= 3):
                    data[i * 100 + j * 10 + k] = data[i * 100 + j * 10 + k] + 0.3
                data[i * 100 + j * 10 + k] = data[i * 100 + j * 10 + k] + 0.5

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


@pytest.mark.xfail(reason="Inconsistent thread_extent", strict=True)
def test_no_hoisting_3():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    dshape_inner = (33, 63)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                ib.scope_attr(tx, "thread_extent", dshape_inner[0])
                ib.scope_attr(bx, "thread_extent", dshape_inner[1])
                with ib.if_scope(tx < 3):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_4():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    dshape_inner = (33, 63)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                ib.scope_attr(tx, "thread_extent", dshape_inner[0])
                with ib.if_scope(tx < 3):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


@pytest.mark.xfail(reason="Inconsistent thread_extent", strict=True)
def test_no_hoisting_5():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    dshape_inner = (33, 63)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            ib.scope_attr(bx, "thread_extent", dshape_inner[1])
            with ib.for_range(0, n, "k") as k:
                ib.scope_attr(tx, "thread_extent", dshape_inner[0])
                with ib.if_scope(tx < 3):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_6():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope((tx + k) < 3):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_no_hoisting_7():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.if_scope((tx + j) < 9):
                with ib.for_range(0, n, "k") as k:
                    with ib.if_scope((tx + k) < 3):
                        data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_2():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    dshape_inner = (33, 63)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    # ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                ib.scope_attr(bx, "thread_extent", dshape[1])
                with ib.if_scope(tx < 3):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.RemoveNoOp()(mod)
    stmt = mod["main"].body

    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)


@pytest.mark.xfail(reason="Inconsistent thread_extent", strict=True)
def test_hoisting_block_scope_3():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    dshape_inner = (33, 63)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            ib.scope_attr(tx, "thread_extent", dshape_inner[0])
            ib.scope_attr(bx, "thread_extent", dshape_inner[1])
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(tx < 3):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_5():
    ib = tvm.tir.ir_builder.create()
    data = ib.pointer("float32", name="data", scope="global")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")
    g = te.var("g")

    ib.scope_attr(data, "storage_scope", "global")
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(data[g] < 3):
                    data[9 * j + 3 * j * k] = data[9 * j + 3 * j * k] + 0.3
                with ib.else_scope():
                    data[9 * j + 3 * j * k] = data[9 * j + 3 * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)

    stmt = new_stmt
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_6():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope((tx + n) < 3):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)


def test_hoisting_block_scope_7():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    data = ib.pointer("float32", name="data")
    l = te.var("l")
    m = te.var("m")
    n = te.var("n")

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope((tx + i) < 3):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 0.3
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k] + 1.3

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    tvm.ir.assert_structural_equal(new_stmt, stmt)

    with tvm.transform.PassContext(
        config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
    ):
        new_stmt = tvm.tir.transform.HoistIfThenElse()(mod)["main"].body
    assert not tvm.ir.structural_equal(new_stmt, stmt)


if __name__ == "__main__":
    tvm.testing.main()
