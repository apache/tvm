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
from tvm import relay
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


def test_hoisting_block_scope_1():
    n = te.size_var("n")
    m = te.size_var("m")
    A = te.placeholder((n, m), name="A")
    k = te.reduce_axis((0, m), "k")
    B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
    s = te.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    BF = s.rfactor(B, ki)
    xo, xi = s[B].split(s[B].op.axis[0], factor=32)
    s[B.op].bind(xo, te.thread_axis("blockIdx.x"))
    s[B.op].bind(xi, te.thread_axis("threadIdx.y"))
    s[B].bind(s[B].op.reduce_axis[0], te.thread_axis("threadIdx.x"))
    s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
    mod = tvm.driver.build_module.schedule_to_module(s, [A, B], "main", None)
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


def test_hoisting_block_scope_4():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    AA = te.compute((n,), lambda *i: A(*i), name="A")
    BB = te.compute((n,), lambda *i: B(*i), name="B")
    T = te.compute(A.shape, lambda *i: AA(*i) + BB(*i), name="T")
    C = te.compute(A.shape, lambda *i: T(*i), name="C")
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
    s[C].vectorize(xi)
    mod = tvm.driver.build_module.schedule_to_module(s, [A, B, C], "main", None)
    mod = tvm.tir.transform.Simplify()(mod)

    stmt = mod["main"].body
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


@pytest.mark.skip()
def test_hoisting_op_conv():
    dtype = "float32"
    dshape = (1, 80, 73, 73)
    kshape = (192, 80, 3, 3)
    padding = (1, 1)
    groups = 1
    dilation = (1, 1)
    kernel_size = (3, 3)
    channels = 192
    scale = 1
    x = relay.var("x", shape=dshape, dtype=dtype)
    w = relay.var("w", shape=kshape, dtype=dtype)
    y = relay.nn.conv2d(
        x,
        w,
        padding=padding,
        dilation=dilation,
        groups=groups,
        channels=channels,
        kernel_size=kernel_size,
    )

    func = relay.Function([x, w], y)
    mod = tvm.IRModule()
    mod["main"] = func
    mod = relay.transform.InferType()(mod)

    data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
    kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)

    params = {"w": tvm.nd.array(kernel)}
    for target, dev in enabled_targets():
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
            m = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
            x = np.random.uniform(size=dshape)
            data_tvm = tvm.nd.array(data)
            m.set_input("x", data_tvm)
            m.run()
            e = m.module.time_evaluator("run", dev, number=300, repeat=3)
            t1 = e(data_tvm).results
            t1 = np.array(t1) * 1000
            print("{} ms".format(t1.mean()))

        with tvm.transform.PassContext(
            opt_level=3, config={"tir.HoistIfThenElse": {"support_block_scope_hosting": True}}
        ):
            lib = relay.build_module.build(mod, target=target, params=params)
            m = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
            x = np.random.uniform(size=dshape)
            data_tvm = tvm.nd.array(data)
            m.set_input("x", data_tvm)
            m.set_input(**params)
            m.run()
            e = m.module.time_evaluator("run", dev, number=300, repeat=3)
            t2 = e(data_tvm).results
            t2 = np.array(t2) * 1000

            print("{} ms".format(t2.mean()))
        tvm.testing.assert_allclose(t1.mean(), t2.mean(), atol=1, rtol=1e-1)


if __name__ == "__main__":
    tvm.testing.main()
