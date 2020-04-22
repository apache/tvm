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
            val = [(op.then_case, op.else_case), ("IfThenElse", tuple(var_list))]
            var_list.clear()
        elif isinstance(op, tvm.tir.For):
            val = [(op.body,), ("For", op.loop_var.name)]
        elif isinstance(op, tvm.tir.AttrStmt):
            val = [(op.body,), ("AttrStmt", op.attr_key, int(op.value))]
        else:
            return
        node_dict[key] = val

    tvm.tir.stmt_functor.post_order_visit(stmt, _visit)
    for key, val in node_dict.items():
        struct[val[1]] = tuple(node_dict[child][1] if child in node_dict
                               else None for child in val[0])

    assert struct == expected_struct, "Structure mismatch: expect %s but got %s" \
                                      % (expected_struct, struct)
    var_list.clear()

def test_basic():
    ib = tvm.tir.ir_builder.create()
    l = te.var('l')
    m = te.var('m')
    n = te.var('n')

    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(ib.likely(i < 2)):
                    ib.emit(tvm.tir.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.tir.Evaluate(n))

    stmt = ib.get()
    new_stmt = tvm.tir.ir_pass.HoistIfThenElse(stmt)
    expected_struct = {('For', 'k'): (None,), ('For', 'j'): (('For', 'k'),),
                       ('IfThenElse', ('i',)): (('For', 'j'), ('For', 'j')),
                       ('For', 'i'): (('IfThenElse', ('i',)),)}
    verify_structure(new_stmt, expected_struct)

def test_no_else():
    ib = tvm.tir.ir_builder.create()
    l = te.var('l')
    m = te.var('m')
    n = te.var('n')

    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(ib.likely(i < 2)):
                    ib.emit(tvm.tir.Evaluate(m))

    stmt = ib.get()
    new_stmt = tvm.tir.ir_pass.HoistIfThenElse(stmt)
    expected_struct = {('For', 'k'): (None,), ('For', 'j'): (('For', 'k'),),
                       ('IfThenElse', ('i',)): (('For', 'j'), None),
                       ('For', 'i'): (('IfThenElse', ('i',)),)}
    verify_structure(new_stmt, expected_struct)

def test_attr_stmt():
    ib = tvm.tir.ir_builder.create()
    dshape = (32, 64)
    data = ib.pointer("float32", name="data")
    l = te.var('l')
    m = te.var('m')
    n = te.var('n')

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", dshape[0])
    ib.scope_attr(bx, "thread_extent", dshape[1])
    with ib.for_range(0, l, "i") as i:
        with ib.for_range(0, m, "j") as j:
            with ib.for_range(0, n, "k") as k:
                with ib.if_scope(tvm.tir.any(i < 4, j >= 8)):
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k]  + 0.5
                with ib.else_scope():
                    data[bx * j + tx * j * k] = data[bx * j + tx * j * k]  + 1.0

    stmt = ib.get()
    new_stmt = tvm.tir.ir_pass.HoistIfThenElse(stmt)
    expected_struct = {('For', 'k'): (None,), ('IfThenElse', ('i', 'j')): (('For', 'k'), ('For', 'k')),
                       ('For', 'j'): (('IfThenElse', ('i', 'j')),), ('For', 'i'): (('For', 'j'),),
                       ('AttrStmt', 'thread_extent', 64): (('For', 'i'),),
                       ('AttrStmt', 'thread_extent', 32): (('AttrStmt', 'thread_extent', 64),)}
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
    new_stmt = tvm.tir.ir_pass.HoistIfThenElse(stmt)
    expected_struct = {('IfThenElse', ('i', 'j')): (None, None), ('For', 'l'): (('IfThenElse', ('i', 'j')),),
                       ('For', 'k'): (('For', 'l'),), ('For', 'j'): (None,), ('IfThenElse', ('i',)): (('For', 'j'), None),
                       ('For', 'i'): (('IfThenElse', ('i',)),)}
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
                        with ib.if_scope(j <5):
                            data[i * 3 + j + k + l] = data[i * 3 + j + k + l] - 1


    with ib.for_range(0, 5, "i") as i:
        with ib.for_range(0, 10, "j") as j:
                with ib.for_range(0, 15, "k") as k:
                    with ib.if_scope(n >= 3):
                        data[i * 3 + j + k] = data[i * 3 + j + k] + 0.6

    stmt = ib.get()
    new_stmt = tvm.tir.ir_pass.HoistIfThenElse(stmt)
    expected_struct = {('IfThenElse', ('i', 'j')): (None, None), ('IfThenElse', ('j',)): (None, None),
                       ('For', 'l'): (None,), ('For', 'k'): (None,), ('For', 'j'): (('For', 'j'),),
                       ('IfThenElse', ('i',)): (('For', 'j'), None), ('For', 'i'): (('IfThenElse', ('i',)),),
                       ('IfThenElse', ('n',)): (('For', 'j'), None)}
    verify_structure(new_stmt, expected_struct)


if __name__ == "__main__":
    test_basic()
    test_no_else()
    test_attr_stmt()
    test_nested_for()
    test_if_block()
