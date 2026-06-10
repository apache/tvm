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
import tvm.testing
from tvm import tirx as tir
from tvm.ir import Op
from tvm.ir.base import assert_structural_equal
from tvm.tirx.expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    Broadcast,
    BufferLoad,
    Call,
    Cast,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    Let,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    ProducerLoad,
    Ramp,
    Reduce,
    Select,
    Shuffle,
    SizeVar,
    StringImm,
    Sub,
    Var,
)
from tvm.tirx.expr_functor import ExprMutator, ExprVisitor

# Basic example variables for testing
n = tir.Var("n", "int32")
m = tir.Var("m", "int32")
x = tir.Var("x", "float32")
y = tir.Var("y", "float32")


class BasicVisitor(ExprVisitor):
    """Default ExprVisitor"""


class ASTLog:
    """Helper class to log AST"""

    def __init__(self) -> None:
        self.log = []
        self.indent = "\t"
        self.level = 0

    def push_scope(self):
        self.level += 1

    def pop_scope(self):
        self.level -= 1

    def add(self, s: str):
        self.log.append(self.indent * self.level + s)

    def __str__(self) -> str:
        return "\n".join(self.log)


class ASTPrinter(ExprVisitor):
    """Print TIR AST in structured format."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_var_(self, op: Var) -> None:
        self.log.add("Var")

    def visit_size_var_(self, op: SizeVar) -> None:
        self.log.add("SizeVar")

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        self.log.add("BufferLoad")
        self.log.push_scope()
        for idx in op.indices:
            self.visit_expr(idx)
        self.log.pop_scope()

    def visit_producer_load_(self, op: ProducerLoad) -> None:
        self.log.add("ProducerLoad")
        self.log.push_scope()
        for idx in op.indices:
            self.visit_expr(idx)
        self.log.pop_scope()

    def visit_let_(self, op: Let) -> None:
        self.log.add("Let")
        self.log.push_scope()
        self.visit_expr(op.var)
        self.visit_expr(op.value)
        self.visit_expr(op.body)
        self.log.pop_scope()

    def visit_call_(self, op: Call) -> None:
        self.log.add("Call")
        self.log.push_scope()
        if isinstance(op.op, Op):
            self.log.add("Op")
        else:
            self.visit_expr(op.op)
        for arg in op.args:
            self.visit_expr(arg)
        self.log.pop_scope()

    def visit_add_(self, op: Add) -> None:
        self.log.add("Add")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_sub_(self, op: Sub) -> None:
        self.log.add("Sub")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_mul_(self, op: Mul) -> None:
        self.log.add("Mul")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_div_(self, op: Div) -> None:
        self.log.add("Div")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_mod_(self, op: Mod) -> None:
        self.log.add("Mod")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_floordiv_(self, op: FloorDiv) -> None:
        self.log.add("FloorDiv")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_floormod_(self, op: FloorMod) -> None:
        self.log.add("FloorMod")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_min_(self, op: Min) -> None:
        self.log.add("Min")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_max_(self, op: Max) -> None:
        self.log.add("Max")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_eq_(self, op: EQ) -> None:
        self.log.add("EQ")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_ne_(self, op: NE) -> None:
        self.log.add("NE")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_lt_(self, op: LT) -> None:
        self.log.add("LT")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_le_(self, op: LE) -> None:
        self.log.add("LE")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_gt_(self, op: GT) -> None:
        self.log.add("GT")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_ge_(self, op: GE) -> None:
        self.log.add("GE")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_and_(self, op: And) -> None:
        self.log.add("And")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_or_(self, op: Or) -> None:
        self.log.add("Or")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.visit_expr(op.b)
        self.log.pop_scope()

    def visit_reduce_(self, op: Reduce) -> None:
        self.log.add("Reduce")
        self.log.push_scope()
        for source in op.source:
            self.visit_expr(source)
        for axis in op.axis:
            self.visit_expr(axis.var)
        self.visit_expr(op.condition)
        self.log.pop_scope()

    def visit_cast_(self, op: Cast) -> None:
        self.log.add("Cast")
        self.log.push_scope()
        self.visit_expr(op.value)
        self.log.pop_scope()

    def visit_not_(self, op: Not) -> None:
        self.log.add("Not")
        self.log.push_scope()
        self.visit_expr(op.a)
        self.log.pop_scope()

    def visit_select_(self, op: Select) -> None:
        self.log.add("Select")
        self.log.push_scope()
        self.visit_expr(op.condition)
        self.visit_expr(op.true_value)
        self.visit_expr(op.false_value)
        self.log.pop_scope()

    def visit_ramp_(self, op: Ramp) -> None:
        self.log.add("Ramp")
        self.log.push_scope()
        self.visit_expr(op.base)
        self.visit_expr(op.stride)
        self.visit_expr(op.lanes)
        self.log.pop_scope()

    def visit_broadcast_(self, op: Broadcast) -> None:
        self.log.add("Broadcast")
        self.log.push_scope()
        self.visit_expr(op.value)
        self.visit_expr(op.lanes)
        self.log.pop_scope()

    def visit_shuffle_(self, op: Shuffle) -> None:
        self.log.add("Shuffle")
        self.log.push_scope()
        for vec in op.vectors:
            self.visit_expr(vec)
        for idx in op.indices:
            self.visit_expr(idx)
        self.log.pop_scope()

    def visit_int_imm_(self, op: IntImm) -> None:
        self.log.add("IntImm")

    def visit_float_imm_(self, op: FloatImm) -> None:
        self.log.add("FloatImm")

    def visit_string_imm_(self, op: StringImm) -> None:
        self.log.add("StringImm")


class BasicMutator(ExprMutator):
    """Default ExprMutator"""


class ASTPostPrinterMutator(ExprMutator):
    """Print TIR AST in the post order format."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_var_(self, op: Var) -> tir.PrimExpr:
        result = super().visit_var_(op)
        self.log.add("Var")
        return result

    def visit_size_var_(self, op: SizeVar) -> tir.PrimExpr:
        result = op
        self.log.add("SizeVar")
        return result

    def visit_buffer_load_(self, op: BufferLoad) -> tir.PrimExpr:
        result = super().visit_buffer_load_(op)
        self.log.add("BufferLoad")
        return result

    def visit_producer_load_(self, op: ProducerLoad) -> tir.PrimExpr:
        result = super().visit_producer_load_(op)
        self.log.add("ProducerLoad")
        return result

    def visit_let_(self, op: Let) -> tir.PrimExpr:
        result = super().visit_let_(op)
        self.log.add("Let")
        return result

    def visit_call_(self, op: Call) -> tir.PrimExpr:
        result = super().visit_call_(op)
        self.log.add("Call")
        return result

    def visit_add_(self, op: Add) -> tir.PrimExpr:
        result = super().visit_add_(op)
        self.log.add("Add")
        return result

    def visit_sub_(self, op: Sub) -> tir.PrimExpr:
        result = super().visit_sub_(op)
        self.log.add("Sub")
        return result

    def visit_mul_(self, op: Mul) -> tir.PrimExpr:
        result = super().visit_mul_(op)
        self.log.add("Mul")
        return result

    def visit_div_(self, op: Div) -> tir.PrimExpr:
        result = super().visit_div_(op)
        self.log.add("Div")
        return result

    def visit_mod_(self, op: Mod) -> tir.PrimExpr:
        result = super().visit_mod_(op)
        self.log.add("Mod")
        return result

    def visit_floordiv_(self, op: FloorDiv) -> tir.PrimExpr:
        result = super().visit_floordiv_(op)
        self.log.add("FloorDiv")
        return result

    def visit_floormod_(self, op: FloorMod) -> tir.PrimExpr:
        result = super().visit_floormod_(op)
        self.log.add("FloorMod")
        return result

    def visit_min_(self, op: Min) -> tir.PrimExpr:
        result = super().visit_min_(op)
        self.log.add("Min")
        return result

    def visit_max_(self, op: Max) -> tir.PrimExpr:
        result = super().visit_max_(op)
        self.log.add("Max")
        return result

    def visit_eq_(self, op: EQ) -> tir.PrimExpr:
        result = super().visit_eq_(op)
        self.log.add("EQ")
        return result

    def visit_ne_(self, op: NE) -> tir.PrimExpr:
        result = super().visit_ne_(op)
        self.log.add("NE")
        return result

    def visit_lt_(self, op: LT) -> tir.PrimExpr:
        result = super().visit_lt_(op)
        self.log.add("LT")
        return result

    def visit_le_(self, op: LE) -> tir.PrimExpr:
        result = super().visit_le_(op)
        self.log.add("LE")
        return result

    def visit_gt_(self, op: GT) -> tir.PrimExpr:
        result = super().visit_gt_(op)
        self.log.add("GT")
        return result

    def visit_ge_(self, op: GE) -> tir.PrimExpr:
        result = super().visit_ge_(op)
        self.log.add("GE")
        return result

    def visit_and_(self, op: And) -> tir.PrimExpr:
        result = super().visit_and_(op)
        self.log.add("And")
        return result

    def visit_or_(self, op: Or) -> tir.PrimExpr:
        result = super().visit_or_(op)
        self.log.add("Or")
        return result

    def visit_reduce_(self, op: Reduce) -> tir.PrimExpr:
        result = super().visit_reduce_(op)
        self.log.add("Reduce")
        return result

    def visit_cast_(self, op: Cast) -> tir.PrimExpr:
        result = super().visit_cast_(op)
        self.log.add("Cast")
        return result

    def visit_not_(self, op: Not) -> tir.PrimExpr:
        result = super().visit_not_(op)
        self.log.add("Not")
        return result

    def visit_select_(self, op: Select) -> tir.PrimExpr:
        result = super().visit_select_(op)
        self.log.add("Select")
        return result

    def visit_ramp_(self, op: Ramp) -> tir.PrimExpr:
        result = super().visit_ramp_(op)
        self.log.add("Ramp")
        return result

    def visit_broadcast_(self, op: Broadcast) -> tir.PrimExpr:
        result = super().visit_broadcast_(op)
        self.log.add("Broadcast")
        return result

    def visit_shuffle_(self, op: Shuffle) -> tir.PrimExpr:
        result = super().visit_shuffle_(op)
        self.log.add("Shuffle")
        return result

    def visit_int_imm_(self, op: IntImm) -> tir.PrimExpr:
        result = super().visit_int_imm_(op)
        self.log.add("IntImm")
        return result

    def visit_float_imm_(self, op: FloatImm) -> tir.PrimExpr:
        result = super().visit_float_imm_(op)
        self.log.add("FloatImm")
        return result

    def visit_string_imm_(self, op: StringImm) -> tir.PrimExpr:
        result = super().visit_string_imm_(op)
        self.log.add("StringImm")
        return result


def basic_check(expr, visitor_str, mutator_str):
    """Helper function to check visitor and mutator on an expression"""

    # Check visitor
    basic_visitor = BasicVisitor()
    basic_visitor.visit_expr(expr)
    # Check AST printer visitor
    log_visitor = ASTPrinter()
    log_visitor.visit_expr(expr)
    assert str(log_visitor.log) == visitor_str

    # Check basic mutator
    basic_mutator = BasicMutator()
    mutated_expr = basic_mutator.visit_expr(expr)
    assert_structural_equal(mutated_expr, expr)

    # Check post-order printer mutator
    post_log_mutator = ASTPostPrinterMutator()
    mutated_expr = post_log_mutator.visit_expr(expr)
    assert_structural_equal(mutated_expr, expr)
    assert str(post_log_mutator.log) == mutator_str


def test_var():
    basic_check(n, "Var", "Var")


def test_size_var():
    sv = tir.SizeVar("sv", "int32")
    basic_check(sv, "SizeVar", "SizeVar")


def test_int_imm():
    basic_check(tir.IntImm("int32", 10), "IntImm", "IntImm")


def test_float_imm():
    basic_check(tir.FloatImm("float32", 1.5), "FloatImm", "FloatImm")


def test_string_imm():
    basic_check(tir.StringImm("hello"), "StringImm", "StringImm")


def test_add():
    add_node = tir.Add(n, m)
    basic_check(add_node, "\n".join(["Add", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "Add"]))


def test_sub():
    sub_node = tir.Sub(n, m)
    basic_check(sub_node, "\n".join(["Sub", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "Sub"]))


def test_mul():
    mul_node = tir.Mul(n, m)
    basic_check(mul_node, "\n".join(["Mul", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "Mul"]))


def test_div():
    div_node = tir.Div(n, m)
    basic_check(div_node, "\n".join(["Div", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "Div"]))


def test_floor_div():
    floor_div_node = tir.FloorDiv(n, m)
    basic_check(
        floor_div_node,
        "\n".join(["FloorDiv", "\tVar", "\tVar"]),
        "\n".join(["Var", "Var", "FloorDiv"]),
    )


def test_floor_mod():
    floor_mod_node = tir.FloorMod(n, m)
    basic_check(
        floor_mod_node,
        "\n".join(["FloorMod", "\tVar", "\tVar"]),
        "\n".join(["Var", "Var", "FloorMod"]),
    )


def test_min():
    min_node = tir.Min(n, m)
    basic_check(min_node, "\n".join(["Min", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "Min"]))


def test_max():
    max_node = tir.Max(n, m)
    basic_check(max_node, "\n".join(["Max", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "Max"]))


def test_eq():
    eq_node = tir.EQ(n, m)
    basic_check(eq_node, "\n".join(["EQ", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "EQ"]))


def test_ne():
    ne_node = tir.NE(n, m)
    basic_check(ne_node, "\n".join(["NE", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "NE"]))


def test_lt():
    lt_node = tir.LT(n, m)
    basic_check(lt_node, "\n".join(["LT", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "LT"]))


def test_le():
    le_node = tir.LE(n, m)
    basic_check(le_node, "\n".join(["LE", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "LE"]))


def test_gt():
    gt_node = tir.GT(n, m)
    basic_check(gt_node, "\n".join(["GT", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "GT"]))


def test_ge():
    ge_node = tir.GE(n, m)
    basic_check(ge_node, "\n".join(["GE", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "GE"]))


def test_and():
    and_node = tir.And(tir.EQ(n, m), tir.LT(n, 10))
    basic_check(
        and_node,
        "\n".join(["And", "\tEQ", "\t\tVar", "\t\tVar", "\tLT", "\t\tVar", "\t\tIntImm"]),
        "\n".join(["Var", "Var", "EQ", "Var", "IntImm", "LT", "And"]),
    )


def test_or():
    or_node = tir.Or(tir.EQ(n, m), tir.LT(n, 10))
    basic_check(
        or_node,
        "\n".join(["Or", "\tEQ", "\t\tVar", "\t\tVar", "\tLT", "\t\tVar", "\t\tIntImm"]),
        "\n".join(["Var", "Var", "EQ", "Var", "IntImm", "LT", "Or"]),
    )


def test_not():
    not_node = tir.Not(tir.EQ(n, m))
    basic_check(
        not_node,
        "\n".join(["Not", "\tEQ", "\t\tVar", "\t\tVar"]),
        "\n".join(["Var", "Var", "EQ", "Not"]),
    )


def test_select():
    select_node = tir.Select(tir.EQ(n, m), n, m)
    basic_check(
        select_node,
        "\n".join(["Select", "\tEQ", "\t\tVar", "\t\tVar", "\tVar", "\tVar"]),
        "\n".join(["Var", "Var", "EQ", "Var", "Var", "Select"]),
    )


def test_cast():
    cast_node = tir.Cast("float32", n)
    basic_check(cast_node, "\n".join(["Cast", "\tVar"]), "\n".join(["Var", "Cast"]))


def test_let():
    let_node = tir.Let(n, tir.IntImm("int32", 10), n + 1)
    basic_check(
        let_node,
        "\n".join(["Let", "\tVar", "\tIntImm", "\tAdd", "\t\tVar", "\t\tIntImm"]),
        "\n".join(["Var", "IntImm", "Var", "IntImm", "Add", "Let"]),
    )


def test_ramp():
    ramp_node = tir.Ramp(n, 1, 4)
    basic_check(
        ramp_node,
        "\n".join(["Ramp", "\tVar", "\tIntImm", "\tIntImm"]),
        "\n".join(["Var", "IntImm", "IntImm", "Ramp"]),
    )


def test_broadcast():
    broadcast_node = tir.Broadcast(n, 4)
    basic_check(
        broadcast_node,
        "\n".join(["Broadcast", "\tVar", "\tIntImm"]),
        "\n".join(["Var", "IntImm", "Broadcast"]),
    )


def test_inherit():
    # The internal class is not instantiated.
    class InternalVisitor(ExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_add_(self, op: Add) -> None:
            self.log.add("InternalAdd")
            self.log.push_scope()
            self.visit_expr(op.a)
            self.visit_expr(op.b)
            self.log.pop_scope()

        def visit_var_(self, op: Var) -> None:
            self.log.add("InternalVar")

    class LeafVisitor(InternalVisitor):
        def visit_add_(self, op: Add) -> None:
            self.log.add("LeafAdd")
            self.log.push_scope()
            self.visit_expr(op.a)
            self.visit_expr(op.b)
            self.log.pop_scope()

    add_node = tir.Add(n, m)
    lv = LeafVisitor()
    lv.visit_expr(add_node)
    assert str(lv.log) == "\n".join(["LeafAdd", "\tInternalVar", "\tInternalVar"])


def test_inherit_with_cls():
    class InternalVisitor(ExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_add_(self, op: Add) -> None:
            self.log.add("InternalAdd")
            self.log.push_scope()
            self.visit_expr(op.a)
            self.visit_expr(op.b)
            self.log.pop_scope()

        def visit_var_(self, op: Var) -> None:
            self.log.add("InternalVar")

    class LeafVisitor(InternalVisitor):
        def visit_add_(self, op: Add) -> None:
            self.log.add("LeafAdd")
            self.log.push_scope()
            self.visit_expr(op.a)
            self.visit_expr(op.b)
            self.log.pop_scope()

    add_node = tir.Add(n, m)
    iv = InternalVisitor()
    iv.visit_expr(add_node)
    assert str(iv.log) == "\n".join(["InternalAdd", "\tInternalVar", "\tInternalVar"])

    lv = LeafVisitor()
    lv.visit_expr(add_node)
    assert str(lv.log) == "\n".join(["LeafAdd", "\tInternalVar", "\tInternalVar"])


def test_call_visitor_super():
    class InternalVisitor(ExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_add_(self, op: Add) -> None:
            self.log.add("InternalAdd")
            super().visit_add_(op)  # call ExprVisitor.visit_add_

        def visit_var_(self, op: Var) -> None:
            self.log.add("InternalVar")

        def visit_int_imm_(self, op: IntImm) -> None:
            self.log.add("InternalIntImm")

    class LeafVisitor(InternalVisitor):
        def visit_add_(self, op: Add) -> None:
            self.log.add("LeafAdd")
            super().visit_add_(op)  # call InternalVisitor.visit_add_

    add_node = tir.Add(n, tir.IntImm("int32", 10))
    iv = InternalVisitor()
    iv.visit_expr(add_node)
    assert str(iv.log) == "\n".join(["InternalAdd", "InternalVar", "InternalIntImm"])

    lv = LeafVisitor()
    lv.visit_expr(add_node)
    assert str(lv.log) == "\n".join(["LeafAdd", "InternalAdd", "InternalVar", "InternalIntImm"])


def test_call_mutator_super():
    class InternalMutator(ExprMutator):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_add_(self, op: Add) -> tir.PrimExpr:
            self.log.add("InternalAdd")
            return super().visit_add_(op)  # call ExprMutator.visit_add_

        def visit_var_(self, op: Var) -> tir.PrimExpr:
            self.log.add("InternalVar")
            return super().visit_var_(op)  # call ExprMutator.visit_var_

        def visit_int_imm_(self, op: IntImm) -> tir.PrimExpr:
            self.log.add("InternalIntImm")
            return super().visit_int_imm_(op)  # call ExprMutator.visit_int_imm_

    class LeafMutator(InternalMutator):
        def visit_add_(self, op: Add) -> tir.PrimExpr:
            self.log.add("LeafAdd")
            return super().visit_add_(op)  # call InternalMutator.visit_add_

    add_node = tir.Add(n, tir.IntImm("int32", 10))
    im = InternalMutator()
    im.visit_expr(add_node)
    assert str(im.log) == "\n".join(["InternalAdd", "InternalVar", "InternalIntImm"])

    lm = LeafMutator()
    lm.visit_expr(add_node)
    assert str(lm.log) == "\n".join(["LeafAdd", "InternalAdd", "InternalVar", "InternalIntImm"])


def test_var_mutation():
    """Test mutating variables in a TIR expression"""

    class VarMutator(ExprMutator):
        def __init__(self, var_map):
            super().__init__()
            self.var_map = var_map

        def visit_var_(self, op: Var) -> tir.PrimExpr:
            if op.name in self.var_map:
                return self.var_map[op.name]
            return op

    # Create a simple expression
    expr = n + m

    # Create a mutator that replaces 'n' with a constant
    var_map = {"n": tir.IntImm("int32", 42)}
    mutator = VarMutator(var_map)
    result = mutator.visit_expr(expr)

    # The result should be 42 + m
    expected = tir.Add(tir.IntImm("int32", 42), m)
    assert_structural_equal(result, expected)


if __name__ == "__main__":
    tvm.testing.main()
