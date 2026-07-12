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
"""
Tests for StmtVisitor and StmtMutator functionality in TVM TIR.
"""

import tvm
import tvm.testing
from tvm import tirx as tir
from tvm.ir import Range
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.expr import EQ, GT, LT, Add, IntImm, Mul, Sub, Var
from tvm.tirx.stmt_functor import StmtExprMutator, StmtExprVisitor, StmtMutator, StmtVisitor


class ASTLog:
    """Helper class to log AST traversal"""

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


class BasicStmtVisitor(StmtVisitor):
    """Default StmtVisitor - doesn't override any methods"""

    pass


class ASTPrinter(StmtVisitor):
    """Print TIR AST in structured format."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_bind_(self, op):
        self.log.add("Bind")
        self.log.push_scope()
        self.visit_expr(op.value)
        self.log.pop_scope()

    def visit_attr_(self, op):
        self.log.add("AttrStmt")
        self.log.push_scope()
        self.visit_expr(op.value)
        self.visit_stmt(op.body)
        self.log.pop_scope()

    def visit_assert_(self, op):
        self.log.add("AssertStmt")
        self.log.push_scope()
        self.visit_expr(op.condition)
        self.visit_expr(op.message)
        self.visit_stmt(op.body)
        self.log.pop_scope()

    def visit_for_(self, op):
        self.log.add("For")
        self.log.push_scope()
        self.visit_expr(op.min)
        self.visit_expr(op.extent)
        self.visit_stmt(op.body)
        self.log.pop_scope()

    def visit_while_(self, op):
        self.log.add("While")
        self.log.push_scope()
        self.visit_expr(op.condition)
        self.visit_stmt(op.body)
        self.log.pop_scope()

    def visit_buffer_store_(self, op):
        self.log.add("BufferStore")
        self.log.push_scope()
        self.visit_expr(op.value)
        for index in op.indices:
            self.visit_expr(index)
        self.log.pop_scope()

    def visit_seqstmt_(self, op):
        self.log.add("SeqStmt")
        self.log.push_scope()
        for stmt in op.seq:
            self.visit_stmt(stmt)
        self.log.pop_scope()

    def visit_evaluate_(self, op):
        self.log.add("Evaluate")
        self.log.push_scope()
        self.visit_expr(op.value)
        self.log.pop_scope()

    def visit_block_(self, op):
        self.log.add("Block")
        self.log.push_scope()
        if op.init is not None:
            self.visit_stmt(op.init)
        self.visit_stmt(op.body)
        self.log.pop_scope()

    def visit_block_realize_(self, op):
        self.log.add("BlockRealize")
        self.log.push_scope()
        for val in op.iter_values:
            self.visit_expr(val)
        self.visit_expr(op.predicate)
        self.visit_stmt(op.block)
        self.log.pop_scope()

    def visit_if_then_else_(self, op):
        self.log.add("IfThenElse")
        self.log.push_scope()
        self.visit_expr(op.condition)
        self.visit_stmt(op.then_case)
        if op.else_case:
            self.visit_stmt(op.else_case)
        self.log.pop_scope()

    def visit_decl_buffer_(self, op):
        self.log.add("DeclBuffer")
        self.log.push_scope()
        self.visit_stmt(op.body)
        self.log.pop_scope()

    def visit_break_(self, op):
        self.log.add("Break")

    def visit_continue_(self, op):
        self.log.add("Continue")

    def visit_op_call_(self, op):
        self.log.add("OpCall")
        self.log.push_scope()
        for arg in op.args:
            if isinstance(arg, tir.BufferRegion):
                self.visit_buffer_region_(arg)
            else:
                self.visit_expr(arg)
        self.log.pop_scope()

    def visit_buffer_region_(self, op):
        self.log.add("BufferRegion")
        self.log.push_scope()
        for r in op.region:
            self.visit_expr(r.min)
            self.visit_expr(r.extent)
        self.log.pop_scope()

    def visit_expr(self, expr):
        """Simple expression visitor that logs expression types."""
        if expr is None:
            return

        if isinstance(expr, Var):
            self.log.add("Var")
        elif isinstance(expr, IntImm):
            self.log.add("IntImm")
        elif isinstance(expr, Add):
            self.log.add("Add")
            self.log.push_scope()
            self.visit_expr(expr.a)
            self.visit_expr(expr.b)
            self.log.pop_scope()
        elif isinstance(expr, Sub):
            self.log.add("Sub")
            self.log.push_scope()
            self.visit_expr(expr.a)
            self.visit_expr(expr.b)
            self.log.pop_scope()
        elif isinstance(expr, Mul):
            self.log.add("Mul")
            self.log.push_scope()
            self.visit_expr(expr.a)
            self.visit_expr(expr.b)
            self.log.pop_scope()
        elif isinstance(expr, EQ):
            self.log.add("EQ")
            self.log.push_scope()
            self.visit_expr(expr.a)
            self.visit_expr(expr.b)
            self.log.pop_scope()
        elif isinstance(expr, LT):
            self.log.add("LT")
            self.log.push_scope()
            self.visit_expr(expr.a)
            self.visit_expr(expr.b)
            self.log.pop_scope()
        elif isinstance(expr, GT):
            self.log.add("GT")
            self.log.push_scope()
            self.visit_expr(expr.a)
            self.visit_expr(expr.b)
            self.log.pop_scope()
        else:
            self.log.add(f"Expr::{type(expr).__name__}")


class ASTPrinterMutator(StmtMutator):
    """Print TIR AST in post-order while mutating."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_bind_(self, op):
        result = super().visit_bind_(op)
        self.log.add("Bind")
        return result

    def visit_attr_(self, op):
        result = super().visit_attr_(op)
        self.log.add("AttrStmt")
        return result

    def visit_assert_(self, op):
        result = super().visit_assert_(op)
        self.log.add("AssertStmt")
        return result

    def visit_for_(self, op):
        result = super().visit_for_(op)
        self.log.add("For")
        return result

    def visit_while_(self, op):
        result = super().visit_while_(op)
        self.log.add("While")
        return result

    def visit_buffer_store_(self, op):
        result = super().visit_buffer_store_(op)
        self.log.add("BufferStore")
        return result

    def visit_seqstmt_(self, op):
        result = super().visit_seqstmt_(op)
        self.log.add("SeqStmt")
        return result

    def visit_evaluate_(self, op):
        result = super().visit_evaluate_(op)
        self.log.add("Evaluate")
        return result

    def visit_block_(self, op):
        result = super().visit_block_(op)
        self.log.add("Block")
        return result

    def visit_block_realize_(self, op):
        result = super().visit_block_realize_(op)
        self.log.add("BlockRealize")
        return result

    def visit_if_then_else_(self, op):
        result = super().visit_if_then_else_(op)
        self.log.add("IfThenElse")
        return result

    def visit_decl_buffer_(self, op):
        result = super().visit_decl_buffer_(op)
        self.log.add("DeclBuffer")
        return result

    def visit_break_(self, op):
        result = super().visit_break_(op)
        self.log.add("Break")
        return result

    def visit_continue_(self, op):
        result = super().visit_continue_(op)
        self.log.add("Continue")
        return result

    def visit_op_call_(self, op):
        result = super().visit_op_call_(op)
        self.log.add("OpCall")
        return result

    def visit_buffer_region_(self, op):
        result = super().visit_buffer_region_(op)
        self.log.add("BufferRegion")
        return result

    def visit_expr(self, expr):
        """Simple expression visitor that logs expression types."""
        if expr is None:
            return expr

        if isinstance(expr, Var):
            self.log.add("Var")
            return expr
        elif isinstance(expr, IntImm):
            self.log.add("IntImm")
            return expr
        elif isinstance(expr, Add):
            a = self.visit_expr(expr.a)
            b = self.visit_expr(expr.b)
            self.log.add("Add")
            if a is expr.a and b is expr.b:
                return expr
            return tir.Add(a, b)
        elif isinstance(expr, Sub):
            a = self.visit_expr(expr.a)
            b = self.visit_expr(expr.b)
            self.log.add("Sub")
            if a is expr.a and b is expr.b:
                return expr
            return tir.Sub(a, b)
        elif isinstance(expr, Mul):
            a = self.visit_expr(expr.a)
            b = self.visit_expr(expr.b)
            self.log.add("Mul")
            if a is expr.a and b is expr.b:
                return expr
            return tir.Mul(a, b)
        elif isinstance(expr, EQ):
            a = self.visit_expr(expr.a)
            b = self.visit_expr(expr.b)
            self.log.add("EQ")
            if a is expr.a and b is expr.b:
                return expr
            return tir.EQ(a, b)
        elif isinstance(expr, LT):
            a = self.visit_expr(expr.a)
            b = self.visit_expr(expr.b)
            self.log.add("LT")
            if a is expr.a and b is expr.b:
                return expr
            return tir.LT(a, b)
        elif isinstance(expr, GT):
            a = self.visit_expr(expr.a)
            b = self.visit_expr(expr.b)
            self.log.add("GT")
            if a is expr.a and b is expr.b:
                return expr
            return tir.GT(a, b)
        else:
            self.log.add(f"Expr::{type(expr).__name__}")
            return expr


class StmtExprASTPrinter(StmtExprVisitor):
    """AST printer using StmtExprVisitor."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_bind_(self, op):
        self.log.add("Bind")
        self.log.push_scope()
        super().visit_bind_(op)
        self.log.pop_scope()

    def visit_attr_(self, op):
        self.log.add("AttrStmt")
        self.log.push_scope()
        super().visit_attr_(op)
        self.log.pop_scope()

    def visit_assert_(self, op):
        self.log.add("AssertStmt")
        self.log.push_scope()
        super().visit_assert_(op)
        self.log.pop_scope()

    def visit_for_(self, op):
        self.log.add("For")
        self.log.push_scope()
        super().visit_for_(op)
        self.log.pop_scope()

    def visit_while_(self, op):
        self.log.add("While")
        self.log.push_scope()
        super().visit_while_(op)
        self.log.pop_scope()

    def visit_buffer_store_(self, op):
        self.log.add("BufferStore")
        self.log.push_scope()
        super().visit_buffer_store_(op)
        self.log.pop_scope()

    def visit_seqstmt_(self, op):
        self.log.add("SeqStmt")
        self.log.push_scope()
        super().visit_seqstmt_(op)
        self.log.pop_scope()

    def visit_evaluate_(self, op):
        self.log.add("Evaluate")
        self.log.push_scope()
        super().visit_evaluate_(op)
        self.log.pop_scope()

    def visit_block_(self, op):
        self.log.add("Block")
        self.log.push_scope()
        super().visit_block_(op)
        self.log.pop_scope()

    def visit_block_realize_(self, op):
        self.log.add("BlockRealize")
        self.log.push_scope()
        super().visit_block_realize_(op)
        self.log.pop_scope()

    def visit_if_then_else_(self, op):
        self.log.add("IfThenElse")
        self.log.push_scope()
        super().visit_if_then_else_(op)
        self.log.pop_scope()

    def visit_decl_buffer_(self, op):
        self.log.add("DeclBuffer")
        self.log.push_scope()
        super().visit_decl_buffer_(op)
        self.log.pop_scope()

    def visit_break_(self, op):
        self.log.add("Break")
        super().visit_break_(op)

    def visit_continue_(self, op):
        self.log.add("Continue")
        super().visit_continue_(op)

    # ExprVisitor methods
    def visit_var_(self, op):
        self.log.add("Var")

    def visit_int_imm_(self, op):
        self.log.add("IntImm")

    def visit_add_(self, op):
        self.log.add("Add")
        self.log.push_scope()
        super().visit_add_(op)
        self.log.pop_scope()

    def visit_sub_(self, op):
        self.log.add("Sub")
        self.log.push_scope()
        super().visit_sub_(op)
        self.log.pop_scope()

    def visit_mul_(self, op):
        self.log.add("Mul")
        self.log.push_scope()
        super().visit_mul_(op)
        self.log.pop_scope()

    def visit_eq_(self, op):
        self.log.add("EQ")
        self.log.push_scope()
        super().visit_eq_(op)
        self.log.pop_scope()

    def visit_lt_(self, op):
        self.log.add("LT")
        self.log.push_scope()
        super().visit_lt_(op)
        self.log.pop_scope()

    def visit_gt_(self, op):
        self.log.add("GT")
        self.log.push_scope()
        super().visit_gt_(op)
        self.log.pop_scope()


class StmtExprMutatorPrinter(StmtExprMutator):
    """AST mutator printer using StmtExprMutator."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_bind_(self, op):
        result = super().visit_bind_(op)
        self.log.add("Bind")
        return result

    def visit_attr_(self, op):
        result = super().visit_attr_(op)
        self.log.add("AttrStmt")
        return result

    def visit_assert_(self, op):
        result = super().visit_assert_(op)
        self.log.add("AssertStmt")
        return result

    def visit_for_(self, op):
        result = super().visit_for_(op)
        self.log.add("For")
        return result

    def visit_while_(self, op):
        result = super().visit_while_(op)
        self.log.add("While")
        return result

    def visit_buffer_store_(self, op):
        result = super().visit_buffer_store_(op)
        self.log.add("BufferStore")
        return result

    def visit_seqstmt_(self, op):
        result = super().visit_seqstmt_(op)
        self.log.add("SeqStmt")
        return result

    def visit_evaluate_(self, op):
        result = super().visit_evaluate_(op)
        self.log.add("Evaluate")
        return result

    def visit_block_(self, op):
        result = super().visit_block_(op)
        self.log.add("Block")
        return result

    def visit_block_realize_(self, op):
        result = super().visit_block_realize_(op)
        self.log.add("BlockRealize")
        return result

    # ExprMutator methods
    def visit_var_(self, op):
        result = super().visit_var_(op)
        self.log.add("Var")
        return result

    def visit_int_imm_(self, op):
        result = super().visit_int_imm_(op)
        self.log.add("IntImm")
        return result

    def visit_add_(self, op):
        result = super().visit_add_(op)
        self.log.add("Add")
        return result

    def visit_sub_(self, op):
        result = super().visit_sub_(op)
        self.log.add("Sub")
        return result

    def visit_mul_(self, op):
        result = super().visit_mul_(op)
        self.log.add("Mul")
        return result

    def visit_eq_(self, op):
        result = super().visit_eq_(op)
        self.log.add("EQ")
        return result

    def visit_lt_(self, op):
        result = super().visit_lt_(op)
        self.log.add("LT")
        return result

    def visit_gt_(self, op):
        result = super().visit_gt_(op)
        self.log.add("GT")
        return result


def basic_check(stmt, visitor_str, mutator_str):
    """Check visitor and mutator behavior on the given statement."""
    # Check basic visitor
    basic_visitor = BasicStmtVisitor()
    basic_visitor.visit_stmt(stmt)

    # Check AST printer visitor
    log_visitor = ASTPrinter()
    log_visitor.visit_stmt(stmt)
    assert str(log_visitor.log) == visitor_str

    # Check AST printer mutator
    log_mutator = ASTPrinterMutator()
    result = log_mutator.visit_stmt(stmt)
    # Check we get back structurally equivalent statement
    tvm.ir.assert_structural_equal(result, stmt)
    assert str(log_mutator.log) == mutator_str


def create_test_statements():
    """Create test statements for various TIR constructs."""
    x = tir.Var("x", "int32")

    # IntImm
    int_imm = tir.IntImm("int32", 10)

    # Simple expression
    add_expr = tir.Add(x, int_imm)

    # Evaluate
    evaluate_stmt = tir.Evaluate(add_expr)

    # Bind + SeqStmt (was LetStmt)
    let_stmt = tir.SeqStmt([tir.Bind(x, int_imm), evaluate_stmt])

    # For loop
    for_loop = tir.For(x, 0, 10, tir.ForKind.SERIAL, evaluate_stmt)

    # While loop
    while_loop = tir.While(tir.LT(x, int_imm), evaluate_stmt)

    # Buffer operations
    buffer_var = tir.Var("buf", "handle")
    buffer = tir.decl_buffer((10,), "int32", buffer_var.name_hint)
    buffer_store = tir.BufferStore(buffer, add_expr, [int_imm])

    # Sequence of statements
    seq_stmt = tir.SeqStmt([evaluate_stmt, for_loop])

    # Block with iteration variables
    iter_var = tir.IterVar(Range(0, 10), x, 0)
    block = tir.SBlock([iter_var], [], [], "block", evaluate_stmt)
    block_realize = tir.SBlockRealize([int_imm], tir.IntImm("bool", 1), block)

    # IfThenElse statement
    if_then_else = tir.IfThenElse(tir.LT(x, int_imm), evaluate_stmt, evaluate_stmt)

    # Break and continue statements inside a for loop
    @T.prim_func(s_tir=True)
    def func(A: T.Buffer((10,), "int32")):
        for x in range(10):
            A[x] = x + 1
            if x == 5:
                break
            continue

    # DeclBuffer
    buffer_decl = tir.DeclBuffer(T.buffer((10,), "int32"), evaluate_stmt)

    # OpCall
    @T.prim_func(s_tir=True)
    def op_call(A: T.Buffer((10,), "int32"), B: T.Buffer((10,), "int32")):
        Tx.add(A, B, 1.0)

    return {
        "evaluate": evaluate_stmt,
        "let": let_stmt,
        "for": for_loop,
        "while": while_loop,
        "buffer_store": buffer_store,
        "seq_stmt": seq_stmt,
        "block_realize": block_realize,
        "if_then_else": if_then_else,
        "for_with_break": func.body,
        "decl_buffer": buffer_decl,
        "op_call": op_call.body,
    }


def test_evaluate():
    """Test evaluate statement."""
    evaluate_stmt = create_test_statements()["evaluate"]
    basic_check(
        evaluate_stmt,
        "\n".join(["Evaluate", "\tAdd", "\t\tVar", "\t\tIntImm"]),
        "\n".join(["Var", "IntImm", "Add", "Evaluate"]),
    )


def test_let():
    """Test let statement (Bind + SeqStmt)."""
    let_stmt = create_test_statements()["let"]
    basic_check(
        let_stmt,
        "\n".join(
            [
                "SeqStmt",
                "\tBind",
                "\t\tIntImm",
                "\tEvaluate",
                "\t\tAdd",
                "\t\t\tVar",
                "\t\t\tIntImm",
            ]
        ),
        "\n".join(["IntImm", "Bind", "Var", "IntImm", "Add", "Evaluate", "SeqStmt"]),
    )


def test_for():
    """Test for loop statement."""
    for_loop = create_test_statements()["for"]
    basic_check(
        for_loop,
        "\n".join(
            ["For", "\tIntImm", "\tIntImm", "\tEvaluate", "\t\tAdd", "\t\t\tVar", "\t\t\tIntImm"]
        ),
        "\n".join(["IntImm", "IntImm", "Var", "IntImm", "Add", "Evaluate", "For"]),
    )


def test_while():
    """Test while loop statement."""
    while_loop = create_test_statements()["while"]
    basic_check(
        while_loop,
        "\n".join(
            [
                "While",
                "\tLT",
                "\t\tVar",
                "\t\tIntImm",
                "\tEvaluate",
                "\t\tAdd",
                "\t\t\tVar",
                "\t\t\tIntImm",
            ]
        ),
        "\n".join(["Var", "IntImm", "LT", "Var", "IntImm", "Add", "Evaluate", "While"]),
    )


def test_buffer_store():
    """Test buffer store statement."""
    buffer_store = create_test_statements()["buffer_store"]
    basic_check(
        buffer_store,
        "\n".join(["BufferStore", "\tAdd", "\t\tVar", "\t\tIntImm", "\tIntImm"]),
        "\n".join(["Var", "IntImm", "Add", "IntImm", "BufferStore"]),
    )


def test_seq_stmt():
    """Test sequence statement."""
    seq_stmt = create_test_statements()["seq_stmt"]
    basic_check(
        seq_stmt,
        "\n".join(
            [
                "SeqStmt",
                "\tEvaluate",
                "\t\tAdd",
                "\t\t\tVar",
                "\t\t\tIntImm",
                "\tFor",
                "\t\tIntImm",
                "\t\tIntImm",
                "\t\tEvaluate",
                "\t\t\tAdd",
                "\t\t\t\tVar",
                "\t\t\t\tIntImm",
            ]
        ),
        "\n".join(
            [
                "Var",
                "IntImm",
                "Add",
                "Evaluate",
                "IntImm",
                "IntImm",
                "Var",
                "IntImm",
                "Add",
                "Evaluate",
                "For",
                "SeqStmt",
            ]
        ),
    )


def test_block_realize():
    """Test block realize statement."""
    block_realize = create_test_statements()["block_realize"]
    basic_check(
        block_realize,
        "\n".join(
            [
                "BlockRealize",
                "\tIntImm",
                "\tIntImm",
                "\tBlock",
                "\t\tEvaluate",
                "\t\t\tAdd",
                "\t\t\t\tVar",
                "\t\t\t\tIntImm",
            ]
        ),
        "\n".join(
            [
                "IntImm",
                "IntImm",
                "IntImm",
                "IntImm",
                "Var",
                "IntImm",
                "Add",
                "Evaluate",
                "Block",
                "BlockRealize",
            ]
        ),
    )


def test_if_then_else():
    """Test if-then-else statement."""
    if_then_else = create_test_statements()["if_then_else"]
    basic_check(
        if_then_else,
        "\n".join(
            [
                "IfThenElse",
                "\tLT",
                "\t\tVar",
                "\t\tIntImm",
                "\tEvaluate",
                "\t\tAdd",
                "\t\t\tVar",
                "\t\t\tIntImm",
                "\tEvaluate",
                "\t\tAdd",
                "\t\t\tVar",
                "\t\t\tIntImm",
            ]
        ),
        "\n".join(
            [
                "Var",
                "IntImm",
                "LT",
                "Var",
                "IntImm",
                "Add",
                "Evaluate",
                "Var",
                "IntImm",
                "Add",
                "Evaluate",
                "IfThenElse",
            ]
        ),
    )


def test_for_with_break_continue():
    """Test for loop with break and continue statements."""
    for_with_break = create_test_statements()["for_with_break"]
    basic_check(
        for_with_break,
        "\n".join(
            [
                "For",
                "\tIntImm",
                "\tIntImm",
                "\tSeqStmt",
                "\t\tBufferStore",
                "\t\t\tAdd",
                "\t\t\t\tVar",
                "\t\t\t\tIntImm",
                "\t\t\tVar",
                "\t\tIfThenElse",
                "\t\t\tEQ",
                "\t\t\t\tVar",
                "\t\t\t\tIntImm",
                "\t\t\tEvaluate",
                "\t\t\t\tExpr::Call",
                "\t\tEvaluate",
                "\t\t\tExpr::Call",
            ]
        ),
        "\n".join(
            [
                "IntImm",
                "IntImm",
                "Var",
                "IntImm",
                "Add",
                "Var",
                "BufferStore",
                "Var",
                "IntImm",
                "EQ",
                "Expr::Call",
                "Evaluate",
                "IfThenElse",
                "Expr::Call",
                "Evaluate",
                "SeqStmt",
                "For",
            ]
        ),
    )


def test_decl_buffer():
    """Test buffer declaration statement."""
    buffer_decl = create_test_statements()["decl_buffer"]
    basic_check(
        buffer_decl,
        "\n".join(["DeclBuffer", "\tEvaluate", "\t\tAdd", "\t\t\tVar", "\t\t\tIntImm"]),
        "\n".join(["Var", "IntImm", "Add", "Evaluate", "DeclBuffer"]),
    )


def test_op_call():
    """Test op call statement"""
    op_call = create_test_statements()["op_call"]
    basic_check(
        op_call,
        "\n".join(
            [
                "OpCall",
                "\tBufferRegion",
                "\t\tIntImm",
                "\t\tIntImm",
                "\tBufferRegion",
                "\t\tIntImm",
                "\t\tIntImm",
                "\tExpr::FloatImm",
            ]
        ),
        "\n".join(
            [
                "IntImm",
                "IntImm",
                "BufferRegion",
                "IntImm",
                "IntImm",
                "BufferRegion",
                "Expr::FloatImm",
                "OpCall",
            ]
        ),
    )


def test_stmt_expr_mutator():
    """Test StmtExprMutator."""
    evaluate_stmt = create_test_statements()["evaluate"]
    mutator = StmtExprMutatorPrinter()
    result = mutator.visit_stmt(evaluate_stmt)
    tvm.ir.assert_structural_equal(result, evaluate_stmt)

    expected = "\n".join(["Var", "IntImm", "Add", "Evaluate"])
    assert str(mutator.log) == expected


def test_stmt_expr_visitor():
    """Test StmtExprVisitor."""
    evaluate_stmt = create_test_statements()["evaluate"]
    visitor = StmtExprASTPrinter()
    visitor.visit_stmt(evaluate_stmt)
    expected = "\n".join(["Evaluate", "\tAdd", "\t\tVar", "\t\tIntImm"])
    assert str(visitor.log) == expected


class NegateIntImmMutator(StmtExprMutator):
    """Mutator that negates all integer immediates."""

    def visit_int_imm_(self, op):
        # Create a new IntImm with negated value
        return tir.IntImm(op.ty.dtype, -op.value)


def test_mutator_transformation():
    """Test that mutator actually transforms the AST."""
    evaluate_stmt = create_test_statements()["evaluate"]
    mutator = NegateIntImmMutator()
    result = mutator.visit_stmt(evaluate_stmt)

    # The original has value 10, the transformed should have -10
    assert isinstance(evaluate_stmt.value, tir.Add)
    assert isinstance(evaluate_stmt.value.b, tir.IntImm)
    assert evaluate_stmt.value.b.value == 10

    assert isinstance(result.value, tir.Add)
    assert isinstance(result.value.b, tir.IntImm)
    assert result.value.b.value == -10


class InheritVsMixin:
    """Test inheriting vs mixing in with StmtVisitor/StmtMutator."""

    class InheritedVisitor(StmtVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_for_(self, op):
            self.log.add("InheritedVisitor::For")
            super().visit_for_(op)

    class DerivedVisitor(InheritedVisitor):
        def visit_for_(self, op):
            self.log.add("DerivedVisitor::For")
            super().visit_for_(op)

    class BaseMutator(StmtMutator):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_for_(self, op):
            self.log.add("BaseMutator::For")
            return super().visit_for_(op)

    class DerivedMutator(BaseMutator):
        def visit_for_(self, op):
            self.log.add("DerivedMutator::For")
            return super().visit_for_(op)


def test_inheritance():
    """Test inheritance with visitor and mutator classes."""
    for_loop = create_test_statements()["for"]

    # Test inherited visitor
    visitor = InheritVsMixin.DerivedVisitor()
    visitor.visit_stmt(for_loop)
    expected = "\n".join(["DerivedVisitor::For", "InheritedVisitor::For"])
    assert str(visitor.log) == expected

    # Test derived mutator
    mutator = InheritVsMixin.DerivedMutator()
    result = mutator.visit_stmt(for_loop)
    tvm.ir.assert_structural_equal(result, for_loop)
    expected = "\n".join(["DerivedMutator::For", "BaseMutator::For"])
    assert str(mutator.log) == expected


if __name__ == "__main__":
    tvm.testing.main()
