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
# ruff: noqa: F841

import tvm
import tvm.testing
from tvm import tirx
from tvm.tirx import (
    EQ,
    GE,
    LT,
    Add,
    AssertStmt,
    Cast,
    Evaluate,
    FloatImm,
    For,
    IfThenElse,
    IntImm,
    Max,
    Min,
    Mul,
    PyStmtExprMutator,
    PyStmtExprMutatorWithAnalyzer,
    PyStmtExprVisitor,
    PyStmtExprVisitorWithAnalyzer,
    Select,
    SeqStmt,
    StringImm,
    Sub,
    Var,
)


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


@tirx.functor.visitor
class ASTPrinter(PyStmtExprVisitor):
    """Print tirx AST in structured format. The shape of Node is ignored."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_var_(self, op: Var) -> None:
        self.log.add("Stmt: Var")
        super().visit_var_(op)

    def visit_add_(self, op: Add) -> None:
        self.log.add("Stmt: Add")
        super().visit_add_(op)


@tirx.functor.visitor
class SimpleExprCounter(PyStmtExprVisitor):
    """Count expressions without recursion"""

    def __init__(self):
        super().__init__()
        self.var_count = 0
        self.add_count = 0
        self.mul_count = 0

    def visit_var_(self, op: Var):
        self.var_count += 1
        # Don't recursively visit children to avoid infinite recursion

    def visit_add_(self, op: Add):
        self.add_count += 1
        # Visit children manually
        super().visit_add_(op)

    def visit_mul_(self, op: Mul):
        self.mul_count += 1
        # Visit children manually
        super().visit_mul_(op)


@tirx.functor.mutator
class VariableReplacer(PyStmtExprMutator):
    """Replace variables with constants"""

    def __init__(self, replacements):
        super().__init__()
        self.replacements = replacements

    def visit_var_(self, op: Var):
        if op.name in self.replacements:
            return IntImm("int32", self.replacements[op.name])
        return op


@tirx.functor.mutator
class AddToSubMutator(PyStmtExprMutator):
    """Convert Add operations to Sub operations"""

    def visit_add_(self, op: Add):
        # First mutate the operands
        a = self.visit_expr(op.a)
        b = self.visit_expr(op.b)
        # Convert Add to Sub
        return Sub(a, b)


@tirx.functor.visitor
class SimpleStmtCounter(PyStmtExprVisitor):
    """Count statements without recursion"""

    def __init__(self):
        super().__init__()
        self.for_count = 0
        self.if_count = 0
        self.evaluate_count = 0

    def visit_for_(self, op: For):
        self.for_count += 1
        super().visit_for_(op)

    def visit_if_then_else_(self, op: IfThenElse):
        self.if_count += 1
        super().visit_if_then_else_(op)

    def visit_evaluate_(self, op: Evaluate):
        self.evaluate_count += 1
        super().visit_evaluate_(op)


@tirx.functor.mutator
class ForLoopUnroller(PyStmtExprMutator):
    """Simple loop unroller for demonstration"""

    def __init__(self, unroll_factor=2):
        super().__init__()
        self.unroll_factor = unroll_factor

    def visit_for_(self, op: For):
        # For demonstration, just return the original for now
        # In a real implementation, we would unroll small loops
        return super().visit_for_(op)


@tirx.functor.visitor
class SimpleStmtExprVisitor(PyStmtExprVisitor):
    """Visitor that handles both statements and expressions"""

    def __init__(self):
        super().__init__()
        self.expr_count = 0
        self.stmt_count = 0
        self.var_names = set()

    def visit_var_(self, op: Var):
        self.var_names.add(op.name)
        self.expr_count += 1

    def visit_evaluate_(self, op: Evaluate):
        self.stmt_count += 1
        # Visit the expression
        self.visit_expr(op.value)


@tirx.functor.mutator
class ComplexMutator(PyStmtExprMutator):
    """Mutator that handles both statements and expressions"""

    def __init__(self):
        super().__init__()
        self.modifications = 0

    def visit_add_(self, op: Add):
        self.modifications += 1
        # Convert a + b to a * 2 + b for demonstration
        a = self.visit_expr(op.a)
        b = self.visit_expr(op.b)
        return Add(Mul(a, IntImm("int32", 2)), b)


@tirx.functor.visitor
class AnalyzerAwareVisitor(PyStmtExprVisitorWithAnalyzer):
    """Record analyzer facts visible from Python visitor callbacks."""

    def __init__(self, var):
        super().__init__()
        self.var = var
        self.facts = []

    def visit_evaluate_(self, op: Evaluate):
        if op.value.same_as(self.var):
            self.facts.append(self.analyzer.can_prove(GE(self.var, IntImm("int32", 0))))
            self.facts.append(self.analyzer.can_prove(LT(self.var, IntImm("int32", 10))))
        super().visit_evaluate_(op)


@tirx.functor.mutator
class AnalyzerAwareMutator(PyStmtExprMutatorWithAnalyzer):
    """Use branch constraints from the analyzer to rewrite proven predicates."""

    def _rewrite_if_proven(self, value):
        if value.dtype == "bool" and self.analyzer.can_prove(value):
            return IntImm("bool", True)
        return value

    def visit_lt_(self, op: LT):
        a = self.visit_expr(op.a)
        b = self.visit_expr(op.b)
        value = op if a.same_as(op.a) and b.same_as(op.b) else LT(a, b, op.span)
        return self._rewrite_if_proven(value)

    def visit_ge_(self, op: GE):
        a = self.visit_expr(op.a)
        b = self.visit_expr(op.b)
        value = op if a.same_as(op.a) and b.same_as(op.b) else GE(a, b, op.span)
        return self._rewrite_if_proven(value)

    def visit_evaluate_(self, op: Evaluate):
        value = self.visit_expr(op.value)
        value = self._rewrite_if_proven(value)
        if value.same_as(op.value):
            return op
        return Evaluate(value, op.span)


@tirx.functor.visitor
class PredicateVisitor(PyStmtExprVisitorWithAnalyzer):
    """Record whether boolean Evaluate nodes are provable in analyzer context."""

    def __init__(self):
        super().__init__()
        self.facts = []

    def visit_evaluate_(self, op: Evaluate):
        if op.value.dtype == "bool":
            self.facts.append(self.analyzer.can_prove(op.value))
        super().visit_evaluate_(op)


@tirx.functor.visitor
class AssertMessageVisitor(PyStmtExprVisitorWithAnalyzer):
    """Record string immediates reached through analyzer-aware assert traversal."""

    def __init__(self):
        super().__init__()
        self.strings = []

    def visit_string_imm_(self, op: StringImm):
        self.strings.append(op.value)


@tirx.functor.mutator
class AssertMessageMutator(PyStmtExprMutatorWithAnalyzer):
    """Rewrite assert message strings through analyzer-aware traversal."""

    def visit_string_imm_(self, op: StringImm):
        if op.value == "bad":
            return StringImm("rewritten")
        return op


def test_basic_visitor():
    """Test the basic AST printer visitor"""
    expr = Add(Var("x", dtype="int32"), Var("y", dtype="int32"))
    printer = ASTPrinter()
    printer.visit_expr(expr)
    assert str(printer.log) == "\n".join(["Stmt: Add", "Stmt: Var", "Stmt: Var"])


def test_simple_expr_counter():
    """Test simple expression counting visitor"""
    x = Var("x", dtype="int32")
    y = Var("y", dtype="int32")

    # Create simple expression: x + y
    expr = Add(x, y)

    counter = SimpleExprCounter()
    counter.visit_expr(expr)

    assert counter.var_count == 2  # x and y
    assert counter.add_count == 1  # one add


def test_variable_replacer():
    """Test expression mutator that replaces variables"""
    x = Var("x", dtype="int32")
    y = Var("y", dtype="int32")
    expr = Add(x, Mul(y, IntImm("int32", 3)))

    replacer = VariableReplacer({"x": 10, "y": 5})
    result = replacer.visit_expr(expr)

    # Should be Add(IntImm(10), Mul(IntImm(5), IntImm(3)))
    assert isinstance(result, Add)
    assert isinstance(result.a, IntImm)
    assert result.a.value == 10
    assert isinstance(result.b, Mul)
    assert isinstance(result.b.a, IntImm)
    assert result.b.a.value == 5


def test_add_to_sub_mutator():
    """Test mutator that converts Add to Sub"""
    x = Var("x", dtype="int32")
    y = Var("y", dtype="int32")
    expr = Add(x, y)

    mutator = AddToSubMutator()
    result = mutator.visit_expr(expr)

    assert isinstance(result, Sub)
    assert isinstance(result.a, Var)
    assert isinstance(result.b, Var)
    assert result.a.name == "x"
    assert result.b.name == "y"


def test_simple_stmt_counter():
    """Test statement visitor that counts statements"""
    i = Var("i", dtype="int32")

    # Create a simple for loop
    loop_body = Evaluate(IntImm("int32", 0))
    for_stmt = For(i, IntImm("int32", 0), IntImm("int32", 10), tirx.ForKind.SERIAL, loop_body)

    counter = SimpleStmtCounter()
    counter.visit_stmt(for_stmt)

    assert counter.for_count == 1  # One for loop
    assert counter.evaluate_count == 1  # One evaluate in the body


def test_if_then_else_visitor():
    """Test visitor with if-then-else statements"""
    x = Var("x", dtype="int32")
    condition = EQ(x, IntImm("int32", 0))
    then_stmt = Evaluate(IntImm("int32", 1))
    else_stmt = Evaluate(IntImm("int32", 2))

    if_stmt = IfThenElse(condition, then_stmt, else_stmt)

    counter = SimpleStmtCounter()
    counter.visit_stmt(if_stmt)

    assert counter.if_count == 1
    assert counter.for_count == 0


def test_simple_stmt_expr_visitor():
    """Test stmt_expr_visitor with mixed statements and expressions"""
    x = Var("x", dtype="int32")
    y = Var("y", dtype="int32")

    # Create an evaluate statement with an expression
    expr = Add(x, y)
    stmt = Evaluate(expr)

    visitor = SimpleStmtExprVisitor()
    visitor.visit_stmt(stmt)

    assert visitor.stmt_count == 1  # One Evaluate statement
    assert visitor.expr_count == 2  # Two variables
    assert "x" in visitor.var_names
    assert "y" in visitor.var_names


def test_complex_mutator():
    """Test stmt_expr_mutator"""
    x = Var("x", dtype="int32")
    y = Var("y", dtype="int32")

    # Expression with Add operations
    expr = Add(x, y)
    stmt = Evaluate(expr)

    mutator = ComplexMutator()
    result = mutator.visit_stmt(stmt)
    print(type(mutator))

    assert mutator.modifications == 1  # One Add operation modified
    assert isinstance(result, Evaluate)

    # Check that the expression was modified
    modified_expr = result.value
    assert isinstance(modified_expr, Add)
    assert isinstance(modified_expr.a, Mul)  # First operand should be multiplied by 2


def test_analyzer_aware_visitor_loop_context():
    """Test that analyzer-aware visitors expose loop bounds to Python callbacks."""
    i = Var("i", dtype="int32")
    stmt = For(
        i,
        IntImm("int32", 0),
        IntImm("int32", 10),
        tirx.ForKind.SERIAL,
        Evaluate(i),
    )

    visitor = AnalyzerAwareVisitor(i)
    visitor.visit_stmt(stmt)

    assert visitor.facts == [True, True]


def test_analyzer_aware_mutator_branch_context():
    """Test that analyzer-aware mutators expose branch predicates to Python callbacks."""
    x = Var("x", dtype="int32")
    stmt = IfThenElse(
        LT(x, IntImm("int32", 4)),
        Evaluate(LT(x, IntImm("int32", 8))),
        Evaluate(GE(x, IntImm("int32", 4))),
    )

    result = AnalyzerAwareMutator().visit_stmt(stmt)

    assert isinstance(result, IfThenElse)
    assert isinstance(result.then_case.value, IntImm)
    assert isinstance(result.else_case.value, IntImm)
    assert result.then_case.value.value == 1
    assert result.else_case.value.value == 1


def test_analyzer_aware_visitor_assert_context():
    """Test that assert constraints are visible to later statements in the same sequence."""
    x = Var("x", dtype="int32")
    stmt = SeqStmt(
        [
            AssertStmt(LT(x, IntImm("int32", 4)), StringImm("ValueError")),
            Evaluate(LT(x, IntImm("int32", 8))),
        ]
    )

    visitor = PredicateVisitor()
    visitor.visit_stmt(stmt)

    assert visitor.facts == [True]


def test_analyzer_aware_assert_visits_error_kind_and_message():
    """Test that analyzer-aware assert traversal covers error kind and message parts."""
    x = Var("x", dtype="int32")
    stmt = AssertStmt(LT(x, IntImm("int32", 4)), StringImm("ValueError"), [StringImm("bad")])

    visitor = AssertMessageVisitor()
    visitor.visit_stmt(stmt)

    assert visitor.strings == ["ValueError", "bad"]

    result = AssertMessageMutator().visit_stmt(stmt)

    assert result.error_kind.value == "ValueError"
    assert [part.value for part in result.message_parts] == ["rewritten"]


def test_analyzer_aware_visitor_pure_bind_context():
    """Test that pure Bind values are visible to later statements in the same sequence."""
    x = Var("x", dtype="int32")
    stmt = SeqStmt(
        [
            tirx.Bind(x, IntImm("int32", 4)),
            Evaluate(GE(x, IntImm("int32", 4))),
        ]
    )

    visitor = PredicateVisitor()
    visitor.visit_stmt(stmt)

    assert visitor.facts == [True]


def test_analyzer_aware_mutator_skips_opaque_bind_context():
    """Test that opaque Bind values are not inserted into the analyzer context."""
    h = Var("h", dtype="handle")
    stmt = SeqStmt(
        [
            tirx.Bind(h, tirx.tvm_stack_alloca("tvm_ffi_any", 1)),
            Evaluate(IntImm("int32", 0)),
        ]
    )

    result = AnalyzerAwareMutator().visit_stmt(stmt)

    assert isinstance(result, tirx.Bind)
    assert result.var.same_as(h)


def test_analyzer_aware_visitor_branch_assert_does_not_leak():
    """Test that assert constraints inside a branch do not leak to following statements."""
    x = Var("x", dtype="int32")
    stmt = SeqStmt(
        [
            IfThenElse(
                LT(x, IntImm("int32", 4)),
                AssertStmt(LT(x, IntImm("int32", 1)), StringImm("ValueError")),
                None,
            ),
            Evaluate(LT(x, IntImm("int32", 1))),
        ]
    )

    visitor = PredicateVisitor()
    visitor.visit_stmt(stmt)

    assert visitor.facts == [False]


def test_analyzer_aware_mutator_select_context():
    """Test that analyzer-aware mutators expose Select branch predicates."""
    x = Var("x", dtype="int32")
    stmt = Evaluate(
        Select(
            LT(x, IntImm("int32", 4)),
            LT(x, IntImm("int32", 8)),
            GE(x, IntImm("int32", 4)),
        )
    )

    result = AnalyzerAwareMutator().visit_stmt(stmt)

    assert isinstance(result.value, IntImm)
    assert result.value.value == 1


def test_analyzer_aware_mutator_if_then_else_call_context():
    """Test that analyzer-aware mutators expose tirx.if_then_else expression predicates."""
    x = Var("x", dtype="int32")
    stmt = Evaluate(
        tirx.if_then_else(
            LT(x, IntImm("int32", 4)),
            LT(x, IntImm("int32", 8)),
            GE(x, IntImm("int32", 4)),
        )
    )

    result = AnalyzerAwareMutator().visit_stmt(stmt)

    assert isinstance(result.value.args[1], IntImm)
    assert isinstance(result.value.args[2], IntImm)
    assert result.value.args[1].value == 1
    assert result.value.args[2].value == 1


def test_different_expr_types():
    """Test visitor with various expression types"""
    x = Var("x", dtype="int32")

    # Test different expression types individually
    exprs = [
        IntImm("int32", 42),
        FloatImm("float32", 3.14),
        StringImm("hello"),
        Cast("float32", x),
        Min(x, IntImm("int32", 10)),
        Max(x, IntImm("int32", 0)),
        LT(x, IntImm("int32", 5)),
    ]

    # Just test that we can create and visit each type
    counter = SimpleExprCounter()
    for expr in exprs:
        try:
            counter.visit_expr(expr)
        except Exception as e:
            # Some expressions might not be supported, that's ok
            pass


def test_decorator_functionality():
    """Test that decorators work correctly"""

    # Test that decorated classes are properly wrapped
    visitor = SimpleExprCounter()
    assert hasattr(visitor, "_outer")  # Should have the wrapper functionality

    mutator = VariableReplacer({})
    assert hasattr(mutator, "_outer")


def test_empty_expressions():
    """Test handling of simple expressions"""
    counter = SimpleExprCounter()

    # Test with just a variable
    x = Var("x", dtype="int32")
    counter.visit_expr(x)

    assert counter.var_count == 1

    # Test with just a constant
    counter = SimpleExprCounter()
    const = IntImm("int32", 5)
    counter.visit_expr(const)

    # Constants don't increase var_count
    assert counter.var_count == 0


def test_stmt_mutator():
    """Test basic statement mutator functionality"""
    x = Var("x", dtype="int32")
    stmt = Evaluate(Add(x, IntImm("int32", 1)))

    unroller = ForLoopUnroller()
    result = unroller.visit_stmt(stmt)

    # Should return the same statement (no actual unrolling implemented)
    assert isinstance(result, Evaluate)


def test_nested_expressions():
    """Test with nested expressions"""
    x = Var("x", dtype="int32")
    y = Var("y", dtype="int32")
    z = Var("z", dtype="int32")

    # Create nested expression: (x + y) * z
    inner_add = Add(x, y)
    expr = Mul(inner_add, z)

    counter = SimpleExprCounter()
    counter.visit_expr(expr)

    assert counter.var_count == 3  # x, y, z
    assert counter.add_count == 1  # one add
    assert counter.mul_count == 1  # one mul


def test_simple_mutations():
    """Test simple expression mutations"""
    x = Var("x", dtype="int32")
    y = Var("y", dtype="int32")

    # Test multiple replacements
    expr = Add(x, y)
    replacer = VariableReplacer({"x": 1, "y": 2})
    result = replacer.visit_expr(expr)

    assert isinstance(result, Add)
    assert isinstance(result.a, IntImm)
    assert isinstance(result.b, IntImm)
    assert result.a.value == 1
    assert result.b.value == 2


if __name__ == "__main__":
    test_basic_visitor()
    tvm.testing.main()
