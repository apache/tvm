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
"""Explicit host selection preserves builder scope and source evaluation."""

import pytest

from tvm import ir, tirx
from tvm.script import parser
from tvm.script.ir_builder import ir as I
from tvm.script import tirx as T


@pytest.mark.parametrize("marker", ["I.constexpr", "T.constexpr", "marked"])
@pytest.mark.parametrize("track_span", [True, False])
def test_marked_if_shares_parent_scope(marker, track_span):
    seen = []

    def choose():
        seen.append("condition")
        return True

    result = parser.parse(
        f"""
@T.prim_func
def main(a: T.handle):
    if {marker}(choose()):
        A = T.match_buffer(a, (4,), "int32")
    else:
        invalid()
    A[0] = 3
""",
        extra_vars={"choose": choose, "marked": I.constexpr},
        track_span=track_span,
    )
    assert seen == ["condition"]
    assert len(result.params) == 1
    assert isinstance(result.body, tirx.BufferStore)
    assert result.body.buffer.same_as(result.params[0])


def test_marked_expressions_are_lazy_operand_valued_and_ordered():
    seen = []

    def operand(value):
        seen.append(value)
        return value

    result = parser.parse(
        """
@T.prim_func
def main():
    T.evaluate(operand(1) if I.constexpr(operand(True)) else invalid())
    T.evaluate(I.constexpr(operand(0)) and invalid())
    T.evaluate(I.constexpr(operand(4)) or invalid())
    T.evaluate(I.constexpr(operand(2)) and operand(7))
    T.evaluate(I.constexpr(operand(0)) or operand(8))
""",
        extra_vars={"operand": operand},
    )
    assert seen == [True, 1, 0, 4, 2, 7, 0, 8]
    assert [node.value.value for node in result.body.seq] == [1, 4, 7, 8]


def test_unmarked_expression_eagerly_constructs_both_ir_arms():
    seen = []

    def operand(value):
        seen.append(value)
        return value

    result = parser.parse(
        """
@T.prim_func
def main(condition: T.bool):
    T.evaluate(operand(1) if condition else operand(2))
    T.evaluate(condition and operand(True))
    T.evaluate(condition or operand(False))
""",
        extra_vars={"operand": operand},
    )
    assert seen == [1, 2, True, False]
    assert isinstance(result.body.seq[0].value, ir.Call)
    assert isinstance(result.body.seq[1].value, tirx.And)
    assert isinstance(result.body.seq[2].value, tirx.Or)


def test_nested_unmarked_statement_retains_ir_frame():
    result = parser.parse("""
@T.prim_func
def main(condition: T.bool):
    if I.constexpr(True):
        if condition:
            T.evaluate(1)
        else:
            T.evaluate(2)
""")
    assert isinstance(result.body, tirx.IfThenElse)


def test_captured_shapes_require_explicit_symbols():
    n = ir.Var("n", "int64")
    source = """
@R.function
def main(x: R.Tensor(shape, "float32")):
    return x
"""
    result = parser.parse(source, extra_vars={"shape": (n, 16)})
    assert result.params[0].ty.shape[0].same_as(n)
    with pytest.raises(Exception):
        parser.parse(source, extra_vars={"shape": ("n", 16)})


def test_dialect_marker_is_same_protocol_identity():
    assert T.constexpr is I.constexpr


@pytest.mark.parametrize("condition", [True, False])
def test_conditional_binding_can_be_assigned_after_skipped_branch(condition):
    result = parser.parse(f"""
@T.prim_func
def main():
    if I.constexpr({condition}):
        x = 1
    x = 2
    T.evaluate(x)
""")
    assert isinstance(result.body, tirx.SeqStmt)
    assert isinstance(result.body.seq[-1], tirx.Evaluate)


@pytest.mark.parametrize("operator", ["and", "or"])
def test_unmarked_logical_chain_preserves_left_association(operator):
    source = f"""
@T.prim_func
def main(a: T.bool, b: T.bool, c: T.bool):
    T.evaluate(a {operator} b {operator} c)
"""
    variables = [tirx.Var(name, "bool") for name in ("a", "b", "c")]
    constructor = tirx.And if operator == "and" else tirx.Or
    expected = tirx.PrimFunc(
        variables, tirx.Evaluate(constructor(constructor(*variables[:2]), variables[2]))
    ).with_attr("global_symbol", "main")
    ir.assert_structural_equal(expected, parser.parse(source))


@pytest.mark.parametrize("track_span", [True, False])
def test_constexpr_keeps_named_expression_unsupported(track_span):
    with pytest.raises(Exception, match="Unsupported expression: NamedExpr"):
        parser.parse(
            """
@T.prim_func
def main():
    if I.constexpr(bool(x := 3)):
        T.evaluate(x)
""",
            track_span=track_span,
        )


def test_ir_optional_binding_survives_skipped_host_assignment():
    result = parser.parse("""
@T.prim_func
def main(condition: T.bool):
    if condition:
        x = T.int32(1)
    if I.constexpr(False):
        x = T.int32(2)
    x = T.int32(3)
    T.evaluate(x)
""")
    assert isinstance(result.body, tirx.SeqStmt)


@pytest.mark.parametrize(
    "expression", ["1 if I.constexpr(x) else 0", "I.constexpr(x) and 1", "I.constexpr(x) or 1"]
)
def test_missing_host_binding_cannot_be_truth_tested(expression):
    from tvm.error import DiagnosticError

    with pytest.raises(DiagnosticError) as caught:
        parser.parse(f"""
@T.prim_func
def main():
    if I.constexpr(False):
        x = 1
    T.evaluate({expression})
""")
    assert isinstance(caught.value.__cause__, NameError)


def test_missing_host_if_binding_raises_before_branch_assignments():
    from tvm.error import DiagnosticError

    with pytest.raises(DiagnosticError) as caught:
        parser.parse("""
@T.prim_func
def main():
    if I.constexpr(False):
        x = 1
    if I.constexpr(x):
        x = 2
    else:
        x = 3
""")
    assert isinstance(caught.value.__cause__, NameError)


def test_host_lambda_local_does_not_capture_optional_binding():
    result = parser.parse("""
@T.prim_func
def main():
    if I.constexpr(False):
        x = 1
    if I.constexpr((lambda x: x)(True)):
        T.evaluate(222)
""")
    assert result.body.value.value == 222
