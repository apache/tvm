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
"""Tests for TVMScript printer Doc objects and the Python doc printer.

Doc construction is mostly exercised through printing; the construction tests
below only cover contracts that the printed output cannot observe.
"""

import pytest
from tvm_ffi.access_path import AccessPath

import tvm
from tvm.script.printer.doc import (
    AssertDoc,
    AssignDoc,
    CallDoc,
    ClassDoc,
    CommentDoc,
    DictDoc,
    DocStringDoc,
    ExprStmtDoc,
    ForDoc,
    FunctionDoc,
    IdDoc,
    IfDoc,
    LambdaDoc,
    ListDoc,
    LiteralDoc,
    OperationDoc,
    OperationKind,
    ReturnDoc,
    ScopeDoc,
    SliceDoc,
    StmtBlockDoc,
    TupleDoc,
    WhileDoc,
)
from tvm.script.printer.doc_printer import to_python_script


def format_script(s: str) -> str:
    """Remove leading and trailing blank lines, and make the minimum indentation 0."""
    s = s.strip("\n")
    non_empty_lines = [line for line in s.splitlines() if line and not line.isspace()]
    if not non_empty_lines:
        return ""
    spaces_to_remove = min(len(line) - len(line.lstrip(" ")) for line in non_empty_lines)
    return "\n".join(line[spaces_to_remove:] for line in s.splitlines()).strip()


# Doc construction contracts


@pytest.mark.parametrize(
    "lhs, rhs, annotation",
    [
        # Either rhs or annotation is required.
        (IdDoc("x"), None, None),
        # Only an IdDoc lhs can carry an annotation.
        (TupleDoc([IdDoc("x"), IdDoc("y")]), IdDoc("u"), IdDoc("int")),
    ],
)
def test_invalid_assign_doc(lhs, rhs, annotation):
    with pytest.raises(ValueError):
        AssignDoc(lhs, rhs, annotation)


def test_if_doc_requires_a_branch():
    with pytest.raises(ValueError):
        IfDoc(IdDoc("x"), [], [])


def test_expr_doc_get_item_wraps_non_tuple():
    index = LiteralDoc(1)
    doc = IdDoc("x")[index]
    assert tuple(doc.indices) == (index,)


def test_function_doc_type_params_default():
    doc = FunctionDoc(IdDoc("f"), [], [], None, [])
    assert list(doc.type_params) == []


def test_stmt_doc_comment():
    doc = ExprStmtDoc(IdDoc("x"))
    assert doc.comment is None

    comment = "test comment"
    doc.comment = comment
    # Make sure the previous statement doesn't set attribute
    # as if it's an ordinary Python object (__slots__ enforces this).
    assert not hasattr(doc, "__dict__") or "comment" not in doc.__dict__
    assert doc.comment == comment


def test_doc_source_paths():
    doc = IdDoc("x")
    assert len(doc.source_paths) == 0

    source_paths = [AccessPath.root(), AccessPath.root().attr("x")]

    doc.source_paths = source_paths
    # This should triggers the __getattr__ and gets a tvm_ffi.Array
    assert not isinstance(doc.source_paths, list)
    assert list(doc.source_paths) == source_paths

    doc.source_paths = []
    assert len(doc.source_paths) == 0


# Expression printing

x, y, z = IdDoc("x"), IdDoc("y"), IdDoc("z")


def op(kind):
    return lambda *operands: OperationDoc(kind, list(operands))


EXPR_CASES = {
    "none": (LiteralDoc(None), "None"),
    "bool": (LiteralDoc(True), "True"),
    "str": (LiteralDoc("test"), '"test"'),
    "str-quote": (LiteralDoc('""'), r'"\"\""'),
    "str-escape": (LiteralDoc("\n\t\\test\r"), r'"\n\t\\test\r"'),
    # TODO: fix the roundatrippable problem caused by utf8
    "str-utf8": pytest.param(LiteralDoc("\x88"), r'"\x88"', marks=pytest.mark.xfail),
    "int": (LiteralDoc(-1), "-1"),
    "float": (LiteralDoc(3.25), "3.25"),
    "id": (IdDoc("test"), "test"),
    "attr": (IdDoc("x").attr("attr"), "x.attr"),
    "index-empty": (IdDoc("x")[()], "x[()]"),
    "index-single": (IdDoc("x")[(LiteralDoc(1),)], "x[1]"),
    "index-multi": (IdDoc("x")[SliceDoc(LiteralDoc(1)), IdDoc("y")], "x[1:, y]"),
    "slice-empty": (IdDoc("x")[SliceDoc()], "x[:]"),
    "slice-stop": (IdDoc("x")[SliceDoc(None, LiteralDoc(2))], "x[:2]"),
    "slice-step": (IdDoc("x")[SliceDoc(None, None, LiteralDoc(3))], "x[::3]"),
    "slice-full": (IdDoc("x")[SliceDoc(LiteralDoc(1), LiteralDoc(2), LiteralDoc(3))], "x[1:2:3]"),
    "call-empty": (CallDoc(IdDoc("f")), "f()"),
    "call-args": (CallDoc(IdDoc("f"), x, y), "f(x, y)"),
    "call-kwargs": (CallDoc(IdDoc("f"), key0=IdDoc("u"), key1=IdDoc("v")), "f(key0=u, key1=v)"),
    "call-both": (CallDoc(IdDoc("f"), x, key0=IdDoc("u")), "f(x, key0=u)"),
    "lambda-no-args": (LambdaDoc([], LiteralDoc(0)), "lambda : 0"),
    "list-empty": (ListDoc([]), "[]"),
    "list": (ListDoc([x, y]), "[x, y]"),
    "tuple-empty": (TupleDoc([]), "()"),
    "tuple-single": (TupleDoc([x]), "(x,)"),
    "tuple": (TupleDoc([x, y]), "(x, y)"),
    "dict-empty": (DictDoc({}), "{}"),
    "dict": (
        DictDoc({LiteralDoc("key_x"): x, LiteralDoc("key_y"): y}),
        '{"key_x": x, "key_y": y}',
    ),
    "if-then-else": (
        op(OperationKind.IfThenElse)(x, LiteralDoc(None), LiteralDoc(1)),
        "None if x else 1",
    ),
}


@pytest.mark.parametrize("doc, expected", EXPR_CASES.values(), ids=EXPR_CASES.keys())
def test_print_expr_doc(doc, expected):
    assert to_python_script(doc) == expected


UNARY_OP_TOKENS = {OperationKind.USub: "-", OperationKind.Invert: "~", OperationKind.Not: "not "}


@pytest.mark.parametrize(
    "op_kind, expected_token", list(UNARY_OP_TOKENS.items()), ids=UNARY_OP_TOKENS.keys()
)
def test_print_unary_operation_doc(op_kind, expected_token):
    doc = OperationDoc(op_kind, [IdDoc("x")])
    assert to_python_script(doc) == f"{expected_token}x"


BINARY_OP_TOKENS = {
    OperationKind.Add: "+",
    OperationKind.Sub: "-",
    OperationKind.Mult: "*",
    OperationKind.Div: "/",
    OperationKind.FloorDiv: "//",
    OperationKind.Mod: "%",
    OperationKind.Pow: "**",
    OperationKind.LShift: "<<",
    OperationKind.RShift: ">>",
    OperationKind.BitAnd: "&",
    OperationKind.BitOr: "|",
    OperationKind.BitXor: "^",
    OperationKind.Lt: "<",
    OperationKind.LtE: "<=",
    OperationKind.Eq: "==",
    OperationKind.NotEq: "!=",
    OperationKind.Gt: ">",
    OperationKind.GtE: ">=",
    OperationKind.And: "and",
    OperationKind.Or: "or",
    OperationKind.MatMul: "@",
}


@pytest.mark.parametrize(
    "op_kind, expected_token", list(BINARY_OP_TOKENS.items()), ids=BINARY_OP_TOKENS.keys()
)
def test_print_binary_operation_doc(op_kind, expected_token):
    doc = OperationDoc(op_kind, [IdDoc("x"), IdDoc("y")])
    assert to_python_script(doc) == f"x {expected_token} y"


SPECIAL_OP_KINDS = {OperationKind.IfThenElse}


def test_operation_doc_test_exhaustive():
    for op_kind in OperationKind:
        if OperationKind._UnaryStart < op_kind < OperationKind._UnaryEnd:
            assert op_kind in UNARY_OP_TOKENS, (
                f"{op_kind.name} not covered in test_print_unary_operation_doc. "
                f"Please add the expected token to UNARY_OP_TOKENS"
            )
        elif OperationKind._BinaryStart < op_kind < OperationKind._BinaryEnd:
            assert op_kind in BINARY_OP_TOKENS, (
                f"{op_kind.name} not covered in test_print_binary_operation_doc. "
                f"Please add the expected token to BINARY_OP_TOKENS"
            )
        elif not op_kind.name.startswith("_"):
            assert op_kind in SPECIAL_OP_KINDS, (
                f"{op_kind.name} not covered in test_print_expr_doc. "
                f"Please add a test case for it to EXPR_CASES and SPECIAL_OP_KINDS"
            )


def generate_expr_precedence_test_cases():
    K = OperationKind
    neg, invert, not_ = op(K.USub), op(K.Invert), op(K.Not)
    add, mult, div, mod, pow_ = op(K.Add), op(K.Mult), op(K.Div), op(K.Mod), op(K.Pow)
    lshift, bit_and, bit_or, bit_xor = op(K.LShift), op(K.BitAnd), op(K.BitOr), op(K.BitXor)
    lt, eq, not_eq = op(K.Lt), op(K.Eq), op(K.NotEq)
    and_, or_, ite = op(K.And), op(K.Or), op(K.IfThenElse)

    test_cases = {
        "attr-call-index": [
            (add(x, y)[z][add(z, z)].attr("name"), "(x + y)[z][z + z].name"),
            (add(x, y.attr("test")), "x + y.test"),
            (x[z].call(y), "x[z](y)"),
            (LambdaDoc([x], x).call(y), "(lambda x: x)(y)"),
        ],
        "power": [
            (pow_(pow_(x, y), z), "(x ** y) ** z"),
            (pow_(x, pow_(y, z)), "x ** y ** z"),
            (pow_(neg(x), neg(y)), "(-x) ** -y"),
        ],
        "unary": [
            (invert(neg(y)), "~-y"),
            (neg(y).attr("test"), "(-y).test"),
            (neg(add(invert(x), neg(y))), "-(~x + -y)"),
        ],
        "add-mult": [
            (mult(x, mult(y, z)), "x * (y * z)"),
            (mult(mult(x, y), z), "x * y * z"),
            (add(x, mod(y, z)), "x + y % z"),
            (div(add(x, y), add(y, z)), "(x + y) / (y + z)"),
        ],
        "shift": [
            (div(x, lshift(y, z)), "x / (y << z)"),
            (lshift(mult(x, y), z), "x * y << z"),
        ],
        "bitwise": [
            (bit_and(bit_or(x, y), bit_or(y, z)), "(x | y) & (y | z)"),
            (bit_or(bit_and(x, y), bit_and(y, z)), "x & y | y & z"),
            (bit_and(bit_xor(x, bit_or(y, z)), z), "(x ^ (y | z)) & z"),
        ],
        "comparison": [
            (not_eq(add(x, y), z), "x + y != z"),
            (lt(x, ite(y, y, y)), "x < (y if y else y)"),
        ],
        "boolean": [
            (not_(and_(x, y)), "not (x and y)"),
            (and_(not_(x), y), "not x and y"),
            (and_(or_(x, y), z), "(x or y) and z"),
            # Parentheses are not necessary, though they may help readability.
            (or_(and_(x, y), z), "x and y or z"),
            (or_(not_(eq(x, y)), lt(y, z)), "not x == y or y < z"),
            (and_(ite(x, y, z), x), "(y if x else z) and x"),
        ],
        "if-then-else": [
            (ite(x, ite(y, y, y), z), "y if y else y if x else z"),
            (ite(ite(x, x, x), y, z), "y if (x if x else x) else z"),
            (ite(x, y, ite(z, z, z)), "y if x else (z if z else z)"),
            (
                ite(LambdaDoc([x], x), LambdaDoc([y], y), LambdaDoc([z], z)),
                "(lambda y: y) if (lambda x: x) else (lambda z: z)",
            ),
        ],
        "lambda": [
            (LambdaDoc([x, y], add(z, z)), "lambda x, y: z + z"),
            (add(LambdaDoc([x, y], z), z), "(lambda x, y: z) + z"),
            (LambdaDoc([x], LambdaDoc([y], z)), "lambda x: lambda y: z"),
        ],
    }

    return [
        pytest.param(*args, id=f"{group_name}-{i}")
        for group_name, cases in test_cases.items()
        for i, args in enumerate(cases)
    ]


@pytest.mark.parametrize("doc, expected", generate_expr_precedence_test_cases())
def test_expr_precedence(doc, expected):
    assert to_python_script(doc) == expected


# Statement printing


def get_func_doc_for_class(name):
    args = [
        AssignDoc(IdDoc("x"), rhs=None, annotation=IdDoc("int")),
        AssignDoc(IdDoc("y"), rhs=LiteralDoc(1), annotation=IdDoc("int")),
    ]
    body = [
        AssignDoc(IdDoc("y"), OperationDoc(OperationKind.Add, [IdDoc("x"), LiteralDoc(1)])),
        AssignDoc(IdDoc("y"), OperationDoc(OperationKind.Sub, [IdDoc("y"), LiteralDoc(1)])),
    ]
    return FunctionDoc(
        name=IdDoc(name),
        args=args,
        decorators=[IdDoc("wrap")],
        return_type=LiteralDoc(None),
        body=body,
    )


def expr_stmts(*names):
    return [ExprStmtDoc(IdDoc(name)) for name in names]


STMT_CASES = {
    "block-empty": (StmtBlockDoc([]), ""),
    "block": (StmtBlockDoc(expr_stmts("x", "y")), "x\ny"),
    "assign-annotated": (AssignDoc(x, y, IdDoc("int")), "x: int = y"),
    "assign-no-rhs": (AssignDoc(x, None, IdDoc("int")), "x: int"),
    "assign-tuple": (AssignDoc(TupleDoc([x, TupleDoc([y, z])]), z, None), "x, (y, z) = z"),
    "if-no-else": (
        IfDoc(IdDoc("pred"), expr_stmts("x"), []),
        """
        if pred:
            x
        """,
    ),
    "if-empty-then": (
        IfDoc(IdDoc("pred"), [], expr_stmts("y")),
        """
        if pred:
            pass
        else:
            y
        """,
    ),
    "while-empty": (
        WhileDoc(IdDoc("pred"), []),
        """
        while pred:
            pass
        """,
    ),
    "for": (
        ForDoc(x, y, expr_stmts("x")),
        """
        for x in y:
            x
        """,
    ),
    "scope-no-lhs": (
        ScopeDoc(None, CallDoc(IdDoc("context")), expr_stmts("x")),
        """
        with context():
            x
        """,
    ),
    "assert-msg": (AssertDoc(LiteralDoc(True), LiteralDoc("msg")), 'assert True, "msg"'),
    "function-empty": (
        FunctionDoc(IdDoc("func"), [], [], None, []),
        """
        def func():
            pass
        """,
    ),
    "function-decorators": (
        FunctionDoc(IdDoc("func"), [], [IdDoc("wrap_outer"), IdDoc("wrap_inner")], None, []),
        """
        @wrap_outer
        @wrap_inner
        def func():
            pass
        """,
    ),
    "function-type-params": (
        FunctionDoc(
            IdDoc("func"),
            [],
            [],
            None,
            [ReturnDoc(IdDoc("T"))],
            type_params=[IdDoc("T"), AssignDoc(IdDoc("U"), rhs=None, annotation=IdDoc("int"))],
        ),
        """
        def func[T, U: int]():
            return T
        """,
    ),
    "class-empty": (
        ClassDoc(IdDoc("TestClass"), [], []),
        """
        class TestClass:
            pass
        """,
    ),
    "class": (
        ClassDoc(
            IdDoc("TestClass"),
            [IdDoc("wrap_outer"), IdDoc("wrap_inner")],
            [get_func_doc_for_class("f1"), get_func_doc_for_class("f2")],
        ),
        """
        @wrap_outer
        @wrap_inner
        class TestClass:
            @wrap
            def f1(x: int, y: int = 1) -> None:
                y = x + 1
                y = y - 1

            @wrap
            def f2(x: int, y: int = 1) -> None:
                y = x + 1
                y = y - 1
        """,
    ),
    "comment-empty": (CommentDoc(""), ""),
    "comment": (
        CommentDoc("test comment 1\ntest comment 2"),
        """
        # test comment 1
        # test comment 2
        """,
    ),
    "comment-trailing-tab": (CommentDoc("test comment\t"), "# test comment\t"),
    "docstring-empty": (DocStringDoc(""), ""),
    "docstring": (
        DocStringDoc("test comment 1\ntest comment 2"),
        '''
        """
        test comment 1
        test comment 2
        """
        ''',
    ),
}


@pytest.mark.parametrize("doc, expected", STMT_CASES.values(), ids=STMT_CASES.keys())
def test_print_stmt_doc(doc, expected):
    if expected.startswith("\n"):
        expected = format_script(expected)
    assert to_python_script(doc) == expected


# Each statement type prints its own comment, so keep one case per type.
DOC_COMMENT_CASES = {
    "assign": (AssignDoc(x, y, IdDoc("int")), "comment", "x: int = y  # comment"),
    "if": (
        IfDoc(x, expr_stmts("y"), expr_stmts("z")),
        "comment line 1\ncomment line 2",
        """
        # comment line 1
        # comment line 2
        if x:
            y
        else:
            z
        """,
    ),
    "while": (
        WhileDoc(LiteralDoc(True), [AssignDoc(x, y)]),
        "comment",
        """
        # comment
        while True:
            x = y
        """,
    ),
    "for": (
        ForDoc(x, y, []),
        "comment",
        """
        # comment
        for x in y:
            pass
        """,
    ),
    "scope": (
        ScopeDoc(x, y, []),
        "comment",
        """
        # comment
        with y as x:
            pass
        """,
    ),
    "expr-stmt": (ExprStmtDoc(x), "comment", "x  # comment"),
    "assert": (AssertDoc(LiteralDoc(True)), "comment", "assert True  # comment"),
    "return": (ReturnDoc(LiteralDoc(1)), "comment", "return 1  # comment"),
    "function": (
        get_func_doc_for_class("f"),
        "comment line 1\n\ncomment line 3",
        '''
        @wrap
        def f(x: int, y: int = 1) -> None:
            """
            comment line 1

            comment line 3
            """
            y = x + 1
            y = y - 1
        ''',
    ),
    "class": (
        ClassDoc(IdDoc("TestClass"), decorators=[IdDoc("wrap")], body=[]),
        "comment",
        '''
        @wrap
        class TestClass:
            """
            comment
            """
            pass
        ''',
    ),
}


@pytest.mark.parametrize(
    "doc, comment, expected", DOC_COMMENT_CASES.values(), ids=DOC_COMMENT_CASES.keys()
)
def test_print_doc_comment(doc, comment, expected):
    doc.comment = comment
    assert to_python_script(doc) == format_script(expected)


def test_print_invalid_multiline_doc_comment():
    doc = ExprStmtDoc(IdDoc("x"))
    doc.comment = "1\n2"
    with pytest.raises(ValueError, match="cannot have newline"):
        to_python_script(doc)


# Underlining


def make_path(name: str) -> AccessPath:
    return AccessPath.root().attr(name)


def make_id_doc(name: str, path_name: str | None = None) -> IdDoc:
    doc = IdDoc(name)
    doc.source_paths = [make_path(name if path_name is None else path_name)]
    return doc


def format_script_with_path_info(s: str, *path_info: str) -> str:
    return "\n".join(path_info) + "\n\n" + format_script(s)


def test_underline_multiple_spans():
    doc = StmtBlockDoc(
        [
            ExprStmtDoc(make_id_doc("foo")),
            ExprStmtDoc(make_id_doc("bar")),
            ExprStmtDoc(OperationDoc(OperationKind.Add, [make_id_doc("foo"), make_id_doc("foo")])),
        ]
    )
    assert to_python_script(
        doc, path_to_underline=[make_path("foo")]
    ) == format_script_with_path_info(
        """
        foo
        ^^^
        bar
        foo + foo
        ^^^   ^^^
    """,
        "Access path: <root>.foo",
    )


def test_underline_multiline():
    doc = StmtBlockDoc(expr_stmts("foo", "bar"))
    doc.source_paths = [make_path("whole_doc")]

    assert to_python_script(
        doc, path_to_underline=[make_path("whole_doc")]
    ) == format_script_with_path_info(
        """
        foo
        ^^^
        bar
        ^^^
    """,
        "Access path: <root>.whole_doc",
    )


@pytest.mark.parametrize(
    "to_underline, expected_text",
    [
        (
            [3],
            """
                x0
                x1
                x2
                x3
                ^^
                x4
                x5
                (... 4 lines skipped ...)
            """,
        ),
        (
            [4],
            """
                (... 2 lines skipped ...)
                x2
                x3
                x4
                ^^
                x5
                x6
                (... 3 lines skipped ...)
            """,
        ),
        (
            [0, 9],
            """
                x0
                ^^
                x1
                x2
                (... 4 lines skipped ...)
                x7
                x8
                x9
                ^^
            """,
        ),
        (
            [0, 3, 9],
            """
                x0
                ^^
                x1
                x2
                x3
                ^^
                x4
                x5
                x6
                x7
                x8
                x9
                ^^
            """,
        ),
        (
            [33],
            """
                x0
                x1
                x2
                x3
                x4
                x5
                x6
                x7
                x8
                x9
            """,
        ),
    ],
)
def test_print_two_context_lines(to_underline, expected_text):
    doc = StmtBlockDoc(
        [ExprStmtDoc(make_id_doc(f"x{i}", "yes" if i in to_underline else "no")) for i in range(10)]
    )
    result = to_python_script(doc, num_context_lines=2, path_to_underline=[make_path("yes")])
    path_info = ["Access path: <root>.yes"]
    if to_underline == [33]:
        path_info.append("Note: No visible object for this path is rendered in TVMScript.")
    assert result == format_script_with_path_info(expected_text, *path_info)


def test_underline_and_print_line_numbers():
    doc = StmtBlockDoc([ExprStmtDoc(make_id_doc(f"line{i + 1}")) for i in range(12)])
    result = to_python_script(doc, print_line_numbers=True, path_to_underline=[make_path("line6")])
    assert result == "Access path: <root>.line6\n\n " + format_script(
        """
            1 line1
            2 line2
            3 line3
            4 line4
            5 line5
            6 line6
              ^^^^^
            7 line7
            8 line8
            9 line9
           10 line10
           11 line11
           12 line12
    """
    )


def test_underline_multi_access_paths():
    doc = StmtBlockDoc([ExprStmtDoc(make_id_doc(f"line{i + 1}")) for i in range(4)])
    result = to_python_script(doc, path_to_underline=[make_path("line1"), make_path("line3")])
    assert result == format_script_with_path_info(
        """
            line1
            ^^^^^
            line2
            line3
            ^^^^^
            line4
    """,
        "Access path: <root>.line1",
        "Access path: <root>.line3",
    )


def test_underline_and_print_line_numbers_with_context():
    doc = StmtBlockDoc([ExprStmtDoc(make_id_doc(f"line{i + 1}")) for i in range(12)])
    result = to_python_script(
        doc, print_line_numbers=True, num_context_lines=2, path_to_underline=[make_path("line8")]
    )
    assert result == format_script_with_path_info(
        """
           (... 5 lines skipped ...)
            6 line6
            7 line7
            8 line8
              ^^^^^
            9 line9
           10 line10
           (... 2 lines skipped ...)
    """,
        "Access path: <root>.line8",
    )


def test_underline_based_on_path_prefix():
    doc = StmtBlockDoc([ExprStmtDoc(make_id_doc("foo")), ExprStmtDoc(make_id_doc("bar"))])
    result = to_python_script(doc, path_to_underline=[make_path("foo").attr("x").attr("y")])
    # There is no document that matches the desired path exactly,
    # but path of "foo" is a prefix of the desired path, and thus should be underlined.
    assert result == format_script_with_path_info(
        """
        foo
        ^^^
        bar
    """,
        "Access path: <root>.foo.x.y",
        "Note: The underlined object is the nearest visible parent of this path.",
    )


def test_longer_prefix_must_win():
    foo_x = IdDoc("foo_x")
    foo_x.source_paths = [make_path("foo").attr("x")]

    doc = StmtBlockDoc(
        [ExprStmtDoc(make_id_doc("foo")), ExprStmtDoc(make_id_doc("bar")), ExprStmtDoc(foo_x)]
    )
    result = to_python_script(doc, path_to_underline=[make_path("foo").attr("x").attr("y")])
    # "foo" should not be underlined because there is a document with a more specific path prefix
    assert result == format_script_with_path_info(
        """
        foo
        bar
        foo_x
        ^^^^^
    """,
        "Access path: <root>.foo.x.y",
        "Note: The underlined object is the nearest visible parent of this path.",
    )


if __name__ == "__main__":
    tvm.testing.main()
