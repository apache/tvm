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
import pytest
import itertools

from tvm.script.printer.doc import (
    AssertDoc,
    AssignDoc,
    CallDoc,
    ClassDoc,
    DictDoc,
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
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")

    non_empty_lines = [line for line in s.splitlines() if line and not line.isspace()]
    if not non_empty_lines:
        # no actual content
        return "\n"

    line_indents = [len(line) - len(line.lstrip(" ")) for line in non_empty_lines]
    spaces_to_remove = min(line_indents)

    cleaned_lines = "\n".join(line[spaces_to_remove:] for line in s.splitlines())
    if not cleaned_lines.endswith("\n"):
        cleaned_lines += "\n"
    return cleaned_lines


@pytest.mark.parametrize(
    "doc,expected",
    [
        (LiteralDoc(None), "None"),
        (LiteralDoc(True), "True"),
        (LiteralDoc(False), "False"),
        (LiteralDoc("test"), '"test"'),
        (LiteralDoc(""), '""'),
        (LiteralDoc('""'), r'"\"\""'),
        (LiteralDoc("\n\t\\test\r"), r'"\n\t\\test\r"'),
        # TODO: fix the roundatrippable problem caused by utf8
        pytest.param(LiteralDoc("\x88"), r'"\x88"', marks=pytest.mark.xfail),
        (LiteralDoc(0), "0"),
        (LiteralDoc(-1), "-1"),
        (LiteralDoc(3.25), "3.25"),
        (LiteralDoc(-0.5), "-0.5"),
    ],
    ids=itertools.count(),
)
def test_print_literal_doc(doc, expected):
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "name",
    [
        "test",
        "_test",
        "TestCase",
        "test_case",
        "test123",
    ],
    ids=itertools.count(),
)
def test_print_id_doc(name):
    doc = IdDoc(name)
    assert to_python_script(doc) == format_script(name)


@pytest.mark.parametrize(
    "attr",
    [
        "attr",
        "_attr",
        "Attr",
        "attr_1",
    ],
    ids=itertools.count(),
)
def test_print_attr_doc(attr):
    doc = IdDoc("x").attr(attr)
    assert to_python_script(doc) == format_script(f"x.{attr}")


@pytest.mark.parametrize(
    "indices, expected",
    [
        (
            (),
            "[()]",
        ),
        (
            (LiteralDoc(1),),
            "[1]",
        ),
        (
            (LiteralDoc(2), IdDoc("x")),
            "[2, x]",
        ),
        (
            (SliceDoc(LiteralDoc(1), LiteralDoc(2)),),
            "[1:2]",
        ),
        (
            (SliceDoc(LiteralDoc(1)), IdDoc("y")),
            "[1:, y]",
        ),
        (
            (SliceDoc(), IdDoc("y")),
            "[:, y]",
        ),
        (
            (IdDoc("x"), IdDoc("y"), IdDoc("z")),
            "[x, y, z]",
        ),
    ],
    ids=itertools.count(),
)
def test_print_index_doc(indices, expected):
    doc = IdDoc("x")[indices]
    assert to_python_script(doc) == format_script(f"x{expected}")


UNARY_OP_TOKENS = {
    OperationKind.USub: "-",
    OperationKind.Invert: "~",
}


@pytest.mark.parametrize(
    "op_kind, expected_token",
    list(UNARY_OP_TOKENS.items()),
    ids=UNARY_OP_TOKENS.keys(),
)
def test_print_unary_operation_doc(op_kind, expected_token):
    doc = OperationDoc(op_kind, [IdDoc("x")])
    assert to_python_script(doc) == format_script(f"{expected_token}x")


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
}


@pytest.mark.parametrize(
    "op_kind, expected_token",
    list(BINARY_OP_TOKENS.items()),
    ids=BINARY_OP_TOKENS.keys(),
)
def test_print_binary_operation_doc(op_kind, expected_token):
    doc = OperationDoc(op_kind, [IdDoc("x"), IdDoc("y")])
    assert to_python_script(doc) == format_script(f"x {expected_token} y")


SPECIAL_OP_CASES = [
    (
        OperationKind.IfThenElse,
        [LiteralDoc(True), LiteralDoc("true"), LiteralDoc("false")],
        '"true" if True else "false"',
    ),
    (
        OperationKind.IfThenElse,
        [IdDoc("x"), LiteralDoc(None), LiteralDoc(1)],
        "None if x else 1",
    ),
]


@pytest.mark.parametrize(
    "op_kind, operands, expected", SPECIAL_OP_CASES, ids=[kind for (kind, *_) in SPECIAL_OP_CASES]
)
def test_print_special_operation_doc(op_kind, operands, expected):
    doc = OperationDoc(op_kind, operands)
    assert to_python_script(doc) == format_script(expected)


def test_operation_doc_test_exhaustive():
    special_op_covered = {k for k, *_ in SPECIAL_OP_CASES}
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
            # Special Op
            assert op_kind in special_op_covered, (
                f"{op_kind.name} not covered in test_print_special_operation_doc. "
                f"Please add the test cases for it to SPECIAL_OP_CASES"
            )


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        (
            (),
            {},
            "()",
        ),
        (
            (),
            {"key0": IdDoc("u")},
            "(key0=u)",
        ),
        (
            (),
            {"key0": IdDoc("u"), "key1": IdDoc("v")},
            "(key0=u, key1=v)",
        ),
        (
            (IdDoc("x"),),
            {},
            "(x)",
        ),
        (
            (IdDoc("x"),),
            {"key0": IdDoc("u")},
            "(x, key0=u)",
        ),
        (
            (IdDoc("x"),),
            {"key0": IdDoc("u"), "key1": IdDoc("v")},
            "(x, key0=u, key1=v)",
        ),
        (
            (IdDoc("x"), (IdDoc("y"))),
            {},
            "(x, y)",
        ),
        (
            (IdDoc("x"), (IdDoc("y"))),
            {"key0": IdDoc("u")},
            "(x, y, key0=u)",
        ),
        (
            (IdDoc("x"), (IdDoc("y"))),
            {"key0": IdDoc("u"), "key1": IdDoc("v")},
            "(x, y, key0=u, key1=v)",
        ),
    ],
    ids=itertools.count(),
)
def test_print_call_doc(args, kwargs, expected):
    doc = CallDoc(IdDoc("f"), *args, **kwargs)
    assert to_python_script(doc) == format_script(f"f{expected}")


@pytest.mark.parametrize(
    "args, expected",
    [
        (
            (),
            "lambda : 0",
        ),
        (
            (IdDoc("x"),),
            "lambda x: 0",
        ),
        (
            (IdDoc("x"), IdDoc("y")),
            "lambda x, y: 0",
        ),
        (
            (IdDoc("x"), IdDoc("y"), IdDoc("z")),
            "lambda x, y, z: 0",
        ),
    ],
    ids=itertools.count(),
)
def test_print_lambda_doc(args, expected):
    doc = LambdaDoc(args, body=LiteralDoc(0))
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "elements, expected",
    [
        (
            (),
            "[]",
        ),
        (
            [IdDoc("x")],
            "[x]",
        ),
        (
            [IdDoc("x"), IdDoc("y")],
            "[x, y]",
        ),
        (
            [IdDoc("x"), IdDoc("y"), IdDoc("z")],
            "[x, y, z]",
        ),
    ],
    ids=itertools.count(),
)
def test_print_list_doc(elements, expected):
    doc = ListDoc(elements)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "elements, expected",
    [
        (
            (),
            "()",
        ),
        (
            [IdDoc("x")],
            "(x,)",
        ),
        (
            [IdDoc("x"), IdDoc("y")],
            "(x, y)",
        ),
        (
            [IdDoc("x"), IdDoc("y"), IdDoc("z")],
            "(x, y, z)",
        ),
    ],
    ids=itertools.count(),
)
def test_print_tuple_doc(elements, expected):
    doc = TupleDoc(elements)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "content, expected",
    [
        (
            {},
            "{}",
        ),
        (
            {LiteralDoc("key_x"): IdDoc("x")},
            '{"key_x": x}',
        ),
        (
            {LiteralDoc("key_x"): IdDoc("x"), LiteralDoc("key_y"): IdDoc("y")},
            '{"key_x": x, "key_y": y}',
        ),
        (
            {
                LiteralDoc("key_x"): IdDoc("x"),
                LiteralDoc("key_y"): IdDoc("y"),
                LiteralDoc("key_z"): IdDoc("z"),
            },
            '{"key_x": x, "key_y": y, "key_z": z}',
        ),
    ],
    ids=itertools.count(),
)
def test_print_dict_doc(content, expected):
    doc = DictDoc(content)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "slice_doc, expected",
    [
        (
            SliceDoc(),
            ":",
        ),
        (
            SliceDoc(LiteralDoc(1)),
            "1:",
        ),
        (
            SliceDoc(None, LiteralDoc(2)),
            ":2",
        ),
        (
            SliceDoc(LiteralDoc(1), LiteralDoc(2)),
            "1:2",
        ),
        (
            SliceDoc(None, None, LiteralDoc(3)),
            "::3",
        ),
        (
            SliceDoc(LiteralDoc(1), None, LiteralDoc(3)),
            "1::3",
        ),
        (
            SliceDoc(None, LiteralDoc(2), LiteralDoc(3)),
            ":2:3",
        ),
        (
            SliceDoc(LiteralDoc(1), LiteralDoc(2), LiteralDoc(3)),
            "1:2:3",
        ),
    ],
    ids=itertools.count(),
)
def test_print_slice_doc(slice_doc, expected):
    doc = IdDoc("x")[slice_doc]
    assert to_python_script(doc) == format_script(f"x[{expected}]")


@pytest.mark.parametrize(
    "stmts, expected",
    [
        (
            [],
            "",
        ),
        (
            [ExprStmtDoc(IdDoc("x"))],
            "x",
        ),
        (
            [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
            """
            x
            y
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_stmt_block_doc(stmts, expected):
    doc = StmtBlockDoc(stmts)
    assert to_python_script(doc).strip() == format_script(expected).strip()


@pytest.mark.parametrize(
    "doc, expected",
    [
        (
            AssignDoc(IdDoc("x"), IdDoc("y"), None),
            "x = y",
        ),
        (
            AssignDoc(IdDoc("x"), IdDoc("y"), IdDoc("int")),
            "x: int = y",
        ),
        (
            AssignDoc(IdDoc("x"), None, IdDoc("int")),
            "x: int",
        ),
        (
            AssignDoc(TupleDoc([IdDoc("x"), IdDoc("y")]), IdDoc("z"), None),
            "x, y = z",
        ),
        (
            AssignDoc(TupleDoc([IdDoc("x"), TupleDoc([IdDoc("y"), IdDoc("z")])]), IdDoc("z"), None),
            "x, (y, z) = z",
        ),
    ],
    ids=itertools.count(),
)
def test_print_assign_doc(doc, expected):
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "then_branch, else_branch, expected",
    [
        (
            [ExprStmtDoc(IdDoc("x"))],
            [],
            """
            if pred:
                x
            """,
        ),
        (
            [],
            [ExprStmtDoc(IdDoc("y"))],
            """
            if pred:
                pass
            else:
                y
            """,
        ),
        (
            [ExprStmtDoc(IdDoc("x"))],
            [ExprStmtDoc(IdDoc("y"))],
            """
            if pred:
                x
            else:
                y
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_if_doc(then_branch, else_branch, expected):
    doc = IfDoc(IdDoc("pred"), then_branch, else_branch)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            [ExprStmtDoc(IdDoc("x"))],
            """
            while pred:
                x
            """,
        ),
        (
            [],
            """
            while pred:
                pass
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_while_doc(body, expected):
    doc = WhileDoc(IdDoc("pred"), body)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            [ExprStmtDoc(IdDoc("x"))],
            """
            for x in y:
                x
            """,
        ),
        (
            [],
            """
            for x in y:
                pass
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_for_doc(body, expected):
    doc = ForDoc(IdDoc("x"), IdDoc("y"), body)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "lhs, body, expected",
    [
        (
            IdDoc("c"),
            [ExprStmtDoc(IdDoc("x"))],
            """
            with context() as c:
                x
            """,
        ),
        (
            IdDoc("c"),
            [],
            """
            with context() as c:
                pass
            """,
        ),
        (
            None,
            [],
            """
            with context():
                pass
            """,
        ),
        (
            None,
            [ExprStmtDoc(IdDoc("x"))],
            """
            with context():
                x
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_scope_doc(lhs, body, expected):
    doc = ScopeDoc(lhs, CallDoc(IdDoc("context")), body)
    assert to_python_script(doc) == format_script(expected)


def test_print_expr_stmt_doc():
    doc = ExprStmtDoc(CallDoc(IdDoc("f"), IdDoc("x")))
    assert to_python_script(doc) == format_script("f(x)")


@pytest.mark.parametrize(
    "msg, expected",
    [
        (
            None,
            """
            assert True
            """,
        ),
        (
            LiteralDoc("test message"),
            """
            assert True, "test message"
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_assert_doc(msg, expected):
    test = LiteralDoc(True)

    doc = AssertDoc(test, msg)

    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            LiteralDoc(None),
            """
            return None
            """,
        ),
        (
            IdDoc("x"),
            """
            return x
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_return_doc(value, expected):
    doc = ReturnDoc(value)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "args, decorators, body, expected",
    [
        (
            [],
            [],
            [],
            """
            def func() -> None:
                pass
            """,
        ),
        (
            [AssignDoc(IdDoc("x"), rhs=None, annotation=IdDoc("int"))],
            [],
            [],
            """
            def func(x: int) -> None:
                pass
            """,
        ),
        (
            [AssignDoc(IdDoc("x"), rhs=LiteralDoc(1), annotation=IdDoc("int"))],
            [],
            [],
            """
            def func(x: int = 1) -> None:
                pass
            """,
        ),
        (
            [],
            [IdDoc("wrap")],
            [],
            """
            @wrap
            def func() -> None:
                pass
            """,
        ),
        (
            [],
            [IdDoc("wrap_outter"), IdDoc("wrap_inner")],
            [],
            """
            @wrap_outter
            @wrap_inner
            def func() -> None:
                pass
            """,
        ),
        (
            [
                AssignDoc(IdDoc("x"), rhs=None, annotation=IdDoc("int")),
                AssignDoc(IdDoc("y"), rhs=LiteralDoc(1), annotation=IdDoc("int")),
            ],
            [IdDoc("wrap")],
            [],
            """
            @wrap
            def func(x: int, y: int = 1) -> None:
                pass
            """,
        ),
        (
            [
                AssignDoc(IdDoc("x"), rhs=None, annotation=IdDoc("int")),
                AssignDoc(IdDoc("y"), rhs=LiteralDoc(1), annotation=IdDoc("int")),
            ],
            [IdDoc("wrap")],
            [
                AssignDoc(IdDoc("y"), OperationDoc(OperationKind.Add, [IdDoc("x"), LiteralDoc(1)])),
                AssignDoc(IdDoc("y"), OperationDoc(OperationKind.Sub, [IdDoc("y"), LiteralDoc(1)])),
            ],
            """
            @wrap
            def func(x: int, y: int = 1) -> None:
                y = x + 1
                y = y - 1
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_function_doc(args, decorators, body, expected):
    doc = FunctionDoc(IdDoc("func"), args, decorators, LiteralDoc(None), body)
    assert to_python_script(doc) == format_script(expected)  # test


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


@pytest.mark.parametrize(
    "decorators, body, expected",
    [
        (
            [],
            [],
            """
            class TestClass:
                pass
            """,
        ),
        (
            [IdDoc("wrap")],
            [],
            """
            @wrap
            class TestClass:
                pass
            """,
        ),
        (
            [IdDoc("wrap_outter"), IdDoc("wrap_inner")],
            [],
            """
            @wrap_outter
            @wrap_inner
            class TestClass:
                pass
            """,
        ),
        (
            [IdDoc("wrap")],
            [get_func_doc_for_class("f1")],
            """
            @wrap
            class TestClass:
                @wrap
                def f1(x: int, y: int = 1) -> None:
                    y = x + 1
                    y = y - 1

            """,
        ),
        (
            [IdDoc("wrap")],
            [get_func_doc_for_class("f1"), get_func_doc_for_class("f2")],
            """
            @wrap
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
    ],
    ids=itertools.count(),
)
def test_print_class_doc(decorators, body, expected):
    doc = ClassDoc(IdDoc("TestClass"), decorators, body)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "doc, comment, expected",
    [
        (
            AssignDoc(IdDoc("x"), IdDoc("y"), IdDoc("int")),
            "comment",
            """
            x: int = y  # comment
            """,
        ),
        (
            IfDoc(IdDoc("x"), [ExprStmtDoc(IdDoc("y"))], [ExprStmtDoc(IdDoc("z"))]),
            "comment",
            """
            # comment
            if x:
                y
            else:
                z
            """,
        ),
        (
            IfDoc(IdDoc("x"), [ExprStmtDoc(IdDoc("y"))], [ExprStmtDoc(IdDoc("z"))]),
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
        (
            WhileDoc(
                LiteralDoc(True),
                [
                    AssignDoc(IdDoc("x"), IdDoc("y")),
                ],
            ),
            "comment",
            """
            # comment
            while True:
                x = y
            """,
        ),
        (
            ForDoc(IdDoc("x"), IdDoc("y"), []),
            "comment",
            """
            # comment
            for x in y:
                pass
            """,
        ),
        (
            ScopeDoc(IdDoc("x"), IdDoc("y"), []),
            "comment",
            """
            # comment
            with y as x:
                pass
            """,
        ),
        (
            ExprStmtDoc(IdDoc("x")),
            "comment",
            """
            x  # comment
            """,
        ),
        (
            AssertDoc(LiteralDoc(True)),
            "comment",
            """
            assert True  # comment
            """,
        ),
        (
            ReturnDoc(LiteralDoc(1)),
            "comment",
            """
            return 1  # comment
            """,
        ),
        (
            get_func_doc_for_class("f"),
            "comment",
            '''
            @wrap
            def f(x: int, y: int = 1) -> None:
                """
                comment
                """
                y = x + 1
                y = y - 1
            ''',
        ),
        (
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
        (
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
        (
            ClassDoc(IdDoc("TestClass"), decorators=[IdDoc("wrap")], body=[]),
            "comment line 1\n\ncomment line 3",
            '''
            @wrap
            class TestClass:
                """
                comment line 1

                comment line 3
                """
                pass
            ''',
        ),
    ],
    ids=itertools.count(),
)
def test_print_doc_comment(doc, comment, expected):
    doc.comment = comment
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "doc",
    [
        AssignDoc(IdDoc("x"), IdDoc("y"), IdDoc("int")),
        ExprStmtDoc(IdDoc("x")),
        AssertDoc(IdDoc("x")),
        ReturnDoc(IdDoc("x")),
    ],
)
def test_print_invalid_multiline_doc_comment(doc):
    doc.comment = "1\n2"
    with pytest.raises(ValueError) as e:
        to_python_script(doc)
    assert "cannot have newline" in str(e.value)
