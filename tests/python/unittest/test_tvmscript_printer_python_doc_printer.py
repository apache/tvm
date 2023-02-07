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
import itertools

import pytest
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
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")

    non_empty_lines = [line for line in s.splitlines() if line and not line.isspace()]
    if not non_empty_lines:
        # no actual content
        return ""

    line_indents = [len(line) - len(line.lstrip(" ")) for line in non_empty_lines]
    spaces_to_remove = min(line_indents)

    cleaned_lines = "\n".join(line[spaces_to_remove:] for line in s.splitlines())
    if not cleaned_lines.endswith("\n"):
        cleaned_lines += "\n"
    return cleaned_lines.strip()


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
    OperationKind.Not: "not ",
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
    OperationKind.And: "and",
    OperationKind.Or: "or",
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
    "args, decorators, return_type, body, expected",
    [
        (
            [],
            [],
            None,
            [],
            """
            def func():
                pass
            """,
        ),
        (
            [AssignDoc(IdDoc("x"), rhs=None, annotation=IdDoc("int"))],
            [],
            IdDoc("int"),
            [],
            """
            def func(x: int) -> int:
                pass
            """,
        ),
        (
            [AssignDoc(IdDoc("x"), rhs=LiteralDoc(1), annotation=IdDoc("int"))],
            [],
            LiteralDoc(None),
            [],
            """
            def func(x: int = 1) -> None:
                pass
            """,
        ),
        (
            [],
            [IdDoc("wrap")],
            LiteralDoc(None),
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
            LiteralDoc(None),
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
            LiteralDoc(None),
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
            LiteralDoc(None),
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
def test_print_function_doc(args, decorators, body, return_type, expected):
    doc = FunctionDoc(IdDoc("func"), args, decorators, return_type, body)
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
    "comment, expected",
    [
        (
            "",
            "",
        ),
        (
            "test comment 1",
            "# test comment 1",
        ),
        (
            "test comment 1\ntest comment 2",
            """
            # test comment 1
            # test comment 2
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_comment_doc(comment, expected):
    doc = CommentDoc(comment)
    assert to_python_script(doc) == format_script(expected)


@pytest.mark.parametrize(
    "comment, expected",
    [
        (
            "",
            "",
        ),
        (
            "test comment 1",
            '''
            """
            test comment 1
            """
            ''',
        ),
        (
            "test comment 1\ntest comment 2",
            '''
            """
            test comment 1
            test comment 2
            """
            ''',
        ),
    ],
    ids=itertools.count(),
)
def test_print_doc_string_doc(comment, expected):
    doc = DocStringDoc(comment)
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


def generate_expr_precedence_test_cases():
    x = IdDoc("x")
    y = IdDoc("y")
    z = IdDoc("z")

    def negative(a):
        return OperationDoc(OperationKind.USub, [a])

    def invert(a):
        return OperationDoc(OperationKind.Invert, [a])

    def not_(a):
        return OperationDoc(OperationKind.Not, [a])

    def add(a, b):
        return OperationDoc(OperationKind.Add, [a, b])

    def sub(a, b):
        return OperationDoc(OperationKind.Sub, [a, b])

    def mult(a, b):
        return OperationDoc(OperationKind.Mult, [a, b])

    def div(a, b):
        return OperationDoc(OperationKind.Div, [a, b])

    def mod(a, b):
        return OperationDoc(OperationKind.Mod, [a, b])

    def pow(a, b):
        return OperationDoc(OperationKind.Pow, [a, b])

    def lshift(a, b):
        return OperationDoc(OperationKind.LShift, [a, b])

    def bit_and(a, b):
        return OperationDoc(OperationKind.BitAnd, [a, b])

    def bit_or(a, b):
        return OperationDoc(OperationKind.BitOr, [a, b])

    def bit_xor(a, b):
        return OperationDoc(OperationKind.BitXor, [a, b])

    def lt(a, b):
        return OperationDoc(OperationKind.Lt, [a, b])

    def eq(a, b):
        return OperationDoc(OperationKind.Eq, [a, b])

    def not_eq(a, b):
        return OperationDoc(OperationKind.NotEq, [a, b])

    def and_(a, b):
        return OperationDoc(OperationKind.And, [a, b])

    def or_(a, b):
        return OperationDoc(OperationKind.Or, [a, b])

    def if_then_else(a, b, c):
        return OperationDoc(OperationKind.IfThenElse, [a, b, c])

    test_cases = {
        "attr-call-index": [
            (
                add(x, y).attr("test"),
                "(x + y).test",
            ),
            (
                add(x, y.attr("test")),
                "x + y.test",
            ),
            (
                x[z].call(y),
                "x[z](y)",
            ),
            (
                x.call(y)[z],
                "x(y)[z]",
            ),
            (
                x.call(y).call(z),
                "x(y)(z)",
            ),
            (
                x.call(y).attr("test"),
                "x(y).test",
            ),
            (
                x.attr("test").call(y),
                "x.test(y)",
            ),
            (
                x.attr("test").attr("test2"),
                "x.test.test2",
            ),
            (
                LambdaDoc([x], x).call(y),
                "(lambda x: x)(y)",
            ),
            (
                add(x, y)[z][add(z, z)].attr("name"),
                "(x + y)[z][z + z].name",
            ),
        ],
        "power": [
            (
                pow(pow(x, y), z),
                "(x ** y) ** z",
            ),
            (
                pow(x, pow(y, z)),
                "x ** y ** z",
            ),
            (
                pow(negative(x), negative(y)),
                "(-x) ** -y",
            ),
            (
                pow(add(x, y), add(y, z)),
                "(x + y) ** (y + z)",
            ),
        ],
        "unary": [
            (
                invert(negative(y)),
                "~-y",
            ),
            (
                negative(y).attr("test"),
                "(-y).test",
            ),
            (
                negative(y.attr("test")),
                "-y.test",
            ),
            (
                mult(negative(x), negative(y)),
                "-x * -y",
            ),
            (
                negative(add(invert(x), negative(y))),
                "-(~x + -y)",
            ),
        ],
        "add-mult": [
            (
                mult(x, mult(y, z)),
                "x * (y * z)",
            ),
            (
                mult(mult(x, y), z),
                "x * y * z",
            ),
            (
                mult(x, add(y, z)),
                "x * (y + z)",
            ),
            (
                mult(add(y, z), x),
                "(y + z) * x",
            ),
            (
                add(x, mod(y, z)),
                "x + y % z",
            ),
            (
                add(mult(y, z), x),
                "y * z + x",
            ),
            (
                add(add(x, y), add(y, z)),
                "x + y + (y + z)",
            ),
            (
                div(add(x, y), add(y, z)),
                "(x + y) / (y + z)",
            ),
        ],
        "shift": [
            (
                div(x, lshift(y, z)),
                "x / (y << z)",
            ),
            (
                mult(lshift(y, z), x),
                "(y << z) * x",
            ),
            (
                lshift(x, mult(y, z)),
                "x << y * z",
            ),
            (
                lshift(mult(x, y), z),
                "x * y << z",
            ),
            (
                lshift(mult(x, y), z),
                "x * y << z",
            ),
            (
                lshift(lshift(x, y), z),
                "x << y << z",
            ),
            (
                lshift(x, lshift(y, z)),
                "x << (y << z)",
            ),
        ],
        "bitwise": [
            (
                add(bit_or(x, y), bit_or(y, z)),
                "(x | y) + (y | z)",
            ),
            (
                bit_and(bit_or(x, y), bit_or(y, z)),
                "(x | y) & (y | z)",
            ),
            (
                bit_or(bit_and(x, y), bit_and(y, z)),
                "x & y | y & z",
            ),
            (
                bit_and(bit_xor(x, bit_or(y, z)), z),
                "(x ^ (y | z)) & z",
            ),
        ],
        "comparison": [
            (
                not_eq(add(x, y), z),
                "x + y != z",
            ),
            (
                eq(pow(x, y), z),
                "x ** y == z",
            ),
            (
                lt(x, div(y, z)),
                "x < y / z",
            ),
            (
                lt(x, if_then_else(y, y, y)),
                "x < (y if y else y)",
            ),
        ],
        "boolean": [
            (
                not_(and_(x, y)),
                "not (x and y)",
            ),
            (
                and_(not_(x), y),
                "not x and y",
            ),
            (
                and_(or_(x, y), z),
                "(x or y) and z",
            ),
            (
                or_(x, or_(y, z)),
                "x or (y or z)",
            ),
            (
                or_(or_(x, y), z),
                "x or y or z",
            ),
            (
                or_(and_(x, y), z),
                # Maybe we should consider adding parentheses here
                # for readability, even though it's not necessary.
                "x and y or z",
            ),
            (
                and_(or_(not_(x), y), z),
                "(not x or y) and z",
            ),
            (
                and_(lt(x, y), lt(y, z)),
                "x < y and y < z",
            ),
            (
                or_(not_(eq(x, y)), lt(y, z)),
                # Same as the previous one, the code here is not
                # readable without parentheses.
                "not x == y or y < z",
            ),
            (
                and_(if_then_else(x, y, z), x),
                "(y if x else z) and x",
            ),
            (
                not_(if_then_else(x, y, z)),
                "not (y if x else z)",
            ),
        ],
        "if-then-else": [
            (
                if_then_else(x, if_then_else(y, y, y), z),
                "y if y else y if x else z",
            ),
            (
                if_then_else(if_then_else(x, x, x), y, z),
                "y if (x if x else x) else z",
            ),
            (
                if_then_else(x, y, if_then_else(z, z, z)),
                "y if x else (z if z else z)",
            ),
            (
                if_then_else(lt(x, x), add(y, y), mult(z, z)),
                "y + y if x < x else z * z",
            ),
            (
                if_then_else(LambdaDoc([x], x), LambdaDoc([y], y), LambdaDoc([z], z)),
                "(lambda y: y) if (lambda x: x) else (lambda z: z)",
            ),
        ],
        "lambda": [
            (
                LambdaDoc([x, y], add(z, z)),
                "lambda x, y: z + z",
            ),
            (
                add(LambdaDoc([x, y], z), z),
                "(lambda x, y: z) + z",
            ),
            (
                LambdaDoc([x, y], add(z, z)).call(x, y),
                "(lambda x, y: z + z)(x, y)",
            ),
            (
                LambdaDoc([x], LambdaDoc([y], z)),
                "lambda x: lambda y: z",
            ),
        ],
    }

    return [
        pytest.param(*args, id=f"{group_name}-{i}")
        for group_name, cases in test_cases.items()
        for i, args in enumerate(cases)
    ]


@pytest.mark.parametrize("doc, expected", generate_expr_precedence_test_cases())
def test_expr_precedence(doc, expected):
    assert to_python_script(doc) == format_script(expected)


if __name__ == "__main__":
    tvm.testing.main()
