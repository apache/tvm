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

from typing import Optional

import pytest
from tvm.runtime import ObjectPath
from tvm.script.printer.doc import (
    ExprStmtDoc,
    IdDoc,
    OperationDoc,
    OperationKind,
    StmtBlockDoc,
)
from tvm.script.printer.doc_printer import to_python_script
from tvm.script import ir as I, tir as T


def make_path(name: str) -> ObjectPath:
    return ObjectPath.root().attr(name)


def make_id_doc(name: str, path_name: Optional[str] = None) -> IdDoc:
    if path_name is None:
        path_name = name
    doc = IdDoc(name)
    doc.source_paths = [make_path(path_name)]
    return doc


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
    return cleaned_lines.strip()


def test_underline_basic():
    doc = StmtBlockDoc(
        [
            ExprStmtDoc(make_id_doc("foo")),
            ExprStmtDoc(OperationDoc(OperationKind.Add, [make_id_doc("bar"), make_id_doc("baz")])),
            ExprStmtDoc(make_id_doc("qux")),
        ]
    )
    assert to_python_script(doc, path_to_underline=[make_path("baz")]) == format_script(
        """
        foo
        bar + baz
              ^^^
        qux
    """
    )


def test_underline_multiple_spans():
    doc = StmtBlockDoc(
        [
            ExprStmtDoc(make_id_doc("foo")),
            ExprStmtDoc(make_id_doc("bar")),
            ExprStmtDoc(OperationDoc(OperationKind.Add, [make_id_doc("foo"), make_id_doc("foo")])),
        ]
    )
    assert to_python_script(doc, path_to_underline=[make_path("foo")]) == format_script(
        """
        foo
        ^^^
        bar
        foo + foo
        ^^^   ^^^
    """
    )


def test_underline_multiple_spans_with_line_numbers():
    doc = StmtBlockDoc(
        [
            ExprStmtDoc(make_id_doc("foo")),
            ExprStmtDoc(make_id_doc("bar")),
            ExprStmtDoc(OperationDoc(OperationKind.Add, [make_id_doc("foo"), make_id_doc("foo")])),
        ]
    )
    assert to_python_script(
        doc, print_line_numbers=True, path_to_underline=[make_path("foo")]
    ) == format_script(
        """
        1 foo
          ^^^
        2 bar
        3 foo + foo
          ^^^   ^^^
    """
    )


def test_underline_multiline():
    doc = StmtBlockDoc(
        [
            ExprStmtDoc(IdDoc("foo")),
            ExprStmtDoc(IdDoc("bar")),
        ]
    )
    doc.source_paths = [make_path("whole_doc")]

    assert to_python_script(doc, path_to_underline=[make_path("whole_doc")]) == format_script(
        """
        foo
        ^^^
        bar
        ^^^
    """
    )


@pytest.mark.parametrize(
    "to_underline, expected_text",
    [
        (
            [0],
            """
                x0
                ^^
                x1
                x2
                (... 7 lines skipped ...)
            """,
        ),
        (
            [1],
            """
                x0
                x1
                ^^
                x2
                x3
                (... 6 lines skipped ...)
            """,
        ),
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
            [6],
            """
                (... 4 lines skipped ...)
                x4
                x5
                x6
                ^^
                x7
                x8
                x9
            """,
        ),
        (
            [9],
            """
                (... 7 lines skipped ...)
                x7
                x8
                x9
                ^^
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
            [0, 6, 9],
            """
                x0
                ^^
                x1
                x2
                x3
                x4
                x5
                x6
                ^^
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
    assert result == format_script(expected_text)


def test_underline_and_print_line_numbers():
    doc = StmtBlockDoc([ExprStmtDoc(make_id_doc(f"line{i + 1}")) for i in range(12)])
    result = to_python_script(doc, print_line_numbers=True, path_to_underline=[make_path("line6")])
    assert (
        result.strip()
        == format_script(
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
        ).strip()
    )


def test_underline_multi_object_paths():
    doc = StmtBlockDoc([ExprStmtDoc(make_id_doc(f"line{i + 1}")) for i in range(10)])
    result = to_python_script(
        doc,
        path_to_underline=[
            make_path("line1"),
            make_path("line3"),
            make_path("line5"),
            make_path("line7"),
            make_path("line9"),
        ],
    )
    assert (
        result.strip()
        == format_script(
            """
            line1
            ^^^^^
            line2
            line3
            ^^^^^
            line4
            line5
            ^^^^^
            line6
            line7
            ^^^^^
            line8
            line9
            ^^^^^
            line10
    """
        ).strip()
    )


def test_underline_and_print_line_numbers_with_context():
    doc = StmtBlockDoc([ExprStmtDoc(make_id_doc(f"line{i + 1}")) for i in range(12)])
    result = to_python_script(
        doc, print_line_numbers=True, num_context_lines=2, path_to_underline=[make_path("line8")]
    )
    assert result == format_script(
        """
           (... 5 lines skipped ...)
            6 line6
            7 line7
            8 line8
              ^^^^^
            9 line9
           10 line10
           (... 2 lines skipped ...)
    """
    )


def test_underline_based_on_path_prefix():
    doc = StmtBlockDoc([ExprStmtDoc(make_id_doc("foo")), ExprStmtDoc(make_id_doc("bar"))])
    result = to_python_script(doc, path_to_underline=[make_path("foo").attr("x").attr("y")])
    # There is no document that matches the desired path exactly,
    # but path of "foo" is a prefix of the desired path, and thus should be underlined.
    assert result == format_script(
        """
        foo
        ^^^
        bar
    """
    )


def test_longer_prefix_must_win():
    foo_x = IdDoc("foo_x")
    foo_x.source_paths = [make_path("foo").attr("x")]

    doc = StmtBlockDoc(
        [ExprStmtDoc(make_id_doc("foo")), ExprStmtDoc(make_id_doc("bar")), ExprStmtDoc(foo_x)]
    )
    result = to_python_script(doc, path_to_underline=[make_path("foo").attr("x").attr("y")])
    # "foo" should not be underlined because there is a document with a more specific path prefix
    assert result == format_script(
        """
        foo
        bar
        foo_x
        ^^^^^
    """
    )


def test_underline_from_obj():
    @T.prim_func
    def func(a: T.int32, b: T.int32):
        T.evaluate(a)
        T.evaluate(b)
        T.evaluate(a)
        T.evaluate(b)
        T.evaluate(a)
        T.evaluate(b)

    result = func.with_attr("global_symbol", "main").script(obj_to_underline=[func.params[0]])
    assert result == format_script(
        """
        # from tvm.script import tir as T

        @T.prim_func
        def main(a: T.int32, b: T.int32):
            T.evaluate(a)
                       ^
            T.evaluate(b)
            T.evaluate(a)
                       ^
            T.evaluate(b)
            T.evaluate(a)
                       ^
            T.evaluate(b)
    """
    )


def test_underline_from_multi_obj():
    @T.prim_func
    def func():
        T.evaluate(-1)
        T.evaluate(1)
        T.evaluate(2)
        T.evaluate(3)
        T.evaluate(4)
        T.evaluate(5)
        T.evaluate(6)
        T.evaluate(7)

    result = func.with_attr("global_symbol", "main").script(
        obj_to_underline=[
            func.body.seq[1],
            func.body.seq[3],
            func.body.seq[5],
            func.body.seq[7],
        ]
    )
    assert result == format_script(
        """
        # from tvm.script import tir as T

        @T.prim_func
        def main():
            T.evaluate(-1)
            T.evaluate(1)
            ^^^^^^^^^^^^^
            T.evaluate(2)
            T.evaluate(3)
            ^^^^^^^^^^^^^
            T.evaluate(4)
            T.evaluate(5)
            ^^^^^^^^^^^^^
            T.evaluate(6)
            T.evaluate(7)
            ^^^^^^^^^^^^^
    """
    )


def test_underline_func():
    @T.prim_func
    def func():
        T.evaluate(0)

    result = func.with_attr("global_symbol", "main").script(
        path_to_underline=[
            ObjectPath.root(),
        ]
    )
    assert result == format_script(
        """
        # from tvm.script import tir as T

        @T.prim_func
        ^^^^^^^^^^^^
        def main():
        ^^^^^^^^^^^
            T.evaluate(0)
            ^^^^^^^^^^^^^
    """
    )


def test_underline_func_in_irmodule():
    @I.ir_module
    class irmodule:
        @T.prim_func
        def func():
            T.evaluate(0)

    result = irmodule.script(
        path_to_underline=[
            ObjectPath.root().attr("functions").map_value(irmodule.get_global_var("func")),
        ]
    )
    assert result == format_script(
        """
        # from tvm.script import ir as I
        # from tvm.script import tir as T

        @I.ir_module
        class Module:
            @T.prim_func
            ^^^^^^^^^^^^
            def func():
            ^^^^^^^^^^^
                T.evaluate(0)
                ^^^^^^^^^^^^^
    """
    )


def test_underline_irmodule():
    @I.ir_module
    class irmodule:
        @T.prim_func
        def func():
            T.evaluate(0)

    result = irmodule.script(
        path_to_underline=[
            ObjectPath.root(),
        ]
    )
    assert result == format_script(
        """
        # from tvm.script import ir as I
        # from tvm.script import tir as T

        @I.ir_module
        ^^^^^^^^^^^^
        class Module:
        ^^^^^^^^^^^^^
            @T.prim_func
            ^^^^^^^^^^^^
            def func():
            ^^^^^^^^^^^
                T.evaluate(0)
                ^^^^^^^^^^^^^
    """
    )
