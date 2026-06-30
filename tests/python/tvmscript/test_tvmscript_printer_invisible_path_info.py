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
from tvm_ffi.access_path import AccessPath

import tvm
from tvm.runtime import script_printer
from tvm.runtime.script_printer import PrinterConfig
from tvm.script.printer.doc import ExprStmtDoc, IdDoc, StmtBlockDoc
from tvm.script.printer.doc_printer import to_python_script


def make_path(name: str) -> AccessPath:
    return AccessPath.root().attr(name)


def format_script(s: str) -> str:
    s = s.strip("\n")

    non_empty_lines = [line for line in s.splitlines() if line and not line.isspace()]
    line_indents = [len(line) - len(line.lstrip(" ")) for line in non_empty_lines]
    spaces_to_remove = min(line_indents)

    return "\n".join(line[spaces_to_remove:] for line in s.splitlines()).strip()


def test_render_invisible_path_info_reports_visible_prefix():
    foo = IdDoc("foo")
    foo.source_paths = [make_path("foo")]

    foo_x = IdDoc("foo_x")
    foo_x.source_paths = [make_path("foo").attr("x")]

    doc = StmtBlockDoc([ExprStmtDoc(foo), ExprStmtDoc(foo_x)])

    script, visible_paths = to_python_script(
        doc,
        path_to_underline=[
            make_path("foo").attr("x").attr("y"),
            make_path("missing").attr("field"),
        ],
        render_invisible_path_info=True,
    )

    assert script == format_script(
        """
        foo
        foo_x
        ^^^^^
    """
    )
    assert visible_paths == [make_path("foo").attr("x"), None]


def test_render_invisible_path_info_is_not_python_printer_config_surface():
    config = PrinterConfig()

    assert not hasattr(config, "render_invisible_path_info")


def test_script_render_invisible_path_info_is_passable_from_python():
    script, visible_paths = tvm.ir.IRModule({}).script(
        path_to_underline=[AccessPath.root()],
        render_invisible_path_info=True,
    )

    assert isinstance(script, str)
    assert len(visible_paths) == 1


def test_structural_equal_reports_hidden_field_suffix():
    lhs = tvm.ir.PrimType("int32")
    rhs = tvm.ir.PrimType("float32")

    try:
        tvm.ir.assert_structural_equal(lhs, rhs)
    except ValueError as err:
        message = str(err)
    else:
        raise AssertionError("Expected structural equality failure")

    hidden_context = (
        "Access path: <root>.dtype\n"
        "Highlighted object: <root>\n"
        "Hidden field: .dtype\n"
        "Note: The hidden field is not rendered in TVMScript, so the underline points to the "
        "nearest visible object in the access path."
    )
    lhs_header = "StructuralEqual check failed, caused by lhs at:\n\n"
    rhs_header = "\n\nand rhs at:\n\n"
    assert message.startswith(lhs_header)
    assert rhs_header in message
    lhs_block, rhs_block = message[len(lhs_header) :].split(rhs_header)
    assert lhs_block.startswith(f"{hidden_context}\n\n")
    assert rhs_block.startswith(f"{hidden_context}\n\n")
    assert "int32" in lhs_block
    assert "float32" in rhs_block
    assert hidden_context in message
    assert message.count("Access path:") == 2
    assert "Visible anchor:" not in message
    assert "Hidden field: .dtype" in message
    assert "Full internal path:" not in message
    assert message.index("Access path: <root>.dtype") < message.index("^^")


def test_structural_equal_uses_invisible_path_info_render(monkeypatch):
    lhs = tvm.ir.PrimType("int32")
    rhs = tvm.ir.PrimType("float32")
    calls = []

    def fake_script(obj, config, render_invisible_path_info=False):
        calls.append((obj, config, render_invisible_path_info))
        assert render_invisible_path_info
        requested_path = config.path_to_underline[0]
        return f"script {len(calls)}", [requested_path]

    monkeypatch.setattr(script_printer, "_script", fake_script)

    with pytest.raises(ValueError) as err:
        tvm.ir.assert_structural_equal(lhs, rhs)

    message = str(err.value)
    assert len(calls) == 2
    assert calls[0][0] == lhs
    assert calls[1][0] == rhs
    assert all(render_invisible_path_info for _, _, render_invisible_path_info in calls)
    assert "StructuralEqual check failed, caused by lhs at:\n\n" in message
    assert "\n\nand rhs at:\n\n" in message
    assert "Access path: <root>.dtype\n\nscript 1" in message
    assert "Access path: <root>.dtype\n\nscript 2" in message
    assert "Hidden field:" not in message
