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
# ruff: noqa: E741, F401, F821, F841, RUF005
import inspect
import re

import pytest

import tvm
import tvm.testing
from tvm import tirx
from tvm.script import from_source
from tvm.script import tirx as T


def check_error(func, rel_lineno):
    """check if TIR script throws error"""
    check_error_re = re.compile(r"^.*# check_error: (.+)$")
    source_code = inspect.getsource(func)
    indent = len(re.match(r"^\s*", source_code).group(0))
    source_code = "@T.prim_func(s_tir=True)\n" + "\n".join(
        line[indent:] for line in source_code.splitlines()
    )
    # Parse errors now raise DiagnosticError with formatted source location.
    with pytest.raises(tvm.error.DiagnosticError) as execinfo:
        from_source(source_code)
    err_str = str(execinfo.value)
    if rel_lineno is None:
        return
    # The error message contains " --> <source>:<lineno>:<col>" formatted by Diagnostics.error().
    # Accept either rel_lineno or rel_lineno+1 to match old tolerance.
    assert f":{rel_lineno}:" in err_str or f":{rel_lineno + 1}:" in err_str, (
        f"Expected error message to contain line {rel_lineno}, got:\n{err_str}"
    )
    error_line = source_code.split("\n")[rel_lineno]
    m = check_error_re.match(error_line)
    if m:
        expected_error_text = m.group(1)
        assert expected_error_text in err_str, (
            f'check_error expects "{expected_error_text}" in error: {err_str}'
        )


def test_buffer_bind():
    def buffer_bind_missing_args(a: T.handle) -> None:
        A = T.match_buffer((16, 16), "float32")  # error

    check_error(buffer_bind_missing_args, 2)


def test_undefined_buffer():
    def undefined_buffer(a: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")

        for i in T.serial(16):
            for j in T.serial(0, 16):
                C[i, j] = 0.0  # error

    check_error(undefined_buffer, 6)


def test_unsupported_function_call():
    def unsupported_function_call(a: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")

        for i in T.const_range(16):  # error
            for j in T.serial(0, 16):
                A[i, j] = 0.0

    check_error(unsupported_function_call, 4)


def test_missing_type_annotation():
    def missing_type_annotation(a) -> None:  # error
        T.evaluate(0.0)

    check_error(missing_type_annotation, 1)


def test_invalid_for_function():
    def invalid_for_function(a: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")
        for i in T.evaluate(0.0):  # error
            for j in T.serial(0, 16):
                A[i, j] = 0.0

    check_error(invalid_for_function, 4)


def test_invalid_block_function():
    def invalid_block_function(a: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")

        with T.evaluate(0.0):  # error
            T.evaluate(1.0)

    check_error(invalid_block_function, 4)


def test_return_not_allowed():
    def return_not_allowed(a: T.handle) -> None:
        return T.evaluate(0)  # error

    check_error(return_not_allowed, 2)


def test_no_body():
    def no_body(a: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")
        T.realize(A, "")  # error

    check_error(no_body, 3)


def test_inconsistent_binding():
    def inconsistent_binding_value() -> None:
        for i, j in T.grid(16, 16):
            vi, vj = T.axis.remap("SS", [i])  # error
            T.evaluate(1.0)

    def inconsistent_binding_type() -> None:
        for i, j in T.grid(16, 16):
            vi, vj = T.axis.remap("S", [i, j])  # error
            T.evaluate(1.0)

    check_error(inconsistent_binding_value, 3)
    check_error(inconsistent_binding_type, 3)


def test_error_remap_args():
    def error_remap_type() -> None:
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi, vj = T.axis.remap("TT", [i, j])  # error
                T.evaluate(1.0)

    def error_remap_value() -> None:
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i + j, j])  # error
                T.evaluate(1.0)

    check_error(error_remap_type, 4)
    check_error(error_remap_value, 4)


def test_invalid_block_axes():
    def invalid_block_axes(a: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi = T.axis.S(i, A)  # error
                T.evaluate(1.0)

    check_error(invalid_block_axes, 5)


def test_duplicate_block_axes():
    def duplicate_block_axes() -> None:
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi = T.axis.S(16, i)
                vi = T.axis.S(16, j)  # error
                T.evaluate(1.0)

    def duplicate_block_axes_remap() -> None:
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi, vi = T.axis.remap("SS", [i, j])  # error
                T.evaluate(1.0)

    check_error(duplicate_block_axes, 5)
    check_error(duplicate_block_axes_remap, 4)


def test_miss_block_bind():
    def miss_block_bind_value() -> None:
        for i, j in T.grid(128, 128):
            with T.sblock():
                vi = T.axis.S(i)  # error
                T.evaluate(1.0)

    check_error(miss_block_bind_value, 4)


def test_invalid_loop_var():
    def invalid_loop_var() -> None:
        for i, j in range(0, 16):  # error
            T.evaluate(1.0)

    check_error(invalid_loop_var, 2)


def test_inconsistent_grid():
    def inconsistent_grid(A: T.Buffer(16)) -> None:
        for i in T.grid(16, 16):  # valid, i is a tuple (iter0, iter1)
            T.evaluate(A[i])  # error

    check_error(inconsistent_grid, 3)


def test_invalid_match_buffer_region():
    def invalid_match_buffer_region() -> None:
        for i, j in T.grid(128, 128):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i, j])
                A = T.match_buffer(vi)  # error
                T.evaluate(1.0)

    check_error(invalid_match_buffer_region, 5)


def test_duplicate_buffer():
    def duplicate_buffer() -> None:
        A = T.sblock_alloc_buffer((128, 128), "float32")
        A = T.sblock_alloc_buffer((128, 128), "float32")  # error

    check_error(duplicate_buffer, 3)


def test_duplicate_block_signature():
    def duplicate_reads() -> None:
        A = T.sblock_alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[0:8, 0:8])
                T.reads(A[0:16, 0:16])  # error
                T.evaluate(1.0)

    def duplicate_writes() -> None:
        A = T.sblock_alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i, j])
                T.writes(A[0:8, 0:8])
                T.writes(A[0:16, 0:16])  # error
                T.evaluate(1.0)

    def duplicate_predicate() -> None:
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i, j])
                T.where(1)
                T.where(0)  # error

    def duplicate_init() -> None:
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i, j])
                with T.init():
                    T.evaluate(1.0)
                with T.init():  # error
                    T.evaluate(1.0)

    def duplicate_axes() -> None:
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i, j])
                vi = T.axis.S(i, 16)  # error
                T.evaluate(1.0)

    def duplicate_sblock_attrs_with_same_key_diff_value() -> None:
        for i, j in T.grid(16, 16):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i, j])
                T.sblock_attr({"key1": "block1"})
                T.sblock_attr({"key1": "block2"})  # error
                T.evaluate(1.0)

    check_error(duplicate_reads, 7)
    check_error(duplicate_writes, 7)
    check_error(duplicate_predicate, 6)
    check_error(duplicate_init, 7)
    check_error(duplicate_axes, 5)
    check_error(duplicate_sblock_attrs_with_same_key_diff_value, 6)


def test_opaque_access_during_complete():
    def opaque_access_during_complete(a: T.handle) -> None:  # error
        A = T.match_buffer(a, (16, 16), "float32")
        for i, j in T.grid(16, 16):
            with T.sblock():
                T.evaluate(T.call_extern("dummy_extern_function", A.data, dtype="int32"))

    check_error(opaque_access_during_complete, None)


def test_convert_slice_to_bufferload():
    def convert_slice_to_bufferload() -> None:
        A = T.sblock_alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.sblock():
                vi, vj = T.axis.remap("SS", [i, j])
                A[vi, vj] = A[vi : vi + 2, vj] + 1  # error

    check_error(convert_slice_to_bufferload, 6)


def test_tvm_exception_catch_from_special_stmt():
    def special_stmt_except() -> None:
        A = T.sblock_alloc_buffer("(128, 128)", "float32")  # error
        T.evaluate(1.0)

    check_error(special_stmt_except, 2)


def test_tvm_exception_catch_from_scope_handler():
    def scope_handler_except() -> None:
        for i in T.serial("1", "1"):  # error
            T.evaluate(1)

    check_error(scope_handler_except, 2)


def test_tvm_exception_catch_from_bare_intrin():
    def intrin_except_unassign(a: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")
        T.evaluate(A)  # error

    check_error(intrin_except_unassign, 3)


def test_tvm_exception_catch_from_assigned_intrin():
    def intrin_except_assign(a: T.handle) -> None:
        A = T.match_buffer(a, (16, 16), "float32")
        A[0, 0] = A[A]  # error

    check_error(intrin_except_assign, 3)


def test_match_buffer_shape_mismatch():
    def buffer_shape_mismatch(a: T.handle) -> None:
        A = T.match_buffer(a, (8, 8))
        for i, j in T.grid(8, 2):
            with T.sblock():
                T.reads([])
                T.writes([A[i, j * 4 : j * 4 + 4]])
                sub_A = T.match_buffer(
                    A[i, j * 4 : j * 4 + 4], (5)
                )  # error: shape mismatched between 4 and 5
                for jj in range(0, 4):
                    sub_A[i, j * 4 + jj] = 1

    check_error(buffer_shape_mismatch, 7)


def test_high_dim_store():
    def high_dim_store() -> None:
        with T.sblock("root"):
            B = T.alloc_buffer((256,), "float32")
            for i, j in T.grid(16, 16):
                B[i, j] = 1.0  # error: Store is only allowed with one index

    check_error(high_dim_store, 5)


def test_block_has_option_vars():
    def block_has_option_vars() -> None:
        with T.sblock("root") as x:  # error: block does not support option_vars
            T.evaluate(0.0)

    check_error(block_has_option_vars, 2)


def test_implicit_root_has_attrs():
    def implicit_root_has_read():
        T.reads([])  # error: implicit root does not support reads
        T.evaluate(0.0)

    def implicit_root_has_write():
        T.writes([])  # error: implicit root does not support writes
        T.evaluate(0.0)

    def implicit_root_has_attrs():
        T.sblock_attr({})  # error: implicit root does not support sblock_attr
        T.evaluate(0.0)

    def implicit_root_has_predicate():
        T.where(True)  # error: implicit root does not support predicate
        T.evaluate(0.0)

    def implicit_root_has_axes():
        v = T.axis.S(0, 0)  # error: implicit root does not support axis define
        T.evaluate(0.0)

    check_error(implicit_root_has_read, 2)
    check_error(implicit_root_has_write, 2)
    check_error(implicit_root_has_attrs, 2)
    check_error(implicit_root_has_predicate, 2)
    check_error(implicit_root_has_axes, 2)


@T.prim_func(s_tir=True)
def elementwise_not_affine(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in T.grid(128, 128, 128, 8):
        with T.sblock("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            vl = T.axis.S(128, l * 16)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@T.prim_func(s_tir=True)
def elementwise_non_single_branch(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    C = T.sblock_alloc_buffer((128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                C[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for k in T.serial(0, 128):
            with T.sblock("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                B[vi, vj, vk] = C[vi, vj, vk] * 2.0


def test_reorder_fail_block():
    sch = tvm.s_tir.Schedule(elementwise_not_affine, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError) as execinfo:
        sch.reorder(l, i)
    expected_sub_error_message = (
        "                            # tirx.SBlock#0\n"
        '                            with T.sblock("B"):\n'
        "                            ^^^^^^^^^^^^^^^^^^^\n"
    )
    assert expected_sub_error_message in str(execinfo.value)


def test_reorder_fail_nested_loop_inner():
    sch = tvm.s_tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError) as execinfo:
        sch.reorder(k, i)
    expected_sub_error_message = (
        "            for i in range(128):\n"
        "                # tirx.For#0\n"
        "                for j in range(128):\n"
        "                ^^^^^^^^^^^^^^^^^^^^\n"
    )
    assert expected_sub_error_message in str(execinfo.value)


def test_fuse_fail_nested_loop_outer():
    sch = tvm.s_tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError) as execinfo:
        sch.fuse(k, i)
    expected_sub_error_message = (
        "            # tirx.For#1\n"
        "            for i in range(128):\n"
        "            ^^^^^^^^^^^^^^^^^^^^\n"
        "                for j in range(128):\n"
    )
    assert expected_sub_error_message in str(execinfo.value)


def test_report_error_root_block():
    sch = tvm.s_tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    root = sch.get_sblock("root")
    with pytest.raises(tvm.s_tir.ScheduleError) as execinfo:
        sch.compute_inline(root)
    expected_sub_error_message = (
        '        # tirx.SBlock#0\n        with T.sblock("root"):\n        ^^^^^^^^^^^^^^^^^^^^^^\n'
    )
    assert expected_sub_error_message in str(execinfo.value)


def test_load_var():
    def load_var_multiple() -> None:
        d = T.float32()
        d[2] = d[2, 1]  # error cannot provide two indices to load

    check_error(load_var_multiple, 3)


def test_store_var():
    def store_var_multiple() -> None:
        d = T.float32()
        d[2, 1] = d[1]  # error cannot provide two indices to store

    check_error(store_var_multiple, 3)


def test_load_handle():
    def load_handle(h: T.handle) -> None:
        h_ = T.match_buffer(h, [1])
        h_[0] = h[0]  # error cannot load from handle

    check_error(load_handle, 3)


def test_store_handle():
    def store_handle(h: T.handle) -> None:
        h_ = T.match_buffer(h, [1])
        h[0] = h_[0]  # error cannot store to handle

    check_error(store_handle, 3)


def test_binop_bad_ast_type():
    def binop_bad_ast_type(h: T.handle):
        h_ = T.match_buffer(h, [1])
        h_[0] = h + [2]  # error rhs should be a primexpr

    check_error(binop_bad_ast_type, 3)


def test_binop_bad_type():
    def binop_bad_type(h: T.handle):
        h_ = T.match_buffer(h, [1])
        h_[0] = h + 2  # error lhs and rhs should be the same type

    check_error(binop_bad_type, 3)


def test_non_integer_typed_block_iter():
    def non_integer_typed_block_iter():
        with T.sblock():
            i = T.axis.S(0.1, 0.1)  # error IterVar requires an integer dtype

    check_error(non_integer_typed_block_iter, 3)


def test_illegal_buffer_slice():
    def strided_buffer_region(A: T.handle):
        # do not allow stride in buffer region
        A = T.match_buffer((128, 128), "int32")
        with T.sblock():
            T.reads([])
            T.writes([A[0:128:2, 0:128:3]])  # error
            T.evaluate(T.call_extern("strided_compute", dtype=""))

    def access_reversed_slice(A: T.handle):
        # do not allow reversed slice step
        A = T.match_buffer((128,), "int32")
        A[0:128:-1] = T.broadcast(1, 128)  # error

    def access_non_const_slice_length(A: T.handle):
        # do not allow non-constant slice length
        A = T.match_buffer((128,), "int32")
        for i in range(4):
            T.evaluate(A[0:i:1])  # error

    check_error(strided_buffer_region, 3)
    check_error(access_reversed_slice, 3)
    check_error(access_non_const_slice_length, 3)


def test_syntax_sugar_fail():
    def loop_syntax_sugar_fail(a: T.handle) -> None:
        A = T.match_buffer(a, (128,))
        for i in T.thread_binding(128, 128):
            A[i] = A[i] * 2.0

    check_error(loop_syntax_sugar_fail, 3)


def test_multi_line_error_report():
    """A parse error whose offending AST node spans several physical source
    lines must render ALL spanned lines (each with its own gutter line number
    and an underline covering the span), not just the first line."""

    # The offending call (`T.axis.remap(...)`) is deliberately split across
    # four physical lines so its AST node spans lineno..end_lineno > lineno.
    source_code = "\n".join(
        [
            "@T.prim_func(s_tir=True)",
            "def f() -> None:",
            "    for i, j in T.grid(16, 16):",
            "        vi, vj = T.axis.remap(",
            '            "S",',
            "            [i, j],",
            "        )  # error",
            "        T.evaluate(1.0)",
        ]
    )

    with pytest.raises(tvm.error.DiagnosticError) as execinfo:
        from_source(source_code)
    err_str = str(execinfo.value)

    # All four spanned source lines must appear in the rendered snippet.
    assert "T.axis.remap(" in err_str, err_str
    assert '"S",' in err_str, err_str
    assert "[i, j]," in err_str, err_str
    # The trailing `)` closing line is also part of the span.
    rendered_lines = err_str.splitlines()
    assert any(" 7 " in line and ")" in line for line in rendered_lines), err_str
    # The underline carets must be present on more than one line (multi-line).
    marker_lines = [line for line in rendered_lines if "^" in line]
    assert len(marker_lines) >= 2, err_str
    # The gutter must show distinct line numbers for the spanned lines.
    assert " 4 " in err_str and " 5 " in err_str and " 6 " in err_str, err_str


def test_format_source_snippet_multi_line():
    """Unit-level check that _format_source_snippet renders every line in a
    multi-line span, with the underline covering start-col..EOL on the first
    line, full interior lines, and col-1..end-col on the last line."""
    from tvm.script.parser.core.diagnostics import _format_source_snippet

    source_lines = [
        "first ignored line\n",
        "    foo(bar,\n",
        "        baz,\n",
        "        qux)\n",
        "last ignored line\n",
    ]
    # Span lines 2..4 (1-based), starting at col 5 ('foo'), ending at col 13
    # (exclusive) on line 4.
    snippet = _format_source_snippet(
        source_lines, lineno=2, col_offset=5, end_lineno=4, end_col_offset=13
    )
    lines = snippet.splitlines()
    # All three spanned source lines must be present.
    assert any("foo(bar," in line for line in lines), snippet
    assert any("baz," in line for line in lines), snippet
    assert any("qux)" in line for line in lines), snippet
    # Underline carets present on the first line under 'foo(bar,'.
    assert "^" in snippet, snippet
    # The line numbers 2, 3, 4 appear in the gutter.
    assert " 2 |" in snippet and " 3 |" in snippet and " 4 |" in snippet, snippet


def test_format_source_snippet_single_line_unchanged():
    """A single-line span (end_lineno == lineno) underlines only the
    [col_offset, end_col_offset) columns on that one line."""
    from tvm.script.parser.core.diagnostics import _format_source_snippet

    source_lines = ["ignored\n", "    abc + def\n", "ignored\n"]
    # Underline just 'abc' (cols 5..8 exclusive) on line 2.
    snippet = _format_source_snippet(
        source_lines, lineno=2, col_offset=5, end_lineno=2, end_col_offset=8
    )
    lines = snippet.splitlines()
    # Exactly one source-text line and one marker line (plus the leading gutter).
    text_lines = [line for line in lines if "abc + def" in line]
    assert len(text_lines) == 1, snippet
    marker_line = next(line for line in lines if "^" in line)
    assert marker_line.count("^") == 3, snippet


if __name__ == "__main__":
    tvm.testing.main()
