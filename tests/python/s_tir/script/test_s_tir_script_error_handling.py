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
"""S-TIR script error handling."""

from __future__ import annotations

import ast
import inspect
import re
import sys
import traceback

import pytest

import tvm
import tvm.testing
from tvm import tirx
from tvm.script import from_source
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def check_error(func, rel_lineno, error_type):
    """Check the original exception class and its real source location."""
    source_code = inspect.getsource(func)
    indent = len(re.match(r"^\s*", source_code).group(0))
    source_code = "@Ts.prim_func\n" + "\n".join(line[indent:] for line in source_code.splitlines())
    with pytest.raises(error_type) as caught:
        from_source(source_code, extra_vars={"T": T, "Ts": Ts})
    assert type(caught.value) is error_type
    if isinstance(caught.value, SyntaxError):
        assert caught.value.filename == "<str>"
        assert caught.value.lineno == rel_lineno + 1
        assert caught.value.offset is not None
        assert caught.value.end_lineno >= caught.value.lineno
        assert caught.value.end_offset is not None
    else:
        frames = [
            frame
            for frame in traceback.extract_tb(caught.value.__traceback__)
            if frame.filename == "<str>"
        ]
        assert frames
        if rel_lineno is not None:
            assert frames[-1].lineno == rel_lineno + 1
    if rel_lineno is not None:
        match = re.match(r"^.*# check_error: (.+)$", source_code.splitlines()[rel_lineno])
        if match:
            assert match.group(1) in str(caught.value)


def test_buffer_bind():
    def buffer_bind_missing_args(A: T.Buffer(dtype="float32")) -> None:  # error
        T.evaluate(0)

    check_error(buffer_bind_missing_args, 1, TypeError)


def test_undefined_buffer():
    def undefined_buffer(A: T.Buffer((16, 16), "float32")) -> None:
        for i in T.serial(16):
            for j in T.serial(0, 16):
                C[i, j] = 0.0  # error  # noqa: F821

    check_error(undefined_buffer, 4, NameError)


def test_unsupported_function_call():
    def unsupported_function_call(A: T.Buffer((16, 16), "float32")) -> None:
        for i in T.const_range(16):  # error
            for j in T.serial(0, 16):
                A[i, j] = 0.0

    check_error(unsupported_function_call, 2, AttributeError)


def test_invalid_for_function():
    def invalid_for_function(A: T.Buffer((16, 16), "float32")) -> None:
        for i in T.evaluate(0.0):  # error
            for j in T.serial(0, 16):
                A[i, j] = 0.0

    check_error(invalid_for_function, 2, TypeError)


def test_invalid_block_function():
    def invalid_block_function(A: T.Buffer((16, 16), "float32")) -> None:
        with T.evaluate(0.0):  # error
            T.evaluate(1.0)

    # Ordinary Python reports a missing context-manager protocol differently before 3.11.
    error_type = AttributeError if sys.version_info < (3, 11) else TypeError
    check_error(invalid_block_function, 2, error_type)


def test_return_not_allowed():
    def return_not_allowed(a: T.handle) -> None:
        return T.evaluate(0)  # error

    check_error(return_not_allowed, 2, NotImplementedError)


def test_no_body():
    def no_body(A: T.Buffer((16, 16), "float32")) -> None:
        T.realize(A, "")  # error

    check_error(no_body, 2, AttributeError)


def test_inconsistent_binding():
    def inconsistent_binding_value() -> None:
        for i, j in T.grid(16, 16):
            vi, vj = Ts.axis.remap("SS", [i])  # error
            T.evaluate(1.0)

    def inconsistent_binding_type() -> None:
        for i, j in T.grid(16, 16):
            vi, vj = Ts.axis.remap("S", [i, j])  # error
            T.evaluate(1.0)

    check_error(inconsistent_binding_value, 3, tvm.error.InternalError)
    check_error(inconsistent_binding_type, 3, tvm.error.InternalError)


def test_error_remap_args():
    def error_remap_type() -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("TT", [i, j])  # error
                T.evaluate(1.0)

    def error_remap_value() -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i + j, j])  # error
                T.evaluate(1.0)

    check_error(error_remap_type, 4, tvm.error.InternalError)
    check_error(error_remap_value, 4, tvm.error.InternalError)


def test_invalid_block_axes():
    def invalid_block_axes(A: T.Buffer((16, 16), "float32")) -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi = Ts.axis.S(i, A)  # error
                T.evaluate(1.0)

    check_error(invalid_block_axes, 4, TypeError)


def test_duplicate_block_axes():
    @Ts.prim_func
    def duplicate_block_axes() -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi = Ts.axis.S(16, i)
                vi = Ts.axis.S(16, j)
                T.evaluate(vi)

    @Ts.prim_func
    def duplicate_block_axes_remap() -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vi = Ts.axis.remap("SS", [i, j])
                T.evaluate(vi)

    # Python spelling does not rename or merge independently created native axes.
    for parsed in (duplicate_block_axes, duplicate_block_axes_remap):
        block = parsed.body.block.body.body.body.block
        assert len(block.iter_vars) == 2
        first, second = (axis.var for axis in block.iter_vars)
        assert not first.same_as(second)
        assert block.body.value.same_as(second)


def test_miss_block_bind():
    def miss_block_bind_value() -> None:
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi = Ts.axis.S(i)  # error
                T.evaluate(1.0)

    check_error(miss_block_bind_value, 4, TypeError)


def test_invalid_loop_var():
    def invalid_loop_var() -> None:
        for i, j in range(0, 16):  # error
            T.evaluate(1.0)

    check_error(invalid_loop_var, 2, ValueError)


def test_inconsistent_grid():
    def inconsistent_grid(A: T.Buffer(16)) -> None:
        for (i,) in T.grid(16, 16):  # error: one explicit target cannot unpack two variables
            T.evaluate(A[i])

    check_error(inconsistent_grid, 2, ValueError)


def test_invalid_match_buffer_region():
    def invalid_match_buffer_region() -> None:
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                A = Ts.match_buffer(vi)  # error
                T.evaluate(1.0)

    check_error(invalid_match_buffer_region, 5, TypeError)


def test_buffer_rebinding_preserves_distinct_allocations():
    @Ts.prim_func
    def rebound_buffer() -> None:
        A = Ts.sblock_alloc_buffer((128, 128), "float32")
        A = Ts.sblock_alloc_buffer((128, 128), "float32")
        A[0, 0] = A[0, 1] + T.float32(1)

    # Python rebinding selects the second buffer and retains both native allocations.
    block = rebound_buffer.body.block
    assert len(block.alloc_buffers) == 2
    first, second = block.alloc_buffers
    assert not first.same_as(second)
    store = block.body
    assert isinstance(store, tirx.BufferStore)
    assert store.buffer.same_as(second)
    assert store.value.a.source.same_as(second)


def test_duplicate_block_signature():
    def duplicate_reads() -> None:
        A = Ts.sblock_alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[0:8, 0:8])
                Ts.reads(A[0:16, 0:16])  # error
                T.evaluate(1.0)

    def duplicate_writes() -> None:
        A = Ts.sblock_alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.writes(A[0:8, 0:8])
                Ts.writes(A[0:16, 0:16])  # error
                T.evaluate(1.0)

    def duplicate_predicate() -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.where(1)
                Ts.where(0)  # error

    def duplicate_init() -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                with Ts.init():
                    T.evaluate(1.0)
                with Ts.init():  # error
                    T.evaluate(1.0)

    @Ts.prim_func
    def duplicate_axes() -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                vi = Ts.axis.S(i, 16)
                T.evaluate(1.0)

    def duplicate_sblock_attrs_with_same_key_diff_value() -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.sblock_attr({"key1": "block1"})
                Ts.sblock_attr({"key1": "block2"})  # error
                T.evaluate(1.0)

    check_error(duplicate_reads, 7, tvm.error.InternalError)
    check_error(duplicate_writes, 7, tvm.error.InternalError)
    check_error(duplicate_predicate, 6, tvm.error.InternalError)
    check_error(duplicate_init, 7, ValueError)
    parsed = duplicate_axes
    axes = parsed.body.block.body.body.body.block.iter_vars
    assert len(axes) == 3
    assert not axes[0].var.same_as(axes[2].var)
    check_error(duplicate_sblock_attrs_with_same_key_diff_value, 6, tvm.error.InternalError)


def test_opaque_access_during_complete():
    def opaque_access_during_complete(A: T.Buffer((16, 16), "float32")) -> None:  # error
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                T.evaluate(T.call_extern("dummy_extern_function", A.data, dtype="int32"))

    check_error(opaque_access_during_complete, None, ValueError)


def test_convert_slice_to_bufferload():
    def convert_slice_to_bufferload() -> None:
        A = Ts.sblock_alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                A[vi, vj] = A[vi : vi + 2, vj] + 1  # error

    check_error(convert_slice_to_bufferload, 6, TypeError)


def test_tvm_exception_catch_from_special_stmt():
    def special_stmt_except() -> None:
        A = Ts.sblock_alloc_buffer("(128, 128)", "float32")  # error
        T.evaluate(1.0)

    check_error(special_stmt_except, 2, TypeError)


def test_tvm_exception_catch_from_scope_handler():
    def scope_handler_except() -> None:
        for i in T.serial("1", "1"):  # error
            T.evaluate(1)

    check_error(scope_handler_except, 2, TypeError)


def test_tvm_exception_catch_from_bare_intrin():
    def intrin_except_unassign(A: T.Buffer((16, 16), "float32")) -> None:
        T.evaluate(A)  # error

    check_error(intrin_except_unassign, 2, tvm.error.InternalError)


def test_tvm_exception_catch_from_assigned_intrin():
    def intrin_except_assign(A: T.Buffer((16, 16), "float32")) -> None:
        A[0, 0] = A[A]  # error

    check_error(intrin_except_assign, 2, tvm.error.InternalError)


def test_match_buffer_shape_mismatch():
    def buffer_shape_mismatch(A: T.Buffer((8, 8))) -> None:
        for i, j in T.grid(8, 2):
            with Ts.sblock():
                Ts.reads([])
                Ts.writes([A[i, j * 4 : j * 4 + 4]])
                sub_A = Ts.match_buffer(
                    A[i, j * 4 : j * 4 + 4], (5)
                )  # error: shape mismatched between 4 and 5
                for jj in range(0, 4):
                    sub_A[i, j * 4 + jj] = 1

    check_error(buffer_shape_mismatch, 6, tvm.error.InternalError)


def test_high_dim_store():
    def high_dim_store() -> None:
        with Ts.sblock("root"):
            B = T.alloc_buffer((256,), "float32")
            for i, j in T.grid(16, 16):
                B[i, j] = 1.0  # error: Store is only allowed with one index

    check_error(high_dim_store, 5, tvm.error.InternalError)


def test_block_has_option_vars():
    def block_has_option_vars() -> None:
        with Ts.sblock("root") as x:  # error: block does not support option_vars
            T.evaluate(0.0)

    check_error(block_has_option_vars, 2, TypeError)


def test_implicit_root_has_attrs():
    def implicit_root_has_read():
        Ts.reads([])  # error: implicit root does not support reads
        T.evaluate(0.0)

    def implicit_root_has_write():
        Ts.writes([])  # error: implicit root does not support writes
        T.evaluate(0.0)

    def implicit_root_has_attrs():
        Ts.sblock_attr({})  # error: implicit root does not support sblock_attr
        T.evaluate(0.0)

    def implicit_root_has_predicate():
        Ts.where(True)  # error: implicit root does not support predicate
        T.evaluate(0.0)

    def implicit_root_has_axes():
        v = Ts.axis.S(0, 0)  # error: implicit root does not support axis define
        T.evaluate(0.0)

    check_error(implicit_root_has_read, 2, ValueError)
    check_error(implicit_root_has_write, 2, ValueError)
    check_error(implicit_root_has_attrs, 2, tvm.error.InternalError)
    check_error(implicit_root_has_predicate, 2, ValueError)
    check_error(implicit_root_has_axes, 2, tvm.error.InternalError)


@Ts.prim_func
def elementwise_not_affine(
    A: T.Buffer((128, 128, 128, 128)), B: T.Buffer((128, 128, 128, 128))
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 8):  # noqa: E741
        with Ts.sblock("B"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            vl = Ts.axis.S(128, l * 16)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.prim_func
def elementwise_non_single_branch(
    A: T.Buffer((128, 128, 128)), B: T.Buffer((128, 128, 128))
) -> None:
    C = Ts.sblock_alloc_buffer((128, 128, 128))

    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128):
            with Ts.sblock("C"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                C[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for k in T.serial(0, 128):
            with Ts.sblock("B"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                B[vi, vj, vk] = C[vi, vj, vk] * 2.0


def test_reorder_fail_block():
    sch = tvm.s_tir.Schedule(elementwise_not_affine, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k, l = sch.get_loops(block_b)  # noqa: E741
    with pytest.raises(tvm.s_tir.ScheduleError) as execinfo:
        sch.reorder(l, i)
    expected_sub_error_message = (
        "                            # s_tir.SBlock#0\n"
        '                            with Ts.sblock("B"):\n'
        "                            ^^^^^^^^^^^^^^^^^^^^\n"
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
        "        # s_tir.SBlock#0\n"
        '        with Ts.sblock("root"):\n'
        "        ^^^^^^^^^^^^^^^^^^^^^^^\n"
    )
    assert expected_sub_error_message in str(execinfo.value)


def test_load_var():
    def load_var_multiple() -> None:
        d = T.float32()
        d[2] = d[2, 1]  # error cannot provide two indices to load

    check_error(load_var_multiple, 3, TypeError)


def test_store_var():
    def store_var_multiple() -> None:
        d = T.float32()
        d[2, 1] = d[1]  # error cannot provide two indices to store

    check_error(store_var_multiple, 3, TypeError)


def test_load_handle():
    def load_handle(h: T.handle, h_: T.Buffer([1])) -> None:
        h_[0] = h[0]  # error cannot load from handle

    check_error(load_handle, 2, TypeError)


def test_store_handle():
    def store_handle(h: T.handle, h_: T.Buffer([1])) -> None:
        h[0] = h_[0]  # error cannot store to handle

    check_error(store_handle, 2, TypeError)


def test_binop_bad_ast_type():
    def binop_bad_ast_type(h: T.handle, h_: T.Buffer([1])):
        h_[0] = h + [2]  # error rhs should be a primexpr  # noqa: RUF005

    check_error(binop_bad_ast_type, 2, TypeError)


def test_binop_bad_type():
    def binop_bad_type(h: T.handle, h_: T.Buffer([1])):
        h_[0] = h + 2  # error lhs and rhs should be the same type

    check_error(binop_bad_type, 2, TypeError)


def test_non_integer_typed_block_iter():
    def non_integer_typed_block_iter():
        with Ts.sblock():
            i = Ts.axis.S(0.1, 0.1)  # error IterVar requires an integer dtype

    check_error(non_integer_typed_block_iter, 3, tvm.error.InternalError)


def test_illegal_buffer_slice():
    def strided_buffer_region(A: T.Buffer((128, 128), "int32")):
        # do not allow stride in buffer region

        with Ts.sblock("block"):
            Ts.reads([])
            Ts.writes([A[0:128:2, 0:128:3]])  # error
            T.evaluate(T.call_extern("strided_compute", dtype=""))

    def access_reversed_slice(A: T.Buffer((128,), "int32")):
        # do not allow reversed slice step

        A[0:128:-1] = T.broadcast(1, 128)  # error

    def access_non_const_slice_length(A: T.Buffer((128,), "int32")):
        # do not allow non-constant slice length

        for i in range(4):
            T.evaluate(A[0:i:1])  # error

    check_error(strided_buffer_region, 6, ValueError)
    check_error(access_reversed_slice, 4, tvm.error.InternalError)
    check_error(access_non_const_slice_length, 5, TypeError)


def test_syntax_sugar_fail():
    def loop_syntax_sugar_fail(A: T.Buffer((128,))) -> None:
        for i in T.thread_binding(128, 128):
            A[i] = A[i] * 2.0

    check_error(loop_syntax_sugar_fail, 2, ValueError)


def test_multi_line_error_report():
    """An original builder failure retains the full source-call traceback range."""

    # The offending call (`Ts.axis.remap(...)`) is deliberately split across
    # four physical lines so its AST node spans lineno..end_lineno > lineno.
    source_code = "\n".join(
        [
            "@Ts.prim_func",
            "def f() -> None:",
            "    for i, j in T.grid(16, 16):",
            "        vi, vj = Ts.axis.remap(",
            '            "S",',
            "            [i, j],",
            "        )  # error",
            "        T.evaluate(1.0)",
        ]
    )

    with pytest.raises(tvm.error.InternalError) as caught:
        from_source(source_code, extra_vars={"T": T, "Ts": Ts})
    frames = [
        frame
        for frame in traceback.extract_tb(caught.value.__traceback__)
        if frame.filename == "<str>"
    ]
    assert frames[-1].lineno == 4
    if getattr(frames[-1], "colno", None) is not None:
        call = next(
            node
            for node in ast.walk(ast.parse(source_code))
            if isinstance(node, ast.Call) and node.lineno == 4
        )
        assert (
            frames[-1].lineno,
            frames[-1].colno,
            frames[-1].end_lineno,
            frames[-1].end_colno,
        ) == (call.lineno, call.col_offset, call.end_lineno, call.end_col_offset)


def test_tir_func_private_manual_global_symbol_fail():
    with pytest.raises(tvm.error.InternalError):

        @Ts.prim_func(private=True)
        def matmul(
            A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])
        ) -> None:
            T.func_attr({"global_symbol": "matmul"})

            for i, j, k in T.grid(128, 128, 128):
                with Ts.sblock("update"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        # should not execute
        assert matmul.__name__ == "matmul"


def test_buffer_input_requires_shape_arg():
    with pytest.raises(TypeError):

        @Ts.prim_func
        def func(A: T.Buffer(dtype="int32")):
            T.evaluate(0)


if __name__ == "__main__":
    tvm.testing.main()
