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
import sys
import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.ir.diagnostics import override_renderer
import inspect


def buffer_bind_missing_args(a: T.handle) -> None:
    A = T.match_buffer((16, 16), "float32")  # error


def test_buffer_bind():
    check_error(buffer_bind_missing_args, 2)


def range_missing_args(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")

    T.attr(A, "realize_scope", "")
    T.realize(A[0:16, 0:16], "")
    for i in T.serial(16):  # error
        for j in T.serial(0, 16):
            A[i, j] = 0.0


def test_range_missing_args():
    check_error(range_missing_args, 6)


def undefined_buffer(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")

    T.attr(A, "realize_scope", "")
    T.realize(C[0:16, 0:16], "")  # error
    for i in T.serial(16):
        for j in T.serial(0, 16):
            A[i, j] = 0.0


def test_undefined_buffer():
    check_error(undefined_buffer, 5)


def unsupported_stmt(a: T.int32) -> None:
    if a > 0:
        print("I love tvm")  # error


def test_unsupported_stmt():
    check_error(unsupported_stmt, 3)


def unsupported_function_call(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")

    T.attr(A, "realize_scope", "")
    T.realize(A[0:16, 0:16], "")
    for i in T.const_range(16):  # error
        for j in T.serial(0, 16):
            A[i, j] = 0.0


def test_unsupported_function_call():
    check_error(unsupported_function_call, 6)


def missing_type_annotation(a) -> None:  # error
    T.evaluate(0.0)


def test_missing_type_annotation():
    check_error(missing_type_annotation, 1)


def invalid_expr_stmt() -> None:
    T.max(1, 2)  # error


def test_invalid_expr_stmt():
    check_error(invalid_expr_stmt, 2)


def invalid_for_function(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")

    for i in T.evaluate(0.0):  # error
        for j in T.serial(0, 16):
            A[i, j] = 0.0


def test_invalid_for_function():
    check_error(invalid_for_function, 4)


def invalid_block_function(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")

    with T.evaluate(0.0):  # error
        T.evaluate(1.0)


def test_invalid_block_function():
    check_error(invalid_block_function, 4)


def return_not_allowed(a: T.handle) -> None:
    return T.evaluate(0)  # error


def test_return_not_allowed():
    check_error(return_not_allowed, 2)


def tir_assert(a: T.handle) -> None:
    T.Assert(0, "")  # error


def test_tir_assert():
    check_error(tir_assert, 2)


def no_body(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    T.realize(A, "")  # error


def test_no_body():
    check_error(no_body, 3)


def allocate_with_buffers() -> None:
    with T.allocate([1], "float32", "") as [A, B]:  # error
        T.evaluate(1.0)


def test_allocate_with_buffers():
    check_error(allocate_with_buffers, 2)


def inconsistent_binding_value() -> None:
    for i, j in T.grid(16, 16):
        vi, vj = T.axis.remap("SS", [i])  # error
        T.evaluate(1.0)


def inconsistent_binding_type() -> None:
    for i, j in T.grid(16, 16):
        vi, vj = T.axis.remap("S", [i, j])  # error
        T.evaluate(1.0)


def test_inconsistent_binding():
    check_error(inconsistent_binding_value, 3)
    check_error(inconsistent_binding_type, 3)


def error_remap_type() -> None:
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("TT", [i, j])  # error
            T.evaluate(1.0)


def error_remap_value() -> None:
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i + j, j])  # error
            T.evaluate(1.0)


def test_error_remap_args():
    check_error(error_remap_type, 4)
    check_error(error_remap_value, 4)


def invalid_block_axes(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    for i, j in T.grid(16, 16):
        with T.block():
            vi = T.axis.S(i, A)  # error
            T.evaluate(1.0)


def test_invalid_block_axes():
    check_error(invalid_block_axes, 5)


def duplicate_block_axes() -> None:
    for i, j in T.grid(16, 16):
        with T.block():
            vi = T.axis.S(16, i)
            vi = T.axis.S(16, j)  # error
            T.evaluate(1.0)


def duplicate_block_axes_remap() -> None:
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vi = T.axis.remap("SS", [i, j])  # error
            T.evaluate(1.0)


def test_duplicate_block_axes():
    check_error(duplicate_block_axes, 5)
    check_error(duplicate_block_axes_remap, 4)


def miss_block_bind_value() -> None:
    for i, j in T.grid(128, 128):
        with T.block():
            vi = T.axis.S(i)  # error
            T.evaluate(1.0)


def test_miss_block_bind():
    check_error(miss_block_bind_value, 4)


def invalid_loop_var() -> None:
    for i, j in range(0, 16):  # error
        T.evaluate(1.0)


def test_invalid_loop_var():
    check_error(invalid_loop_var, 2)


def inconsistent_grid() -> None:
    for i in T.grid(16, 16):  # error
        T.evaluate(1.0)


def test_inconsistent_grid():
    check_error(inconsistent_grid, 2)


def invalid_match_buffer_region() -> None:
    for i, j in T.grid(128, 128):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            A = T.match_buffer(vi)  # error
            T.evaluate(1.0)


def test_invalid_match_buffer_region():
    check_error(invalid_match_buffer_region, 5)


def duplicate_buffer() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            A = T.alloc_buffer((128, 128), "float32")  # error
            T.evaluate(1.0)


def test_duplicate_buffer():
    check_error(duplicate_buffer, 6)


def duplicate_reads() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[0:8, 0:8])
            T.reads(A[0:16, 0:16])  # error
            T.evaluate(1.0)


def duplicate_writes() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            T.writes(A[0:8, 0:8])
            T.writes(A[0:16, 0:16])  # error
            T.evaluate(1.0)


def duplicate_predicate() -> None:
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            T.where(1)
            T.where(0)  # error


def duplicate_annotations() -> None:
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            T.block_attr({})
            T.block_attr({})  # error


def duplicate_init() -> None:
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            with T.init():
                T.evaluate(1.0)
            with T.init():  # error
                T.evaluate(1.0)


def duplicate_axes() -> None:
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            vi = T.axis.S(i, 16)  # error
            T.evaluate(1.0)


def test_duplicate_block_signature():
    check_error(duplicate_reads, 7)
    check_error(duplicate_writes, 7)
    check_error(duplicate_predicate, 6)
    check_error(duplicate_annotations, 6)
    check_error(duplicate_init, 7)
    check_error(duplicate_axes, 5)


def opaque_access_during_complete(a: T.handle) -> None:  # error
    A = T.match_buffer(a, (16, 16), "float32")
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            T.evaluate(T.load("float32", A.data, vi * 16 + vj))


def test_opaque_access_during_complete():
    check_error(opaque_access_during_complete, 1)


def convert_slice_to_bufferload() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            A[vi, vj] = A[vi : vi + 2, vj] + 1  # error


def test_convert_slice_to_bufferload():
    check_error(convert_slice_to_bufferload, 6)


def error_index_type() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            A[vi, vj] = A[vi, 0.0] + 1  # error


def error_bufferslice_index_type() -> None:
    A = T.alloc_buffer((1,), "float32")
    B = T.alloc_buffer((16, 16), "float32")
    C = T.alloc_buffer((16, 16), "float32")
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, A[0]]  # error


def test_error_index_type():
    check_error(error_index_type, 6)
    check_error(error_bufferslice_index_type, 8)


def error_index_with_stop() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            A[vi, vj] = A[vi, 1:10] + 1  # error


def error_bufferslice_index_with_stop() -> None:
    A = T.alloc_buffer((1,), "int32")
    B = T.alloc_buffer((16, 16), "float32")
    C = T.alloc_buffer((16, 16), "float32")
    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, A[0:1]]  # error


def test_error_index_with_stop_slice():
    check_error(error_index_with_stop, 6)
    check_error(error_bufferslice_index_with_stop, 8)


def mismatch_args() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    with T.block():
        T.reads(A[0, 0], A[1, 1])  # error
        T.evaluate(1.0)


def test_mismatch_args():
    check_error(mismatch_args, 4)


def special_stmt_except() -> None:
    A = T.alloc_buffer("(128, 128)", "float32")  # error
    T.evaluate(1.0)


def scope_handler_except() -> None:
    for i in T.serial("1", "1"):  # error
        T.evaluate(1)


def intrin_except_unassign(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    T.evaluate(A)  # error


def intrin_except_assign(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32")
    A[0, 0] = T.load(A, A, A)  # error


def test_tvm_exception_catch():
    # test catching c++ side exception
    check_error(special_stmt_except, 2)
    check_error(scope_handler_except, 2)
    check_error(intrin_except_unassign, 3)
    check_error(intrin_except_assign, 3)


def buffer_shape_mismatch(a: T.handle) -> None:
    A = T.match_buffer(a, (8, 8))
    for i, j in T.grid(8, 2):
        with T.block():
            T.reads([])
            T.writes([A[i, j * 4 : j * 4 + 4]])
            sub_A = T.match_buffer(
                A[i, j * 4 : j * 4 + 4], (5)
            )  # error: shape mismatched between 4 and 5
            for jj in range(0, 4):
                sub_A[i, j * 4 + jj] = 1


def test_match_buffer_shape_mismatch():
    check_error(buffer_shape_mismatch, 7)


def high_dim_store() -> None:
    with T.block("root"):
        B = T.allocate([256], "float32", "global")
        for i, j in T.grid(16, 16):
            B[i, j] = 1.0  # error: Store is only allowed with one index


def test_high_dim_store():
    check_error(high_dim_store, 5)


def block_has_option_vars() -> None:
    with T.block("root") as x:  # error: block does not support option_vars
        T.evaluate(0.0)


def test_block_has_option_vars():
    check_error(block_has_option_vars, 2)


def check_error(func, rel_lineno):
    # Override the default renderer to accumulate errors
    errors = []

    def render(e):
        for d in e.diagnostics:
            errors.append(d)

    override_renderer(render)
    # The diagnostic context throws an exception when it gets an error
    try:
        source_code = inspect.getsource(func)
        source_code = "@T.prim_func\n" + source_code
        tvm.script.from_source(source_code)
    except tvm.error.DiagnosticError as e:
        pass
    assert len(errors) == 1, errors
    for d in errors:
        assert (
            d.span.line - 1 == rel_lineno
        ), f"Expected error to be on line {rel_lineno}, but it was on {d.span.line - 1}"


# TODO(Siyuan): block iter errors.


@T.prim_func
def elementwise_not_affine(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in T.grid(128, 128, 128, 8):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            vl = T.axis.S(128, l * 16)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@T.prim_func
def elementwise_non_single_branch(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    C = T.alloc_buffer((128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                C[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for k in T.serial(0, 128):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                B[vi, vj, vk] = C[vi, vj, vk] * 2.0


def test_reorder_fail_block():
    sch = tir.Schedule(elementwise_not_affine, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError) as execinfo:
        sch.reorder(l, i)
    expected_sub_error_message = (
        "            # tir.Block#0\n"
        '            with T.block("B"):\n'
        "            ^^^^^^^^^^^^^^^^^^\n"
    )
    print("expected: ", expected_sub_error_message, "actual: ", str(execinfo.value))
    assert expected_sub_error_message in str(execinfo.value)


def test_reorder_fail_nested_loop_inner():
    sch = tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError) as execinfo:
        sch.reorder(k, i)
    expected_sub_error_message = (
        "        for i in T.serial(0, 128):\n"
        "            # tir.For#0\n"
        "            for j in T.serial(0, 128):\n"
        "            ^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
    )
    print("expected: ", expected_sub_error_message, "actual: ", str(execinfo.value))
    assert expected_sub_error_message in str(execinfo.value)


def test_fuse_fail_nested_loop_outer():
    sch = tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError) as execinfo:
        sch.fuse(k, i)
    expected_sub_error_message = (
        "        # tir.For#1\n"
        "        for i in T.serial(0, 128):\n"
        "        ^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
        "            for j in T.serial(0, 128):\n"
    )
    print("expected: ", expected_sub_error_message, "actual: ", str(execinfo.value))
    assert expected_sub_error_message in str(execinfo.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
