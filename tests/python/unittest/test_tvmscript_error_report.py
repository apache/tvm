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
import tvm
from tvm import tir
from tvm.script import ty, from_source
from tvm.ir.diagnostics import override_renderer
import inspect


def buffer_bind_missing_args(a: ty.handle) -> None:
    A = tir.match_buffer((16, 16), "float32")  # error


def test_buffer_bind():
    check_error(buffer_bind_missing_args, 2)


def range_missing_args(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")

    tir.attr(A, "realize_scope", "")
    tir.realize(A[0:16, 0:16], "")
    for i in tir.serial(16):  # error
        for j in tir.serial(0, 16):
            A[i, j] = 0.0


def test_range_missing_args():
    check_error(range_missing_args, 6)


def undefined_buffer(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")

    tir.attr(A, "realize_scope", "")
    tir.realize(C[0:16, 0:16], "")  # error
    for i in tir.serial(16):
        for j in tir.serial(0, 16):
            A[i, j] = 0.0


def test_undefined_buffer():
    check_error(undefined_buffer, 5)


def unsupported_stmt(a: ty.int32) -> None:
    if a > 0:
        print("I love tvm")  # error


def test_unsupported_stmt():
    check_error(unsupported_stmt, 3)


def unsupported_function_call(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")

    tir.attr(A, "realize_scope", "")
    tir.realize(A[0:16, 0:16], "")
    for i in tir.const_range(16):  # error
        for j in tir.serial(0, 16):
            A[i, j] = 0.0


def test_unsupported_function_call():
    check_error(unsupported_function_call, 6)


def missing_type_annotation(a) -> None:  # error
    tir.evaluate(0.0)


def test_missing_type_annotation():
    check_error(missing_type_annotation, 1)


def invalid_expr_stmt() -> None:
    tir.max(1, 2)  # error


def test_invalid_expr_stmt():
    check_error(invalid_expr_stmt, 2)


def invalid_for_function(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")

    for i in tir.evaluate(0.0):  # error
        for j in tir.serial(0, 16):
            A[i, j] = 0.0


def test_invalid_for_function():
    check_error(invalid_for_function, 4)


def invalid_block_function(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")

    with tir.evaluate(0.0):  # error
        tir.evaluate(1.0)


def test_invalid_block_function():
    check_error(invalid_block_function, 4)


def return_not_allowed(a: ty.handle) -> None:
    return tir.evaluate(0)  # error


def test_return_not_allowed():
    check_error(return_not_allowed, 2)


def tir_assert(a: ty.handle) -> None:
    tir.Assert(0, "")  # error


def test_tir_assert():
    check_error(tir_assert, 2)


def no_body(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    tir.realize(A, "")  # error


def test_no_body():
    check_error(no_body, 3)


def allocate_with_buffers() -> None:
    with tir.allocate([1], "float32", "") as [A, B]:  # error
        tir.evaluate(1.0)


def test_allocate_with_buffers():
    check_error(allocate_with_buffers, 2)


def inconsistent_binding() -> None:
    with tir.block([128, 128]) as [vi]:  # error
        tir.evaluate(1.0)


def test_inconsistent_binding():
    check_error(inconsistent_binding, 2)


def invalid_block_axes(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    with tir.block([A]) as [vi]:  # error
        tir.evaluate(1.0)


def test_invalid_block_axes():
    check_error(invalid_block_axes, 3)


def miss_block_bind() -> None:
    with tir.block([16, 16]) as [vi, vj]:  # error
        tir.bind(vi, 1)
        tir.evaluate(1.0)


def test_miss_block_bind():
    check_error(miss_block_bind, 2)


def invalid_loop_var() -> None:
    for i, j in range(0, 16):  # error
        tir.evaluate(1.0)


def test_invalid_loop_var():
    check_error(invalid_loop_var, 2)


def inconsistent_grid() -> None:
    for i in tir.grid(16, 16):  # error
        tir.evaluate(1.0)


def test_inconsistent_grid():
    check_error(inconsistent_grid, 2)


def invalid_match_buffer_region() -> None:
    with tir.block([16, 16]) as [vi, vj]:
        A = tir.match_buffer_region(vi)  # error
        tir.evaluate(1.0)


def test_invalid_match_buffer_region():
    check_error(invalid_match_buffer_region, 3)


def duplicate_buffer() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        A = tir.alloc_buffer((128, 128), "float32")  # error
        tir.evaluate(1.0)


def test_duplicate_buffer():
    check_error(duplicate_buffer, 4)


def duplicate_reads() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        tir.reads(A[0:8, 0:8])
        tir.reads(A[0:16, 0:16])  # error
        tir.evaluate(1.0)


def duplicate_writes() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        tir.writes(A[0:8, 0:8])
        tir.writes(A[0:16, 0:16])  # error
        tir.evaluate(1.0)


def duplicate_predicate() -> None:
    with tir.block([16, 16]) as [vi, vj]:
        tir.where(1)
        tir.where(0)  # error


def duplicate_annotations() -> None:
    with tir.block([16, 16]) as [vi, vj]:
        tir.block_attr({})
        tir.block_attr({})  # error


def duplicate_init() -> None:
    with tir.block([16, 16]) as [vi, vj]:
        with tir.init():
            tir.evaluate(1.0)
        with tir.init():  # error
            tir.evaluate(1.0)


def test_duplicate_block_signature():
    check_error(duplicate_reads, 5)
    check_error(duplicate_writes, 5)
    check_error(duplicate_predicate, 4)
    check_error(duplicate_annotations, 4)
    check_error(duplicate_init, 5)


def opaque_access_during_complete(a: ty.handle) -> None:  # error
    A = tir.match_buffer(a, (16, 16), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        tir.evaluate(tir.load("float32", A.data, vi * 16 + vj))


def test_opaque_access_during_complete():
    check_error(opaque_access_during_complete, 1)


def convert_slice_to_bufferload() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        A[vi, vj] = A[vi : vi + 2, vj] + 1  # error


def test_convert_slice_to_bufferload():
    check_error(convert_slice_to_bufferload, 4)


def error_index_type() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        A[vi, vj] = A[vi, 0.0] + 1  # error


def error_bufferslice_index_type() -> None:
    A = tir.alloc_buffer((1,), "float32")
    B = tir.alloc_buffer((16, 16), "float32")
    C = tir.alloc_buffer((16, 16), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        C[vi, vj] = B[vi, A[0]]  # error


def test_error_index_type():
    check_error(error_index_type, 4)
    check_error(error_bufferslice_index_type, 6)


def error_index_with_stop() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        A[vi, vj] = A[vi, 1:10] + 1  # error


def error_bufferslice_index_with_stop() -> None:
    A = tir.alloc_buffer((1,), "int32")
    B = tir.alloc_buffer((16, 16), "float32")
    C = tir.alloc_buffer((16, 16), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        C[vi, vj] = B[vi, A[0:1]]  # error


def test_error_index_with_stop_slice():
    check_error(error_index_with_stop, 4)
    check_error(error_bufferslice_index_with_stop, 6)


def mismatch_args() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([16, 16]) as [vi, vj]:
        tir.reads(A[0, 0], A[1, 1])  # error
        tir.evaluate(1.0)


def test_mismatch_args():
    check_error(mismatch_args, 4)


def special_stmt_except() -> None:
    A = tir.alloc_buffer("(128, 128)", "float32")  # error
    with tir.block([16, 16]) as [vi, vj]:
        tir.evaluate(1.0)


def scope_handler_except() -> None:
    for i in tir.serial("1", "1"):  # error
        tir.evaluate(1)


def intrin_except_unassign(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    tir.evaluate(A)  # error


def intrin_except_assign(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    A[0, 0] = tir.load(A, A, A)  # error


def test_tvm_exception_catch():
    # test catching c++ side exception
    check_error(special_stmt_except, 2)
    check_error(scope_handler_except, 2)
    check_error(intrin_except_unassign, 3)
    check_error(intrin_except_assign, 3)


def check_error(module, rel_lineno):
    # Override the default renderer to accumulate errors
    _, start_line = inspect.getsourcelines(module)
    lineno = start_line + rel_lineno - 1
    errors = []

    def render(e):
        for d in e.diagnostics:
            errors.append(d)

    override_renderer(render)
    # The diagnostic context throws an exception when it gets an error
    try:
        mod = from_source(module)
    except tvm.error.DiagnosticError as e:
        pass
    assert len(errors) == 1, errors
    for d in errors:
        assert (
            d.span.line == lineno
        ), f"Expected error to be on line {lineno}, but it was on {d.span.line}"


if __name__ == "__main__":
    test_buffer_bind()
    test_range_missing_args()
    test_undefined_buffer()
    test_unsupported_stmt()
    test_unsupported_function_call()
    test_missing_type_annotation()
    test_invalid_expr_stmt()
    test_invalid_for_function()
    test_invalid_block_function()
    test_return_not_allowed()
    test_tir_assert()
    test_no_body()
    test_allocate_with_buffers()
    test_inconsistent_binding()
    test_invalid_block_axes()
    test_miss_block_bind()
    test_invalid_loop_var()
    test_inconsistent_grid()
    test_invalid_match_buffer_region()
    test_duplicate_buffer()
    test_duplicate_block_signature()
    test_opaque_access_during_complete()
    test_convert_slice_to_bufferload()
    test_error_index_type()
    test_error_index_with_stop_slice()
    test_mismatch_args()
    test_tvm_exception_catch()
