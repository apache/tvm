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
