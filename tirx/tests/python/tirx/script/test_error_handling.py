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

"""TIRx script error handling."""

from __future__ import annotations

import inspect
import re
import sys
import traceback

import pytest

import tvm_tirx
from tvm_tirx.script import from_source
from tvm_tirx.script import tirx as T


def test_optional_annotation_requires_jit_at_the_source_parameter():
    # Ordinary argument validation must reject Optional at its original parameter location.
    with pytest.raises(TypeError, match="^T.Optional is only supported by @T.jit$") as caught:

        @T.function
        def invalid(value: T.Optional(T.handle)):
            T.evaluate(0)

    assert type(caught.value) is TypeError
    lines, first = inspect.getsourcelines(
        test_optional_annotation_requires_jit_at_the_source_parameter
    )
    index, line = next((i, text) for i, text in enumerate(lines) if "def invalid(" in text)
    frames = traceback.extract_tb(caught.value.__traceback__)
    source = [
        frame for frame in frames if frame.filename == __file__ and frame.lineno == first + index
    ]
    assert source
    if getattr(source[-1], "colno", None) is not None:
        column = line.index("value:")
        assert (source[-1].colno, source[-1].end_lineno, source[-1].end_colno) == (
            column,
            first + index,
            column + len("value: T.Optional(T.handle)"),
        )


def test_scalar_assign_error_not_swallowed():
    """Regression: genuine errors (non-TypeError) from tensor_store during
    scalar-assignment sugar must propagate, not be silently swallowed.

    Before the fix, both eval_expr and tensor_store were wrapped in a single
    broad ``except Exception: pass``, so any error from tensor_store would be
    swallowed and the assignment would silently fall through to eval_assign."""
    from unittest.mock import patch

    original = tvm_tirx.tirx.script.ir_builder.parser_protocol.tensor_store

    def bomb(*args, **kwargs):
        # Intercept only the scalar-assignment path (indices == [0])
        if args[1] == [0]:
            raise ValueError("boom")
        return original(*args, **kwargs)

    # The ValueError propagates unchanged. A broad ``except Exception`` here
    # previously swallowed it and fell through to eval_assign.
    with patch("tvm_tirx.tirx.script.ir_builder.parser_protocol.tensor_store", side_effect=bomb):
        try:

            @T.function
            def func():
                T.device_entry()
                v: T.int32
                v = v + T.int32(1)  # noqa: F821

        except ValueError as error:
            assert str(error) == "boom"
        else:
            pytest.fail("tensor_store error was swallowed")


def check_error(func, rel_lineno, error_type):
    """Check the original exception class and its real source location."""
    source_code = inspect.getsource(func)
    indent = len(re.match(r"^\s*", source_code).group(0))
    source_code = "@T.function\n" + "\n".join(line[indent:] for line in source_code.splitlines())
    with pytest.raises(error_type) as caught:
        from_source(source_code, extra_vars={"T": T})
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


def test_tensor_bind():
    def tensor_bind_missing_args(A: T.Tensor(dtype="float32")) -> None:  # error
        T.evaluate(0)

    check_error(tensor_bind_missing_args, 1, TypeError)


def test_undefined_tensor():
    def undefined_tensor(A: T.Tensor((16, 16), "float32")) -> None:
        for i in T.serial(16):
            for j in T.serial(0, 16):
                C[i, j] = 0.0  # error  # noqa: F821

    check_error(undefined_tensor, 4, NameError)


def test_unsupported_function_call():
    def unsupported_function_call(A: T.Tensor((16, 16), "float32")) -> None:
        for i in T.const_range(16):  # error
            for j in T.serial(0, 16):
                A[i, j] = 0.0

    check_error(unsupported_function_call, 2, AttributeError)


def test_invalid_for_function():
    def invalid_for_function(A: T.Tensor((16, 16), "float32")) -> None:
        for i in T.evaluate(0.0):  # error
            for j in T.serial(0, 16):
                A[i, j] = 0.0

    check_error(invalid_for_function, 2, TypeError)


def test_invalid_block_function():
    def invalid_block_function(A: T.Tensor((16, 16), "float32")) -> None:
        with T.evaluate(0.0):  # error
            T.evaluate(1.0)

    # Ordinary Python reports a missing context-manager protocol differently before 3.11.
    error_type = AttributeError if sys.version_info < (3, 11) else TypeError
    check_error(invalid_block_function, 2, error_type)


def test_return_not_allowed():
    def return_not_allowed(a: T.handle) -> None:
        return T.evaluate(0)  # error

    check_error(return_not_allowed, 2, TypeError)


def test_no_body():
    def no_body(A: T.Tensor((16, 16), "float32")) -> None:
        T.realize(A, "")  # error

    check_error(no_body, 2, AttributeError)


def test_invalid_loop_var():
    def invalid_loop_var() -> None:
        for i, j in range(0, 16):  # error
            T.evaluate(1.0)

    check_error(invalid_loop_var, 2, ValueError)


def test_inconsistent_grid():
    def inconsistent_grid(A: T.Tensor(16, "float32")) -> None:
        for (i,) in T.grid(16, 16):  # error: one explicit target cannot unpack two variables
            T.evaluate(A[i])

    check_error(inconsistent_grid, 2, ValueError)


def test_tvm_exception_catch_from_scope_handler():
    def scope_handler_except() -> None:
        for i in T.serial("1", "1"):  # error
            T.evaluate(1)

    check_error(scope_handler_except, 2, TypeError)


def test_tvm_exception_catch_from_bare_intrin():
    def intrin_except_unassign(A: T.Tensor((16, 16), "float32")) -> None:
        T.evaluate(A)  # error

    check_error(intrin_except_unassign, 2, RuntimeError)


def test_tvm_exception_catch_from_assigned_intrin():
    def intrin_except_assign(A: T.Tensor((16, 16), "float32")) -> None:
        A[0, 0] = A[A]  # error

    check_error(intrin_except_assign, 2, TypeError)


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
    def load_handle(h: T.handle, h_: T.Tensor([1], "float32")) -> None:
        h_[0] = h[0]  # error cannot load from handle

    check_error(load_handle, 2, TypeError)


def test_store_handle():
    def store_handle(h: T.handle, h_: T.Tensor([1], "float32")) -> None:
        h[0] = h_[0]  # error cannot store to handle

    check_error(store_handle, 2, TypeError)


def test_binop_bad_ast_type():
    def binop_bad_ast_type(h: T.handle, h_: T.Tensor([1], "float32")):
        h_[0] = h + [2]  # error rhs should be a primexpr  # noqa: RUF005

    check_error(binop_bad_ast_type, 2, TypeError)


def test_binop_bad_type():
    def binop_bad_type(h: T.handle, h_: T.Tensor([1], "float32")):
        h_[0] = h + 2  # error lhs and rhs should be the same type

    check_error(binop_bad_type, 2, TypeError)


def test_tensor_input_requires_shape_arg():
    with pytest.raises(TypeError):

        @T.function
        def func(A: T.Tensor(dtype="int32")):
            T.evaluate(0)
