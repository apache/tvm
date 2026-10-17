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
"""Scoped transport of already selected JIT inputs into builder execution.

JIT owns validation, defaults and caching. These contexts carry only the root
function's selected values (including explicit None absences), restoring the prior
values after nested parsing or an exception. Native builder frames own no JIT
state, and ordinary parsing does not import the TIRx JIT entry point.
"""

from __future__ import annotations

import ast
import typing
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


def optional_annotation_body(node: ast.expr, environment: Mapping[str, Any]) -> ast.expr | None:
    """Recognize standard optional syntax without evaluating its contained annotation.

    Only the actual typing.Optional object or typing module is accepted. Attribute
    lookup on arbitrary receivers stays opaque, just as for language namespaces.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        if isinstance(node.right, ast.Constant) and node.right.value is None:
            inner = node.left
        elif isinstance(node.left, ast.Constant) and node.left.value is None:
            inner = node.right
        else:
            return None
        if not (isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr)):
            return inner
    if isinstance(node, ast.Subscript):
        marker = node.value
        if isinstance(marker, ast.Name):
            optional = environment.get(marker.id) is typing.Optional
        else:
            optional = (
                isinstance(marker, ast.Attribute)
                and isinstance(marker.value, ast.Name)
                and environment.get(marker.value.id) is typing
                and marker.attr == "Optional"
            )
        if optional:
            return node.slice
    return None


# Per-execution inputs, never a second owner of native function or parameter state.
# Host/JIT values and callable annotation adapters have open-ended Python types.
_SPECIALIZATION: ContextVar[tuple[str | None, dict[str, Any]] | None] = ContextVar(
    "tvm_parser_specialization", default=None
)


@contextmanager
def use_specialization(name: str | None, const_args: Mapping[str, Any] | None) -> Iterator[None]:
    """Pass fixed JIT arguments to one root builder execution.

    Parameters
    ----------
    name : str or None
        Source name of the standalone root function. Readers must request this
        exact name. None is used when the parsed root is not a function.
    const_args : Mapping[str, Any] or None
        Parameter names mapped to fixed values, including explicit None values for
        None-typed optional parameters. None selects ordinary parsing; an empty
        mapping still selects specialization with no compile-time values.
        The mapping is copied on entry, without copying its values.

    Yields
    ------
    None
        The enclosed builder execution sees this selection. The previous
        selection is restored on exit, including nested parsing and exceptions.
    """
    token = _SPECIALIZATION.set(None if const_args is None else (name, dict(const_args)))
    try:
        yield
    finally:
        _SPECIALIZATION.reset(token)


def read_specialization_bindings(name: str) -> dict[str, Any] | None:
    """Read the current root's fixed JIT arguments.

    Parameters
    ----------
    name : str
        Source function name to match against the active root selection.

    Returns
    -------
    dict[str, Any] or None
        Parameter names mapped to fixed values when the root name matches. An empty
        dictionary means an active specialization with no selected values. None
        means ordinary parsing or a different root name. The returned dictionary
        is borrowed from the current context and should not be mutated.
    """
    context = _SPECIALIZATION.get()
    return context[1] if context is not None and context[0] == name else None
