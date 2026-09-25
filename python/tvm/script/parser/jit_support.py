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

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

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
        omitted optional parameters. None selects ordinary parsing; an empty
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


def unwrap_annotation(annotation: Any, const_args: Mapping[str, Any] | None = None) -> Any:
    """Read an optional runtime annotation only within a JIT specialization.

    Parameters
    ----------
    annotation : Any
        Evaluated runtime annotation. An object implementing the callable
        ``__tvm_optional_annotation__`` adapter supplies its contained annotation;
        all other objects pass through unchanged.
    const_args : Mapping[str, Any] or None, optional
        Parameter names mapped to fixed values for the active root. None, the
        default, means ordinary parsing and forbids optional annotation adapters. Any
        mapping, including an empty mapping, enables the adapter. Its contents
        are not inspected here.

    Returns
    -------
    Any
        The adapter's result, or the identical input object when no adapter exists.

    Raises
    ------
    TypeError
        If an optional annotation adapter is used without specialization.

    Notes
    -----
    Generated specialization calls this after checking selected values and
    absences, so omitted parameters never evaluate their annotation. Ordinary
    argument construction delegates annotation validation to the language variant.
    Adapter lookup and execution exceptions propagate unchanged.
    """
    unwrap = getattr(annotation, "__tvm_optional_annotation__", None)
    if unwrap is not None:
        if const_args is None:
            raise TypeError("T.Optional is only supported by @T.jit")
        return unwrap()
    return annotation
