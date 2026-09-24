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
from typing import Any, TypeVar

# Per-execution inputs, never a second owner of native function or parameter state.
_Value = TypeVar("_Value")

# Host/JIT values and callable annotation adapters have open-ended Python types.
_SPECIALIZATION: ContextVar[tuple[str | None, dict[str, Any]] | None] = ContextVar(
    "tvm_parser_specialization", default=None
)


@contextmanager
def use_specialization(name: str | None, bindings: Mapping[str, Any] | None) -> Iterator[None]:
    """Pass selected JIT bindings to one root builder execution.

    ``None`` denotes ordinary parsing; an empty mapping is a specialization
    with no compile-time values. Copy inputs so nested execution cannot alter
    the caller's selection.
    """
    token = _SPECIALIZATION.set(None if bindings is None else (name, dict(bindings)))
    try:
        yield
    finally:
        _SPECIALIZATION.reset(token)


def read_specialization_bindings(name: str) -> dict[str, Any] | None:
    """Read the current root's bindings, or None outside specialization."""
    context = _SPECIALIZATION.get()
    return context[1] if context is not None and context[0] == name else None


def unwrap_annotation(annotation: Any, specialization: Mapping[str, Any] | None = None) -> Any:
    """Read an optional runtime annotation only within a JIT specialization.

    Generated specialization calls this after checking selected values and
    absences, so omitted parameters never evaluate their annotation. Ordinary
    argument construction delegates annotation validation to the language variant.
    """
    unwrap = getattr(annotation, "__tvm_optional_annotation__", None)
    if unwrap is not None:
        if specialization is None:
            raise TypeError("T.Optional is only supported by @T.jit")
        return unwrap()
    return annotation


def require_constexpr_binding(value: _Value, name: str) -> _Value:
    """Require an explicit captured value for a constexpr parameter."""
    from tvm.script.ir_builder.base import MISSING

    if value is MISSING:
        raise TypeError(f"constexpr parameter {name!r} requires a specialization binding")
    return value
