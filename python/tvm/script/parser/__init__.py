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
"""Canonical TVMScript parser with language variant namespace initialization."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

_NAMESPACES: dict[str, object] = {}
_NAMESPACE_INITIALIZERS: list[Callable[[], None]] = []
_NAMESPACE_ALIASES: set[str] = {"I", "ir"}
_ENTRY_EXPORTS = (
    "from_source",
    "ir_module",
    "make_decorator",
    "make_macro_decorator",
    "parse",
    "pyfunc",
)
__all__ = [*_ENTRY_EXPORTS, "register_namespace", "register_namespace_initializer"]
_initialized = False
_initializing = False


def register_namespace(alias: str, namespace: object) -> None:
    """Register an opaque fixed namespace; its first alias is the syntax key root.

    Parameters
    ----------
    alias : str
        Source name for the namespace, such as ``"I"`` or ``"M"``. The first
        registered alias for an object is its canonical syntax-policy prefix;
        additional aliases for that object resolve to the same prefix.
    namespace : object
        Namespace object exposing construction operations or source decorators.
        It is borrowed without copying or inspecting its members.

    Returns
    -------
    None
        Registration updates the namespaces available to subsequent parses.

    Notes
    -----
    Each parse borrows these namespaces. Replacing an alias affects future parses
    only; registration enters no builder frame and inspects no namespace members.
    """
    _NAMESPACES[alias] = namespace


def register_namespace_initializer(
    initializer: Callable[[], None], *, aliases: tuple[str, ...] = ()
) -> None:
    """Register a lazy bootstrap callback without importing the language variant.

    Parameters
    ----------
    initializer : Callable[[], None]
        Callback that registers the language variant's namespaces. It receives
        no arguments and is called during the parser's first initialization.
        If initialization has already completed, a newly registered callback
        runs immediately. Re-registering the same callback object does not
        enqueue or invoke it again.
    aliases : tuple[str, ...], optional
        Names whose attribute lookup should trigger initialization. Defaults
        to an empty tuple. These names advertise lazy namespaces; the callback
        must still register their actual objects with :func:`register_namespace`.
        Aliases are added even when the callback was previously registered.

    Returns
    -------
    None
        The callback is registered, and invoked immediately when the parser
        is already initialized.
    """
    _NAMESPACE_ALIASES.update(aliases)
    if any(existing is initializer for existing in _NAMESPACE_INITIALIZERS):
        return
    _NAMESPACE_INITIALIZERS.append(initializer)
    if _initialized:
        initializer()


def _initialize() -> None:
    global _initialized, _initializing
    if _initialized or _initializing:
        return
    _initializing = True
    try:
        importlib.import_module(__name__ + ".ir")
        register_namespace("TypeVar", TypeVar)
        for initialize in _NAMESPACE_INITIALIZERS:
            initialize()
        _initialized = True
    finally:
        _initializing = False


def __getattr__(name: str) -> Any:
    if name not in _ENTRY_EXPORTS and name not in _NAMESPACES and name not in _NAMESPACE_ALIASES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _initialize()
    if name in _ENTRY_EXPORTS:
        return getattr(importlib.import_module(__name__ + ".entry"), name)
    if name in _NAMESPACES:
        return _NAMESPACES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
