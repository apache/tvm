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
"""Static syntax policies keyed by registered builder namespace paths.

Each language variant registers its namespace through ``register_namespace`` and supplies
explicit canonical keys such as ``T.int32`` or ``R.Tensor`` at its marker sites.
The first registered alias names that namespace. Source aliases normalize to this
same root; only direct root members select special syntax. Parser consumers read
the internal metadata tables directly. Ordinary captured callables, local aliases,
instance methods and descriptors do not acquire syntax policies through identity
or receiver inference.

Tables retain static syntax facts only, never source functions, captures, frames
or constructed results. Registration decorators return the same callable.
Shared builder operations are documented in
``tvm.script.ir_builder.parser_protocol``; concrete language variants own registration
and namespace initialization.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum, auto
from typing import Any, TypeVar

__all__ = ["register_mutable_decl", "register_scalar_annotation"]

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


class DefinitionKind(IntEnum):
    """Construction behavior of a definition's canonical source decorator."""

    FUNCTION = auto()
    MACRO = auto()
    PYTHON = auto()


SCALAR_ANNOTATION_DTYPE: dict[str, object] = {}
MUTABLE_CELL_DECL: dict[str, frozenset[str]] = {}
DEFINITION_KIND: dict[str, DefinitionKind] = {}


def register_scalar_annotation(
    namespace_path: str,
    constructor: _Callable,
    *,
    dtype: object = None,
) -> _Callable:
    """Register the dtype of a fixed scalar annotation without evaluating it.

    Parameters
    ----------
    namespace_path : str
        Canonical registered namespace alias and exported callable path, such
        as ``"T.int32"``. A later registration at this path replaces its dtype.
    constructor : Callable
        Callable providing the eager construction operation. Registration
        records its syntax path without invoking or wrapping this callable.
    dtype : object, optional
        Static scalar dtype for an explicit PEP 695 symbol bound. None (the
        default) leaves the annotation without a supported scalar bound dtype.

    Returns
    -------
    Callable
        The exact ``constructor`` object.

    Notes
    -----
    This metadata does not give constructor calls special assignment semantics.
    Scalar runtime parameters retain their ordinary annotation construction.

    .. code:: python

        register_scalar_annotation("T.int32", T.int32, dtype="int32")
        # Source: def f[n: T.int32](...):
        # Builder: n = X.resolve_type_var_("n", dtype="int32")
    """
    SCALAR_ANNOTATION_DTYPE[namespace_path] = dtype
    return constructor


def register_mutable_decl(
    namespace_path: str, *, syntax: str = "call"
) -> Callable[[_Callable], _Callable]:
    """Register persistent mutable storage syntax at a fixed namespace path.

    Parameters
    ----------
    namespace_path : str
        Canonical registered namespace alias and exported constructor path.
    syntax : str, optional
        Declaration form: ``"call"`` (the default) for a constructor call,
        ``"annotation"`` for an annotated local declaration, or ``"parameter"``
        for a function parameter annotation. Repeated registration adds forms
        for the same path without removing existing ones.

    Returns
    -------
    Callable
        Registration decorator that returns its constructor unchanged.

    Raises
    ------
    ValueError
        If ``syntax`` is not one of the three supported forms.

    Notes
    -----
    Explicit declarations take precedence over mutable target updates;
    storage creation and updates stay outside ordinary ``bind_``.

    .. code:: python

        register_mutable_decl("T.int32", syntax="annotation")(T.int32)
        # Source: x: T.int32 = 0
        # Builder: x = X.decl_mutable_cell_(0, ty=X.int32, name="x")
    """
    if syntax not in ("call", "annotation", "parameter"):
        raise ValueError("Mutable declaration syntax must be call, annotation or parameter")

    def decorate(constructor: _Callable) -> _Callable:
        MUTABLE_CELL_DECL[namespace_path] = MUTABLE_CELL_DECL.get(
            namespace_path, frozenset()
        ) | frozenset((syntax,))
        return constructor

    return decorate
