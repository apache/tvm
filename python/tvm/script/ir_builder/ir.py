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
"""Package tvm.script.ir_builder.ir"""

from typing import NoReturn, TypeVar

from tvm.ir import GlobalInfo, Span, Var
from tvm.runtime import Object as tvm_Object

from . import _ffi_api
from .base import IRBuilder, SpanEntry
from .frame import IRModuleFrame

T = TypeVar("T")


def dynamic(name: str, dtype: str = "int64", *, span: SpanEntry | Span | None = None) -> Var:
    """Create a fresh primitive symbolic variable, independently of builder scope.

    Parameters
    ----------
    name : str
        The symbol's display name. Repeated names do not share identity.
    dtype : str
        Primitive dtype, defaulting to int64.
    span : SpanEntry, Span or None
        Source location of the symbol.

    Returns
    -------
    Var
        A fresh symbol. Reuse this object to share dimensions across annotations
        and function bodies, including outside an ``I.ir_module`` definition.
    """
    return Var(name, dtype, span.span if isinstance(span, SpanEntry) else span)


def constexpr(value: object) -> NoReturn:
    """Mark host control syntax or a JIT specialization annotation.

    Parameters
    ----------
    value : object
        Source expression to evaluate with ordinary Python semantics in a
        supported control-flow position. The marker itself can also appear
        as an annotation identifying a JIT specialization parameter.

    Raises
    ------
    TypeError
        If invoked directly instead of being recognized in parsed source.

    Notes
    -----
    The parser recognizes this marker through its fixed namespace path and
    removes it before execution. Host operators retain ordinary Python behavior.
    Direct invocation raises TypeError; no builder frame or IR is constructed.

    .. code:: python

        # Source
        if I.constexpr(enabled):
            T.evaluate(1)
        # Builder
        if enabled:
            X.emit_(X.evaluate(1))
    """
    raise TypeError("constexpr is a parser syntax marker, not a runtime operation")


def meta_var(value: T) -> T:
    """Return a Python metadata value without binding, naming or relocating it.

    Parameters
    ----------
    value : T
        Any host or IR object, including an unpackable sequence.

    Returns
    -------
    T
        The exact input object. No frame is required, no IR is emitted, and
        existing names and source locations are retained.

    .. code:: python

        # Source and generated Python (the value retains its identity)
        value = I.meta_var(existing_value)
        a, b = I.meta_var((left, right))
    """
    return value


def module_attrs(attrs: dict[str, tvm_Object], allow_overwrite=False) -> None:
    """Specify the attrs of the ir_module frame.
    Parameters
    ----------
    attrs: Dict[str, Object]
        The module attrs.
    allow_overwrite: bool
        Whether allow overwrite the existing attrs.
    """
    return _ffi_api.ModuleAttrs(attrs, allow_overwrite)  # type: ignore[attr-defined] # pylint: disable=no-member


def module_get_attr(attr_key: str) -> tvm_Object | None:
    """Get the specified attr of the ir_module frame.
    Parameters
    ----------
    attr_key: str
        The key of the attr to be retrieved.
    Returns
    -------
    attr: Optional[Object]
        The specified module attr or None if not found.
    """
    return _ffi_api.ModuleGetAttr(attr_key)  # type: ignore[attr-defined] # pylint: disable=no-member


def module_set_attr(
    attr_key: str, attr_value: tvm_Object | None, allow_overwrite: bool = False
) -> None:
    """Set the specified attr of the ir_module frame.
    Parameters
    ----------
    attr_key: str
        The key of the attr to be set.
    attr_value: Optional[Object]
        The value of the attr to be set.
    allow_overwrite: bool
        Whether allow overwrite the existing attr.
    """
    return _ffi_api.ModuleSetAttr(attr_key, attr_value, allow_overwrite)  # type: ignore[attr-defined] # pylint: disable=no-member


def module_global_infos(global_infos: dict[str, list[GlobalInfo]]) -> None:
    """Specify the global infos of the ir_module frame.

    Parameters
    ----------
    global_infos: Dict[str, List[GlobalInfo]]
        The module global infos.
    """
    return _ffi_api.ModuleGlobalInfos(global_infos)


def _global_infos():
    """Read metadata from the nearest active native module frame."""
    if IRBuilder.is_in_scope():
        for frame in reversed(IRBuilder.current().frames):
            if isinstance(frame, IRModuleFrame):
                return frame.global_infos
    raise ValueError("Global-info lookup requires an enclosing module frame")


def _get_dialect_builder(name: str):
    """Resolve a registered lazy dialect export for the shared builder package."""
    import importlib
    import sys

    from tvm.script import _DIALECT_REGISTRY

    if name in _DIALECT_REGISTRY:
        module = importlib.import_module(f"tvm.script.ir_builder.{name}")
        setattr(sys.modules["tvm.script.ir_builder"], name, module)
        return module
    raise AttributeError(f"module 'tvm.script.ir_builder' has no attribute {name!r}")
