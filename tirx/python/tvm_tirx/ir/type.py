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
"""Unified type system in the project."""

import sys
from collections.abc import Callable
from types import ModuleType
from typing import ClassVar

import tvm_ffi

from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.runtime import MemoryScope

from . import _ffi_api
from .base import Node, Scriptable


@tvm_ffi.register_object("ir.Type")
class Type(Node, Scriptable):
    """The base class of all types.

    ``__expr_methods__`` names methods exposed on expressions of this type.
    Each method receives the type instance and then the expression operand:
    ``expr.method(...)`` is equivalent to ``expr.ty.method(expr, ...)``.
    ``__expr_properties__`` maps read-only expression property names to
    getters taking ``(type_instance, expression)``. Ordinary expression
    attributes take precedence, then declared properties, then methods.
    Subclasses inherit these declarations or replace them explicitly.
    """

    __expr_methods__: tuple[str, ...] = ()
    __expr_properties__: ClassVar[dict[str, Callable]] = {}

    @staticmethod
    def missing():
        """Construct a MissingType for missing type information."""
        return _ffi_api.TypeMissing()

    @staticmethod
    def Missing():
        """Construct a MissingType for missing type information."""
        return _ffi_api.TypeMissing()

    def __eq__(self, other):
        """Compare two types for structural equivalence."""
        return bool(tvm_ffi.structural_equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Compares two TVM types by referential equality."""
        return self.is_(other)


@tvm_ffi.register_object("ir.MissingType")
class MissingType(Type):
    """Type information that has not been supplied or computed.

    Unlike AnyType or Void, this is not a concrete type and must be resolved
    before a boundary that requires fully typed IR.
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.MissingType)


@tvm_ffi.register_object("ir.AnyType")
class AnyType(Type):
    """The top type, which admits any value."""

    def __init__(self, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.AnyType, loc)


@tvm_ffi.register_object("ir.TensorRegionType")
class TensorRegionType(Type):
    """The type of a multi-dimensional :class:`tvm_tirx.ir.TensorRegion`."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.TensorRegionType)


@tvm_ffi.register_object("ir.OpaqueType")
class OpaqueType(Type):
    """Type marker for opaque values that must be removed from finished IR."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.OpaqueType)


@tvm_ffi.register_object("ir.PrimType")
class PrimType(Type):
    """Primitive data type in the low level IR

    Parameters
    ----------
    dtype : str
        The runtime data type relates to the primtype.
    """

    def __init__(self, dtype):
        self.__init_handle_by_constructor__(_ffi_api.PrimType, dtype)

    def __call__(self, expr=None):
        """Create a fresh variable, or convert a primitive expression to this type."""
        from .expr import Var
        from .prim import _ffi_api as prim_ffi_api

        if expr is None:
            return Var("", self)
        if isinstance(expr, str) and str(self.dtype) not in ("bool", "void", ""):
            expr = float(expr)
        return prim_ffi_api._cast(expr, self, UNKNOWN_LOC)

    def __getitem__(self, shape):
        """Describe local storage with this element type and the given shape."""
        return _PrimTypeArrayAnnotation(self, shape if isinstance(shape, tuple) else (shape,))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.dtype == other
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        dtype = self.dtype
        return hash((dtype.type_code, dtype.bits, dtype.lanes))

    def __str__(self):
        return str(self.dtype)

    def matches_code(self, *codes) -> bool:
        """Return whether this type has any of the given DLPack dtype codes."""
        type_code = self.dtype.type_code
        return any(type_code == int(code) for code in codes)

    def matches_element_type(self, code, bits: int) -> bool:
        """Return whether this type has the given scalar element code and bits."""
        dtype = self.dtype
        return dtype.type_code == int(code) and dtype.bits == bits

    def is_scalar(self) -> bool:
        """Return whether this type has exactly one fixed lane."""
        return self.dtype.lanes == 1


class _PrimTypeArrayAnnotation:
    """A shaped primitive annotation, interpreted by a dialect's declaration builder."""

    __slots__ = ("dtype", "shape")

    def __init__(self, dtype: PrimType, shape: tuple):
        self.dtype = dtype
        self.shape = shape


@tvm_ffi.register_object("ir.StringType")
class StringType(Type):
    """Semantic string type, independent of the function's calling convention."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.StringType)


@tvm_ffi.register_object("ir.PtrType")
class PtrType(Type):
    """PtrType used in the low-level TIR.

    Parameters
    ----------
    element_type : tvm_tirx.ir.Type
        The type of pointer's element.

    scope : str or tvm_tirx.runtime.MemoryScope
        The storage scope into which the pointer addresses.
    """

    def __init__(self, element_type, scope=MemoryScope("global"), *, loc=UNKNOWN_LOC):
        self.__init_handle_by_constructor__(_ffi_api.PtrType, _normalize_type(element_type), scope)
        if loc is not UNKNOWN_LOC:
            # Keep the script constructor's source-context behavior on the class alias.
            from tvm_tirx.script.ir_builder.base import at  # pylint: disable=import-outside-toplevel

            at(loc, self)


@tvm_ffi.register_object("ir.TupleType")
class TupleType(Type):
    """The type of tuple values.

    Parameters
    ----------
    fields : List[Type]
        The fields in the tuple
    """

    def __init__(self, fields, loc: Location = UNKNOWN_LOC):
        self.__init_handle_by_constructor__(
            _ffi_api.TupleType, [_normalize_type(field) for field in fields], loc
        )


@tvm_ffi.register_object("ir.FuncType")
class FuncType(Type):
    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, and a return type.

    Parameters
    ----------
    arg_types : List[tvm_tirx.ir.Type]
        The argument types

    ret_type : tvm_tirx.ir.Type
        The return type.
    """

    def __init__(self, arg_types, ret_type):
        self.__init_handle_by_constructor__(
            _ffi_api.FuncType,
            [_normalize_type(arg) for arg in arg_types],
            _normalize_type(ret_type),
        )


def _normalize_type(value) -> Type:
    """Normalize supported explicit type arguments without invoking arbitrary factories.

    Bare concrete Type classes denote their nullary instances. Primitive dtype
    strings become PrimType values without constructing a variable or requiring
    a builder frame.
    """
    if isinstance(value, Type):
        return value
    if isinstance(value, type) and issubclass(value, Type) and value is not Type:
        return value()
    if isinstance(value, str | tvm_ffi.dtype):
        return PrimType(value)
    raise TypeError(f"Expected an IR type or primitive dtype, got {type(value).__name__}")


def _init_type_api(namespace: str, target_module: str | ModuleType) -> None:
    """Expose loaded Type bindings as short aliases in a builder namespace.

    Dialect bindings precede root ``ir`` bindings. Explicit exports always win.
    This only aliases registered classes: it never imports bindings, constructs
    instances, or generates constructors. Call it again after loading additional
    bindings to fill missing names or replace generated root aliases with dialect
    aliases. Module-local provenance distinguishes generated aliases from exports.
    """
    module = sys.modules[target_module] if isinstance(target_module, str) else target_module
    exports = vars(module)
    generated = exports.setdefault("_generated_type_aliases", {})
    root_index = Type.__tvm_ffi_type_info__.type_index
    type_keys = [str(key) for key in tvm_ffi.registry.get_registered_type_keys()]
    dialect_aliases = set()
    for prefix in dict.fromkeys((namespace, "ir")):
        for type_key in type_keys:
            owner, _, name = type_key.rpartition(".")
            if owner != prefix or type_key == "ir.Type":
                continue
            info = tvm_ffi.core._lookup_or_register_type_info_from_type_key(type_key)
            if root_index not in info.type_ancestors:
                continue
            alias = name[:-4] if name.endswith("Type") else name
            if prefix == namespace:
                # Reserve unbound dialect names so a later repeat can fill them.
                dialect_aliases.add(alias)
            elif alias in dialect_aliases:
                continue
            if info.type_cls is None or not alias:
                continue
            if alias in exports:
                previous = generated.get(alias)
                if previous is None or exports[alias] is not previous[1]:
                    continue
                if prefix != namespace or previous[0] != "ir" or namespace == "ir":
                    continue
            exports[alias] = info.type_cls
            generated[alias] = (prefix, info.type_cls)
            if "__all__" in exports and alias not in exports["__all__"]:
                exports["__all__"] = [*exports["__all__"], alias]
