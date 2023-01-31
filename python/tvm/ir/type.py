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
from enum import IntEnum

import tvm
import tvm._ffi
from tvm.runtime import Scriptable

from . import _ffi_api
from .base import Node


class Type(Node, Scriptable):
    """The base class of all types."""

    def __eq__(self, other):
        """Compare two types for structural equivalence."""
        return bool(tvm.ir.structural_equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Compares two Relay types by referential equality."""
        return super().__eq__(other)


class TypeKind(IntEnum):
    """Possible kinds of TypeVars."""

    Type = 0
    ShapeVar = 1
    BaseType = 2
    Constraint = 4
    AdtHandle = 5
    TypeData = 6


@tvm._ffi.register_object("PrimType")
class PrimType(Type):
    """Primitive data type in the low level IR

    Parameters
    ----------
    dtype : str
        The runtime data type relates to the primtype.
    """

    def __init__(self, dtype):
        self.__init_handle_by_constructor__(_ffi_api.PrimType, dtype)


@tvm._ffi.register_object("PointerType")
class PointerType(Type):
    """PointerType used in the low-level TIR.

    Parameters
    ----------
    element_type : tvm.ir.Type
        The type of pointer's element.

    storage_scope : str
        The storage scope into which the pointer addresses.
    """

    def __init__(self, element_type, storage_scope=""):
        self.__init_handle_by_constructor__(_ffi_api.PointerType, element_type, storage_scope)


@tvm._ffi.register_object("TypeVar")
class TypeVar(Type):
    """Type parameter in functions.

    A type variable represents a type placeholder which will
    be filled in later on. This allows the user to write
    functions which are generic over types.

    Parameters
    ----------
    name_hint: str
        The name of the type variable. This name only acts as a hint, and
        is not used for equality.

    kind : Optional[TypeKind]
        The kind of the type parameter.
    """

    def __init__(self, name_hint, kind=TypeKind.Type):
        self.__init_handle_by_constructor__(_ffi_api.TypeVar, name_hint, kind)

    def __call__(self, *args):
        """Create a type call from this type.

        Parameters
        ----------
        args: List[Type]
            The arguments to the type call.

        Returns
        -------
        call: Type
            The result type call.
        """
        # pylint: disable=import-outside-toplevel
        from .type_relation import TypeCall

        return TypeCall(self, args)


@tvm._ffi.register_object("GlobalTypeVar")
class GlobalTypeVar(Type):
    """A global type variable that is used for defining new types or type aliases.

    Parameters
    ----------
    name_hint: str
        The name of the type variable. This name only acts as a hint, and
        is not used for equality.

    kind : Optional[TypeKind]
        The kind of the type parameter.
    """

    def __init__(self, name_hint, kind=TypeKind.AdtHandle):
        self.__init_handle_by_constructor__(_ffi_api.GlobalTypeVar, name_hint, kind)

    def __call__(self, *args):
        """Create a type call from this type.

        Parameters
        ----------
        args: List[Type]
            The arguments to the type call.

        Returns
        -------
        call: Type
            The result type call.
        """
        # pylint: disable=import-outside-toplevel
        from .type_relation import TypeCall

        return TypeCall(self, args)


@tvm._ffi.register_object("TupleType")
class TupleType(Type):
    """The type of tuple values.

    Parameters
    ----------
    fields : List[Type]
        The fields in the tuple
    """

    def __init__(self, fields):
        self.__init_handle_by_constructor__(_ffi_api.TupleType, fields)


@tvm._ffi.register_object("TypeConstraint")
class TypeConstraint(Type):
    """Abstract class representing a type constraint."""


@tvm._ffi.register_object("FuncType")
class FuncType(Type):
    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, and a return type.

    We can informally write them as:
    `forall (type_params), (arg_types) -> ret_type where type_constraints`

    Parameters
    ----------
    arg_types : List[tvm.relay.Type]
        The argument types

    ret_type : tvm.relay.Type
        The return type.

    type_params : Optional[List[tvm.relay.TypeVar]]
        The type parameters

    type_constraints : Optional[List[tvm.relay.TypeConstraint]]
        The type constraints.
    """

    def __init__(self, arg_types, ret_type, type_params=None, type_constraints=None):
        if type_params is None:
            type_params = []
        if type_constraints is None:
            type_constraints = []
        self.__init_handle_by_constructor__(
            _ffi_api.FuncType, arg_types, ret_type, type_params, type_constraints
        )


@tvm._ffi.register_object("IncompleteType")
class IncompleteType(Type):
    """Incomplete type during type inference.

    kind : Optional[TypeKind]
        The kind of the incomplete type.
    """

    def __init__(self, kind=TypeKind.Type):
        self.__init_handle_by_constructor__(_ffi_api.IncompleteType, kind)


@tvm._ffi.register_object("relay.RefType")
class RelayRefType(Type):
    """Reference Type in relay.

    Parameters
    ----------
    value: Type
        The value type.
    """

    def __init__(self, value):
        self.__init_handle_by_constructor__(_ffi_api.RelayRefType, value)
