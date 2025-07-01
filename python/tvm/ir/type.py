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
import tvm
import tvm.ffi
from tvm.runtime import Scriptable

from . import _ffi_api
from .base import Node


@tvm.ffi.register_object("ir.Type")
class Type(Node, Scriptable):
    """The base class of all types."""

    def __eq__(self, other):
        """Compare two types for structural equivalence."""
        return bool(tvm.ir.structural_equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Compares two TVM types by referential equality."""
        return super().__eq__(other)


@tvm.ffi.register_object("ir.PrimType")
class PrimType(Type):
    """Primitive data type in the low level IR

    Parameters
    ----------
    dtype : str
        The runtime data type relates to the primtype.
    """

    def __init__(self, dtype):
        self.__init_handle_by_constructor__(_ffi_api.PrimType, dtype)


@tvm.ffi.register_object("ir.PointerType")
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


@tvm.ffi.register_object("ir.TupleType")
class TupleType(Type):
    """The type of tuple values.

    Parameters
    ----------
    fields : List[Type]
        The fields in the tuple
    """

    def __init__(self, fields):
        self.__init_handle_by_constructor__(_ffi_api.TupleType, fields)


@tvm.ffi.register_object("ir.FuncType")
class FuncType(Type):
    """Function type.

    A function type consists of a list of type parameters to enable
    the definition of generic functions,
    a set of type constraints which we omit for the time being,
    a sequence of argument types, and a return type.

    Parameters
    ----------
    arg_types : List[tvm.ir.Type]
        The argument types

    ret_type : tvm.ir.Type
        The return type.
    """

    def __init__(self, arg_types, ret_type):
        self.__init_handle_by_constructor__(
            _ffi_api.FuncType,
            arg_types,
            ret_type,
        )


@tvm.ffi.register_object("ir.TensorMapType")
class TensorMapType(Type):
    """TensorMapType used in the low-level TIR.

    Parameters
    ----------
    span : tvm.ir.Span
        The span information.
    """

    def __init__(self, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.TensorMapType, span  # pylint: disable=no-member
        )
