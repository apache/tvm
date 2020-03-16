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
# pylint: disable=invalid-name, unused-import
"""The type nodes of the Relay language."""
from tvm.ir import Type, TypeKind, TypeVar, GlobalTypeVar
from tvm.ir import TypeConstraint, FuncType, TupleType, IncompleteType
from tvm.ir import TypeCall, TypeRelation, TensorType, RelayRefType as RefType

from .base import RelayNode, register_relay_node
from . import _make

Any = _make.Any

def type_has_any(tensor_type):
    """Check whether type has any as a shape.

    tensor_type : Type
        The type to be inspected

    Returns
    -------
    has_any : bool
        The check result.
    """
    return _make.IsDynamic(tensor_type)


def ShapeVar(name):
    """A helper which constructs a type var of which the shape kind.

    Parameters
    ----------
    name : str

    Returns
    -------
    type_var : tvm.relay.TypeVar
        The shape variable.
    """
    return TypeVar(name, kind=TypeKind.ShapeVar)


def scalar_type(dtype):
    """Creates a scalar type.

    This function returns TensorType((), dtype)

    Parameters
    ----------
    dtype : str
        The content data type.

    Returns
    -------
    s_type : tvm.relay.TensorType
        The result type.
    """
    return TensorType((), dtype)
