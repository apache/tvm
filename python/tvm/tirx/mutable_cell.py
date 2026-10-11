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
"""Nonescaping, initialized local mutable cells.

A cell handle identifies one lexical allocation. Only load and store consume
handles; loads are values, and copying a value never aliases its cell. The
storage-scope string defaults to "local"; other tags are unsupported.
"""

from tvm.ir import Call, Expr, Var, const
from tvm.ir.location import UNKNOWN_LOC

from .type import MutableCellType


def _element_value(value, element_type):
    if not isinstance(value, Expr):
        value = const(value, element_type.dtype)
    return value


def mutable_cell_alloc(element_type, value=0, scope="local", *, loc=UNKNOWN_LOC):
    """Create a fresh initialized cell allocation expression, to be bound once.

    Primitive literals are converted to the element type. Expression initializers
    must already have the exact element type. No address or storage buffer exists.
    """
    ty = MutableCellType(element_type, scope)
    call = Call("tirx.mutable_cell_alloc", [_element_value(value, ty.element_type)], ty=ty, loc=loc)
    call.validate()
    return call


def mutable_cell_load(cell, *, loc=UNKNOWN_LOC):
    """Read the value of a cell at this expression's evaluation point."""
    call = Call("tirx.mutable_cell_load", [cell], loc=loc)
    call.validate()
    return call


def mutable_cell_store(cell, value, *, loc=UNKNOWN_LOC):
    """Update a cell; the resulting void expression must be evaluated as a statement."""
    if not isinstance(cell, Var) or not isinstance(cell.ty, MutableCellType):
        raise TypeError("mutable_cell_store expects a Var with MutableCellType")
    call = Call(
        "tirx.mutable_cell_store", [cell, _element_value(value, cell.ty.element_type)], loc=loc
    )
    call.validate()
    return call


def is_mutable_cell_load(value):
    """Whether an expression names the current contents of a local cell."""
    return (
        isinstance(value, Call)
        and getattr(value.op, "name", None) == "tirx.mutable_cell_load"
        and len(value.args) == 1
        and isinstance(value.args[0], Var)
        and isinstance(value.args[0].ty, MutableCellType)
    )
