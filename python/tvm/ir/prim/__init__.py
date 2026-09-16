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
"""Primitive expression nodes shared by TVM IR dialects."""

from ..expr import Expr
from . import _ffi_api


def expr_deep_equal(lhs: Expr, rhs: Expr) -> bool:
    """Deeply compare two nested expressions.

    Parameters
    ----------
    lhs : Expr
        The left operand.

    rhs : Expr
        The right operand.

    Returns
    -------
    result : bool
        The comparison result

    Note
    ----

    This function does not remap variable bindings, it will not
    return true for (let x = 1 in x + 1) vs (let y = 1 in y + 1), unless x.same_as(y).
    Use py:func:`tvm_ffi.structural_equal` to handle structural variable remapping.

    Due to the restriction of not remapping variables, this function can run
    faster than StructuralEqual and can be used as a utility function during arithmetic
    simplifications.

    Always consider py:func:`tvm_ffi.structural_equal` first, which handles
    the structural remapping.

    See Also
    --------
    tvm_ffi.structural_equal
    """
    return _ffi_api.expr_deep_equal(lhs, rhs)  # type: ignore


_EXPR_NAMES = {
    "StringImm",
    "Cast",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "FloorDiv",
    "FloorMod",
    "Min",
    "Max",
    "EQ",
    "NE",
    "LT",
    "LE",
    "GT",
    "GE",
    "And",
    "Or",
    "Not",
    "Select",
    "Let",
    "Ramp",
    "Broadcast",
    "Shuffle",
}


def __getattr__(name):
    # Keep the historical tvm.tirx classes as the single Python definitions
    # during this mechanical C++ ownership move.  Importing lazily avoids the
    # tvm.ir <-> tvm.tirx initialization cycle while exposing the new public
    # tvm.ir.prim spelling.
    if name in _EXPR_NAMES:
        from tvm.tirx import expr  # pylint: disable=import-outside-toplevel

        return getattr(expr, name)
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | _EXPR_NAMES)
