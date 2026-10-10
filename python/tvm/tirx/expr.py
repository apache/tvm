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
# pylint: disable=redefined-builtin
"""TIR expression nodes.

Each expression node have subfields that can be visited from python side.
For example, you can use addexp.a to get the left operand of an Add node.

.. code-block:: python

  x = tvm.tirx.Var("n", "int32")
  y = x + 2
  assert(isinstance(y, tvm.tirx.Add))
  assert(y.a == x)
"""

from tvm import ir
from tvm.ir import Expr
from tvm.ir import const as const
from tvm.ir._overload_prim_expr import (  # noqa: F401
    EqualOp,
    ExprOp,
    NotEqualOp,
    div_ambiguity_error,
)
from tvm.ir.base import Location
from tvm.ir.prim import convert as convert

# Retain historical imports as aliases of the canonical shared definitions.
from tvm.ir.prim.expr import (  # noqa: F401
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    BinaryOpExpr,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    Broadcast,
    Cast,
    CmpExpr,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    Let,
    LogicalExpr,
    LShift,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    Ramp,
    RShift,
    Select,
    Shuffle,
    Sub,
)
from tvm.runtime import ObjectConvertible

from . import _ffi_api


class IntImmEnum(ObjectConvertible):
    """Lazily evaluate an IntImm in case
    the constructor is not available in runtime.

    Parameters
    ----------
    value : int
        The enum value

    loc : Location or None, optional
        The location of the cast in the source.
    """

    def __init__(self, value: int, loc: Location | None = None) -> None:
        self.value = value
        self.loc = loc

    def asobject(self) -> "IntImm":
        """Convert object."""
        return IntImm("int32", self.value, self.loc)  # type: ignore


Var = ir.Var


def TensorLoad(buffer: Var, indices: list[Expr], loc: Location | None = None) -> ir.TensorLoad:
    """Construct a validated buffer load.

    Parameters
    ----------
    buffer : Var
        The buffer to be loaded.

    indices : List[Expr]
        The buffer indices to load values from.

    loc : Location or None, optional
        The location of this expression in the source code.

    """

    return _ffi_api.TensorLoad(buffer, indices, loc)


class CallEffectKind:
    """Possible kinds of Call effects."""

    # only expose up to opaque
    ExprAnnotation = IntImmEnum(0)
    Pure = IntImmEnum(1)
    ReadState = IntImmEnum(2)
    UpdateState = IntImmEnum(3)
    Opaque = UpdateState
