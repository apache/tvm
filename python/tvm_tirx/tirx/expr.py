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

  x = tvm_tirx.tirx.Var("n", "int32")
  y = x + 2
  assert(isinstance(y, tvm_tirx.tirx.Add))
  assert(y.a == x)
"""

from tvm_tirx import ir
from tvm_tirx.ir import Expr as Expr
from tvm_tirx.ir import const as const
from tvm_tirx.ir._prim_expr_overload import (  # noqa: F401
    EqualOp,
    ExprOp,
    NotEqualOp,
    div_ambiguity_error,
)
from tvm_tirx.ir.prim import convert as convert

# Retain historical imports as aliases of the canonical shared definitions.
from tvm_tirx.ir.prim.expr import (  # noqa: F401
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
    Cast,
    CmpExpr,
    Div,
    FloorDiv,
    FloorMod,
    Let,
    LogicalExpr,
    LShift,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    RShift,
    Select,
    Sub,
)

Var = ir.Var
