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
"""The tir expression operation registration"""

from typing import Type

from tvm import tir
from tvm.tir import IntImm

from .._core import OpMethod, doc, register_op


def _register_expr_op(ty: Type):  # pylint: disable=invalid-name
    ty._dispatch_type = ty  # pylint: disable=protected-access

    def _and(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if isinstance(b, bool):
            b = IntImm("bool", b)
        return tir.And(a, b)

    def _or(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if isinstance(b, bool):
            b = IntImm("bool", b)
        return tir.Or(a, b)

    def r(op: Type, i: int, m: OpMethod):  # pylint: disable=invalid-name
        register_op(ty, op, i)(m)

    for i in [0, 1]:
        # Case 1. binop
        r(doc.Add, i, tir.Add)
        r(doc.Sub, i, tir.Sub)
        r(doc.Mult, i, tir.Mul)
        r(doc.Div, i, tir.Div)
        r(doc.FloorDiv, i, tir.FloorDiv)
        r(doc.Mod, i, tir.FloorMod)
        r(doc.LShift, i, lambda a, b: a << b)
        r(doc.RShift, i, lambda a, b: a >> b)
        r(doc.BitOr, i, lambda a, b: a | b)
        r(doc.BitXor, i, lambda a, b: a ^ b)
        r(doc.BitAnd, i, lambda a, b: a & b)
        # doc.MatMult <-- not implemented
        # doc.Pow <-- not implemented
        # Case 2. cmpop
        r(doc.Eq, i, tir.EQ)
        r(doc.NotEq, i, tir.NE)
        r(doc.Lt, i, tir.LT)
        r(doc.LtE, i, tir.LE)
        r(doc.Gt, i, tir.GT)
        r(doc.GtE, i, tir.GE)
        # doc.Is <-- not implemented
        # doc.IsNot <-- not implemented
        # doc.In <-- not implemented
        # doc.NotIn <-- not implemented
        # Case 3. boolop
        r(doc.And, i, _and)
        r(doc.Or, i, _or)
    for i in [0]:
        #  Case 4. unaryop
        r(doc.Invert, i, lambda a: ~a)
        r(doc.Not, i, tir.Not)
        r(doc.UAdd, i, lambda a: +a)
        r(doc.USub, i, lambda a: -a)


_register_expr_op(tir.PrimExpr)
_register_expr_op(tir.IterVar)
