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
"""The tirx expression operation registration"""

import tvm
from tvm import tirx
from tvm.ir import PrimType
from tvm.runtime import DataTypeCode
from tvm.script.parser._core import OpMethod, doc, register_op
from tvm.tirx import IntImm
from tvm.tirx.expr import FloatImm


def _register_expr_op(ty: type):  # pylint: disable=invalid-name
    ty._dispatch_type = ty  # pylint: disable=protected-access

    def _expr_ty(expr):
        ty = expr.ty if tvm.ir.is_prim_expr(expr) else None
        if not isinstance(ty, PrimType):
            ty = expr.expr_ty()
        if not isinstance(ty, PrimType):
            raise TypeError(f"Expected a PrimType expression, but got {ty}")
        return ty

    def _and(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if isinstance(b, bool):
            b = IntImm("bool", b)
        if not _expr_ty(a).is_scalar() or not _expr_ty(b).is_scalar():
            return a & b
        else:
            return tirx.And(a, b)

    def _or(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if isinstance(b, bool):
            b = IntImm("bool", b)
        if not _expr_ty(a).is_scalar() or not _expr_ty(b).is_scalar():
            return a | b
        else:
            return tirx.Or(a, b)

    def _get_type_str(ty: PrimType):
        dtype_str = str(ty.dtype)
        if ty.is_scalar():
            return dtype_str
        index = dtype_str.find("x")
        return dtype_str[0:index]

    def _auto_broadcast(a, b, op):
        if isinstance(a, int):
            if tvm.ir.is_prim_expr(b) or hasattr(b, "expr_ty"):
                b_ty = _expr_ty(b)
                if b_ty.matches_code(DataTypeCode.INT, DataTypeCode.UINT, DataTypeCode.BOOL):
                    a = IntImm(_get_type_str(b_ty), a)
                elif b_ty.matches_code(DataTypeCode.FLOAT):
                    a = FloatImm(_get_type_str(b_ty), a)
            elif isinstance(b, float):
                a = FloatImm("float32", a)
            else:
                a = IntImm("int32", a)
        elif isinstance(a, float):
            b_ty = _expr_ty(b)
            if b_ty.matches_code(DataTypeCode.FLOAT):
                a = FloatImm(_get_type_str(b_ty), a)
            else:
                a = FloatImm("float32", a)

        assert tvm.ir.is_prim_expr(a), "Operand should be a Expr."
        if isinstance(b, int):
            a_ty = _expr_ty(a)
            if a_ty.matches_code(DataTypeCode.INT, DataTypeCode.UINT, DataTypeCode.BOOL):
                b = IntImm(_get_type_str(a_ty), b)
            elif a_ty.matches_code(DataTypeCode.FLOAT):
                b = FloatImm(_get_type_str(a_ty), b)
        elif isinstance(b, float):
            b = FloatImm(_get_type_str(_expr_ty(a)), b)

        a_ty = _expr_ty(a)
        b_ty = _expr_ty(b)
        if a_ty.dtype.lanes == b_ty.dtype.lanes:
            return op(a, b)
        elif a_ty.is_scalar() and a_ty.dtype.lanes != b_ty.dtype.lanes:
            broadcast_a = tirx.Broadcast(a, b_ty.dtype.lanes)
            return op(broadcast_a, b)
        elif b_ty.is_scalar() and a_ty.dtype.lanes != b_ty.dtype.lanes:
            broadcast_b = tirx.Broadcast(b, a_ty.dtype.lanes)
            return op(a, broadcast_b)
        else:
            raise TypeError("do not know how to deal with it.")

    def _eq(a, b):
        return _auto_broadcast(a, b, tirx.EQ)

    def _ne(a, b):
        return _auto_broadcast(a, b, tirx.NE)

    def _lt(a, b):
        return _auto_broadcast(a, b, tirx.LT)

    def _le(a, b):
        return _auto_broadcast(a, b, tirx.LE)

    def _gt(a, b):
        return _auto_broadcast(a, b, tirx.GT)

    def _ge(a, b):
        return _auto_broadcast(a, b, tirx.GE)

    def r(op: type, i: int, m: OpMethod):  # pylint: disable=invalid-name
        register_op(ty, op, i)(m)

    for i in [0, 1]:
        # Case 1. binop
        # doc.Add <-- is overloaded
        # doc.Sub <-- is overloaded
        # doc.Mult <-- is overloaded
        # doc.Div <-- is overloaded
        # doc.FloorDiv <-- is overloaded
        # doc.Mod <-- is overloaded
        # doc.LShift <-- is overloaded
        # doc.RShift <-- is overloaded
        # doc.BitOr <-- is overloaded
        # doc.BitXor <-- is overloaded
        # doc.BitAnd <-- is overloaded
        # doc.MatMult <-- not implemented
        # doc.Pow <-- not implemented
        # Case 2. cmpop
        r(doc.Eq, i, _eq)
        r(doc.NotEq, i, _ne)
        r(doc.Lt, i, _lt)
        r(doc.LtE, i, _le)
        r(doc.Gt, i, _gt)
        r(doc.GtE, i, _ge)
        # doc.Is <-- not implemented
        # doc.IsNot <-- not implemented
        # doc.In <-- not implemented
        # doc.NotIn <-- not implemented
        # Case 3. boolop
        r(doc.And, i, _and)
        r(doc.Or, i, _or)
    for i in [0]:
        #  Case 4. unaryop
        # doc.Invert <-- is overloaded
        r(doc.Not, i, tirx.Not)
        # doc.UAdd <-- is overloaded
        # doc.USub <-- is overloaded


_register_expr_op(tirx.Expr)
_register_expr_op(tirx.IterVar)
