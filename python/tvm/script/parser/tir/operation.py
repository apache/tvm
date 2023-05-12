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
from tvm._ffi.runtime_ctypes import DataType, DataTypeCode
from tvm.tir import IntImm
from tvm.tir.expr import FloatImm

from .._core import OpMethod, doc, register_op


def _register_expr_op(ty: Type):  # pylint: disable=invalid-name
    ty._dispatch_type = ty  # pylint: disable=protected-access

    def _and(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if isinstance(b, bool):
            b = IntImm("bool", b)
        if DataType(a.dtype).lanes > 1 or DataType(b.dtype).lanes > 1:
            return a & b
        else:
            return tir.And(a, b)

    def _or(a, b):
        if isinstance(a, bool):
            a = IntImm("bool", a)
        if isinstance(b, bool):
            b = IntImm("bool", b)
        if DataType(a.dtype).lanes > 1 or DataType(b.dtype).lanes > 1:
            return a | b
        else:
            return tir.Or(a, b)

    def _get_type_str(dtype: str):
        if DataType(dtype).lanes == 1:
            return dtype
        index = dtype.find("x")
        return dtype[0:index]

    def _auto_broadcast(a, b, op):

        if isinstance(a, int):
            if hasattr(b, "dtype"):
                if (
                    DataType(b.dtype).type_code == DataTypeCode.INT
                    or DataType(b.dtype).type_code == DataTypeCode.UINT
                ):
                    a = IntImm(_get_type_str(b.dtype), a)
                elif DataType(b.dtype).type_code == DataTypeCode.FLOAT:
                    a = FloatImm(_get_type_str(b.dtype), a)
            elif isinstance(b, float):
                a = FloatImm("float32", a)
            else:
                a = IntImm("int32", a)
        elif isinstance(a, float):
            if DataType(b.dtype).type_code == DataTypeCode.FLOAT:
                a = FloatImm(_get_type_str(b.dtype), a)
            else:
                a = FloatImm("float32", a)

        assert isinstance(a, tir.PrimExpr), "Operand should be a PrimExpr."
        if isinstance(b, int):
            if (
                DataType(a.dtype).type_code == DataTypeCode.INT
                or DataType(a.dtype).type_code == DataTypeCode.UINT
            ):
                b = IntImm(_get_type_str(a.dtype), b)
            elif DataType(a.dtype).type_code == DataTypeCode.FLOAT:
                b = FloatImm(_get_type_str(a.dtype), b)
        elif isinstance(b, float):
            b = FloatImm(_get_type_str(a.dtype), b)

        if DataType(a.dtype).lanes == DataType(b.dtype).lanes:
            return op(a, b)
        elif DataType(a.dtype).lanes == 1 and DataType(a.dtype).lanes != DataType(b.dtype).lanes:
            broadcast_a = tir.Broadcast(a, DataType(b.dtype).lanes)
            return op(broadcast_a, b)
        elif DataType(b.dtype).lanes == 1 and DataType(a.dtype).lanes != DataType(b.dtype).lanes:
            broadcast_b = tir.Broadcast(b, DataType(a.dtype).lanes)
            return op(a, broadcast_b)
        else:
            raise TypeError("do not know how to deal with it.")

    def _eq(a, b):
        return _auto_broadcast(a, b, tir.EQ)

    def _ne(a, b):
        return _auto_broadcast(a, b, tir.NE)

    def _lt(a, b):
        return _auto_broadcast(a, b, tir.LT)

    def _le(a, b):
        return _auto_broadcast(a, b, tir.LE)

    def _gt(a, b):
        return _auto_broadcast(a, b, tir.GT)

    def _ge(a, b):
        return _auto_broadcast(a, b, tir.GE)

    def r(op: Type, i: int, m: OpMethod):  # pylint: disable=invalid-name
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
        r(doc.Not, i, tir.Not)
        # doc.UAdd <-- is overloaded
        # doc.USub <-- is overloaded


_register_expr_op(tir.PrimExpr)
_register_expr_op(tir.IterVar)
