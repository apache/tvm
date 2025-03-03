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

import inspect

import pytest

import tvm
import tvm.testing
from tvm import te, tir
from tvm.tir import floordiv as fld
from tvm.tir import floormod as flm
from tvm.tir import truncdiv as tdiv
from tvm.tir import truncmod as tmod

from tvm.script import tir as T


class TestCase:
    def __init__(self, before, expected, preconditions=None):
        if isinstance(before, tir.expr.EqualOp):
            before = before.asobject()
        if isinstance(expected, tir.expr.EqualOp):
            expected = expected.asobject()

        self.before = self._convert(before)
        self.expected = self._convert(expected)
        self.preconditions = preconditions

    @staticmethod
    def _convert(expr):
        if isinstance(expr, tir.expr.EqualOp):
            return expr.asobject()
        elif isinstance(expr, int):
            return T.int32(expr)
        elif isinstance(expr, float):
            return T.float32(expr)
        else:
            return expr

    @property
    def constraint(self):
        if self.preconditions is None:
            return True
        elif isinstance(self.preconditions, tvm.ir.PrimExpr):
            return self.preconditions
        else:
            return tvm.tir.all(*self.preconditions)

    @property
    def __name__(self):
        return str(self.before)


class BaseCompare:
    extensions = tvm.arith.Extension.NoExtensions

    def test_simplify(self, test_case):
        analyzer = tvm.arith.Analyzer()
        analyzer.enabled_extensions = self.extensions

        if inspect.isclass(test_case.expected) and issubclass(test_case.expected, Exception):
            with pytest.raises(test_case.expected):
                with analyzer.constraint_scope(test_case.constraint):
                    analyzer.rewrite_simplify(test_case.before)
        else:
            with analyzer.constraint_scope(test_case.constraint):
                after = analyzer.rewrite_simplify(test_case.before)

            assert tvm.ir.structural_equal(after, test_case.expected), (
                f"Rewrite didn't match expected.\n"
                f"Before   = {test_case.before}\n"
                f"After    = {after}\n"
                f"Expected = {test_case.expected}"
            )


class TestVector(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    x64 = te.var("x", dtype="int64")
    vx = te.var("vx", dtype="int32x2")
    vc = te.var("vc", dtype="uint1")
    test_case = tvm.testing.parameter(
        # Add rules
        TestCase(tvm.tir.Ramp(x, 1, 4) + tvm.tir.Ramp(y, 2, 4), tvm.tir.Ramp(x + y, 3, 4)),
        TestCase(tvm.tir.Ramp(x, 1, 2) + y, tvm.tir.Ramp(x + y, 1, 2)),
        TestCase(y + tvm.tir.Ramp(x, 1, 2), tvm.tir.Ramp(y + x, 1, 2)),
        TestCase(
            tvm.tir.Ramp(x, 1, tir.vscale() * 4) + tvm.tir.Ramp(y, 2, tir.vscale() * 4),
            tvm.tir.Ramp(x + y, 3, tir.vscale() * 4),
        ),
        TestCase(y.astype("int32x2") + x.astype("int32x2"), (y + x).astype("int32x2")),
        TestCase(tvm.tir.Broadcast(0, 4) + y, tvm.tir.Broadcast(y, 4)),
        # int64 lanes
        TestCase(
            tvm.tir.Broadcast(x, 4) + tvm.tir.Ramp(0, 1, tvm.tir.IntImm(dtype="int64", value=4)),
            tvm.tir.Ramp(x, 1, 4),
        ),
        TestCase(
            tvm.tir.Broadcast(x, tvm.tir.IntImm(dtype="int64", value=4)) + tvm.tir.Ramp(0, 1, 4),
            tvm.tir.Ramp(x, 1, 4),
        ),
        # int64 iterators with int32 lanes
        TestCase(
            tvm.tir.Broadcast(x64, 4) + tvm.tir.Ramp(tvm.tir.IntImm(dtype="int64", value=0), 1, 4),
            tvm.tir.Ramp(x64, 1, 4),
        ),
        TestCase(
            tvm.tir.Broadcast(0, tir.vscale() * 8) + y, tvm.tir.Broadcast(y, tir.vscale() * 8)
        ),
        TestCase(
            tvm.tir.Ramp(x, 1, 4).astype("float32x4") + tvm.tir.Broadcast(0.0, 4),
            tvm.tir.Ramp(x, 1, 4).astype("float32x4"),
        ),
        # Sub rules
        TestCase(tvm.tir.Ramp(x, 4, 4) - tvm.tir.Ramp(y, 2, 4), tvm.tir.Ramp(x - y, 2, 4)),
        TestCase(tvm.tir.Ramp(x, 1, 2) - y, tvm.tir.Ramp(x - y, 1, 2)),
        TestCase(y - tvm.tir.Ramp(x, 1, 2), tvm.tir.Ramp(y - x, -1, 2)),
        TestCase(y.astype("int32x2") - x.astype("int32x2"), (y - x).astype("int32x2")),
        # Mul rules
        TestCase(y.astype("int32x2") * x.astype("int32x2"), (y * x).astype("int32x2")),
        TestCase(tvm.tir.Ramp(x, 4, 4) * 2, tvm.tir.Ramp(x * 2, 8, 4)),
        TestCase(2 * tvm.tir.Ramp(x, 4, 4), tvm.tir.Ramp(x * 2, 8, 4)),
        TestCase(tvm.tir.Broadcast(0, 4) * x, tvm.tir.Broadcast(0, 4)),
        TestCase(tvm.tir.Broadcast(0.0, 4) * x, tvm.tir.Broadcast(0.0, 4)),
        ## DivMod rules
        # trunc div
        TestCase(tdiv(y.astype("int32x2"), x.astype("int32x2")), tdiv(y, x).astype("int32x2")),
        TestCase(tdiv(tvm.tir.Ramp(x, 4, 4), 2), tvm.tir.Ramp(tdiv(x, 2), 2, 4)),
        TestCase(
            tdiv(tvm.tir.Ramp(x, 4, tir.vscale() * 5), 2),
            tvm.tir.Ramp(tdiv(x, 2), 2, tir.vscale() * 5),
        ),
        TestCase(tdiv(tvm.tir.Ramp(x * 8 + 1, 1, 4), 8), x.astype("int32x4"), x >= 0),
        TestCase(tdiv(tvm.tir.Ramp(x * 8 + 15, 1, 4), 8), tdiv(tvm.tir.Ramp(x * 8 + 15, 1, 4), 8)),
        # trunc mod
        TestCase(tmod(y.astype("int32x2"), x.astype("int32x2")), tmod(y, x).astype("int32x2")),
        TestCase(tmod(tvm.tir.Ramp(x, 4, 4), 2), tvm.tir.Broadcast(tmod(x, 2), 4)),
        TestCase(tmod(tvm.tir.Ramp(x * 8 + 1, 1, 4), 8), tvm.tir.Ramp(1, 1, 4), x >= 0),
        TestCase(
            tmod(tvm.tir.Ramp(x * 8 + 1, 1, tir.vscale() * 4), 8),
            tmod(tvm.tir.Ramp(1, 1, tir.vscale() * 4), 8),
            x >= 0,
        ),
        TestCase(tmod(tvm.tir.Ramp(x * 8 + 1, 15, 4), 8), tmod(tvm.tir.Ramp(1, 15, 4), 8), x >= 0),
        # floor div
        TestCase(fld(y.astype("int32x2"), x.astype("int32x2")), fld(y, x).astype("int32x2")),
        TestCase(fld(tvm.tir.Ramp(x, 4, 4), 2), tvm.tir.Ramp(fld(x, 2), 2, 4)),
        TestCase(
            fld(tvm.tir.Ramp(x, 4, tir.vscale() * 4), 2),
            tvm.tir.Ramp(fld(x, 2), 2, tir.vscale() * 4),
        ),
        TestCase(fld(tvm.tir.Ramp(x * 8 + 1, 1, 4), 8), (x).astype("int32x4")),
        TestCase(fld(tvm.tir.Ramp(x * 8 + 15, 1, 4), 8), fld(tvm.tir.Ramp(x * 8 + 15, 1, 4), 8)),
        TestCase(
            fld(tvm.tir.Ramp(x, 8, 5), tvm.tir.Broadcast(4, 5)), tvm.tir.Ramp(fld(x, 4), 2, 5)
        ),
        TestCase(
            fld(tvm.tir.Ramp(x, 8, tir.vscale() * 4), tvm.tir.Broadcast(4, tir.vscale() * 4)),
            tvm.tir.Ramp(fld(x, 4), 2, tir.vscale() * 4),
        ),
        TestCase(
            fld(tvm.tir.Ramp(flm(x * 4, 256), 1, 4), tvm.tir.Broadcast(8, 4)),
            tvm.tir.Broadcast(fld(flm(x * 4, 256), 8), 4),
        ),
        TestCase(
            fld(tvm.tir.Ramp(x, 7, 4), tvm.tir.Broadcast(4, 4)),
            fld(tvm.tir.Ramp(x, 7, 4), tvm.tir.Broadcast(4, 4)),
        ),
        TestCase(
            fld(tvm.tir.Ramp(x * 8, 1, 4), tvm.tir.Broadcast(4, 4)), tvm.tir.Broadcast(x * 2, 4)
        ),
        TestCase(
            fld(tvm.tir.Ramp(x * 8, 1, tir.vscale() * 4), tvm.tir.Broadcast(4, tir.vscale() * 4)),
            fld(tvm.tir.Ramp(x * 8, 1, tir.vscale() * 4), tvm.tir.Broadcast(4, tir.vscale() * 4)),
        ),
        TestCase(
            fld(tvm.tir.Ramp(x * 8, 3, 4), tvm.tir.Broadcast(4, 4)),
            fld(tvm.tir.Ramp(x * 8, 3, 4), tvm.tir.Broadcast(4, 4)),
        ),
        TestCase(
            fld(tvm.tir.Ramp(x * 8 + 15, 1, 4), tvm.tir.Broadcast(4, 4)),
            fld(tvm.tir.Ramp(x * 8 + 15, 1, 4), tvm.tir.Broadcast(4, 4)),
        ),
        TestCase(
            fld(tvm.tir.Ramp(x * 4, 1, 4), tvm.tir.Broadcast(64, 4)),
            tvm.tir.Broadcast(fld(x, 16), 4),
        ),
        TestCase(
            fld(tvm.tir.Ramp(x * 8, 2, 4), tvm.tir.Broadcast(64, 4)),
            tvm.tir.Broadcast(fld(x, 8), 4),
        ),
        TestCase(
            fld(tvm.tir.Ramp(x * 4, 1, 5), tvm.tir.Broadcast(64, 5)),
            fld(tvm.tir.Ramp(x * 4, 1, 5), tvm.tir.Broadcast(64, 5)),
        ),  # Example negative case: x = 15; [60, 61, 62, 63, 64] / 64 = [0, 0, 0, 0, 1]
        TestCase(
            fld(tvm.tir.Ramp(x * 4 + 3, 1, 4), tvm.tir.Broadcast(64, 4)),
            fld(tvm.tir.Ramp(x * 4 + 3, 1, 4), tvm.tir.Broadcast(64, 4)),
        ),  # Example negative case: x = 15; [63, 64, 65, 66] % 64 = [0, 1, 1, 1]
        TestCase(
            fld(tvm.tir.Ramp(x * 7, 1, 4), tvm.tir.Broadcast(64, 4)),
            fld(tvm.tir.Ramp(x * 7, 1, 4), tvm.tir.Broadcast(64, 4)),
        ),  # Example negative case: x = 9; [63, 70, 77, 84] % 64 = [0, 1, 1, 1]
        # floor mod
        TestCase(flm(y.astype("int32x2"), x.astype("int32x2")), flm(y, x).astype("int32x2")),
        TestCase(flm(tvm.tir.Ramp(x, 4, 4), 2), tvm.tir.Broadcast(flm(x, 2), 4)),
        TestCase(
            flm(tvm.tir.Ramp(x, 4, tir.vscale() * 8), 2),
            tvm.tir.Broadcast(flm(x, 2), tir.vscale() * 8),
        ),
        TestCase(flm(tvm.tir.Ramp(x * 8 + 1, 1, 4), 8), tvm.tir.Ramp(1, 1, 4)),
        TestCase(
            flm(tvm.tir.Ramp(x * 8 + 1, 1, tir.vscale() * 4), 8),
            flm(tvm.tir.Ramp(1, 1, tir.vscale() * 4), 8),
        ),
        TestCase(flm(tvm.tir.Ramp(x * 8 + 1, 15, 4), 8), flm(tvm.tir.Ramp(1, 15, 4), 8)),
        TestCase(
            flm(tvm.tir.Ramp(x, 8, 4), tvm.tir.Broadcast(4, 4)), tvm.tir.Broadcast(flm(x, 4), 4)
        ),
        TestCase(
            flm(tvm.tir.Ramp(x, 7, 4), tvm.tir.Broadcast(4, 4)),
            flm(tvm.tir.Ramp(x, 7, 4), tvm.tir.Broadcast(4, 4)),
        ),
        TestCase(flm(tvm.tir.Ramp(x * 8, 1, 4), tvm.tir.Broadcast(4, 4)), tvm.tir.Ramp(0, 1, 4)),
        TestCase(
            flm(tvm.tir.Ramp(x * 8, 1, 5), tvm.tir.Broadcast(4, 5)),
            flm(tvm.tir.Ramp(0, 1, 5), tvm.tir.Broadcast(4, 5)),
        ),
        TestCase(
            flm(tvm.tir.Ramp(x * 8 + 7, 1, 4), tvm.tir.Broadcast(4, 4)),
            flm(tvm.tir.Ramp(3, 1, 4), tvm.tir.Broadcast(4, 4)),
        ),
        TestCase(
            flm(tvm.tir.Ramp(x * 4, 1, 4), tvm.tir.Broadcast(64, 4)),
            tvm.tir.Ramp(flm(x * 4, 64), 1, 4),
        ),
        TestCase(
            flm(tvm.tir.Ramp(x * 8, 2, 4), tvm.tir.Broadcast(64, 4)),
            tvm.tir.Ramp(flm(x * 8, 64), 2, 4),
        ),
        TestCase(
            flm(tvm.tir.Ramp(x * 4, 1, 5), tvm.tir.Broadcast(64, 5)),
            flm(tvm.tir.Ramp(x * 4, 1, 5), tvm.tir.Broadcast(64, 5)),
        ),  # Example negative case: x = 15; [60, 61, 62, 63, 64] % 64 = [60, 61, 62, 63, 0]
        TestCase(
            flm(tvm.tir.Ramp(x * 4 + 3, 1, 4), tvm.tir.Broadcast(64, 4)),
            flm(tvm.tir.Ramp(x * 4 + 3, 1, 4), tvm.tir.Broadcast(64, 4)),
        ),  # Example negative case: x = 15; [63, 64, 65, 66] % 64 = [63, 0, 1, 2]
        TestCase(
            flm(tvm.tir.Ramp(x * 2, 1, 8), tvm.tir.Broadcast(20, 8)),
            flm(tvm.tir.Ramp(x * 2, 1, 8), tvm.tir.Broadcast(20, 8)),
        ),  # Example negative case: x = 9; [18, 19, 20, ..., 25] % 20 = [18, 19, 0, 1, ..., 5]
        TestCase(
            flm(tvm.tir.Ramp(x * 7, 1, 4), tvm.tir.Broadcast(64, 4)),
            flm(tvm.tir.Ramp(x * 7, 1, 4), tvm.tir.Broadcast(64, 4)),
        ),  # Example negative case: x = 9; [63, 70, 77, 84] % 64 = [63, 6, 13, 20]
        # Min/Max rules
        TestCase(
            tvm.te.min(y.astype("int32x2"), x.astype("int32x2")), tvm.te.min(y, x).astype("int32x2")
        ),
        TestCase(
            tvm.te.min(tvm.te.min(vx, y.astype("int32x2")), x.astype("int32x2")),
            tvm.te.min(vx, tvm.te.min(y, x).astype("int32x2")),
        ),
        TestCase(
            tvm.te.max(y.astype("int32x2"), x.astype("int32x2")), tvm.te.max(y, x).astype("int32x2")
        ),
        TestCase(
            tvm.te.max(tvm.te.max(vx, y.astype("int32x2")), x.astype("int32x2")),
            tvm.te.max(vx, tvm.te.max(y, x).astype("int32x2")),
        ),
        ## Logical rules
        TestCase(y.astype("int32x2").equal(x.astype("int32x2")), (y.equal(x)).astype("uint1x2")),
        TestCase(
            tvm.tir.NE(y.astype("int32x2"), (x.astype("int32x2"))),
            (tvm.tir.NE(y, x)).astype("uint1x2"),
        ),
        TestCase(y.astype("int32x2") > x.astype("int32x2"), (x < y).astype("uint1x2")),
        TestCase(y.astype("int32x2") >= x.astype("int32x2"), (x <= y).astype("uint1x2")),
        TestCase(y.astype("int32x2") < x.astype("int32x2"), (y < x).astype("uint1x2")),
        TestCase(y.astype("int32x2") <= x.astype("int32x2"), (y <= x).astype("uint1x2")),
        TestCase(
            tvm.tir.And(y.astype("int32x2") <= x.astype("int32x2"), vc.astype("uint1x2")),
            (tvm.tir.And(y <= x, vc)).astype("uint1x2"),
        ),
        TestCase(
            tvm.tir.Or(y.astype("int32x2") <= x.astype("int32x2"), vc.astype("uint1x2")),
            (tvm.tir.Or(y <= x, vc)).astype("uint1x2"),
        ),
    )


class TestSelect(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    test_case = tvm.testing.parameter(
        # Add rules
        TestCase(
            tvm.tir.Select(x < 0, y, 0) + tvm.tir.Select(x < 0, 1, z),
            tvm.tir.Select(x < 0, y + 1, z),
        ),
        TestCase(
            tvm.tir.Select(x < 0, y, 1) - tvm.tir.Select(x < 0, 1, z),
            tvm.tir.Select(x < 0, y + (-1), 1 - z),
        ),
        TestCase(tvm.tir.Select(x < 0, y, z) - y, tvm.tir.Select(x < 0, 0, z - y)),
        TestCase(tvm.tir.Select(x < 0, y, z) - z, tvm.tir.Select(x < 0, y - z, 0)),
        TestCase(
            tvm.te.min(tvm.tir.Select(x < 0, y, 0), tvm.tir.Select(x < 0, 1, z)),
            tvm.tir.Select(x < 0, tvm.te.min(y, 1), tvm.te.min(0, z)),
        ),
        TestCase(
            tvm.te.max(tvm.tir.Select(x < 0, y, 0), tvm.tir.Select(x < 0, 1, z)),
            tvm.tir.Select(x < 0, tvm.te.max(y, 1), tvm.te.max(0, z)),
        ),
        TestCase(tvm.tir.Select(x * 3 + 1 != 0, y, z), y),
        TestCase(tvm.tir.Select(x * 3 + 1 == 0, y, z), z),
        TestCase(tvm.tir.Select(x > 0, y + 1, y + 1), y + 1),
    )


class TestCancellation(BaseCompare):
    var_int8 = tir.Var("var_int8", "int8")
    var_int32 = tir.Var("var_int32", "int32")
    var_int64 = tir.Var("var_int64", "int64")
    var_uint8 = tir.Var("var_uint8", "uint8")
    var_uint32 = tir.Var("var_uint32", "uint32")
    var_uint64 = tir.Var("var_uint64", "uint64")

    test_case = tvm.testing.parameter(
        TestCase(tir.const(5, "int64") - tir.const(5, "int64"), tir.const(0, "int64")),
        TestCase(tir.const(5, "uint8") - tir.const(5, "uint8"), tir.const(0, "uint8")),
        TestCase(var_int8 - var_int8, tir.const(0, "int8")),
        TestCase(var_int32 - var_int32, tir.const(0, "int32")),
        TestCase(var_int64 - var_int64, tir.const(0, "int64")),
        TestCase(var_uint8 - var_uint8, tir.const(0, "uint8")),
        TestCase(var_uint32 - var_uint32, tir.const(0, "uint32")),
        TestCase(var_uint64 - var_uint64, tir.const(0, "uint64")),
        TestCase(tir.EQ(tir.const(5, "int64"), tir.const(5, "int64")), tir.const(True, "bool")),
        TestCase(tir.EQ(tir.const(5, "uint8"), tir.const(5, "uint8")), tir.const(True, "bool")),
        TestCase(tir.EQ(var_int8, var_int8), tir.const(True, "bool")),
        TestCase(tir.EQ(var_int32, var_int32), tir.const(True, "bool")),
        TestCase(tir.EQ(var_int64, var_int64), tir.const(True, "bool")),
        TestCase(tir.EQ(var_uint8, var_uint8), tir.const(True, "bool")),
        TestCase(tir.EQ(var_uint32, var_uint32), tir.const(True, "bool")),
        TestCase(tir.EQ(var_uint64, var_uint64), tir.const(True, "bool")),
        TestCase(tir.NE(tir.const(5, "int64"), tir.const(5, "int64")), tir.const(False, "bool")),
        TestCase(tir.NE(tir.const(5, "uint8"), tir.const(5, "uint8")), tir.const(False, "bool")),
        TestCase(tir.NE(var_int8, var_int8), tir.const(False, "bool")),
        TestCase(tir.NE(var_int32, var_int32), tir.const(False, "bool")),
        TestCase(tir.NE(var_int64, var_int64), tir.const(False, "bool")),
        TestCase(tir.NE(var_uint8, var_uint8), tir.const(False, "bool")),
        TestCase(tir.NE(var_uint32, var_uint32), tir.const(False, "bool")),
        TestCase(tir.NE(var_uint64, var_uint64), tir.const(False, "bool")),
    )


class TestAddIndex(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    test_case = tvm.testing.parameter(
        TestCase(x + (y - x), y),
        TestCase(x - (y + 1) + (y + 1), x),
        TestCase((x - 10) + (10 - z), x - z),
        TestCase((x - y) + (z - x), z - y),
        TestCase(tvm.te.min(x, y - z) + z, tvm.te.min(x + z, y)),
        TestCase(tvm.te.min(x - z, y) + z, tvm.te.min(x, y + z)),
        TestCase(tvm.te.max(x, y - 10) + 10, tvm.te.max(x + 10, y)),
        TestCase(tvm.te.max(x - 11, y) + 11, tvm.te.max(x, y + 11)),
        TestCase(tvm.te.max(x, y * 2) + tvm.te.min(x, y * 2), x + y * 2),
        TestCase(tvm.te.min(x, y * 2) + tvm.te.max(x, y * 2), x + y * 2),
        TestCase(tvm.te.max(x, y + 2) + (-2), tvm.te.max(x + (-2), y)),
        TestCase(tvm.te.min(x, y + 2) + (-2), tvm.te.min(x + (-2), y)),
        TestCase(tvm.te.min(x + 2, y + 3) + (-2), tvm.te.min(x, y + 1)),
        TestCase(tvm.te.max(0, 1 - x * 4) + x * 4, tvm.te.max(x * 4, 1)),
        TestCase(tvm.te.max(2 - x * 4, 0) + x * 4, tvm.te.max(x * 4, 2)),
        TestCase(tvm.te.min(0, 1 - x * 4) + x * 4, tvm.te.min(x * 4, 1)),
        TestCase(tvm.te.min(2 - x * 4, 0) + x * 4, tvm.te.min(x * 4, 2)),
        TestCase(x * y + x * 10, x * (y + 10)),
        TestCase(y * x + x * 10, x * (y + 10)),
        TestCase(y * x + 10 * x, x * (y + 10)),
        TestCase(x * y + 10 * x, x * (y + 10)),
        TestCase((2 * z) + tvm.te.min(x, y - (2 * z)), tvm.te.min(x + (z * 2), y)),
        TestCase(y * x + x, x * (y + 1)),
        TestCase(x * y + x, x * (y + 1)),
        TestCase((x + 10) + 13, x + 23),
        TestCase((x + 10) + (13 + z), x + z + 23),
        TestCase(x * y + 10 * x, x * (y + 10)),
        TestCase(y * x + x * 3, x * (y + 3)),
        TestCase(x + 3 + y, x + y + 3),
        TestCase((3 - y) + x, x - y + 3),
        # canonicalization
        TestCase(x + 2 + 3 + 4 + x, x * 2 + 9),
        TestCase(x + 2 + 3 + 4 + x * 3, x * 4 + 9),
        # DivMod rules
        # trunc div
        TestCase(y * tmod(x, 8) + 10 * tmod(x, 8), tmod(x, 8) * (y + 10)),
        TestCase(tdiv(x, 8) * 8 + tmod(x, 8), x),
        # floor div
        TestCase(y * flm(x, 8) + 10 * flm(x, 8), flm(x, 8) * (y + 10)),
        TestCase(fld(x, 8) * 8 + flm(x, 8), x),
        TestCase(fld(flm(x, 2) + 7, 2) + fld(x, 2), fld(x + 7, 2)),
    )


class TestSubIndex(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    a, b = tvm.tir.Any(), tvm.tir.Any()

    test_case = tvm.testing.parameter(
        TestCase(x + y - y, x),
        TestCase(x + y - x, y),
        TestCase(x - (y + x), 0 - y),
        TestCase(x - (x + y), 0 - y),
        TestCase(tvm.te.min(x, y) - x, tvm.te.min(0, y - x)),
        TestCase(tvm.te.min(x, y) - y, tvm.te.min(x - y, 0)),
        TestCase(tvm.te.max(x, y) - x, tvm.te.max(0, y - x)),
        TestCase(tvm.te.max(x, y) - y, tvm.te.max(x - y, 0)),
        TestCase(x - tvm.te.min(x, y), tvm.te.max(0, x - y)),
        TestCase(y - tvm.te.min(x, y), tvm.te.max(y - x, 0)),
        TestCase(x - tvm.te.max(x, y), tvm.te.min(0, x - y)),
        TestCase(y - tvm.te.max(x, y), tvm.te.min(y - x, 0)),
        # mul co-efficient foldng
        TestCase(x - x, 0),
        TestCase(a - a, 0),
        TestCase(a - b, a - b),
        TestCase(x * y - x, x * (y + (-1))),
        TestCase(x * y - 10 * x, x * (y + (-10))),
        TestCase(y * x - x * z, x * (y - z)),
        TestCase(y * x - z * x, x * (y - z)),
        TestCase(x + 10 - 20, x + (-10)),
        # 4-operands pattern
        TestCase((x + y) - (x + z), y - z),
        TestCase((y + x) - (x + z), y - z),
        TestCase((x + y) - (z + x), y - z),
        TestCase((y + x) - (z + x), y - z),
        TestCase(tvm.te.min(x + y, z) - x, tvm.te.min(y, z - x)),
        TestCase(tvm.te.min(y + x, z) - x, tvm.te.min(y, z - x)),
        TestCase(tvm.te.min(z, x + y) - x, tvm.te.min(z - x, y)),
        TestCase(tvm.te.min(z, y + x) - x, tvm.te.min(z - x, y)),
        TestCase(tvm.te.max(x + y, z) - x, tvm.te.max(y, z - x)),
        TestCase(tvm.te.max(y + x, z) - x, tvm.te.max(y, z - x)),
        TestCase(tvm.te.max(z, x + y) - x, tvm.te.max(z - x, y)),
        TestCase(tvm.te.max(z, y + x) - x, tvm.te.max(z - x, y)),
        TestCase(x - tvm.te.min(x + y, z), tvm.te.max(0 - y, x - z)),
        TestCase(x - tvm.te.min(y + x, z), tvm.te.max(0 - y, x - z)),
        TestCase(x - tvm.te.min(z, x + y), tvm.te.max(x - z, 0 - y)),
        TestCase(x - tvm.te.min(z, y + x), tvm.te.max(x - z, 0 - y)),
        TestCase(tvm.te.min(x, y) - tvm.te.min(y, x), 0),
        TestCase(tvm.te.max(x, y) - tvm.te.max(y, x), 0),
        TestCase(tvm.te.min(x, y) - tvm.te.min(x + 10, y + 10), -10),
        TestCase(tvm.te.min(x + 10, y + 1) - tvm.te.min(x, y - 9), 10),
        TestCase(x - tvm.te.max(x + y, 0), tvm.te.min(0 - y, x)),
        TestCase(x - tvm.te.max(0, x + y), tvm.te.min(x, 0 - y)),
        TestCase(x - tvm.te.min(x + y, 0), tvm.te.max(0 - y, x)),
        TestCase(x - tvm.te.min(0, x + y), tvm.te.max(x, 0 - y)),
        # DivMod patterns
        # truc div
        TestCase(x - tdiv(x, 3) * 3, tmod(x, 3)),
        TestCase(tdiv(x + 5, 3) - tdiv(x, 3), tdiv(tmod(x, 3) + 5, 3), x >= 0),
        TestCase(tdiv(x + 5, 3) - tdiv(x + 1, 3), tdiv(tmod(x + 1, 3) + 4, 3), x >= -1),
        TestCase(y - tdiv(y, (-5)) * (-5), tmod(y, 5)),
        TestCase(tdiv(y, 3) * 3 - y, 0 - tmod(y, 3)),
        TestCase(y - tdiv(y - 6, 5) * 5, tmod(y + (-6), 5) + 6),
        TestCase(tdiv(y - 6, 5) * 5 - y, (-6) - tmod(y + (-6), 5)),
        TestCase(y - tdiv(y + z, 5) * 5, tmod(y + z, 5) - z),
        TestCase(tdiv(y + z, 5) * 5 - y, z - tmod(y + z, 5)),
        TestCase(y - tdiv(y - z, 5) * 5, tmod(y - z, 5) + z),
        TestCase(tdiv(y - z, 5) * 5 - y, 0 - tmod(y - z, 5) - z),
        TestCase(y * 3 - tdiv(y, 2) * 6, tmod(y, 2) * 3),
        TestCase(tdiv(y, 3) * 6 - y * 2, tmod(y, 3) * (-2)),
        TestCase(y * 5 - tdiv(y + z, 2) * 10, (tmod(y + z, 2) - z) * 5),
        TestCase(y * 5 - tdiv(y - z, 2) * 10, (tmod(y - z, 2) + z) * 5),
        TestCase(tdiv(y + z, 3) * 6 - y * 2, (z - tmod(y + z, 3)) * 2),
        TestCase(tdiv(y - z, 3) * 6 - y * 2, (0 - tmod(y - z, 3) - z) * 2),
        TestCase(5 * y - tdiv(y + z, 2) * 10, (tmod(y + z, 2) - z) * 5),
        TestCase(5 * y - 10 * tdiv(y - z, 2), (tmod(y - z, 2) + z) * 5),
        TestCase(6 * tdiv(y + z, 3) - y * 2, (z - tmod(y + z, 3)) * 2),
        TestCase(tdiv(y - z, 3) * 6 - 2 * y, (0 - tmod(y - z, 3) - z) * 2),
        # floor div
        TestCase(x - fld(x, 3) * 3, flm(x, 3)),
        TestCase(fld(x + 5, 3) - fld(x, 3), fld(flm(x, 3) + 5, 3)),
        TestCase(fld(x + 5, 3) - fld(x + 2, 3), fld(flm(x + 2, 3), 3) + 1),
        TestCase(fld(y, 3) * 3 - y, 0 - flm(y, 3)),
        TestCase(y - fld(y - 6, 5) * 5, flm(y + 4, 5) + 6),
        TestCase(fld(y - 6, 5) * 5 - y, (-6) - flm(y + 4, 5)),
        TestCase(y - fld(y + z, 5) * 5, flm(y + z, 5) - z),
        TestCase(fld(y + z, 5) * 5 - y, z - flm(y + z, 5)),
        TestCase(y - fld(y - z, 5) * 5, flm(y - z, 5) + z),
        TestCase(fld(y - z, 5) * 5 - y, 0 - flm(y - z, 5) - z),
        TestCase(y * 3 - fld(y, 2) * 6, flm(y, 2) * 3),
        TestCase(fld(y, 3) * 6 - y * 2, flm(y, 3) * (-2)),
        TestCase(y * 5 - fld(y + z, 2) * 10, (flm(y + z, 2) - z) * 5),
        TestCase(y * 5 - fld(y - z, 2) * 10, (flm(y - z, 2) + z) * 5),
        TestCase(fld(y + z, 3) * 6 - y * 2, (z - flm(y + z, 3)) * 2),
        TestCase(fld(y - z, 3) * 6 - y * 2, (0 - flm(y - z, 3) - z) * 2),
        TestCase(5 * y - fld(y + z, 2) * 10, (flm(y + z, 2) - z) * 5),
        TestCase(5 * y - 10 * fld(y - z, 2), (flm(y - z, 2) + z) * 5),
        TestCase(6 * fld(y + z, 3) - y * 2, (z - flm(y + z, 3)) * 2),
        TestCase(fld(y - z, 3) * 6 - 2 * y, (0 - flm(y - z, 3) - z) * 2),
    )


class TestMulIndex(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    test_case = tvm.testing.parameter(
        TestCase((x + 2) * 3, x * 3 + 6),
        TestCase((x * 2) * 3, x * 6),
        TestCase(tvm.te.min(x, y) * tvm.te.max(x, y), x * y),
        TestCase(tvm.te.max(x, y) * tvm.te.min(x, y), x * y),
        TestCase((x - y) * (-2), (y - x) * 2),
    )


class TestDivIndex(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    non_negative = [x >= 0, y >= 0, z >= 0]

    test_case = tvm.testing.parameter(
        TestCase(tdiv(x, x), 1),
        TestCase(tdiv(tdiv(x, 2), 3), tdiv(x, 6)),
        TestCase(tdiv(tdiv(x, 2) + 1, 3), tdiv(x + 2, 6), non_negative),
        TestCase(tdiv(x * 2, 4), tdiv(x, 2)),
        TestCase(tdiv(x * 4, 2), x * 2),
        TestCase(tdiv(x * 4 + y, 2), x * 2 + tdiv(y, 2), non_negative),
        TestCase(tdiv(tvm.te.min(x * 6, y), 2), tvm.te.min(x * 3, tdiv(y, 2)), non_negative),
        TestCase(tdiv(tvm.te.max(x * 6, y), 2), tvm.te.max(x * 3, tdiv(y, 2)), non_negative),
        TestCase(tdiv(y + x * 4, 2), tdiv(y, 2) + x * 2, non_negative),
        TestCase(tdiv(tvm.te.min(y, x * 6), 2), tvm.te.min(tdiv(y, 2), x * 3), non_negative),
        TestCase(tdiv(tvm.te.max(y, x * 6), 2), tvm.te.max(tdiv(y, 2), x * 3), non_negative),
        # 3-operands
        TestCase(tdiv(x * 6 + y + z, 2), x * 3 + tdiv(y + z, 2), non_negative),
        TestCase(tdiv(x * 6 - y + (y + 3), 2), x * 3 + 1, non_negative),
        TestCase(tdiv(x * 6 + (y + 3) - y, 2), x * 3 + 1, non_negative),
        TestCase(tdiv(y + x * 6 + z, 2), x * 3 + tdiv(y + z, 2), non_negative),
        TestCase(tdiv(x + 4, 2), tdiv(x, 2) + 2, non_negative),
        TestCase(tdiv(x + y, x), tdiv(y, x) + 1, non_negative),
        TestCase(tdiv(y + x, x), tdiv(y, x) + 1, non_negative),
        TestCase(tdiv((x + y) + z, x), tdiv(y + z, x) + 1, non_negative),
        TestCase(tdiv((y + x) + z, x), tdiv(y + z, x) + 1, non_negative),
        TestCase(tdiv(y + (x + z), x), tdiv(y + z, x) + 1, non_negative),
        TestCase(tdiv(y + (z + x), x), tdiv(y + z, x) + 1, non_negative),
        TestCase(tdiv(x * y, y), x, non_negative),
        TestCase(tdiv(y * x, y), x, non_negative),
        TestCase(tdiv(x * z + y, z), x + tdiv(y, z), non_negative),
        TestCase(tdiv(z * x + y, z), x + tdiv(y, z), non_negative),
        TestCase(tdiv(y + x * z, z), tdiv(y, z) + x, non_negative),
        TestCase(tdiv(y + z * x, z), tdiv(y, z) + x, non_negative),
    )


class TestFloordivIndex(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    test_case = tvm.testing.parameter(
        TestCase(fld(fld(x, 2), 3), fld(x, 6)),
        TestCase(fld(fld(x, 2) + 1, 3), fld(x + 2, 6)),
        TestCase(fld(x - flm(x, 21), 21), fld(x, 21)),
        TestCase(fld(x * 2, 4), fld(x, 2)),
        TestCase(fld(x * 4, 2), x * 2),
        TestCase(fld(x * 8 + 7, 16), fld(x, 2)),
        TestCase(fld(x * 8 + 39, 16), fld(x, 2) + 2),
        TestCase(fld(x * 8 - 1, 16), fld(x * 8 + -1, 16)),
        TestCase(fld(x * 8 - 9, 16), fld(x, 2) + -1),
        # TODO(Lunderberg): Remove the necessity for the preconditions
        # in this section.  They shouldn't be necessary for floordiv,
        # where they would be required for truncdiv.
        TestCase(fld(x * 360 + y, 16), x * 22, [x >= 0, x < 2, y >= 0, y < 7]),
        TestCase(fld(x * 360 + y, 25), x * 14, [x >= 0, x < 2, y >= 0, y < 7]),
        TestCase(fld(x * 360 - 8, 25), fld(x * 360 + -8, 25)),
        TestCase(fld(x * 4 + y, 2), x * 2 + fld(y, 2)),
        TestCase(fld(tvm.te.min(x * 6, y), 2), tvm.te.min(x * 3, fld(y, 2))),
        TestCase(fld(tvm.te.max(x * 6, y), 2), tvm.te.max(x * 3, fld(y, 2))),
        TestCase(fld(y + x * 4, 2), x * 2 + fld(y, 2)),
        TestCase(fld(tvm.te.min(y, x * 6), 2), tvm.te.min(fld(y, 2), x * 3)),
        TestCase(fld(tvm.te.max(y, x * 6), 2), tvm.te.max(fld(y, 2), x * 3)),
        # 3-operands
        #
        # TODO(Lunderberg): Remove the necessity for the preconditions
        # in this section.  They shouldn't be required, since floordiv
        # has translational symmetry, even for negative.
        TestCase(fld(x * 6 + y + z, 2), x * 3 + fld(y + z, 2)),
        TestCase(fld(x * 6 - y + (y + 3), 2), x * 3 + 1),
        TestCase(fld(x * 6 + (y + 3) - y, 2), x * 3 + 1),
        TestCase(fld(y + x * 6 + z, 2), x * 3 + fld(y + z, 2)),
        TestCase(fld(x + 4, 2), fld(x, 2) + 2),
        TestCase(fld(x + y, x), fld(y, x) + 1, x >= 0),
        TestCase(fld(y + x, x), fld(y, x) + 1, x >= 0),
        TestCase(fld((x + y) + z, x), fld(y + z, x) + 1, x >= 0),
        TestCase(fld((y + x) + z, x), fld(y + z, x) + 1, x >= 0),
        TestCase(fld(y + (x + z), x), fld(y + z, x) + 1, x >= 0),
        TestCase(fld(y + (z + x), x), fld(y + z, x) + 1, x >= 0),
        TestCase(fld(x * y, y), x, y >= 0),
        TestCase(fld(y * x, y), x, y >= 0),
        TestCase(fld(x * z + y, z), x + fld(y, z), z >= 0),
        TestCase(fld(x * z * 2 + y, z * 2), x + fld(y, z * 2), z * 2 >= 0),
        TestCase(fld(z * x + y, z), x + fld(y, z), z >= 0),
        TestCase(fld(y + x * z, z), fld(y, z) + x, z >= 0),
        TestCase(fld(y + z * x, z), fld(y, z) + x, z >= 0),
        TestCase(fld(x * 32 + y, 64), fld(x, 2), [y >= 0, y < 32]),
        TestCase(fld(x * 128 + y * 4 + z, 512), fld(x, 4), [y >= 0, y < 32, z >= 0, z < 4]),
    )


class TestModIndex(BaseCompare):
    x, y, nx, ny, z = te.var("x"), te.var("y"), te.var("nx"), te.var("ny"), te.var("z")

    test_case = tvm.testing.parameter(
        # TODO(Lunderberg): Loosen these preconditions.  When there's
        # a single term whose factor is divisible by the denominator,
        # the sign of the argument doesn't matter.
        TestCase(tmod(x * 10, 2), 0, x >= 0),
        TestCase(tmod(x * 10 + y, 2), tmod(y, 2), [x >= 0, y >= 0]),
        TestCase(tmod(x + 10, 2), tmod(x, 2), x >= 0),
        TestCase(tmod(x + y * 10, 2), tmod(x, 2), [x >= 0, y >= 0]),
        TestCase(tmod(x * 10 + 1 + y * 2 + 2, 2), 1, [x >= 0, y >= 0]),
        TestCase(tmod(x * 10, -2), 0, x <= 0),
        TestCase(tmod(x * 10 + y, -2), tmod(y, 2), [x >= 0, y >= 0]),
        TestCase(tmod(x + 10, -2), tmod(x, 2), x >= 0),
        TestCase(tmod(x + y * 10, -2), tmod(x, 2), [x >= 0, y >= 0]),
        TestCase(tmod(x * 10 + 1 + y * 2 + 2, -2), 1, [x >= 0, y >= 0]),
        TestCase(tmod(x * (-10), 2), 0),
        TestCase(tmod(x * (-10) + y, 2), tmod(x * (-10) + y, 2)),
        TestCase(tmod(x + (-10), 2), tmod(x + (-10), 2)),
        TestCase(tmod(x + y * (-10), 2), tmod(x + y * (-10), 2)),
        TestCase(tmod(x * (-10), -2), 0),
        TestCase(tmod(nx * 10, 2), 0),
        TestCase(tmod(nx * (-10) + y, 2), tmod(y, 2), [nx <= 0, y >= 0]),
        TestCase(tmod(x + ny * (-10), 2), tmod(x, 2), [x >= 0, ny <= 0]),
        TestCase(tmod(nx * (-10) + 1 + ny * (-2) + 2, 2), 1, [nx <= 0, ny <= 0]),
        TestCase(tmod(nx * 10, -2), 0),
        TestCase(tmod(nx * (-10) + y, -2), tmod(y, 2), [nx <= 0, y >= 0]),
        TestCase(tmod(x + ny * (-10), -2), tmod(x, 2), [x >= 0, ny <= 0]),
    )


class TestFloormodIndex(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    test_case = tvm.testing.parameter(
        TestCase(flm(x * 10, 2), 0),
        TestCase(flm(x * 9600, 6400), flm(x * 3200, 6400)),
        TestCase(flm(x * 10 + y, 2), flm(y, 2)),
        TestCase(flm(x * 360 + y, 16), flm(x * 8 + y, 16)),
        TestCase(flm(x + 10, 2), flm(x, 2)),
        TestCase(flm(x + y * 10, 2), flm(x, 2)),
        TestCase(flm(x + y * 360, 16), flm(x + y * 8, 16)),
        TestCase(flm(x * (-10), 2), 0),
        TestCase(flm(x * (-10) + y, 2), flm(y, 2)),
        TestCase(flm(x + (-10), 2), flm(x, 2)),
        TestCase(flm(x + y * (-10), 2), flm(x, 2)),
        TestCase(flm(x * 32 + y, 64), flm(x, 2) * 32 + y, [y >= 0, y < 32]),
        TestCase(flm(x * 32 - y, 64), flm(x * 32 - y, 64), [y >= 0, y < 32]),
        TestCase(flm(x * z * 2 + y, z * 2), flm(y, z * 2), z * 2 >= 0),
        # NOTE: the followng case is covered by canonical simplify
        # long range simplifcation in general can be covered by canonical simplify
        # TestCase(flm(x * 10 + 1 + y * 2 + 2, 2), 1),
    )


class TestFloorModTwo(BaseCompare):
    """Special-case simplifications for FloorMod(expr,2)

    Because FloorMod(expr,2) has only two possible values, it can be
    simplified more aggressively than most FloorMod expressions.  Some
    of these have analogues for other denominators (e.g. x%3 + (x+1)%3
    + (x+2)%3 == 0 + 1 + 2), but they don't appear as often and
    require identifying more related terms in order to apply.

    (x + c1)//2 - (x+c2)//2 => (x%2)*( c1%2 - c1%2 ) + (c1//2 - c2//2)

    We should not introduce extra negative coeficient to iterators
    however during simplification
    """

    x, y, z = te.var("x"), te.var("y"), te.var("z")
    test_case = tvm.testing.parameter(
        # Removing offsets from floormod
        TestCase(flm(x, 2) + flm(x + 1, 2), 1),
        TestCase(flm(x + 1, 2) + flm(x, 2), 1),
        # Difference of floordiv yields floormod
        TestCase(fld(x + 1, 2) - fld(x, 2), flm(x, 2)),
        TestCase(fld(x, 2) - fld(x - 1, 2), flm(x, 2) * -1 + 1),
        TestCase(fld(x + 5, 2) - fld(x - 2, 2), flm(x, 2) + 3),
        TestCase(fld(x + 5, 2) - fld(x - 3, 2), 4),
        TestCase(fld(flm(x, 2) + 1, 2), flm(x, 2)),
        # Sum of floordiv and floormod to yield floordiv
        TestCase(fld(x + 1, 2) - flm(x, 2), fld(x, 2)),
        TestCase(fld(x, 2) + flm(x, 2), fld(x + 1, 2)),
        # regression: although we can rewrite (x + 1) %2 => 1 - x%2
        # doing so would introduce negative co-efficient to iterators
        # which makes later iter map detection harder, in principle we
        # should not introduce additional negative signs of iterator in rewriting
        TestCase(flm(x + 1, 2), flm(x + 1, 2)),
        TestCase(flm(x + 5, 2), flm(x + 1, 2)),
        TestCase(flm(x + 1, 2) * 8192, flm(x + 1, 2) * 8192, [x >= 0, x < 2]),
    )


class TestFloorModPadded(BaseCompare):
    """Special-case simplifications for divisibility proof
    such that (x - x % k) must be divisible by k
    """

    x, y = te.var("x"), te.var("y")
    test_case = tvm.testing.parameter(
        TestCase(flm(x - flm(x, 9), 9), 0),
        TestCase(flm(x - flm(x, -9), 9), 0),
        TestCase(flm(x + flm(-x, 9), 9), 0),
        TestCase(flm(x + flm(8 * x, 9), 9), 0),
        TestCase(flm(x - flm(x, y), y), 0),
        TestCase(flm(x - flm(x, -y), y), 0),
        TestCase(flm(x + flm(-x, y), y), 0),
    )


class TestMinIndex(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    test_case = tvm.testing.parameter(
        # const int bound
        TestCase(tvm.te.min(tmod(x, 2), tmod(y, 2) + 10), tmod(x, 2)),
        TestCase(tvm.te.min(flm(x, 2), flm(y, 2) + 10), flm(x, 2)),
        TestCase(tvm.te.min(x + 1, x + 10), x + 1),
        TestCase(tvm.te.min(x + 111, x + 10), x + 10),
        TestCase(tvm.te.min(x + 1, x), x),
        TestCase(tvm.te.min(x, x + 2), x),
        TestCase(tvm.te.min(1 - x, 2 - x), 1 - x),
        TestCase(tvm.te.min(3 - x, 2 - x), 2 - x),
        TestCase(tvm.te.min(tvm.te.max(x, y), tvm.te.min(x, y)), tvm.te.min(x, y)),
        TestCase(tvm.te.min(tvm.te.max(x, y), tvm.te.min(y, x)), tvm.te.min(x, y)),
        TestCase(tvm.te.min(tvm.te.max(x, y), x), x),
        TestCase(tvm.te.min(tvm.te.max(y, x), x), x),
        TestCase(tvm.te.min(tvm.te.min(x, y), x), tvm.te.min(x, y)),
        TestCase(tvm.te.min(tvm.te.min(x, y), y), tvm.te.min(x, y)),
        TestCase(tvm.te.min(x, tvm.te.max(x, y)), x),
        TestCase(tvm.te.min(x, tvm.te.max(y, x)), x),
        TestCase(tvm.te.min(x, tvm.te.min(x, y)), tvm.te.min(x, y)),
        TestCase(tvm.te.min(y, tvm.te.min(x, y)), tvm.te.min(x, y)),
        TestCase(tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), y), tvm.te.min(tvm.te.min(x, y), z)),
        TestCase(
            tvm.te.min(tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), x * 2), y),
            tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), x * 2),
        ),
        TestCase(
            tvm.te.min(tvm.te.min(tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), x * 2), z * 2), y),
            tvm.te.min(tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), x * 2), z * 2),
        ),
        TestCase(tvm.te.min(tvm.te.max(x, y), tvm.te.max(x, z)), tvm.te.max(tvm.te.min(y, z), x)),
        TestCase(tvm.te.min(tvm.te.max(x, y), tvm.te.max(z, x)), tvm.te.max(tvm.te.min(y, z), x)),
        TestCase(tvm.te.min(tvm.te.max(y, x), tvm.te.max(x, z)), tvm.te.max(tvm.te.min(y, z), x)),
        TestCase(tvm.te.min(tvm.te.max(y, x), tvm.te.max(z, x)), tvm.te.max(tvm.te.min(y, z), x)),
        TestCase(tvm.te.min(y + x, z + x), tvm.te.min(y, z) + x),
        TestCase(tvm.te.min(y + x, x + z), tvm.te.min(y, z) + x),
        TestCase(tvm.te.min(x + y, z + x), tvm.te.min(y, z) + x),
        TestCase(tvm.te.min(x + y, x + z), tvm.te.min(y, z) + x),
        TestCase(tvm.te.min(x - y, x - z), x - tvm.te.max(y, z)),
        TestCase(tvm.te.min(y - x, z - x), tvm.te.min(y, z) - x),
        TestCase(tvm.te.min(tvm.te.min(x, 1), 10), tvm.te.min(x, 1)),
        TestCase(tvm.te.min(tvm.te.min(x, 11), 10), tvm.te.min(x, 10)),
        TestCase(tvm.te.min(x * 3, 9), tvm.te.min(x, 3) * 3),
        TestCase(tvm.te.min(x * 2, 0), tvm.te.min(x, 0) * 2),
        TestCase(tvm.te.min(0 - x * 2, 0), tvm.te.max(x, 0) * -2),
        TestCase(tvm.te.min(3 - x, 2), 3 - tvm.te.max(x, 1)),
        TestCase(tvm.te.min(x * (-2), -4), tvm.te.max(x, 2) * -2),
        TestCase(tvm.te.min(x * (-2), 4), tvm.te.max(x, -2) * -2),
        TestCase(tvm.te.min(x * (0), 4), 0),
        TestCase(tvm.te.min(x * (0), -4), -4),
        # DivMod rules
        # truc div
        TestCase(tvm.te.min(tdiv(x + 3, 4) * 4, x), x),
        TestCase(tvm.te.min(x, tdiv(x + 3, 4) * 4), x),
        TestCase(tvm.te.min(tdiv(x + 3, 4) * 4, tvm.te.max(x, 4)), tvm.te.max(x, 4), x > 0),
        TestCase(tvm.te.min(tvm.te.max(x, 4), tdiv(x + 3, 4) * 4), tvm.te.max(x, 4), x > 0),
        TestCase(tvm.te.min(tdiv(x, 10), tdiv(y, 10)), tdiv(tvm.te.min(x, y), 10)),
        TestCase(tvm.te.min(tdiv(x, (-10)), tdiv(y, (-10))), tdiv(tvm.te.max(x, y), (-10))),
        # floor div
        TestCase(tvm.te.min(fld(x + 3, 4) * 4, x), x),
        TestCase(tvm.te.min(x, fld(x + 3, 4) * 4), x),
        TestCase(tvm.te.min(x, fld(x, 4) * 4), fld(x, 4) * 4),
        TestCase(tvm.te.min(fld(x + 3, 4) * 4, tvm.te.max(x, 4)), tvm.te.max(x, 4), x > 0),
        TestCase(tvm.te.min(tvm.te.max(x, 4), fld(x + 3, 4) * 4), tvm.te.max(x, 4), x > 0),
        TestCase(tvm.te.min(fld(x, 10), fld(y, 10)), fld(tvm.te.min(x, y), 10)),
        TestCase(tvm.te.min(fld(x, (-10)), fld(y, (-10))), fld(tvm.te.max(x, y), (-10))),
    )


class TestMaxIndex(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    test_case = tvm.testing.parameter(
        # const int bound
        TestCase(tvm.te.max(tmod(x, 2), tmod(y, 2) + 10), tmod(y, 2) + 10),
        TestCase(tvm.te.max(flm(x, 2), flm(y, 2) + 10), flm(y, 2) + 10),
        TestCase(tvm.te.max(x + 1, x + 10), x + 10),
        TestCase(tvm.te.max(x + 111, x + 10), x + 111),
        TestCase(tvm.te.max(x + 1, x), x + 1),
        TestCase(tvm.te.max(x, x + 2), x + 2),
        TestCase(tvm.te.max(1 - x, 2 - x), 2 - x),
        TestCase(tvm.te.max(3 - x, 2 - x), 3 - x),
        TestCase(tvm.te.max(tvm.te.min(x, y), tvm.te.max(x, y)), tvm.te.max(x, y)),
        TestCase(tvm.te.max(tvm.te.min(x, y), tvm.te.max(y, x)), tvm.te.max(x, y)),
        TestCase(tvm.te.max(tvm.te.min(x, y), x), x),
        TestCase(tvm.te.max(tvm.te.min(y, x), x), x),
        TestCase(tvm.te.max(tvm.te.max(x, y), x), tvm.te.max(x, y)),
        TestCase(tvm.te.max(tvm.te.max(x, y), y), tvm.te.max(x, y)),
        TestCase(tvm.te.max(x, tvm.te.min(x, y)), x),
        TestCase(tvm.te.max(x, tvm.te.min(y, x)), x),
        TestCase(tvm.te.max(x, tvm.te.max(x, y)), tvm.te.max(x, y)),
        TestCase(tvm.te.max(y, tvm.te.max(x, y)), tvm.te.max(x, y)),
        TestCase(tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), y), tvm.te.max(tvm.te.max(x, y), z)),
        TestCase(
            tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), x * 2), y),
            tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), x * 2),
        ),
        TestCase(
            tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), x * 2), z * 2), y),
            tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), x * 2), z * 2),
        ),
        TestCase(tvm.te.max(tvm.te.min(x, y), tvm.te.min(x, z)), tvm.te.min(tvm.te.max(y, z), x)),
        TestCase(tvm.te.max(tvm.te.min(x, y), tvm.te.min(z, x)), tvm.te.min(tvm.te.max(y, z), x)),
        TestCase(tvm.te.max(tvm.te.min(y, x), tvm.te.min(x, z)), tvm.te.min(tvm.te.max(y, z), x)),
        TestCase(tvm.te.max(tvm.te.min(y, x), tvm.te.min(z, x)), tvm.te.min(tvm.te.max(y, z), x)),
        TestCase(tvm.te.max(y + x, z + x), tvm.te.max(y, z) + x),
        TestCase(tvm.te.max(y + x, x + z), tvm.te.max(y, z) + x),
        TestCase(tvm.te.max(x + y, z + x), tvm.te.max(y, z) + x),
        TestCase(tvm.te.max(x + y, x + z), tvm.te.max(y, z) + x),
        TestCase(tvm.te.max(x - y, x - z), x - tvm.te.min(y, z)),
        TestCase(tvm.te.max(y - x, z - x), tvm.te.max(y, z) - x),
        TestCase(tvm.te.max(tvm.te.max(x, 1), 10), tvm.te.max(x, 10)),
        TestCase(tvm.te.max(tvm.te.max(x, 11), 10), tvm.te.max(x, 11)),
        TestCase(tvm.te.max(x * 3, 9), tvm.te.max(x, 3) * 3),
        TestCase(tvm.te.max(3 - x, 1), 3 - tvm.te.min(x, 2)),
        TestCase(tvm.te.max(x * 2, 0), tvm.te.max(x, 0) * 2),
        TestCase(tvm.te.max(0 - x * 2, 0), tvm.te.min(x, 0) * -2),
        TestCase(tvm.te.max(x * (-2), -4), tvm.te.min(x, 2) * -2),
        TestCase(tvm.te.max(x * (-2), 4), tvm.te.min(x, -2) * -2),
        TestCase(tvm.te.max(x * (0), 4), 4),
        TestCase(tvm.te.max(x * (0), -4), 0),
        # DivMod rules
        # truc div
        TestCase(tvm.te.max(tdiv(x, 10), tdiv(y, 10)), tdiv(tvm.te.max(x, y), 10)),
        TestCase(tvm.te.max(tdiv(x, (-10)), tdiv(y, (-10))), tdiv(tvm.te.min(x, y), (-10))),
        TestCase(tvm.te.max(tdiv(x + 3, 4) * 4, x), tdiv(x + 3, 4) * 4),
        # floordiv
        TestCase(tvm.te.max(fld(x, 10), fld(y, 10)), fld(tvm.te.max(x, y), 10)),
        TestCase(tvm.te.max(fld(x, (-10)), fld(y, (-10))), fld(tvm.te.min(x, y), (-10))),
        TestCase(tvm.te.max(fld(x + 3, 4) * 4, x), fld(x + 3, 4) * 4),
        TestCase(tvm.te.max(fld(x, 4) * 4, x), x),
        TestCase(tvm.te.max(x, fld(x, 4) * 4), x),
    )


class TestScalableIndex(BaseCompare):
    x, y = te.var("x"), te.var("y")
    test_case = tvm.testing.parameter(
        # MinNode
        TestCase(tvm.te.min(x + tir.vscale() * 4, x), x),
        TestCase(tvm.te.min(x - tir.vscale() * 4, x), x + tir.vscale() * -4),
        TestCase(tvm.te.min(x + tir.vscale() * 4, x + tir.vscale() * 8), tir.vscale() * 4 + x),
        TestCase(tvm.te.min(x + tir.vscale() * 4 - flm(4, tir.vscale() * 4), x), x),
        TestCase(tvm.te.min(tir.vscale() * x, tir.vscale() * y), tir.vscale() * x, x < y),
        # MaxNode
        TestCase(tvm.te.max(x + tir.vscale() * 4, x), x + tir.vscale() * 4),
        TestCase(tvm.te.max(x - tir.vscale() * 4, x), x),
        TestCase(tvm.te.max(x + tir.vscale() * 4, x + tir.vscale() * 4), x + tir.vscale() * 4),
        TestCase(
            tvm.te.max(x + tir.vscale() * 4 - flm(4, tir.vscale() * 4), x),
            x + tir.vscale() * 4 - flm(4, tir.vscale() * 4),
        ),
        TestCase(tvm.te.max(tir.vscale() * x, tir.vscale() * y), tir.vscale() * x, x > y),
        # FloorDiv
        TestCase(fld(x * tir.vscale() * 4 + y, tir.vscale() * 4), x + fld(y, tir.vscale() * 4)),
        TestCase(fld(x, tir.vscale() * 4), 0, [x >= 0, x < tir.vscale() * 4]),
        # FloorMod
        TestCase(flm(x * tir.vscale() * 4 + y, tir.vscale() * 4), flm(y, tir.vscale() * 4)),
        TestCase(flm(x, tir.vscale() * 4), x, [x >= 0, x < tir.vscale() * 4]),
    )

    def test_simplify(self, test_case):
        with tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+sve"):
            super().test_simplify(test_case)


class TestComparisons(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    test_case = tvm.testing.parameter(
        # const int bound
        TestCase((tmod(x, 2) + 10).equal(0), tvm.tir.const(0, "bool")),
        TestCase(tvm.tir.NE(tmod(x, 2) + 10, 0), tvm.tir.const(1, "bool")),
        TestCase(tmod(x, 2) + 10 > 1, tvm.tir.const(1, "bool")),
        TestCase(tmod(x, 2) + 10 <= 1, tvm.tir.const(0, "bool")),
        TestCase(flm(x, 2) + 2 > 1, tvm.tir.const(1, "bool")),
        TestCase(flm(x, 2) + 10 <= 1, tvm.tir.const(0, "bool")),
        TestCase(x * 3 + 10 == 0, tvm.tir.const(0, "bool")),
        TestCase(x * 3 + 10 != 0, tvm.tir.const(1, "bool")),
        # canonicalization
        TestCase((x - 10).equal(0), x.equal(10)),
        TestCase((10 - x).equal(0), x.equal(10)),
        TestCase((x * y).equal(0), tvm.tir.Or(x.equal(0), y.equal(0))),
        # Write LT as LE for integer arguments, if possible
        TestCase(x - 1 < y, x <= y),
        TestCase(x + (-1) < y, x <= y),
        TestCase(x < y - (-1), x <= y),
        TestCase(x < y + 1, x <= y),
        TestCase(x + 2 < y + 3, x <= y),
        TestCase(x - 3 < y - 2, x <= y),
        TestCase(x - 3 < y + (-2), x <= y),
        TestCase(x + (-3) < y - 2, x <= y),
        # Merge constants on the LHS/RHS of a LT expression.
        TestCase(x + 10 < y + 10, x < y),
        TestCase(x + 5 < y + 10, x < y + 5),
        TestCase(x + 10 < y + 5, x + 5 < y),
        TestCase(x - 5 < y - 10, x + 5 < y),
        TestCase(x - 10 < y - 5, x < y + 5),
        TestCase(x < y - 10, x + 10 < y),
        TestCase(x - 10 < y, x < y + 10),
        # cmp bound
        TestCase(x + y < x + z, y < z),
        TestCase(x + y < z + x, y < z),
        TestCase(y + x < x + z, y < z),
        TestCase(y + x < z + x, y < z),
        TestCase(y - x < z - x, y < z),
        TestCase(x - y < x - z, z < y),
        TestCase(x < z + x, tvm.tir.LT(0, z)),
        TestCase(x < x + z, tvm.tir.LT(0, z)),
        TestCase(100 < x + 1, tvm.tir.LT(99, x)),
        TestCase(1 < 100 - x, tvm.tir.LT(x, 99)),
        TestCase(x * 3 < y * 3, x < y),
        TestCase(x * (-3) < y * (-3), y < x),
        TestCase(x * 3 >= y * 3, y <= x),
        TestCase(x * 4 >= 2, tvm.tir.LE(1, x)),
        TestCase(x * 2 >= 50, tvm.tir.LE(25, x)),
        TestCase(x * 4 <= 2, x <= 0),
        TestCase((0 - x * 3) <= 0, tvm.tir.LE(0, x)),
        TestCase((0 - x * 3) >= 0, tvm.tir.LE(x, 0)),
        TestCase(2 * x <= 0, x <= 0),
        TestCase(x * 2 >= 3, tvm.tir.LE(2, x)),
        TestCase(x * 2 >= 2, tvm.tir.LE(1, x)),
        TestCase(x * 2 >= 1, tvm.tir.LE(1, x)),
        TestCase(x * 2 >= 0, tvm.tir.LE(0, x)),
        TestCase(x * 2 >= -1, tvm.tir.LE(0, x)),
        TestCase(x * 2 >= -2, tvm.tir.LE(-1, x)),
        TestCase(x * 2 >= -3, tvm.tir.LE(-1, x)),
        TestCase(x * 2 <= 3, tvm.tir.LE(x, 1)),
        TestCase(x * 2 <= 2, tvm.tir.LE(x, 1)),
        TestCase(x * 2 <= 1, tvm.tir.LE(x, 0)),
        TestCase(x * 2 <= 0, tvm.tir.LE(x, 0)),
        TestCase(x * 2 <= -1, tvm.tir.LE(x, -1)),
        TestCase(x * 2 <= -2, tvm.tir.LE(x, -1)),
        TestCase(x * 2 <= -3, tvm.tir.LE(x, -2)),
        TestCase(x * (-2) >= 3, tvm.tir.LE(x, -2)),
        TestCase(x * (-2) >= 2, tvm.tir.LE(x, -1)),
        TestCase(x * (-2) >= 1, tvm.tir.LE(x, -1)),
        TestCase(x * (-2) >= 0, tvm.tir.LE(x, 0)),
        TestCase(x * (-2) >= -1, tvm.tir.LE(x, 0)),
        TestCase(x * (-2) >= -2, tvm.tir.LE(x, 1)),
        TestCase(x * (-2) >= -3, tvm.tir.LE(x, 1)),
        TestCase(x * (-2) <= 3, tvm.tir.LE(-1, x)),
        TestCase(x * (-2) <= 2, tvm.tir.LE(-1, x)),
        TestCase(x * (-2) <= 1, tvm.tir.LE(0, x)),
        TestCase(x * (-2) <= 0, tvm.tir.LE(0, x)),
        TestCase(x * (-2) <= -1, tvm.tir.LE(1, x)),
        TestCase(x * (-2) <= -2, tvm.tir.LE(1, x)),
        TestCase(x * (-2) <= -3, tvm.tir.LE(2, x)),
        # DivMod rules
        # truc div
        TestCase(tdiv(x, 2) < 3, x < 6),
        TestCase(3 < tdiv(x, 2), tvm.tir.LT(7, x)),
        TestCase(tdiv(x, 3) >= 0, tvm.tir.LE(-2, x)),
        TestCase(tdiv(x, 2) >= 1, tvm.tir.LE(2, x)),
        TestCase(tdiv(x, 2) >= 0, tvm.tir.LE(-1, x)),
        TestCase(tdiv(x, 2) >= -1, tvm.tir.LE(-3, x)),
        TestCase(tdiv(x, 2) <= 1, tvm.tir.LE(x, 3)),
        TestCase(tdiv(x, 2) <= 0, tvm.tir.LE(x, 1)),
        TestCase(tdiv(x, 2) <= -1, tvm.tir.LE(x, -2)),
        TestCase(tdiv(x, 4) * 4 < x, tvm.tir.LT(0, tmod(x, 4))),
        TestCase(tdiv(x, 4) * 4 >= x, tvm.tir.LE(tmod(x, 4), 0)),
        TestCase(tdiv(x, 4) * 4 < x + y, tvm.tir.LT(0, tmod(x, 4) + y)),
        TestCase(tdiv(x, 4) * 4 < x - y, tvm.tir.LT(y, tmod(x, 4))),
        TestCase(tdiv(x + 2, 4) * 4 >= x, tvm.tir.LE(tmod(x + 2, 4), 2)),
        TestCase(tdiv(x + 2, 4) * 4 >= x + y, tvm.tir.LE(tmod(x + 2, 4) + y, 2)),
        TestCase(tdiv(x + 2, 4) * 4 >= x - y, tvm.tir.LE(tmod(x + 2, 4), y + 2)),
        # floor div
        TestCase(fld(x, 2) < 3, x < 6),
        TestCase(3 < fld(x, 2), tvm.tir.LT(7, x)),
        TestCase(-3 < fld(x, 2), tvm.tir.LT(-5, x)),
        TestCase(fld(x, 3) >= 0, tvm.tir.LE(0, x)),
        TestCase(fld(x, 2) >= 1, tvm.tir.LE(2, x)),
        TestCase(fld(x, 2) >= 0, tvm.tir.LE(0, x)),
        TestCase(fld(x, 2) >= -1, tvm.tir.LE(-2, x)),
        TestCase(fld(x, 2) <= 1, tvm.tir.LE(x, 3)),
        TestCase(fld(x, 2) <= 0, tvm.tir.LE(x, 1)),
        TestCase(fld(x, 2) <= -1, tvm.tir.LE(x, -1)),
        TestCase(fld(x, 4) * 4 < x, tvm.tir.LT(0, flm(x, 4))),
        TestCase(fld(x, 4) * 4 >= x, tvm.tir.EQ(flm(x, 4), 0)),
        TestCase(fld(x, 4) * 4 < x + y, tvm.tir.LT(0, flm(x, 4) + y)),
        TestCase(fld(x, 4) * 4 < x - y, tvm.tir.LT(y, flm(x, 4))),
        TestCase(fld(x + 2, 4) * 4 >= x, tvm.tir.LE(flm(x + 2, 4), 2)),
        TestCase(fld(x + 2, 4) * 4 >= x + y, tvm.tir.LE(flm(x + 2, 4) + y, 2)),
        TestCase(fld(x + 2, 4) * 4 >= x - y, tvm.tir.LE(flm(x + 2, 4), y + 2)),
        # End DivMod Rules
        # merging flm/fld into known value
        TestCase(tir.all(fld(x, 8) == 3, flm(x, 8) == 4), x == 28),
        TestCase(tir.all(flm(x, 8) == 4, fld(x, 8) == 3), x == 28),
        TestCase(tir.all(fld(x, 8) == -3, flm(x, 8) == 4), x == -20),
        TestCase(tir.all(flm(x, 8) == 4, fld(x, 8) == -3), x == -20),
        # Rewrite based on definition of integer division
        TestCase(tir.all(T.int32(0) <= x - y * 5, x - y * 5 < 5), y == fld(x, 5)),
        TestCase(tir.all(x - y * 5 < 5, T.int32(0) <= x - y * 5), y == fld(x, 5)),
        # Narrow upper bound using floormod
        TestCase(tir.all(x < 20, flm(x, 5) < 2), tir.all(x < 17, flm(x, 5) < 2)),
        TestCase(tir.all(x < 18, flm(x, 5) < 2), tir.all(x < 17, flm(x, 5) < 2)),
        TestCase(tir.all(x <= 19, flm(x, 5) < 2), tir.all(x < 17, flm(x, 5) < 2)),
        TestCase(tir.all(x <= 18, flm(x, 5) < 2), tir.all(x < 17, flm(x, 5) < 2)),
        TestCase(tir.all(x < -20, flm(x, 5) < 2), tir.all(x < -23, flm(x, 5) < 2)),
        TestCase(tir.all(x < 18 - 40, flm(x, 5) < 2), tir.all(x < 17 - 40, flm(x, 5) < 2)),
        TestCase(tir.all(x <= -21, flm(x, 5) < 2), tir.all(x < -23, flm(x, 5) < 2)),
        TestCase(tir.all(x <= -22, flm(x, 5) < 2), tir.all(x < -23, flm(x, 5) < 2)),
        # No change if the floormod cannot help narrow the upper bound
        TestCase(tir.all(x < 16, flm(x, 5) < 2), tir.all(x < 16, flm(x, 5) < 2)),
        TestCase(tir.all(x <= 15, flm(x, 5) < 2), tir.all(x <= 15, flm(x, 5) < 2)),
        # Merge a known floordiv and an upper bound of floormod into a value range
        TestCase(
            tir.all(fld(x, 10) == 5, flm(x, 10) < 7),
            tir.all(T.int32(50) <= x, x < 57),
        ),
        TestCase(
            tir.all(fld(x, 10) == 5, flm(x, 10) <= 7),
            tir.all(T.int32(50) <= x, x <= 57),
        ),
        TestCase(
            tir.all(fld(x, 10) == -5, flm(x, 10) < 7),
            tir.all(T.int32(-50) <= x, x < -43),
        ),
        TestCase(
            tir.all(fld(x, 10) == -5, flm(x, 10) <= 7),
            tir.all(T.int32(-50) <= x, x <= -43),
        ),
        # Merge a known floordiv and an lower bound of floormod into a value range
        TestCase(
            tir.all(fld(x, 10) == 5, T.int32(7) < flm(x, 10)),
            tir.all(T.int32(57) < x, x < 60),
        ),
        TestCase(
            tir.all(fld(x, 10) == 5, T.int32(7) <= flm(x, 10)),
            tir.all(T.int32(57) <= x, x < 60),
        ),
        TestCase(
            tir.all(fld(x, 10) == -5, T.int32(7) < flm(x, 10)),
            tir.all(T.int32(-43) < x, x < -40),
        ),
        TestCase(
            tir.all(fld(x, 10) == -5, T.int32(7) <= flm(x, 10)),
            tir.all(T.int32(-43) <= x, x < -40),
        ),
        TestCase(tvm.te.min(x, 11) < 10, x < 10),
        TestCase(tvm.te.min(x, 8) < 10, tvm.tir.const(1, "bool")),
        TestCase(tvm.te.max(8, x) > 10, tvm.tir.LT(10, x)),
        TestCase(x + 1 < tvm.te.max(8, x), x < 7),
        TestCase(x < 11, tvm.tir.const(1, "bool"), x <= 10),
        TestCase(x <= 10, tvm.tir.const(1, "bool"), x <= 10),
        TestCase(z <= 5, tvm.tir.const(1, "bool"), z <= 5),
        TestCase(x + y <= 10, tvm.tir.const(1, "bool"), [x <= 10, y <= 0]),
        TestCase(x + y >= -10, tvm.tir.const(1, "bool"), [x >= 0, y >= -10]),
        TestCase(z - 5 <= y + 10, tvm.tir.const(1, "bool"), [z <= 5, y >= -10]),
        TestCase(tvm.tir.all(x > -1, z <= x + 5), tvm.tir.const(1, "bool"), [x >= 0, z <= 5]),
        TestCase(x * y <= 0, tvm.tir.const(1, "bool"), [x >= 0, y <= 0]),
        TestCase((x + 1) * (y - 1) < 0, tvm.tir.const(1, "bool"), [x >= 0, y <= 0]),
        TestCase(y * y >= 0, tvm.tir.const(1, "bool"), y <= 0),
        TestCase(x * 6 <= -3, tvm.tir.const(0, "bool"), x >= 0),
        TestCase(tmod(y - 1, 3) == 0, tmod(y + (-1), 3) == 0),
    )


class TestComparisonOfProductAndSum(BaseCompare):
    extensions = tvm.arith.Extension.ComparisonOfProductAndSum

    x, y, z = te.var("x"), te.var("y"), te.var("z")

    test_case = tvm.testing.parameter(
        # Special inequality cases
        TestCase(
            x * y < (x + y) * 2048,
            tvm.tir.const(1, "bool"),
            [x > 0, y > 0, x < 2048],
        ),
        TestCase(
            x * y < (x + y) * 2048,
            tvm.tir.const(1, "bool"),
            [x > 0, y > 0, x < 4096, y < 4096],
        ),
        TestCase(
            # Both sides are divisible by 8192
            x * y * 8192 < (y + x) * 16777216,
            tvm.tir.const(1, "bool"),
            [x > 0, y > 0, x < 4096, y < 4096],
        ),
        TestCase(
            # The two sides have co-prime factors, but the bounds are
            # still sufficient to prove the inequality.
            x * y * 59 < (y + x) * 176128,
            tvm.tir.const(1, "bool"),
            [x > 0, y > 0, x < 4096, y < 4096],
        ),
    )


class TestLogical(BaseCompare):
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    test_case = tvm.testing.parameter(
        TestCase(tvm.tir.And(tvm.tir.EQ(x, y), tvm.tir.NE(x, y)), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(tvm.tir.NE(x, y), tvm.tir.EQ(x, y)), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(x > 1, tvm.tir.Not(x > 1)), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(x <= y, y < x), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(y < x, x <= y), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(x < 1, 0 < x), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(x < 0, 1 < x), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(x < 1, 1 <= x), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(x <= 1, 1 < x), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(1 <= x, x < 1), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(1 < x, x <= 1), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(x <= 1, 2 <= x), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(2 <= x, x <= 1), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.And(x == 1, x != 2), x == 1),
        TestCase(tvm.tir.And(x == 1, x == 2), tvm.tir.const(False, "bool")),
        TestCase(tvm.tir.Or(tvm.tir.EQ(x, y), tvm.tir.NE(x, y)), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(tvm.tir.NE(x, y), tvm.tir.EQ(x, y)), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(x > y, tvm.tir.Not(x > y)), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(x <= y, y < x), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(y < x, y >= x), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(x < 1, 0 < x), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(0 < x, x < 1), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(x < 1, 1 <= x), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(x <= 1, 1 < x), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(1 <= x, x < 1), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(1 < x, x <= 1), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(x <= 1, 2 <= x), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(2 <= x, x <= 1), tvm.tir.const(True, "bool")),
        TestCase(tvm.tir.Or(x != 1, x == 2), x != 1),
        TestCase(tvm.tir.Or(x != 1, x != 2), tvm.tir.const(True, "bool")),
        TestCase(
            tvm.tir.Or(x == 1, tvm.tir.Or(y == 1, z == 1)),
            tvm.tir.Or(tvm.tir.Or(x == 1, y == 1), z == 1),
        ),
        TestCase(
            tvm.tir.And(x == 1, tvm.tir.And(y == 1, z == 1)),
            tvm.tir.And(tvm.tir.And(x == 1, y == 1), z == 1),
        ),
    )


class TestLet(BaseCompare):
    x, y = te.var("x"), te.var("y")
    z = tvm.tir.Let(x, 1, x + 1)

    test_case = tvm.testing.parameter(
        TestCase(z + z, 4),
    )


class TestCast(BaseCompare):
    def _generate_tests():
        x = te.var("x")
        dtypes = ["float32", "float16", "int32", "int8", "bool"]
        for dtype1 in dtypes:
            yield TestCase(tvm.tir.Cast(dtype1, x - x), tvm.tir.const(0, dtype1))
            yield TestCase(tvm.tir.Cast(dtype1, x == x), tvm.tir.const(1, dtype1))
            for dtype2 in dtypes:
                for i in [0, 1, 2, 3]:
                    if i <= 1 or (dtype1 != "bool" and dtype2 != "bool"):
                        yield TestCase(
                            tvm.tir.Cast(dtype1, tvm.tir.const(i, dtype2)), tvm.tir.const(i, dtype1)
                        )

    test_case = tvm.testing.parameter(*_generate_tests())


class TestShiftLeft(BaseCompare):
    z = tvm.tir.op.call_intrin("int32", "tir.shift_left", 1, 10)
    test_case = tvm.testing.parameter(
        TestCase(z, tvm.tir.const(1 << 10, "int32")),
    )


class TestDivZero(BaseCompare):
    ramp = tvm.tir.Ramp(1, 1, 2)
    broadcast = tvm.tir.Broadcast(0, 2)

    test_case = tvm.testing.parameter(
        TestCase(tvm.tir.Div(ramp, broadcast), tvm.error.TVMError),
        TestCase(tvm.tir.Mod(ramp, broadcast), tvm.error.TVMError),
        TestCase(tvm.tir.FloorDiv(ramp, broadcast), tvm.error.TVMError),
        TestCase(tvm.tir.FloorMod(ramp, broadcast), tvm.error.TVMError),
    )


class TestSubBufferload(BaseCompare):
    buf = tvm.tir.decl_buffer([1], dtype="float32")
    load = tvm.tir.BufferLoad(buf, [0])

    test_case = tvm.testing.parameter(
        TestCase(load - load, 0.0),
    )


class TestIfThenElse(BaseCompare):
    x = te.var("x", "int32")

    test_case = tvm.testing.parameter(
        TestCase(
            tvm.tir.if_then_else(x < 5, tvm.tir.if_then_else(x > 1, 1, 0), 0),
            tvm.tir.if_then_else(tvm.tir.And(tvm.tir.LT(x, 5), tvm.tir.LT(1, x)), 1, 0),
        ),
        TestCase(
            tvm.tir.if_then_else(x > 2, tvm.tir.if_then_else(x > 1, 1, 0), 0),
            tvm.tir.if_then_else(tvm.tir.LT(2, x), 1, 0),
        ),
    )


class TestCLZ(BaseCompare):
    test_case = tvm.testing.parameter(
        TestCase(tvm.tir.call_intrin("int32", "tir.clz", 0), T.int32(32)),
        TestCase(tvm.tir.call_intrin("int32", "tir.clz", 1), T.int32(31)),
        TestCase(tvm.tir.call_intrin("int32", "tir.clz", 2), T.int32(30)),
        TestCase(tvm.tir.call_intrin("int32", "tir.clz", 128), T.int32(24)),
        TestCase(tvm.tir.call_intrin("int32", "tir.clz", tvm.tir.IntImm("int64", 0)), T.int32(64)),
        TestCase(tvm.tir.call_intrin("int32", "tir.clz", tvm.tir.IntImm("int64", 1)), T.int32(63)),
        TestCase(tvm.tir.call_intrin("int32", "tir.clz", tvm.tir.IntImm("int64", 2)), T.int32(62)),
        TestCase(
            tvm.tir.call_intrin("int32", "tir.clz", tvm.tir.IntImm("int64", 128)), T.int32(56)
        ),
    )


if __name__ == "__main__":
    tvm.testing.main()
