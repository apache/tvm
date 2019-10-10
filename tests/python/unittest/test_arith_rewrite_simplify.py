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
import tvm

class RewriteChecker:
    def __init__(self):
        self.analyzer = tvm.arith.Analyzer()

    def verify(self, data, expected):
        res = self.analyzer.rewrite_simplify(data)
        assert tvm.ir_pass.Equal(res, expected), "data={}, res={}, expected={}".format(data, res, expected)


def test_vector_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    # Add rules
    ck.verify(tvm.expr.Ramp(x, 1, 4) + tvm.expr.Ramp(y, 2, 4),
              tvm.expr.Ramp(x + y, 3, 4))
    ck.verify(tvm.expr.Ramp(x, 1, 2) + y,
              tvm.expr.Ramp(x + y, 1, 2))
    ck.verify(y + tvm.expr.Ramp(x, 1, 2) ,
              tvm.expr.Ramp(y + x, 1, 2))
    ck.verify(y.astype("int32x2") + x.astype("int32x2"),
              (y + x).astype("int32x2"))
    # Sub rules
    ck.verify(tvm.expr.Ramp(x, 4, 4) - tvm.expr.Ramp(y, 2, 4),
              tvm.expr.Ramp(x - y, 2, 4))
    ck.verify(tvm.expr.Ramp(x, 1, 2) - y,
              tvm.expr.Ramp(x - y, 1, 2))
    ck.verify(y - tvm.expr.Ramp(x, 1, 2) ,
              tvm.expr.Ramp(y - x, -1, 2))
    ck.verify(y.astype("int32x2") - x.astype("int32x2"),
              (y - x).astype("int32x2"))

    # Mul rules
    ck.verify(y.astype("int32x2") * x.astype("int32x2"),
              (y * x).astype("int32x2"))
    ck.verify(tvm.expr.Ramp(x, 4, 4) * 2,
              tvm.expr.Ramp(x * 2, 8, 4))
    ck.verify(2 * tvm.expr.Ramp(x, 4, 4),
              tvm.expr.Ramp(x * 2, 8, 4))

    ## DivMod rules
    tdiv = tvm.truncdiv
    tmod = tvm.truncmod
    # truc div
    ck.verify(tdiv(y.astype("int32x2"), x.astype("int32x2")),
              tdiv(y, x).astype("int32x2"))
    ck.verify(tdiv(tvm.expr.Ramp(x, 4, 4), 2),
              tvm.expr.Ramp(tdiv(x, 2), 2, 4))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify(tdiv(tvm.expr.Ramp(x * 8 + 1, 1, 4), 8),
              (x).astype("int32x4"))
    ck.verify(tdiv(tvm.expr.Ramp(x * 8 + 15, 1, 4), 8),
              tdiv(tvm.expr.Ramp(x * 8 + 15, 1, 4), 8))
    ck.verify(tmod(y.astype("int32x2"), x.astype("int32x2")),
              tmod(y, x).astype("int32x2"))
    ck.verify(tmod(tvm.expr.Ramp(x, 4, 4), 2),
              tvm.expr.Broadcast(tmod(x, 2), 4))
    ck.verify(tmod(tvm.expr.Ramp(x * 8 + 1, 1, 4), 8),
              tvm.expr.Ramp(1, 1, 4))
    ck.verify(tmod(tvm.expr.Ramp(x * 8 + 1, 15, 4), 8),
              tmod(tvm.expr.Ramp(1, 15, 4), 8))

    # floor div
    fld = tvm.floordiv
    flm = tvm.floormod
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-10, 1000), override=True)
    ck.verify(fld(y.astype("int32x2"), x.astype("int32x2")),
              fld(y, x).astype("int32x2"))
    ck.verify(fld(tvm.expr.Ramp(x, 4, 4), 2),
              tvm.expr.Ramp(fld(x, 2), 2, 4))
    ck.verify(fld(tvm.expr.Ramp(x * 8 + 1, 1, 4), 8),
              (x).astype("int32x4"))
    ck.verify(fld(tvm.expr.Ramp(x * 8 + 15, 1, 4), 8),
              fld(tvm.expr.Ramp(x * 8 + 15, 1, 4), 8))
    ck.verify(flm(y.astype("int32x2"), x.astype("int32x2")),
              flm(y, x).astype("int32x2"))
    ck.verify(flm(tvm.expr.Ramp(x, 4, 4), 2),
              tvm.expr.Broadcast(flm(x, 2), 4))
    ck.verify(flm(tvm.expr.Ramp(x * 8 + 1, 1, 4), 8),
              tvm.expr.Ramp(1, 1, 4))
    ck.verify(flm(tvm.expr.Ramp(x * 8 + 1, 15, 4), 8),
              flm(tvm.expr.Ramp(1, 15, 4), 8))

    # Min/Max rules
    vx = tvm.var("vx", dtype="int32x2")
    vc = tvm.var("vc", dtype="uint1")
    ck.verify(tvm.min(y.astype("int32x2"), x.astype("int32x2")),
              tvm.min(y, x).astype("int32x2"))
    ck.verify(tvm.min(tvm.min(vx, y.astype("int32x2")), x.astype("int32x2")),
              tvm.min(vx, tvm.min(y, x).astype("int32x2")))
    ck.verify(tvm.max(y.astype("int32x2"), x.astype("int32x2")),
              tvm.max(y, x).astype("int32x2"))
    ck.verify(tvm.max(tvm.max(vx, y.astype("int32x2")), x.astype("int32x2")),
              tvm.max(vx, tvm.max(y, x).astype("int32x2")))

    ## Logical rules
    ck.verify(y.astype("int32x2").equal(x.astype("int32x2")),
              (y.equal(x)).astype("uint1x2"))
    ck.verify(tvm.expr.NE(y.astype("int32x2"), (x.astype("int32x2"))),
              (tvm.expr.NE(y, x)).astype("uint1x2"))
    ck.verify(y.astype("int32x2") > x.astype("int32x2"),
              (x < y).astype("uint1x2"))
    ck.verify(y.astype("int32x2") >= x.astype("int32x2"),
              (x <= y).astype("uint1x2"))
    ck.verify(y.astype("int32x2") < x.astype("int32x2"),
              (y < x).astype("uint1x2"))
    ck.verify(y.astype("int32x2") <= x.astype("int32x2"),
              (y <= x).astype("uint1x2"))
    ck.verify(tvm.expr.And(y.astype("int32x2") <= x.astype("int32x2"), vc.astype("uint1x2")),
              (tvm.expr.And(y <= x, vc)).astype("uint1x2"))
    ck.verify(tvm.expr.Or(y.astype("int32x2") <= x.astype("int32x2"), vc.astype("uint1x2")),
              (tvm.expr.Or(y <= x, vc)).astype("uint1x2"))


def test_select_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    # Add rules
    ck.verify(tvm.expr.Select(x < 0, y, 0) + tvm.expr.Select(x < 0, 1, z),
              tvm.expr.Select(x < 0, y + 1, z))
    ck.verify(tvm.expr.Select(x < 0, y, 1) - tvm.expr.Select(x < 0, 1, z),
              tvm.expr.Select(x < 0, y + (-1), 1 - z))
    ck.verify(tvm.expr.Select(x < 0, y, z) - y,
              tvm.expr.Select(x < 0, 0, z - y))
    ck.verify(tvm.expr.Select(x < 0, y, z) - z,
              tvm.expr.Select(x < 0, y - z, 0))
    ck.verify(tvm.min(tvm.expr.Select(x < 0, y, 0), tvm.expr.Select(x < 0, 1, z)),
              tvm.expr.Select(x < 0, tvm.min(y, 1), tvm.min(0, z)))
    ck.verify(tvm.max(tvm.expr.Select(x < 0, y, 0), tvm.expr.Select(x < 0, 1, z)),
              tvm.expr.Select(x < 0, tvm.max(y, 1), tvm.max(0, z)))

    ck.verify(tvm.expr.Select(x * 3 + 1 != 0, y, z), y)
    ck.verify(tvm.expr.Select(x * 3 + 1 == 0, y, z), z)
    ck.verify(tvm.expr.Select(x > 0, y + 1, y + 1), y + 1)


def test_add_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")

    ck.verify(x + (y - x), y)
    ck.verify(x - (y + 1) + (y + 1), x)
    ck.verify((x - 10) + (10 - z), x - z)
    ck.verify((x - y) + (z - x), z - y)

    ck.verify(tvm.min(x, y - z) + z, tvm.min(x + z, y))
    ck.verify(tvm.min(x - z, y) + z, tvm.min(x, y + z))
    ck.verify(tvm.max(x, y - 10) + 10, tvm.max(x + 10, y))
    ck.verify(tvm.max(x - 11, y) + 11, tvm.max(x, y + 11))

    ck.verify(tvm.max(x, y * 2) + tvm.min(x, y * 2), x + y * 2);
    ck.verify(tvm.min(x, y * 2) + tvm.max(x, y * 2), x + y * 2);

    ck.verify(tvm.max(x, y + 2) + (-2), tvm.max(x + (-2), y));
    ck.verify(tvm.min(x, y + 2) + (-2), tvm.min(x + (-2), y));
    ck.verify(tvm.min(x + 2, y + 3) + (-2), tvm.min(x, y + 1));

    ck.verify(tvm.max(0, 1 - x * 4) + x * 4, tvm.max(x * 4, 1))
    ck.verify(tvm.max(2 - x * 4, 0) + x * 4, tvm.max(x * 4, 2))

    ck.verify(tvm.min(0, 1 - x * 4) + x * 4, tvm.min(x * 4, 1))
    ck.verify(tvm.min(2 - x * 4, 0) + x * 4, tvm.min(x * 4, 2))

    ck.verify(x * y + x * 10, x * (y + 10))
    ck.verify(y * x + x * 10, x * (y + 10))
    ck.verify(y * x + 10 * x, x * (y + 10))
    ck.verify(x * y + 10 * x, x * (y + 10))


    # canonicalization
    ck.verify(x + 2 + 3 + 4 + x, x * 2 + 9);
    ck.verify(x + 2 + 3 + 4 + x * 3, x * 4 + 9);

    # DivMod rules
    tdiv = tvm.truncdiv
    tmod = tvm.truncmod
    # truc div
    ck.verify(y * tmod(x, 8) + 10 * tmod(x, 8), tmod(x, 8) * (y + 10))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1, 1000), override=True)
    ck.verify(tdiv(x, 8) * 8 + tmod(x, 8), x)

    # floor div
    fld = tvm.floordiv
    flm = tvm.floormod
    ck.verify(y * flm(x, 8) + 10 * flm(x, 8), flm(x, 8) * (y + 10))
    ck.verify(fld(x, 8) * 8 + flm(x, 8), x)



def test_sub_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")

    ck.verify(x + y - y, x)
    ck.verify(x + y - x, y)
    ck.verify(x - (y + x), 0 - y)
    ck.verify(x - (x + y), 0 - y)

    ck.verify(tvm.min(x, y) - x, tvm.min(0, y - x))
    ck.verify(tvm.min(x, y) - y, tvm.min(x - y, 0))
    ck.verify(tvm.max(x, y) - x, tvm.max(0, y - x))
    ck.verify(tvm.max(x, y) - y, tvm.max(x - y, 0))

    ck.verify(x - tvm.min(x, y), tvm.max(0, x - y))
    ck.verify(y - tvm.min(x, y), tvm.max(y - x, 0))
    ck.verify(x - tvm.max(x, y), tvm.min(0, x - y))
    ck.verify(y - tvm.max(x, y), tvm.min(y - x, 0))

    # mul co-efficient foldng
    ck.verify(x - x, 0)
    ck.verify(x * y - x, x * (y + (-1)))
    ck.verify(x * y - 10 * x, x * (y + (-10)))
    ck.verify(y * x - x * z, x * (y - z))
    ck.verify(y * x - z * x, x * (y - z))

    ck.verify(x + 10 - 20, x + (-10))

    # 4-operands pattern
    ck.verify((x + y) - (x + z), y - z)
    ck.verify((y + x) - (x + z), y - z)
    ck.verify((x + y) - (z + x), y - z)
    ck.verify((y + x) - (z + x), y - z)

    ck.verify(tvm.min(x + y, z) - x, tvm.min(y, z - x))
    ck.verify(tvm.min(y + x, z) - x, tvm.min(y, z - x))
    ck.verify(tvm.min(z, x + y) - x, tvm.min(z - x, y))
    ck.verify(tvm.min(z, y + x) - x, tvm.min(z - x, y))

    ck.verify(tvm.max(x + y, z) - x, tvm.max(y, z - x))
    ck.verify(tvm.max(y + x, z) - x, tvm.max(y, z - x))
    ck.verify(tvm.max(z, x + y) - x, tvm.max(z - x, y))
    ck.verify(tvm.max(z, y + x) - x, tvm.max(z - x, y))

    ck.verify(x - tvm.min(x + y, z), tvm.max(0 - y, x - z))
    ck.verify(x - tvm.min(y + x, z), tvm.max(0 - y, x - z))
    ck.verify(x - tvm.min(z, x + y), tvm.max(x - z, 0 - y))
    ck.verify(x - tvm.min(z, y + x), tvm.max(x - z, 0 - y))

    ck.verify(tvm.min(x, y) - tvm.min(y, x), 0)
    ck.verify(tvm.max(x, y) - tvm.max(y, x), 0)
    ck.verify(tvm.min(x, y) - tvm.min(x + 10, y + 10), -10)
    ck.verify(tvm.min(x + 10, y + 1) - tvm.min(x, y - 9), 10)

    # DivMod patterns
    # truc div
    tdiv = tvm.truncdiv
    tmod = tvm.truncmod
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify(x - tdiv(x, 3) * 3, tmod(x, 3))

    ck.verify(tdiv(x + 5, 3) - tdiv(x, 3), tdiv(tmod(x, 3) + 5, 3))
    ck.verify(tdiv(x + 5, 3) - tdiv(x + 1, 3), tdiv(tmod(x + 1, 3) + 4, 3))

    ck.verify(y - tdiv(y, (-5)) * (-5), tmod(y, 5))
    ck.verify(tdiv(y, 3) * 3 - y, 0 - tmod(y, 3))
    ck.verify(y - tdiv(y - 6, 5) * 5, tmod(y + (-6), 5) + 6)
    ck.verify(tdiv(y - 6, 5) * 5 - y, (-6) - tmod(y + (-6), 5))
    ck.verify(y - tdiv(y + z, 5) * 5, tmod(y + z, 5) - z)
    ck.verify(tdiv(y + z, 5) * 5 - y, z - tmod(y + z, 5))
    ck.verify(y - tdiv(y - z, 5) * 5, tmod(y - z, 5) + z)
    ck.verify(tdiv(y - z, 5) * 5 - y, 0 - tmod(y - z, 5) - z)

    ck.verify(y * 3 - tdiv(y, 2) * 6, tmod(y, 2) * 3)
    ck.verify(tdiv(y, 3) * 6 - y * 2, tmod(y, 3) * (-2))
    ck.verify(y * 5 - tdiv(y + z, 2) * 10, (tmod(y + z, 2) - z) * 5)
    ck.verify(y * 5 - tdiv(y - z, 2) * 10, (tmod(y - z, 2) + z) * 5)
    ck.verify(tdiv(y + z, 3) * 6 - y * 2, (z - tmod(y + z, 3)) * 2)
    ck.verify(tdiv(y - z, 3) * 6 - y * 2, (0 - tmod(y - z, 3) - z) * 2)
    ck.verify(5 * y - tdiv(y + z, 2) * 10, (tmod(y + z, 2) - z) * 5)
    ck.verify(5 * y - 10 * tdiv(y - z, 2), (tmod(y - z, 2) + z) * 5)
    ck.verify(6 * tdiv(y + z, 3) - y * 2, (z - tmod(y + z, 3)) * 2)
    ck.verify(tdiv(y - z, 3) * 6 - 2 * y, (0 - tmod(y - z, 3) - z) * 2)

    # floor div
    fld = tvm.floordiv
    flm = tvm.floormod
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1000, 1000), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(-1000, 1000), override=True)
    ck.verify(x - fld(x, 3) * 3, flm(x, 3))
    ck.verify(fld(x + 5, 3) - fld(x, 3), fld(flm(x, 3) + 5, 3))
    ck.verify(fld(x + 5, 3) - fld(x + 2, 3), fld(flm(x + 2, 3), 3) + 1)

    ck.verify(fld(y, 3) * 3 - y, 0 - flm(y, 3))
    ck.verify(y - fld(y - 6, 5) * 5, flm(y + (-6), 5) + 6)
    ck.verify(fld(y - 6, 5) * 5 - y, (-6) - flm(y + (-6), 5))
    ck.verify(y - fld(y + z, 5) * 5, flm(y + z, 5) - z)
    ck.verify(fld(y + z, 5) * 5 - y, z - flm(y + z, 5))
    ck.verify(y - fld(y - z, 5) * 5, flm(y - z, 5) + z)
    ck.verify(fld(y - z, 5) * 5 - y, 0 - flm(y - z, 5) - z)
    ck.verify(y * 3 - fld(y, 2) * 6, flm(y, 2) * 3)
    ck.verify(fld(y, 3) * 6 - y * 2, flm(y, 3) * (-2))
    ck.verify(y * 5 - fld(y + z, 2) * 10, (flm(y + z, 2) - z) * 5)
    ck.verify(y * 5 - fld(y - z, 2) * 10, (flm(y - z, 2) + z) * 5)
    ck.verify(fld(y + z, 3) * 6 - y * 2, (z - flm(y + z, 3)) * 2)
    ck.verify(fld(y - z, 3) * 6 - y * 2, (0 - flm(y - z, 3) - z) * 2)
    ck.verify(5 * y - fld(y + z, 2) * 10, (flm(y + z, 2) - z) * 5)
    ck.verify(5 * y - 10 * fld(y - z, 2), (flm(y - z, 2) + z) * 5)
    ck.verify(6 * fld(y + z, 3) - y * 2, (z - flm(y + z, 3)) * 2)
    ck.verify(fld(y - z, 3) * 6 - 2 * y, (0 - flm(y - z, 3) - z) * 2)


def test_mul_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    ck.verify((x + 2) * 3, x * 3 + 6)
    ck.verify((x * 2) * 3, x * 6)
    ck.verify(tvm.min(x, y) * tvm.max(x, y), x * y)
    ck.verify(tvm.max(x, y) * tvm.min(x, y), x * y)
    ck.verify((x - y) * (-2), (y - x) * 2)


def test_div_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    tdiv = tvm.truncdiv
    tmod = tvm.truncmod

    ck.verify(tdiv(x, x), 1)
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(z, tvm.arith.ConstIntBound(0, 1000), override=True)

    ck.verify(tdiv(tdiv(x, 2), 3), tdiv(x, 6))
    ck.verify(tdiv(tdiv(x, 2) + 1, 3), tdiv(x + 2, 6))
    ck.verify(tdiv(x * 2, 4), tdiv(x, 2))
    ck.verify(tdiv(x * 4, 2), x * 2)

    ck.verify(tdiv(x * 4 + y, 2), x * 2 + tdiv(y, 2))
    ck.verify(tdiv(tvm.min(x * 6, y), 2), tvm.min(x * 3, tdiv(y, 2)))
    ck.verify(tdiv(tvm.max(x * 6, y), 2), tvm.max(x * 3, tdiv(y, 2)))

    ck.verify(tdiv(y + x * 4, 2), tdiv(y, 2) + x * 2)
    ck.verify(tdiv(tvm.min(y, x * 6), 2), tvm.min(tdiv(y, 2), x * 3))
    ck.verify(tdiv(tvm.max(y, x * 6), 2), tvm.max(tdiv(y, 2), x * 3))

    # 3-operands
    ck.verify(tdiv(x * 6 + y + z, 2), x * 3 + tdiv(y + z, 2))
    ck.verify(tdiv(x * 6 - y + (y + 3), 2), x * 3 + 1)
    ck.verify(tdiv(x * 6 + (y + 3) - y, 2), x * 3 + 1)
    ck.verify(tdiv(y + x * 6 + z, 2), x * 3 + tdiv(y + z, 2))
    ck.verify(tdiv(x + 4, 2), tdiv(x, 2) + 2)

    ck.verify(tdiv(x + y, x), tdiv(y, x) + 1)
    ck.verify(tdiv(y + x, x), tdiv(y, x) + 1)
    ck.verify(tdiv((x + y) + z, x), tdiv(y + z, x) + 1)
    ck.verify(tdiv((y + x) + z, x), tdiv(y + z, x) + 1)
    ck.verify(tdiv(y + (x + z), x), tdiv(y + z, x) + 1)
    ck.verify(tdiv(y + (z + x), x), tdiv(y + z, x) + 1)

    ck.verify(tdiv(x * y, y), x)
    ck.verify(tdiv(y * x, y), x)

    ck.verify(tdiv(x * z + y, z), x + tdiv(y, z))
    ck.verify(tdiv(z * x + y, z), x + tdiv(y, z))
    ck.verify(tdiv(y + x * z, z), tdiv(y, z) + x)
    ck.verify(tdiv(y + z * x, z), tdiv(y, z) + x)


def test_floordiv_index_simplify():
    # short name for floordiv
    fld = tvm.floordiv
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")

    ck.verify(fld(fld(x, 2), 3), fld(x, 6))
    ck.verify(fld(fld(x, 2) + 1, 3), fld(x + 2, 6))

    ck.verify(fld(x * 2, 4), fld(x, 2))
    ck.verify(fld(x * 4, 2), x * 2)

    ck.verify(fld(x * 4 + y, 2), x * 2 + fld(y, 2))
    ck.verify(fld(tvm.min(x * 6, y), 2), tvm.min(x * 3, fld(y, 2)))
    ck.verify(fld(tvm.max(x * 6, y), 2), tvm.max(x * 3, fld(y, 2)))

    ck.verify(fld(y + x * 4, 2), fld(y, 2) + x * 2)
    ck.verify(fld(tvm.min(y, x * 6), 2), tvm.min(fld(y, 2), x * 3))
    ck.verify(fld(tvm.max(y, x * 6), 2), tvm.max(fld(y, 2), x * 3))

    # 3-operands
    ck.verify(fld(x * 6 + y + z, 2), x * 3 + fld(y + z, 2))
    ck.verify(fld(x * 6 - y + (y + 3), 2), x * 3 + 1)
    ck.verify(fld(x * 6 + (y + 3) - y, 2), x * 3 + 1)
    ck.verify(fld(y + x * 6 + z, 2), x * 3 + fld(y + z, 2))
    ck.verify(fld(x + 4, 2), fld(x, 2) + 2)

    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify(fld(x + y, x), fld(y, x) + 1)
    ck.verify(fld(y + x, x), fld(y, x) + 1)
    ck.verify(fld((x + y) + z, x), fld(y + z, x) + 1)
    ck.verify(fld((y + x) + z, x), fld(y + z, x) + 1)
    ck.verify(fld(y + (x + z), x), fld(y + z, x) + 1)
    ck.verify(fld(y + (z + x), x), fld(y + z, x) + 1)

    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(z, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify(fld(x * y, y), x)
    ck.verify(fld(y * x, y), x)
    ck.verify(fld(x * z + y, z), x + fld(y, z))
    ck.verify(fld(z * x + y, z), x + fld(y, z))
    ck.verify(fld(y + x * z, z), fld(y, z) + x)
    ck.verify(fld(y + z * x, z), fld(y, z) + x)


def test_mod_index_simplify():
    ck = RewriteChecker()
    x, y, nx, ny, z = tvm.var("x"), tvm.var("y"), tvm.var("nx"), tvm.var("ny"), tvm.var("z")
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(nx, tvm.arith.ConstIntBound(-1000, 0), override=True)
    ck.analyzer.update(ny, tvm.arith.ConstIntBound(-1000, 0), override=True)
    tdiv = tvm.truncdiv
    tmod = tvm.truncmod

    ck.verify(tmod(x * 10, 2), 0)
    ck.verify(tmod(x * 10 + y, 2), tmod(y, 2))
    ck.verify(tmod(x + 10, 2), tmod(x, 2))
    ck.verify(tmod(x + y * 10, 2), tmod(x, 2))
    ck.verify(tmod(x* 10 + 1 + y * 2 + 2, 2), 1)
    ck.verify(tmod(x * 10, -2), 0)
    ck.verify(tmod(x * 10 + y, -2), tmod(y, 2))
    ck.verify(tmod(x + 10, -2), tmod(x, 2))
    ck.verify(tmod(x + y * 10, -2), tmod(x, 2))
    ck.verify(tmod(x* 10 + 1 + y * 2 + 2, -2), 1)

    ck.verify(tmod(x * (-10), 2), 0)
    ck.verify(tmod(x * (-10) + y, 2), tmod(x * (-10) + y, 2))
    ck.verify(tmod(x + (-10), 2), tmod(x + (-10), 2))
    ck.verify(tmod(x + y * (-10), 2), tmod(x + y * (-10), 2))
    ck.verify(tmod(x * (-10), -2), 0)

    ck.verify(tmod(nx * 10, 2), 0)
    ck.verify(tmod(nx * (-10) + y, 2), tmod(y, 2))
    ck.verify(tmod(x + ny * (-10), 2), tmod(x, 2))
    ck.verify(tmod(nx * (-10) + 1 + ny * (-2) + 2, 2), 1)
    ck.verify(tmod(nx * 10, -2), 0)
    ck.verify(tmod(nx * (-10) + y, -2), tmod(y, 2))
    ck.verify(tmod(x + ny * (-10), -2), tmod(x, 2))


def test_floormod_index_simplify():
    # short name for floordiv
    flm = tvm.floormod
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    ck = RewriteChecker()
    x, y, nx, ny, z = tvm.var("x"), tvm.var("y"), tvm.var("nx"), tvm.var("ny"), tvm.var("z")

    ck.verify(flm(x * 10, 2), 0)
    ck.verify(flm(x * 10 + y, 2), flm(y, 2))
    ck.verify(flm(x + 10, 2), flm(x, 2))
    ck.verify(flm(x + y * 10, 2), flm(x, 2))
    ck.verify(flm(x* 10 + 1 + y * 2 + 2, 2), 1)
    ck.verify(flm(x * (-10), 2), 0)
    ck.verify(flm(x * (-10) + y, 2), flm(y, 2))
    ck.verify(flm(x + (-10), 2), flm(x, 2))
    ck.verify(flm(x + y * (-10), 2), flm(x, 2))


def test_min_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    fld = tvm.floordiv
    flm = tvm.floormod
    tdiv = tvm.truncdiv
    tmod = tvm.truncmod
    # const int bound
    ck.verify(tvm.min(tmod(x, 2), tmod(y, 2) + 10), tmod(x, 2))
    ck.verify(tvm.min(flm(x, 2), flm(y, 2) + 10), flm(x, 2))

    ck.verify(tvm.min(x + 1, x + 10), x + 1)
    ck.verify(tvm.min(x + 111, x + 10), x + 10)
    ck.verify(tvm.min(x + 1, x), x)
    ck.verify(tvm.min(x, x + 2), x)
    ck.verify(tvm.min(1 - x, 2 - x), 1 - x)
    ck.verify(tvm.min(3 - x, 2 - x), 2 - x)

    ck.verify(tvm.min(tvm.max(x, y), tvm.min(x, y)), tvm.min(x, y))
    ck.verify(tvm.min(tvm.max(x, y), tvm.min(y, x)), tvm.min(x, y))

    ck.verify(tvm.min(tvm.max(x, y), x), x)
    ck.verify(tvm.min(tvm.max(y, x), x), x)
    ck.verify(tvm.min(tvm.min(x, y), x), tvm.min(x, y))
    ck.verify(tvm.min(tvm.min(x, y), y), tvm.min(x, y))

    ck.verify(tvm.min(x, tvm.max(x, y)), x)
    ck.verify(tvm.min(x, tvm.max(y, x)), x)
    ck.verify(tvm.min(x, tvm.min(x, y)), tvm.min(x, y))
    ck.verify(tvm.min(y, tvm.min(x, y)), tvm.min(x, y))

    ck.verify(tvm.min(tvm.min(tvm.min(x, y), z), y),
              tvm.min(tvm.min(x, y), z))
    ck.verify(tvm.min(tvm.min(tvm.min(tvm.min(x, y), z), x * 2), y),
              tvm.min(tvm.min(tvm.min(x, y), z), x * 2))
    ck.verify(tvm.min(tvm.min(tvm.min(tvm.min(tvm.min(x, y), z), x * 2), z * 2), y),
              tvm.min(tvm.min(tvm.min(tvm.min(x, y), z), x * 2), z * 2))

    ck.verify(tvm.min(tvm.max(x, y), tvm.max(x, z)), tvm.max(tvm.min(y, z), x))
    ck.verify(tvm.min(tvm.max(x, y), tvm.max(z, x)), tvm.max(tvm.min(y, z), x))
    ck.verify(tvm.min(tvm.max(y, x), tvm.max(x, z)), tvm.max(tvm.min(y, z), x))
    ck.verify(tvm.min(tvm.max(y, x), tvm.max(z, x)), tvm.max(tvm.min(y, z), x))

    ck.verify(tvm.min(y + x, z + x), tvm.min(y, z) + x)
    ck.verify(tvm.min(y + x, x + z), tvm.min(y, z) + x)
    ck.verify(tvm.min(x + y, z + x), tvm.min(y, z) + x)
    ck.verify(tvm.min(x + y, x + z), tvm.min(y, z) + x)

    ck.verify(tvm.min(x - y, x - z), x - tvm.max(y, z))
    ck.verify(tvm.min(y - x, z - x), tvm.min(y, z) - x)

    ck.verify(tvm.min(tvm.min(x, 1), 10), tvm.min(x, 1))
    ck.verify(tvm.min(tvm.min(x, 11), 10), tvm.min(x, 10))

    ck.verify(tvm.min(x * 3, 9), tvm.min(x, 3) * 3)
    ck.verify(tvm.min(3 - x, 2), 3 - tvm.max(x,  1))

    # DivMod rules
    # truc div
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000))
    ck.verify(tvm.min(tdiv(x + 3, 4) * 4, x), x)
    ck.verify(tvm.min(tdiv(x + 3, 4) * 4, tvm.max(x, 4)), tvm.max(x, 4))
    ck.verify(tvm.min(x, tdiv(x + 3, 4) * 4), x)
    ck.verify(tvm.min(tvm.max(x, 4), tdiv(x + 3, 4) * 4), tvm.max(x, 4))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1000, 1000), True)
    ck.verify(tvm.min(tdiv(x, 10), tdiv(y, 10)), tdiv(tvm.min(x, y), 10))
    ck.verify(tvm.min(tdiv(x, (-10)), tdiv(y, (-10))),
              tdiv(tvm.max(x, y), (-10)))

    # floor div
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1000, 1000), True)
    ck.verify(tvm.min(fld(x + 3, 4) * 4, x), x)
    ck.verify(tvm.min(fld(x + 3, 4) * 4, tvm.max(x, 4)), tvm.max(x, 4))
    ck.verify(tvm.min(x, fld(x + 3, 4) * 4), x)
    ck.verify(tvm.min(x, fld(x, 4) * 4), fld(x, 4) * 4)
    ck.verify(tvm.min(tvm.max(x, 4), fld(x + 3, 4) * 4), tvm.max(x, 4))
    ck.verify(tvm.min(fld(x, 10), fld(y, 10)), fld(tvm.min(x, y), 10))
    ck.verify(tvm.min(fld(x, (-10)), fld(y, (-10))), fld(tvm.max(x, y), (-10)))


def test_max_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    flm = tvm.floormod
    fld = tvm.floordiv
    tdiv = tvm.truncdiv
    tmod = tvm.truncmod
    # const int bound
    ck.verify(tvm.max(tmod(x, 2), tmod(y, 2) + 10), tmod(y, 2) + 10)
    ck.verify(tvm.max(flm(x, 2), flm(y, 2) + 10), flm(y, 2) + 10)

    ck.verify(tvm.max(x + 1, x + 10), x + 10)
    ck.verify(tvm.max(x + 111, x + 10), x + 111)
    ck.verify(tvm.max(x + 1, x), x + 1)
    ck.verify(tvm.max(x, x + 2), x + 2)
    ck.verify(tvm.max(1 - x, 2 - x), 2 - x)
    ck.verify(tvm.max(3 - x, 2 - x), 3 - x)

    ck.verify(tvm.max(tvm.min(x, y), tvm.max(x, y)), tvm.max(x, y))
    ck.verify(tvm.max(tvm.min(x, y), tvm.max(y, x)), tvm.max(x, y))

    ck.verify(tvm.max(tvm.min(x, y), x), x)
    ck.verify(tvm.max(tvm.min(y, x), x), x)
    ck.verify(tvm.max(tvm.max(x, y), x), tvm.max(x, y))
    ck.verify(tvm.max(tvm.max(x, y), y), tvm.max(x, y))

    ck.verify(tvm.max(x, tvm.min(x, y)), x)
    ck.verify(tvm.max(x, tvm.min(y, x)), x)
    ck.verify(tvm.max(x, tvm.max(x, y)), tvm.max(x, y))
    ck.verify(tvm.max(y, tvm.max(x, y)), tvm.max(x, y))

    ck.verify(tvm.max(tvm.max(tvm.max(x, y), z), y),
              tvm.max(tvm.max(x, y), z))
    ck.verify(tvm.max(tvm.max(tvm.max(tvm.max(x, y), z), x * 2), y),
              tvm.max(tvm.max(tvm.max(x, y), z), x * 2))
    ck.verify(tvm.max(tvm.max(tvm.max(tvm.max(tvm.max(x, y), z), x * 2), z * 2), y),
              tvm.max(tvm.max(tvm.max(tvm.max(x, y), z), x * 2), z * 2))

    ck.verify(tvm.max(tvm.min(x, y), tvm.min(x, z)), tvm.min(tvm.max(y, z), x))
    ck.verify(tvm.max(tvm.min(x, y), tvm.min(z, x)), tvm.min(tvm.max(y, z), x))
    ck.verify(tvm.max(tvm.min(y, x), tvm.min(x, z)), tvm.min(tvm.max(y, z), x))
    ck.verify(tvm.max(tvm.min(y, x), tvm.min(z, x)), tvm.min(tvm.max(y, z), x))

    ck.verify(tvm.max(y + x, z + x), tvm.max(y, z) + x)
    ck.verify(tvm.max(y + x, x + z), tvm.max(y, z) + x)
    ck.verify(tvm.max(x + y, z + x), tvm.max(y, z) + x)
    ck.verify(tvm.max(x + y, x + z), tvm.max(y, z) + x)

    ck.verify(tvm.max(x - y, x - z), x - tvm.min(y, z))
    ck.verify(tvm.max(y - x, z - x), tvm.max(y, z) - x)

    ck.verify(tvm.max(tvm.max(x, 1), 10), tvm.max(x, 10))
    ck.verify(tvm.max(tvm.max(x, 11), 10), tvm.max(x, 11))

    ck.verify(tvm.max(x * 3, 9), tvm.max(x, 3) * 3)
    ck.verify(tvm.max(3 - x, 1), 3 - tvm.min(x,  2))

    # DivMod rules
    # truc div
    ck.verify(tvm.max(tdiv(x, 10), tdiv(y, 10)), tdiv(tvm.max(x, y), 10))
    ck.verify(tvm.max(tdiv(x, (-10)), tdiv(y, (-10))), tdiv(tvm.min(x, y), (-10)))
    ck.verify(tvm.max(tdiv(x + 3, 4) * 4, x), tdiv(x + 3, 4) * 4)

    # floordiv
    ck.verify(tvm.max(fld(x, 10), fld(y, 10)), fld(tvm.max(x, y), 10))
    ck.verify(tvm.max(fld(x, (-10)), fld(y, (-10))), fld(tvm.min(x, y), (-10)))
    ck.verify(tvm.max(fld(x + 3, 4) * 4, x), fld(x + 3, 4) * 4)
    ck.verify(tvm.max(fld(x, 4) * 4, x), x)
    ck.verify(tvm.max(x, fld(x, 4) * 4), x)


def test_cmp_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    flm = tvm.floormod
    fld = tvm.floordiv
    tdiv = tvm.truncdiv
    tmod = tvm.truncmod
    # const int bound
    ck.verify((tmod(x, 2) + 10).equal(0), tvm.const(0, "bool"))
    ck.verify(tvm.expr.NE(tmod(x, 2) + 10, 0), tvm.const(1, "bool"))
    ck.verify(tmod(x, 2) + 10 > 1, tvm.const(1, "bool"))
    ck.verify(tmod(x, 2) + 10 <= 1, tvm.const(0, "bool"))
    ck.verify(flm(x, 2) + 2 > 1, tvm.const(1, "bool"))
    ck.verify(flm(x, 2) + 10 <= 1, tvm.const(0, "bool"))

    ck.verify(x * 3 + 10 == 0, tvm.const(0, "bool"))
    ck.verify(x * 3 + 10 != 0, tvm.const(1, "bool"))

    # canonicalization
    ck.verify((x - 10).equal(0), x.equal(10))
    ck.verify((10 - x).equal(0), x.equal(10))
    ck.verify((x * y).equal(0), tvm.expr.Or(x.equal(0), y.equal(0)))

    # cmp bound
    ck.verify(x + y < x + z, y < z)
    ck.verify(x + y < z + x, y < z)
    ck.verify(y + x < x + z, y < z)
    ck.verify(y + x < z + x, y < z)
    ck.verify(y - x < z - x, y < z)
    ck.verify(x - y < x - z, z < y)

    ck.verify(x < z + x, tvm.expr.LT(0, z))
    ck.verify(x < x + z, tvm.expr.LT(0, z))

    ck.verify(100 < x + 1, tvm.expr.LT(99, x))
    ck.verify(1 < 100 - x, tvm.expr.LT(x, 99))
    ck.verify(x * 3 < y * 3, x < y)
    ck.verify(x * (-3) < y * (-3), y < x)
    ck.verify(x * 3 >= y * 3, y <= x)

    ck.verify(x * 4 >= 2, tvm.expr.LE(1, x))
    ck.verify(x * 2 >= 50, tvm.expr.LE(25, x))
    ck.verify(x * 4 <= 2, x <= 0)
    ck.verify((0 - x * 3) <= 0, tvm.expr.LE(0, x))
    ck.verify((0 - x * 3) >= 0, tvm.expr.LE(x, 0))
    ck.verify(2 * x <= 0, x <= 0)

    ck.verify(x * 2 >= 3, tvm.expr.LE(2, x))
    ck.verify(x * 2 >= 2, tvm.expr.LE(1, x))
    ck.verify(x * 2 >= 1, tvm.expr.LE(1, x))
    ck.verify(x * 2 >= 0, tvm.expr.LE(0, x))
    ck.verify(x * 2 >= -1, tvm.expr.LE(0, x))
    ck.verify(x * 2 >= -2, tvm.expr.LE(-1, x))
    ck.verify(x * 2 >= -3, tvm.expr.LE(-1, x))

    ck.verify(x * 2 <= 3, tvm.expr.LE(x, 1))
    ck.verify(x * 2 <= 2, tvm.expr.LE(x, 1))
    ck.verify(x * 2 <= 1, tvm.expr.LE(x, 0))
    ck.verify(x * 2 <= 0, tvm.expr.LE(x, 0))
    ck.verify(x * 2 <= -1, tvm.expr.LE(x, -1))
    ck.verify(x * 2 <= -2, tvm.expr.LE(x, -1))
    ck.verify(x * 2 <= -3, tvm.expr.LE(x, -2))

    ck.verify(x * (-2) >= 3, tvm.expr.LE(x, -2))
    ck.verify(x * (-2) >= 2, tvm.expr.LE(x, -1))
    ck.verify(x * (-2) >= 1, tvm.expr.LE(x, -1))
    ck.verify(x * (-2) >= 0, tvm.expr.LE(x, 0))
    ck.verify(x * (-2) >= -1, tvm.expr.LE(x, 0))
    ck.verify(x * (-2) >= -2, tvm.expr.LE(x, 1))
    ck.verify(x * (-2) >= -3, tvm.expr.LE(x, 1))

    ck.verify(x * (-2) <= 3, tvm.expr.LE(-1, x))
    ck.verify(x * (-2) <= 2, tvm.expr.LE(-1, x))
    ck.verify(x * (-2) <= 1, tvm.expr.LE(0, x))
    ck.verify(x * (-2) <= 0, tvm.expr.LE(0, x))
    ck.verify(x * (-2) <= -1, tvm.expr.LE(1, x))
    ck.verify(x * (-2) <= -2, tvm.expr.LE(1, x))
    ck.verify(x * (-2) <= -3, tvm.expr.LE(2, x))

    # DivMod rules
    # truc div
    ck.verify(tdiv(x, 2) < 3, x < 6)
    ck.verify(3 < tdiv(x, 2), tvm.expr.LT(7, x))
    ck.verify(tdiv(x, 3) >= 0, tvm.expr.LE(-2, x))
    ck.verify(tdiv(x, 2) >= 1, tvm.expr.LE(2, x))
    ck.verify(tdiv(x, 2) >= 0, tvm.expr.LE(-1, x))
    ck.verify(tdiv(x, 2) >= -1, tvm.expr.LE(-3, x))

    ck.verify(tdiv(x, 2) <= 1, tvm.expr.LE(x, 3))
    ck.verify(tdiv(x, 2) <= 0, tvm.expr.LE(x, 1))
    ck.verify(tdiv(x, 2) <= -1, tvm.expr.LE(x, -2))

    ck.verify(tdiv(x, 4) * 4 < x, tvm.expr.LT(0, tmod(x, 4)))
    ck.verify(tdiv(x, 4) * 4 >= x, tvm.expr.LE(tmod(x, 4), 0))

    ck.verify(tdiv(x, 4) * 4 < x + y, tvm.expr.LT(0, tmod(x, 4) + y))
    ck.verify(tdiv(x, 4) * 4 < x - y, tvm.expr.LT(y, tmod(x, 4)))

    ck.verify(tdiv(x + 2, 4) * 4 >= x, tvm.expr.LE(tmod(x + 2, 4), 2))
    ck.verify(tdiv(x + 2, 4) * 4 >= x + y, tvm.expr.LE(tmod(x + 2, 4) + y, 2))
    ck.verify(tdiv(x + 2, 4) * 4 >= x - y, tvm.expr.LE(tmod(x + 2, 4) + (-2), y))

    # floor div
    ck.verify(fld(x, 2) < 3, x < 6)
    ck.verify(3 < fld(x, 2), tvm.expr.LT(7, x))
    ck.verify(-3 < fld(x, 2), tvm.expr.LT(-5, x))
    ck.verify(fld(x, 3) >= 0, tvm.expr.LE(0, x))
    ck.verify(fld(x, 2) >= 1, tvm.expr.LE(2, x))
    ck.verify(fld(x, 2) >= 0, tvm.expr.LE(0, x))
    ck.verify(fld(x, 2) >= -1, tvm.expr.LE(-2, x))

    ck.verify(fld(x, 2) <= 1, tvm.expr.LE(x, 3))
    ck.verify(fld(x, 2) <= 0, tvm.expr.LE(x, 1))
    ck.verify(fld(x, 2) <= -1, tvm.expr.LE(x, -1))

    ck.verify(fld(x, 4) * 4 < x, tvm.expr.LT(0, flm(x, 4)))
    ck.verify(fld(x, 4) * 4 >= x, tvm.expr.LE(flm(x, 4), 0))

    ck.verify(fld(x, 4) * 4 < x + y, tvm.expr.LT(0, flm(x, 4) + y))
    ck.verify(fld(x, 4) * 4 < x - y, tvm.expr.LT(y, flm(x, 4)))

    ck.verify(fld(x + 2, 4) * 4 >= x, tvm.expr.LE(flm(x + 2, 4), 2))
    ck.verify(fld(x + 2, 4) * 4 >= x + y, tvm.expr.LE(flm(x + 2, 4) + y, 2))
    ck.verify(fld(x + 2, 4) * 4 >= x - y, tvm.expr.LE(flm(x + 2, 4) + (-2), y))
    # End DivMod Rules

    ck.verify(tvm.min(x, 11) < 10, x < 10)
    ck.verify(tvm.min(x, 8) < 10, tvm.const(1, "bool"))
    ck.verify(tvm.max(8, x) > 10, tvm.expr.LT(10, x))
    ck.verify(x + 1 < tvm.max(8, x), x < 7)

    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 10), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(-10, 0), override=True)
    ck.analyzer.update(z, tvm.arith.ConstIntBound(-5, 5), override=True)

    ck.verify(x < 11, tvm.const(1, "bool"))
    ck.verify(x <= 10, tvm.const(1, "bool"))
    ck.verify(z <= 5, tvm.const(1, "bool"))
    ck.verify(x + y <= 10, tvm.const(1, "bool"))
    ck.verify(x + y >= -10, tvm.const(1, "bool"))
    ck.verify(z - 5 <= y + 10, tvm.const(1, "bool"))
    ck.verify(tvm.all(x > -1, z <= x + 5), tvm.const(1, "bool"))
    ck.verify(x*y <= 0, tvm.const(1, "bool"))
    ck.verify((x + 1)*(y - 1) < 0, tvm.const(1, "bool"))
    ck.verify(y*y >= 0, tvm.const(1, "bool"))
    ck.verify(x*6 <= -3, tvm.const(0, "bool"))
    ck.verify(tmod(y - 1, 3) == 0, tmod(y + (-1), 3) == 0)


def test_logical_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")

    ck.verify(tvm.expr.And(tvm.expr.EQ(x, y), tvm.expr.NE(x, y)),
              tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(tvm.expr.NE(x, y), tvm.expr.EQ(x, y)),
              tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(x > 1, tvm.expr.Not(x > 1)), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(x <= y, y < x), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(y < x, y <= x), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(x < 1, 0 < x), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(x < 0, 1 < x), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(x < 1, 1 <= x), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(x <= 1, 1 < x), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(1 <= x, x < 1), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(1 < x, x <= 1), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(x <= 1, 2 <= x), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(2 <= x, x <= 1), tvm.const(False, "bool"))
    ck.verify(tvm.expr.And(x == 1, x != 2), x == 1)


    ck.verify(tvm.expr.Or(tvm.expr.EQ(x, y), tvm.expr.NE(x, y)),
              tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(tvm.expr.NE(x, y), tvm.expr.EQ(x, y)),
              tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(x > y, tvm.expr.Not(x > y)), tvm.const(True, "bool"))

    ck.verify(tvm.expr.Or(x <= y, y < x), tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(y < x, y >= x), tvm.const(True, "bool"))

    ck.verify(tvm.expr.Or(x < 1, 0 < x), tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(0 < x, x < 1), tvm.const(True, "bool"))

    ck.verify(tvm.expr.Or(x < 1, 1 <= x), tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(x <= 1, 1 < x), tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(1 <= x, x < 1), tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(1 < x, x <= 1), tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(x <= 1, 2 <= x), tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(2 <= x, x <= 1), tvm.const(True, "bool"))
    ck.verify(tvm.expr.Or(x != 1, x == 2), x != 1)

def test_let_simplify():
    ck = RewriteChecker()
    x, y = tvm.var("x"), tvm.var("y")
    z = tvm.expr.Let(x, 1, x + 1)
    ck.verify(z + z, 4)

def test_cast_simplify():
    ck = RewriteChecker()
    x = tvm.var("x")

    dtypes = ["float32", "float16", "int32", "int8", "bool"]
    for dtype1 in dtypes:
        ck.verify(tvm.expr.Cast(dtype1, x - x), tvm.const(0, dtype1))
        ck.verify(tvm.expr.Cast(dtype1, x == x), tvm.const(1, dtype1))
        for dtype2 in dtypes:
            for i in [0, 1, 2, 3]:
                ck.verify(tvm.expr.Cast(dtype1, tvm.const(i, dtype2)), tvm.const(i, dtype1))

if __name__ == "__main__":
    test_floordiv_index_simplify()
    test_floormod_index_simplify()
    test_cmp_simplify()
    test_vector_simplify()
    test_add_index_simplify()
    test_sub_index_simplify()
    test_mul_index_simplify()
    test_div_index_simplify()
    test_max_index_simplify()
    test_min_index_simplify()
    test_mod_index_simplify()
    test_select_simplify()
    test_logical_simplify()
    test_let_simplify()
    test_cast_simplify()
