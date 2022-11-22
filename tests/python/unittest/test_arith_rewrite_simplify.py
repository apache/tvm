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
import pytest
import tvm
from tvm import te, tir


class RewriteChecker:
    def __init__(self):
        self.analyzer = tvm.arith.Analyzer()

    def verify(self, data, expected):
        res = self.analyzer.rewrite_simplify(data)
        assert tvm.ir.structural_equal(res, expected), "data={}, res={}, expected={}".format(
            data, res, expected
        )


def test_vector_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    # Add rules
    ck.verify(tvm.tir.Ramp(x, 1, 4) + tvm.tir.Ramp(y, 2, 4), tvm.tir.Ramp(x + y, 3, 4))
    ck.verify(tvm.tir.Ramp(x, 1, 2) + y, tvm.tir.Ramp(x + y, 1, 2))
    ck.verify(y + tvm.tir.Ramp(x, 1, 2), tvm.tir.Ramp(y + x, 1, 2))
    ck.verify(y.astype("int32x2") + x.astype("int32x2"), (y + x).astype("int32x2"))
    ck.verify(tvm.tir.Broadcast(0, 4) + y, tvm.tir.Broadcast(y, 4))
    ck.verify(
        tvm.tir.Ramp(x, 1, 4).astype("float32x4") + tvm.tir.Broadcast(0.0, 4),
        tvm.tir.Ramp(x, 1, 4).astype("float32x4"),
    )
    # Sub rules
    ck.verify(tvm.tir.Ramp(x, 4, 4) - tvm.tir.Ramp(y, 2, 4), tvm.tir.Ramp(x - y, 2, 4))
    ck.verify(tvm.tir.Ramp(x, 1, 2) - y, tvm.tir.Ramp(x - y, 1, 2))
    ck.verify(y - tvm.tir.Ramp(x, 1, 2), tvm.tir.Ramp(y - x, -1, 2))
    ck.verify(y.astype("int32x2") - x.astype("int32x2"), (y - x).astype("int32x2"))

    # Mul rules
    ck.verify(y.astype("int32x2") * x.astype("int32x2"), (y * x).astype("int32x2"))
    ck.verify(tvm.tir.Ramp(x, 4, 4) * 2, tvm.tir.Ramp(x * 2, 8, 4))
    ck.verify(2 * tvm.tir.Ramp(x, 4, 4), tvm.tir.Ramp(x * 2, 8, 4))
    ck.verify(tvm.tir.Broadcast(0, 4) * x, tvm.tir.Broadcast(0, 4))
    ck.verify(tvm.tir.Broadcast(0.0, 4) * x, tvm.tir.Broadcast(0.0, 4))

    ## DivMod rules
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    # truc div
    ck.verify(tdiv(y.astype("int32x2"), x.astype("int32x2")), tdiv(y, x).astype("int32x2"))
    ck.verify(tdiv(tvm.tir.Ramp(x, 4, 4), 2), tvm.tir.Ramp(tdiv(x, 2), 2, 4))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify(tdiv(tvm.tir.Ramp(x * 8 + 1, 1, 4), 8), (x).astype("int32x4"))
    ck.verify(tdiv(tvm.tir.Ramp(x * 8 + 15, 1, 4), 8), tdiv(tvm.tir.Ramp(x * 8 + 15, 1, 4), 8))
    # truc mod
    ck.verify(tmod(y.astype("int32x2"), x.astype("int32x2")), tmod(y, x).astype("int32x2"))
    ck.verify(tmod(tvm.tir.Ramp(x, 4, 4), 2), tvm.tir.Broadcast(tmod(x, 2), 4))
    ck.verify(tmod(tvm.tir.Ramp(x * 8 + 1, 1, 4), 8), tvm.tir.Ramp(1, 1, 4))
    ck.verify(tmod(tvm.tir.Ramp(x * 8 + 1, 15, 4), 8), tmod(tvm.tir.Ramp(1, 15, 4), 8))

    # floor div
    fld = tvm.te.floordiv
    flm = tvm.te.floormod
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-10, 1000), override=True)
    ck.verify(fld(y.astype("int32x2"), x.astype("int32x2")), fld(y, x).astype("int32x2"))
    ck.verify(fld(tvm.tir.Ramp(x, 4, 4), 2), tvm.tir.Ramp(fld(x, 2), 2, 4))
    ck.verify(fld(tvm.tir.Ramp(x * 8 + 1, 1, 4), 8), (x).astype("int32x4"))
    ck.verify(fld(tvm.tir.Ramp(x * 8 + 15, 1, 4), 8), fld(tvm.tir.Ramp(x * 8 + 15, 1, 4), 8))
    ck.verify(fld(tvm.tir.Ramp(x, 8, 5), tvm.tir.Broadcast(4, 5)), tvm.tir.Ramp(fld(x, 4), 2, 5))
    ck.verify(
        fld(tvm.tir.Ramp(flm(x * 4, 256), 1, 4), tvm.tir.Broadcast(8, 4)),
        tvm.tir.Broadcast(fld(flm(x * 4, 256), 8), 4),
    )
    ck.verify(
        fld(tvm.tir.Ramp(x, 7, 4), tvm.tir.Broadcast(4, 4)),
        fld(tvm.tir.Ramp(x, 7, 4), tvm.tir.Broadcast(4, 4)),
    )
    ck.verify(fld(tvm.tir.Ramp(x * 8, 1, 4), tvm.tir.Broadcast(4, 4)), tvm.tir.Broadcast(x * 2, 4))
    ck.verify(
        fld(tvm.tir.Ramp(x * 8, 3, 4), tvm.tir.Broadcast(4, 4)),
        fld(tvm.tir.Ramp(x * 8, 3, 4), tvm.tir.Broadcast(4, 4)),
    )
    ck.verify(
        fld(tvm.tir.Ramp(x * 8 + 15, 1, 4), tvm.tir.Broadcast(4, 4)),
        fld(tvm.tir.Ramp(x * 8 + 15, 1, 4), tvm.tir.Broadcast(4, 4)),
    )
    ck.verify(
        fld(tvm.tir.Ramp(x * 4, 1, 4), tvm.tir.Broadcast(64, 4)), tvm.tir.Broadcast(fld(x, 16), 4)
    )
    ck.verify(
        fld(tvm.tir.Ramp(x * 8, 2, 4), tvm.tir.Broadcast(64, 4)), tvm.tir.Broadcast(fld(x, 8), 4)
    )
    ck.verify(
        fld(tvm.tir.Ramp(x * 4, 1, 5), tvm.tir.Broadcast(64, 5)),
        fld(tvm.tir.Ramp(x * 4, 1, 5), tvm.tir.Broadcast(64, 5)),
    )  # Example negative case: x = 15; [60, 61, 62, 63, 64] / 64 = [0, 0, 0, 0, 1]
    ck.verify(
        fld(tvm.tir.Ramp(x * 4 + 3, 1, 4), tvm.tir.Broadcast(64, 4)),
        fld(tvm.tir.Ramp(x * 4 + 3, 1, 4), tvm.tir.Broadcast(64, 4)),
    )  # Example negative case: x = 15; [63, 64, 65, 66] % 64 = [0, 1, 1, 1]
    ck.verify(
        fld(tvm.tir.Ramp(x * 7, 1, 4), tvm.tir.Broadcast(64, 4)),
        fld(tvm.tir.Ramp(x * 7, 1, 4), tvm.tir.Broadcast(64, 4)),
    )  # Example negative case: x = 9; [63, 70, 77, 84] % 64 = [0, 1, 1, 1]

    # floor mod
    ck.verify(flm(y.astype("int32x2"), x.astype("int32x2")), flm(y, x).astype("int32x2"))
    ck.verify(flm(tvm.tir.Ramp(x, 4, 4), 2), tvm.tir.Broadcast(flm(x, 2), 4))
    ck.verify(flm(tvm.tir.Ramp(x * 8 + 1, 1, 4), 8), tvm.tir.Ramp(1, 1, 4))
    ck.verify(flm(tvm.tir.Ramp(x * 8 + 1, 15, 4), 8), flm(tvm.tir.Ramp(1, 15, 4), 8))
    ck.verify(flm(tvm.tir.Ramp(x, 8, 4), tvm.tir.Broadcast(4, 4)), tvm.tir.Broadcast(flm(x, 4), 4))
    ck.verify(
        flm(tvm.tir.Ramp(x, 7, 4), tvm.tir.Broadcast(4, 4)),
        flm(tvm.tir.Ramp(x, 7, 4), tvm.tir.Broadcast(4, 4)),
    )
    ck.verify(flm(tvm.tir.Ramp(x * 8, 1, 4), tvm.tir.Broadcast(4, 4)), tvm.tir.Ramp(0, 1, 4))
    ck.verify(
        flm(tvm.tir.Ramp(x * 8, 1, 5), tvm.tir.Broadcast(4, 5)),
        flm(tvm.tir.Ramp(0, 1, 5), tvm.tir.Broadcast(4, 5)),
    )
    ck.verify(
        flm(tvm.tir.Ramp(x * 8 + 7, 1, 4), tvm.tir.Broadcast(4, 4)),
        flm(tvm.tir.Ramp(3, 1, 4), tvm.tir.Broadcast(4, 4)),
    )
    ck.verify(
        flm(tvm.tir.Ramp(x * 4, 1, 4), tvm.tir.Broadcast(64, 4)), tvm.tir.Ramp(flm(x * 4, 64), 1, 4)
    )
    ck.verify(
        flm(tvm.tir.Ramp(x * 8, 2, 4), tvm.tir.Broadcast(64, 4)), tvm.tir.Ramp(flm(x * 8, 64), 2, 4)
    )
    ck.verify(
        flm(tvm.tir.Ramp(x * 4, 1, 5), tvm.tir.Broadcast(64, 5)),
        flm(tvm.tir.Ramp(x * 4, 1, 5), tvm.tir.Broadcast(64, 5)),
    )  # Example negative case: x = 15; [60, 61, 62, 63, 64] % 64 = [60, 61, 62, 63, 0]
    ck.verify(
        flm(tvm.tir.Ramp(x * 4 + 3, 1, 4), tvm.tir.Broadcast(64, 4)),
        flm(tvm.tir.Ramp(x * 4 + 3, 1, 4), tvm.tir.Broadcast(64, 4)),
    )  # Example negative case: x = 15; [63, 64, 65, 66] % 64 = [63, 0, 1, 2]
    ck.verify(
        flm(tvm.tir.Ramp(x * 2, 1, 8), tvm.tir.Broadcast(20, 8)),
        flm(tvm.tir.Ramp(x * 2, 1, 8), tvm.tir.Broadcast(20, 8)),
    )  # Example negative case: x = 9; [18, 19, 20, ..., 25] % 20 = [18, 19, 0, 1, ..., 5]
    ck.verify(
        flm(tvm.tir.Ramp(x * 7, 1, 4), tvm.tir.Broadcast(64, 4)),
        flm(tvm.tir.Ramp(x * 7, 1, 4), tvm.tir.Broadcast(64, 4)),
    )  # Example negative case: x = 9; [63, 70, 77, 84] % 64 = [63, 6, 13, 20]

    # Min/Max rules
    vx = te.var("vx", dtype="int32x2")
    vc = te.var("vc", dtype="uint1")
    ck.verify(
        tvm.te.min(y.astype("int32x2"), x.astype("int32x2")), tvm.te.min(y, x).astype("int32x2")
    )
    ck.verify(
        tvm.te.min(tvm.te.min(vx, y.astype("int32x2")), x.astype("int32x2")),
        tvm.te.min(vx, tvm.te.min(y, x).astype("int32x2")),
    )
    ck.verify(
        tvm.te.max(y.astype("int32x2"), x.astype("int32x2")), tvm.te.max(y, x).astype("int32x2")
    )
    ck.verify(
        tvm.te.max(tvm.te.max(vx, y.astype("int32x2")), x.astype("int32x2")),
        tvm.te.max(vx, tvm.te.max(y, x).astype("int32x2")),
    )

    ## Logical rules
    ck.verify(y.astype("int32x2").equal(x.astype("int32x2")), (y.equal(x)).astype("uint1x2"))
    ck.verify(
        tvm.tir.NE(y.astype("int32x2"), (x.astype("int32x2"))), (tvm.tir.NE(y, x)).astype("uint1x2")
    )
    ck.verify(y.astype("int32x2") > x.astype("int32x2"), (x < y).astype("uint1x2"))
    ck.verify(y.astype("int32x2") >= x.astype("int32x2"), (x <= y).astype("uint1x2"))
    ck.verify(y.astype("int32x2") < x.astype("int32x2"), (y < x).astype("uint1x2"))
    ck.verify(y.astype("int32x2") <= x.astype("int32x2"), (y <= x).astype("uint1x2"))
    ck.verify(
        tvm.tir.And(y.astype("int32x2") <= x.astype("int32x2"), vc.astype("uint1x2")),
        (tvm.tir.And(y <= x, vc)).astype("uint1x2"),
    )
    ck.verify(
        tvm.tir.Or(y.astype("int32x2") <= x.astype("int32x2"), vc.astype("uint1x2")),
        (tvm.tir.Or(y <= x, vc)).astype("uint1x2"),
    )


def test_select_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    # Add rules
    ck.verify(
        tvm.tir.Select(x < 0, y, 0) + tvm.tir.Select(x < 0, 1, z), tvm.tir.Select(x < 0, y + 1, z)
    )
    ck.verify(
        tvm.tir.Select(x < 0, y, 1) - tvm.tir.Select(x < 0, 1, z),
        tvm.tir.Select(x < 0, y + (-1), 1 - z),
    )
    ck.verify(tvm.tir.Select(x < 0, y, z) - y, tvm.tir.Select(x < 0, 0, z - y))
    ck.verify(tvm.tir.Select(x < 0, y, z) - z, tvm.tir.Select(x < 0, y - z, 0))
    ck.verify(
        tvm.te.min(tvm.tir.Select(x < 0, y, 0), tvm.tir.Select(x < 0, 1, z)),
        tvm.tir.Select(x < 0, tvm.te.min(y, 1), tvm.te.min(0, z)),
    )
    ck.verify(
        tvm.te.max(tvm.tir.Select(x < 0, y, 0), tvm.tir.Select(x < 0, 1, z)),
        tvm.tir.Select(x < 0, tvm.te.max(y, 1), tvm.te.max(0, z)),
    )

    ck.verify(tvm.tir.Select(x * 3 + 1 != 0, y, z), y)
    ck.verify(tvm.tir.Select(x * 3 + 1 == 0, y, z), z)
    ck.verify(tvm.tir.Select(x > 0, y + 1, y + 1), y + 1)


def test_add_index_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    ck.verify(x + (y - x), y)
    ck.verify(x - (y + 1) + (y + 1), x)
    ck.verify((x - 10) + (10 - z), x - z)
    ck.verify((x - y) + (z - x), z - y)

    ck.verify(tvm.te.min(x, y - z) + z, tvm.te.min(x + z, y))
    ck.verify(tvm.te.min(x - z, y) + z, tvm.te.min(x, y + z))
    ck.verify(tvm.te.max(x, y - 10) + 10, tvm.te.max(x + 10, y))
    ck.verify(tvm.te.max(x - 11, y) + 11, tvm.te.max(x, y + 11))

    ck.verify(tvm.te.max(x, y * 2) + tvm.te.min(x, y * 2), x + y * 2)
    ck.verify(tvm.te.min(x, y * 2) + tvm.te.max(x, y * 2), x + y * 2)

    ck.verify(tvm.te.max(x, y + 2) + (-2), tvm.te.max(x + (-2), y))
    ck.verify(tvm.te.min(x, y + 2) + (-2), tvm.te.min(x + (-2), y))
    ck.verify(tvm.te.min(x + 2, y + 3) + (-2), tvm.te.min(x, y + 1))

    ck.verify(tvm.te.max(0, 1 - x * 4) + x * 4, tvm.te.max(x * 4, 1))
    ck.verify(tvm.te.max(2 - x * 4, 0) + x * 4, tvm.te.max(x * 4, 2))

    ck.verify(tvm.te.min(0, 1 - x * 4) + x * 4, tvm.te.min(x * 4, 1))
    ck.verify(tvm.te.min(2 - x * 4, 0) + x * 4, tvm.te.min(x * 4, 2))

    ck.verify(x * y + x * 10, x * (y + 10))
    ck.verify(y * x + x * 10, x * (y + 10))
    ck.verify(y * x + 10 * x, x * (y + 10))
    ck.verify(x * y + 10 * x, x * (y + 10))

    ck.verify((2 * z) + tvm.te.min(x, y - (2 * z)), tvm.te.min(x + (z * 2), y))
    ck.verify(y * x + x, x * (y + 1))
    ck.verify(x * y + x, x * (y + 1))
    ck.verify((x + 10) + 13, x + 23)
    ck.verify((x + 10) + (13 + z), x + z + 23)
    ck.verify(x * y + 10 * x, x * (y + 10))
    ck.verify(y * x + x * 3, x * (y + 3))
    ck.verify(x + 3 + y, x + y + 3)
    ck.verify((3 - y) + x, x - y + 3)

    # canonicalization
    ck.verify(x + 2 + 3 + 4 + x, x * 2 + 9)
    ck.verify(x + 2 + 3 + 4 + x * 3, x * 4 + 9)

    # DivMod rules
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    # truc div
    ck.verify(y * tmod(x, 8) + 10 * tmod(x, 8), tmod(x, 8) * (y + 10))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1, 1000), override=True)
    ck.verify(tdiv(x, 8) * 8 + tmod(x, 8), x)

    # floor div
    fld = tvm.te.floordiv
    flm = tvm.te.floormod
    ck.verify(y * flm(x, 8) + 10 * flm(x, 8), flm(x, 8) * (y + 10))
    ck.verify(fld(x, 8) * 8 + flm(x, 8), x)
    ck.verify(fld(flm(x, 2) + 7, 2) + fld(x, 2), fld(x + 7, 2))


def test_sub_index_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    a, b = tvm.tir.Any(), tvm.tir.Any()

    ck.verify(x + y - y, x)
    ck.verify(x + y - x, y)
    ck.verify(x - (y + x), 0 - y)
    ck.verify(x - (x + y), 0 - y)

    ck.verify(tvm.te.min(x, y) - x, tvm.te.min(0, y - x))
    ck.verify(tvm.te.min(x, y) - y, tvm.te.min(x - y, 0))
    ck.verify(tvm.te.max(x, y) - x, tvm.te.max(0, y - x))
    ck.verify(tvm.te.max(x, y) - y, tvm.te.max(x - y, 0))

    ck.verify(x - tvm.te.min(x, y), tvm.te.max(0, x - y))
    ck.verify(y - tvm.te.min(x, y), tvm.te.max(y - x, 0))
    ck.verify(x - tvm.te.max(x, y), tvm.te.min(0, x - y))
    ck.verify(y - tvm.te.max(x, y), tvm.te.min(y - x, 0))

    # mul co-efficient foldng
    ck.verify(x - x, 0)
    ck.verify(a - a, 0)
    ck.verify(a - b, a - b)
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

    ck.verify(tvm.te.min(x + y, z) - x, tvm.te.min(y, z - x))
    ck.verify(tvm.te.min(y + x, z) - x, tvm.te.min(y, z - x))
    ck.verify(tvm.te.min(z, x + y) - x, tvm.te.min(z - x, y))
    ck.verify(tvm.te.min(z, y + x) - x, tvm.te.min(z - x, y))

    ck.verify(tvm.te.max(x + y, z) - x, tvm.te.max(y, z - x))
    ck.verify(tvm.te.max(y + x, z) - x, tvm.te.max(y, z - x))
    ck.verify(tvm.te.max(z, x + y) - x, tvm.te.max(z - x, y))
    ck.verify(tvm.te.max(z, y + x) - x, tvm.te.max(z - x, y))

    ck.verify(x - tvm.te.min(x + y, z), tvm.te.max(0 - y, x - z))
    ck.verify(x - tvm.te.min(y + x, z), tvm.te.max(0 - y, x - z))
    ck.verify(x - tvm.te.min(z, x + y), tvm.te.max(x - z, 0 - y))
    ck.verify(x - tvm.te.min(z, y + x), tvm.te.max(x - z, 0 - y))

    ck.verify(tvm.te.min(x, y) - tvm.te.min(y, x), 0)
    ck.verify(tvm.te.max(x, y) - tvm.te.max(y, x), 0)
    ck.verify(tvm.te.min(x, y) - tvm.te.min(x + 10, y + 10), -10)
    ck.verify(tvm.te.min(x + 10, y + 1) - tvm.te.min(x, y - 9), 10)

    # DivMod patterns
    # truc div
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
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
    fld = tvm.te.floordiv
    flm = tvm.te.floormod
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
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    ck.verify((x + 2) * 3, x * 3 + 6)
    ck.verify((x * 2) * 3, x * 6)
    ck.verify(tvm.te.min(x, y) * tvm.te.max(x, y), x * y)
    ck.verify(tvm.te.max(x, y) * tvm.te.min(x, y), x * y)
    ck.verify((x - y) * (-2), (y - x) * 2)


def test_div_index_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod

    ck.verify(tdiv(x, x), 1)
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(z, tvm.arith.ConstIntBound(0, 1000), override=True)

    ck.verify(tdiv(tdiv(x, 2), 3), tdiv(x, 6))
    ck.verify(tdiv(tdiv(x, 2) + 1, 3), tdiv(x + 2, 6))
    ck.verify(tdiv(x * 2, 4), tdiv(x, 2))
    ck.verify(tdiv(x * 4, 2), x * 2)

    ck.verify(tdiv(x * 4 + y, 2), x * 2 + tdiv(y, 2))
    ck.verify(tdiv(tvm.te.min(x * 6, y), 2), tvm.te.min(x * 3, tdiv(y, 2)))
    ck.verify(tdiv(tvm.te.max(x * 6, y), 2), tvm.te.max(x * 3, tdiv(y, 2)))

    ck.verify(tdiv(y + x * 4, 2), tdiv(y, 2) + x * 2)
    ck.verify(tdiv(tvm.te.min(y, x * 6), 2), tvm.te.min(tdiv(y, 2), x * 3))
    ck.verify(tdiv(tvm.te.max(y, x * 6), 2), tvm.te.max(tdiv(y, 2), x * 3))

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
    fld = tvm.te.floordiv
    flm = tvm.te.floormod
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    ck.verify(fld(fld(x, 2), 3), fld(x, 6))
    ck.verify(fld(fld(x, 2) + 1, 3), fld(x + 2, 6))
    ck.verify(fld(x - flm(x, 21), 21), fld(x, 21))

    ck.verify(fld(x * 2, 4), fld(x, 2))
    ck.verify(fld(x * 4, 2), x * 2)
    ck.verify(fld(x * 8 + 7, 16), fld(x, 2))
    ck.verify(fld(x * 8 + 39, 16), fld(x, 2) + 2)
    ck.verify(fld(x * 8 - 1, 16), fld(x * 8 + -1, 16))
    ck.verify(fld(x * 8 - 9, 16), fld(x, 2) + -1)

    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 7), override=True)
    ck.verify(fld(x * 360 + y, 16), x * 22)
    ck.verify(fld(x * 360 + y, 25), x * 14)
    ck.verify(fld(x * 360 - 8, 25), fld(x * 360 + -8, 25))

    ck.verify(fld(x * 4 + y, 2), x * 2 + fld(y, 2))
    ck.verify(fld(tvm.te.min(x * 6, y), 2), tvm.te.min(x * 3, fld(y, 2)))
    ck.verify(fld(tvm.te.max(x * 6, y), 2), tvm.te.max(x * 3, fld(y, 2)))

    ck.verify(fld(y + x * 4, 2), x * 2 + fld(y, 2))
    ck.verify(fld(tvm.te.min(y, x * 6), 2), tvm.te.min(fld(y, 2), x * 3))
    ck.verify(fld(tvm.te.max(y, x * 6), 2), tvm.te.max(fld(y, 2), x * 3))

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

    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 31), override=True)
    ck.analyzer.update(z, tvm.arith.ConstIntBound(0, 3), override=True)
    ck.verify(fld(x * 32 + y, 64), fld(x, 2))
    ck.verify(fld(x * 128 + y * 4 + z, 512), fld(x, 4))


def test_mod_index_simplify():
    ck = RewriteChecker()
    x, y, nx, ny, z = te.var("x"), te.var("y"), te.var("nx"), te.var("ny"), te.var("z")
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(nx, tvm.arith.ConstIntBound(-1000, 0), override=True)
    ck.analyzer.update(ny, tvm.arith.ConstIntBound(-1000, 0), override=True)
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod

    ck.verify(tmod(x * 10, 2), 0)
    ck.verify(tmod(x * 10 + y, 2), tmod(y, 2))
    ck.verify(tmod(x + 10, 2), tmod(x, 2))
    ck.verify(tmod(x + y * 10, 2), tmod(x, 2))
    ck.verify(tmod(x * 10 + 1 + y * 2 + 2, 2), 1)
    ck.verify(tmod(x * 10, -2), 0)
    ck.verify(tmod(x * 10 + y, -2), tmod(y, 2))
    ck.verify(tmod(x + 10, -2), tmod(x, 2))
    ck.verify(tmod(x + y * 10, -2), tmod(x, 2))
    ck.verify(tmod(x * 10 + 1 + y * 2 + 2, -2), 1)

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
    flm = tvm.te.floormod
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    ck = RewriteChecker()
    x, y, nx, ny, z = te.var("x"), te.var("y"), te.var("nx"), te.var("ny"), te.var("z")

    ck.verify(flm(x * 10, 2), 0)
    ck.verify(flm(x * 9600, 6400), flm(x * 3200, 6400))
    ck.verify(flm(x * 10 + y, 2), flm(y, 2))
    ck.verify(flm(x * 360 + y, 16), flm(x * 8 + y, 16))
    ck.verify(flm(x + 10, 2), flm(x, 2))
    ck.verify(flm(x + y * 10, 2), flm(x, 2))
    ck.verify(flm(x + y * 360, 16), flm(x + y * 8, 16))
    ck.verify(flm(x * 10 + 1 + y * 2 + 2, 2), 1)
    ck.verify(flm(x * (-10), 2), 0)
    ck.verify(flm(x * (-10) + y, 2), flm(y, 2))
    ck.verify(flm(x + (-10), 2), flm(x, 2))
    ck.verify(flm(x + y * (-10), 2), flm(x, 2))

    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 31), override=True)
    ck.verify(flm(x * 32 + y, 64), flm(x, 2) * 32 + y)
    ck.verify(flm(x * 32 - y, 64), flm(x * 32 - y, 64))


def test_min_index_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    fld = tvm.te.floordiv
    flm = tvm.te.floormod
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    # const int bound
    ck.verify(tvm.te.min(tmod(x, 2), tmod(y, 2) + 10), tmod(x, 2))
    ck.verify(tvm.te.min(flm(x, 2), flm(y, 2) + 10), flm(x, 2))

    ck.verify(tvm.te.min(x + 1, x + 10), x + 1)
    ck.verify(tvm.te.min(x + 111, x + 10), x + 10)
    ck.verify(tvm.te.min(x + 1, x), x)
    ck.verify(tvm.te.min(x, x + 2), x)
    ck.verify(tvm.te.min(1 - x, 2 - x), 1 - x)
    ck.verify(tvm.te.min(3 - x, 2 - x), 2 - x)

    ck.verify(tvm.te.min(tvm.te.max(x, y), tvm.te.min(x, y)), tvm.te.min(x, y))
    ck.verify(tvm.te.min(tvm.te.max(x, y), tvm.te.min(y, x)), tvm.te.min(x, y))

    ck.verify(tvm.te.min(tvm.te.max(x, y), x), x)
    ck.verify(tvm.te.min(tvm.te.max(y, x), x), x)
    ck.verify(tvm.te.min(tvm.te.min(x, y), x), tvm.te.min(x, y))
    ck.verify(tvm.te.min(tvm.te.min(x, y), y), tvm.te.min(x, y))

    ck.verify(tvm.te.min(x, tvm.te.max(x, y)), x)
    ck.verify(tvm.te.min(x, tvm.te.max(y, x)), x)
    ck.verify(tvm.te.min(x, tvm.te.min(x, y)), tvm.te.min(x, y))
    ck.verify(tvm.te.min(y, tvm.te.min(x, y)), tvm.te.min(x, y))

    ck.verify(tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), y), tvm.te.min(tvm.te.min(x, y), z))
    ck.verify(
        tvm.te.min(tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), x * 2), y),
        tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), x * 2),
    )
    ck.verify(
        tvm.te.min(tvm.te.min(tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), x * 2), z * 2), y),
        tvm.te.min(tvm.te.min(tvm.te.min(tvm.te.min(x, y), z), x * 2), z * 2),
    )

    ck.verify(tvm.te.min(tvm.te.max(x, y), tvm.te.max(x, z)), tvm.te.max(tvm.te.min(y, z), x))
    ck.verify(tvm.te.min(tvm.te.max(x, y), tvm.te.max(z, x)), tvm.te.max(tvm.te.min(y, z), x))
    ck.verify(tvm.te.min(tvm.te.max(y, x), tvm.te.max(x, z)), tvm.te.max(tvm.te.min(y, z), x))
    ck.verify(tvm.te.min(tvm.te.max(y, x), tvm.te.max(z, x)), tvm.te.max(tvm.te.min(y, z), x))

    ck.verify(tvm.te.min(y + x, z + x), tvm.te.min(y, z) + x)
    ck.verify(tvm.te.min(y + x, x + z), tvm.te.min(y, z) + x)
    ck.verify(tvm.te.min(x + y, z + x), tvm.te.min(y, z) + x)
    ck.verify(tvm.te.min(x + y, x + z), tvm.te.min(y, z) + x)

    ck.verify(tvm.te.min(x - y, x - z), x - tvm.te.max(y, z))
    ck.verify(tvm.te.min(y - x, z - x), tvm.te.min(y, z) - x)

    ck.verify(tvm.te.min(tvm.te.min(x, 1), 10), tvm.te.min(x, 1))
    ck.verify(tvm.te.min(tvm.te.min(x, 11), 10), tvm.te.min(x, 10))

    ck.verify(tvm.te.min(x * 3, 9), tvm.te.min(x, 3) * 3)
    ck.verify(tvm.te.min(x * 2, 0), tvm.te.min(x, 0) * 2)
    ck.verify(tvm.te.min(0 - x * 2, 0), tvm.te.max(x, 0) * -2)
    ck.verify(tvm.te.min(3 - x, 2), 3 - tvm.te.max(x, 1))
    ck.verify(tvm.te.min(x * (-2), -4), tvm.te.max(x, 2) * -2)
    ck.verify(tvm.te.min(x * (-2), 4), tvm.te.max(x, -2) * -2)
    ck.verify(tvm.te.min(x * (0), 4), 0)
    ck.verify(tvm.te.min(x * (0), -4), -4)

    # DivMod rules
    # truc div
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000))
    ck.verify(tvm.te.min(tdiv(x + 3, 4) * 4, x), x)
    ck.verify(tvm.te.min(tdiv(x + 3, 4) * 4, tvm.te.max(x, 4)), tvm.te.max(x, 4))
    ck.verify(tvm.te.min(x, tdiv(x + 3, 4) * 4), x)
    ck.verify(tvm.te.min(tvm.te.max(x, 4), tdiv(x + 3, 4) * 4), tvm.te.max(x, 4))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1000, 1000), True)
    ck.verify(tvm.te.min(tdiv(x, 10), tdiv(y, 10)), tdiv(tvm.te.min(x, y), 10))
    ck.verify(tvm.te.min(tdiv(x, (-10)), tdiv(y, (-10))), tdiv(tvm.te.max(x, y), (-10)))

    # floor div
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1000, 1000), True)
    ck.verify(tvm.te.min(fld(x + 3, 4) * 4, x), x)
    ck.verify(tvm.te.min(fld(x + 3, 4) * 4, tvm.te.max(x, 4)), tvm.te.max(x, 4))
    ck.verify(tvm.te.min(x, fld(x + 3, 4) * 4), x)
    ck.verify(tvm.te.min(x, fld(x, 4) * 4), fld(x, 4) * 4)
    ck.verify(tvm.te.min(tvm.te.max(x, 4), fld(x + 3, 4) * 4), tvm.te.max(x, 4))
    ck.verify(tvm.te.min(fld(x, 10), fld(y, 10)), fld(tvm.te.min(x, y), 10))
    ck.verify(tvm.te.min(fld(x, (-10)), fld(y, (-10))), fld(tvm.te.max(x, y), (-10)))


def test_max_index_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    flm = tvm.te.floormod
    fld = tvm.te.floordiv
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    # const int bound
    ck.verify(tvm.te.max(tmod(x, 2), tmod(y, 2) + 10), tmod(y, 2) + 10)
    ck.verify(tvm.te.max(flm(x, 2), flm(y, 2) + 10), flm(y, 2) + 10)

    ck.verify(tvm.te.max(x + 1, x + 10), x + 10)
    ck.verify(tvm.te.max(x + 111, x + 10), x + 111)
    ck.verify(tvm.te.max(x + 1, x), x + 1)
    ck.verify(tvm.te.max(x, x + 2), x + 2)
    ck.verify(tvm.te.max(1 - x, 2 - x), 2 - x)
    ck.verify(tvm.te.max(3 - x, 2 - x), 3 - x)

    ck.verify(tvm.te.max(tvm.te.min(x, y), tvm.te.max(x, y)), tvm.te.max(x, y))
    ck.verify(tvm.te.max(tvm.te.min(x, y), tvm.te.max(y, x)), tvm.te.max(x, y))

    ck.verify(tvm.te.max(tvm.te.min(x, y), x), x)
    ck.verify(tvm.te.max(tvm.te.min(y, x), x), x)
    ck.verify(tvm.te.max(tvm.te.max(x, y), x), tvm.te.max(x, y))
    ck.verify(tvm.te.max(tvm.te.max(x, y), y), tvm.te.max(x, y))

    ck.verify(tvm.te.max(x, tvm.te.min(x, y)), x)
    ck.verify(tvm.te.max(x, tvm.te.min(y, x)), x)
    ck.verify(tvm.te.max(x, tvm.te.max(x, y)), tvm.te.max(x, y))
    ck.verify(tvm.te.max(y, tvm.te.max(x, y)), tvm.te.max(x, y))

    ck.verify(tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), y), tvm.te.max(tvm.te.max(x, y), z))
    ck.verify(
        tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), x * 2), y),
        tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), x * 2),
    )
    ck.verify(
        tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), x * 2), z * 2), y),
        tvm.te.max(tvm.te.max(tvm.te.max(tvm.te.max(x, y), z), x * 2), z * 2),
    )

    ck.verify(tvm.te.max(tvm.te.min(x, y), tvm.te.min(x, z)), tvm.te.min(tvm.te.max(y, z), x))
    ck.verify(tvm.te.max(tvm.te.min(x, y), tvm.te.min(z, x)), tvm.te.min(tvm.te.max(y, z), x))
    ck.verify(tvm.te.max(tvm.te.min(y, x), tvm.te.min(x, z)), tvm.te.min(tvm.te.max(y, z), x))
    ck.verify(tvm.te.max(tvm.te.min(y, x), tvm.te.min(z, x)), tvm.te.min(tvm.te.max(y, z), x))

    ck.verify(tvm.te.max(y + x, z + x), tvm.te.max(y, z) + x)
    ck.verify(tvm.te.max(y + x, x + z), tvm.te.max(y, z) + x)
    ck.verify(tvm.te.max(x + y, z + x), tvm.te.max(y, z) + x)
    ck.verify(tvm.te.max(x + y, x + z), tvm.te.max(y, z) + x)

    ck.verify(tvm.te.max(x - y, x - z), x - tvm.te.min(y, z))
    ck.verify(tvm.te.max(y - x, z - x), tvm.te.max(y, z) - x)

    ck.verify(tvm.te.max(tvm.te.max(x, 1), 10), tvm.te.max(x, 10))
    ck.verify(tvm.te.max(tvm.te.max(x, 11), 10), tvm.te.max(x, 11))

    ck.verify(tvm.te.max(x * 3, 9), tvm.te.max(x, 3) * 3)
    ck.verify(tvm.te.max(3 - x, 1), 3 - tvm.te.min(x, 2))
    ck.verify(tvm.te.max(x * 2, 0), tvm.te.max(x, 0) * 2)
    ck.verify(tvm.te.max(0 - x * 2, 0), tvm.te.min(x, 0) * -2)
    ck.verify(tvm.te.max(x * (-2), -4), tvm.te.min(x, 2) * -2)
    ck.verify(tvm.te.max(x * (-2), 4), tvm.te.min(x, -2) * -2)
    ck.verify(tvm.te.max(x * (0), 4), 4)
    ck.verify(tvm.te.max(x * (0), -4), 0)

    # DivMod rules
    # truc div
    ck.verify(tvm.te.max(tdiv(x, 10), tdiv(y, 10)), tdiv(tvm.te.max(x, y), 10))
    ck.verify(tvm.te.max(tdiv(x, (-10)), tdiv(y, (-10))), tdiv(tvm.te.min(x, y), (-10)))
    ck.verify(tvm.te.max(tdiv(x + 3, 4) * 4, x), tdiv(x + 3, 4) * 4)

    # floordiv
    ck.verify(tvm.te.max(fld(x, 10), fld(y, 10)), fld(tvm.te.max(x, y), 10))
    ck.verify(tvm.te.max(fld(x, (-10)), fld(y, (-10))), fld(tvm.te.min(x, y), (-10)))
    ck.verify(tvm.te.max(fld(x + 3, 4) * 4, x), fld(x + 3, 4) * 4)
    ck.verify(tvm.te.max(fld(x, 4) * 4, x), x)
    ck.verify(tvm.te.max(x, fld(x, 4) * 4), x)


def test_cmp_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    flm = tvm.te.floormod
    fld = tvm.te.floordiv
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    # const int bound
    ck.verify((tmod(x, 2) + 10).equal(0), tvm.tir.const(0, "bool"))
    ck.verify(tvm.tir.NE(tmod(x, 2) + 10, 0), tvm.tir.const(1, "bool"))
    ck.verify(tmod(x, 2) + 10 > 1, tvm.tir.const(1, "bool"))
    ck.verify(tmod(x, 2) + 10 <= 1, tvm.tir.const(0, "bool"))
    ck.verify(flm(x, 2) + 2 > 1, tvm.tir.const(1, "bool"))
    ck.verify(flm(x, 2) + 10 <= 1, tvm.tir.const(0, "bool"))

    ck.verify(x * 3 + 10 == 0, tvm.tir.const(0, "bool"))
    ck.verify(x * 3 + 10 != 0, tvm.tir.const(1, "bool"))

    # canonicalization
    ck.verify((x - 10).equal(0), x.equal(10))
    ck.verify((10 - x).equal(0), x.equal(10))
    ck.verify((x * y).equal(0), tvm.tir.Or(x.equal(0), y.equal(0)))

    # cmp bound
    ck.verify(x + y < x + z, y < z)
    ck.verify(x + y < z + x, y < z)
    ck.verify(y + x < x + z, y < z)
    ck.verify(y + x < z + x, y < z)
    ck.verify(y - x < z - x, y < z)
    ck.verify(x - y < x - z, z < y)

    ck.verify(x < z + x, tvm.tir.LT(0, z))
    ck.verify(x < x + z, tvm.tir.LT(0, z))

    ck.verify(100 < x + 1, tvm.tir.LT(99, x))
    ck.verify(1 < 100 - x, tvm.tir.LT(x, 99))
    ck.verify(x * 3 < y * 3, x < y)
    ck.verify(x * (-3) < y * (-3), y < x)
    ck.verify(x * 3 >= y * 3, y <= x)

    ck.verify(x * 4 >= 2, tvm.tir.LE(1, x))
    ck.verify(x * 2 >= 50, tvm.tir.LE(25, x))
    ck.verify(x * 4 <= 2, x <= 0)
    ck.verify((0 - x * 3) <= 0, tvm.tir.LE(0, x))
    ck.verify((0 - x * 3) >= 0, tvm.tir.LE(x, 0))
    ck.verify(2 * x <= 0, x <= 0)

    ck.verify(x * 2 >= 3, tvm.tir.LE(2, x))
    ck.verify(x * 2 >= 2, tvm.tir.LE(1, x))
    ck.verify(x * 2 >= 1, tvm.tir.LE(1, x))
    ck.verify(x * 2 >= 0, tvm.tir.LE(0, x))
    ck.verify(x * 2 >= -1, tvm.tir.LE(0, x))
    ck.verify(x * 2 >= -2, tvm.tir.LE(-1, x))
    ck.verify(x * 2 >= -3, tvm.tir.LE(-1, x))

    ck.verify(x * 2 <= 3, tvm.tir.LE(x, 1))
    ck.verify(x * 2 <= 2, tvm.tir.LE(x, 1))
    ck.verify(x * 2 <= 1, tvm.tir.LE(x, 0))
    ck.verify(x * 2 <= 0, tvm.tir.LE(x, 0))
    ck.verify(x * 2 <= -1, tvm.tir.LE(x, -1))
    ck.verify(x * 2 <= -2, tvm.tir.LE(x, -1))
    ck.verify(x * 2 <= -3, tvm.tir.LE(x, -2))

    ck.verify(x * (-2) >= 3, tvm.tir.LE(x, -2))
    ck.verify(x * (-2) >= 2, tvm.tir.LE(x, -1))
    ck.verify(x * (-2) >= 1, tvm.tir.LE(x, -1))
    ck.verify(x * (-2) >= 0, tvm.tir.LE(x, 0))
    ck.verify(x * (-2) >= -1, tvm.tir.LE(x, 0))
    ck.verify(x * (-2) >= -2, tvm.tir.LE(x, 1))
    ck.verify(x * (-2) >= -3, tvm.tir.LE(x, 1))

    ck.verify(x * (-2) <= 3, tvm.tir.LE(-1, x))
    ck.verify(x * (-2) <= 2, tvm.tir.LE(-1, x))
    ck.verify(x * (-2) <= 1, tvm.tir.LE(0, x))
    ck.verify(x * (-2) <= 0, tvm.tir.LE(0, x))
    ck.verify(x * (-2) <= -1, tvm.tir.LE(1, x))
    ck.verify(x * (-2) <= -2, tvm.tir.LE(1, x))
    ck.verify(x * (-2) <= -3, tvm.tir.LE(2, x))

    # DivMod rules
    # truc div
    ck.verify(tdiv(x, 2) < 3, x < 6)
    ck.verify(3 < tdiv(x, 2), tvm.tir.LT(7, x))
    ck.verify(tdiv(x, 3) >= 0, tvm.tir.LE(-2, x))
    ck.verify(tdiv(x, 2) >= 1, tvm.tir.LE(2, x))
    ck.verify(tdiv(x, 2) >= 0, tvm.tir.LE(-1, x))
    ck.verify(tdiv(x, 2) >= -1, tvm.tir.LE(-3, x))

    ck.verify(tdiv(x, 2) <= 1, tvm.tir.LE(x, 3))
    ck.verify(tdiv(x, 2) <= 0, tvm.tir.LE(x, 1))
    ck.verify(tdiv(x, 2) <= -1, tvm.tir.LE(x, -2))

    ck.verify(tdiv(x, 4) * 4 < x, tvm.tir.LT(0, tmod(x, 4)))
    ck.verify(tdiv(x, 4) * 4 >= x, tvm.tir.LE(tmod(x, 4), 0))

    ck.verify(tdiv(x, 4) * 4 < x + y, tvm.tir.LT(0, tmod(x, 4) + y))
    ck.verify(tdiv(x, 4) * 4 < x - y, tvm.tir.LT(y, tmod(x, 4)))

    ck.verify(tdiv(x + 2, 4) * 4 >= x, tvm.tir.LE(tmod(x + 2, 4), 2))
    ck.verify(tdiv(x + 2, 4) * 4 >= x + y, tvm.tir.LE(tmod(x + 2, 4) + y, 2))
    ck.verify(tdiv(x + 2, 4) * 4 >= x - y, tvm.tir.LE(tmod(x + 2, 4) + (-2), y))

    # floor div
    ck.verify(fld(x, 2) < 3, x < 6)
    ck.verify(3 < fld(x, 2), tvm.tir.LT(7, x))
    ck.verify(-3 < fld(x, 2), tvm.tir.LT(-5, x))
    ck.verify(fld(x, 3) >= 0, tvm.tir.LE(0, x))
    ck.verify(fld(x, 2) >= 1, tvm.tir.LE(2, x))
    ck.verify(fld(x, 2) >= 0, tvm.tir.LE(0, x))
    ck.verify(fld(x, 2) >= -1, tvm.tir.LE(-2, x))

    ck.verify(fld(x, 2) <= 1, tvm.tir.LE(x, 3))
    ck.verify(fld(x, 2) <= 0, tvm.tir.LE(x, 1))
    ck.verify(fld(x, 2) <= -1, tvm.tir.LE(x, -1))

    ck.verify(fld(x, 4) * 4 < x, tvm.tir.LT(0, flm(x, 4)))
    ck.verify(fld(x, 4) * 4 >= x, tvm.tir.EQ(flm(x, 4), 0))

    ck.verify(fld(x, 4) * 4 < x + y, tvm.tir.LT(0, flm(x, 4) + y))
    ck.verify(fld(x, 4) * 4 < x - y, tvm.tir.LT(y, flm(x, 4)))

    ck.verify(fld(x + 2, 4) * 4 >= x, tvm.tir.LE(flm(x + 2, 4), 2))
    ck.verify(fld(x + 2, 4) * 4 >= x + y, tvm.tir.LE(flm(x + 2, 4) + y, 2))
    ck.verify(fld(x + 2, 4) * 4 >= x - y, tvm.tir.LE(flm(x + 2, 4) + (-2), y))
    # End DivMod Rules

    # merging flm/fld into known value
    ck.verify(tir.all(fld(x, 8) == 3, flm(x, 8) == 4), x == 28)
    ck.verify(tir.all(flm(x, 8) == 4, fld(x, 8) == 3), x == 28)
    ck.verify(tir.all(fld(x, 8) == -3, flm(x, 8) == 4), x == -20)
    ck.verify(tir.all(flm(x, 8) == 4, fld(x, 8) == -3), x == -20)

    # Rewrite based on definition of integer division
    ck.verify(tir.all(tvm.runtime.convert(0) <= x - y * 5, x - y * 5 < 5), y == fld(x, 5))
    ck.verify(tir.all(x - y * 5 < 5, tvm.runtime.convert(0) <= x - y * 5), y == fld(x, 5))

    # Narrow upper bound using floormod
    ck.verify(tir.all(x < 20, flm(x, 5) < 2), tir.all(x < 17, flm(x, 5) < 2))
    ck.verify(tir.all(x < 18, flm(x, 5) < 2), tir.all(x < 17, flm(x, 5) < 2))
    ck.verify(tir.all(x <= 19, flm(x, 5) < 2), tir.all(x < 17, flm(x, 5) < 2))
    ck.verify(tir.all(x <= 18, flm(x, 5) < 2), tir.all(x < 17, flm(x, 5) < 2))
    ck.verify(tir.all(x < -20, flm(x, 5) < 2), tir.all(x < -23, flm(x, 5) < 2))
    ck.verify(tir.all(x < 18 - 40, flm(x, 5) < 2), tir.all(x < 17 - 40, flm(x, 5) < 2))
    ck.verify(tir.all(x <= -21, flm(x, 5) < 2), tir.all(x < -23, flm(x, 5) < 2))
    ck.verify(tir.all(x <= -22, flm(x, 5) < 2), tir.all(x < -23, flm(x, 5) < 2))
    # No change if the floormod cannot help narrow the upper bound
    ck.verify(tir.all(x < 16, flm(x, 5) < 2), tir.all(x < 16, flm(x, 5) < 2))
    ck.verify(tir.all(x <= 15, flm(x, 5) < 2), tir.all(x <= 15, flm(x, 5) < 2))

    # Merge a known floordiv and an upper bound of floormod into a value range
    ck.verify(
        tir.all(fld(x, 10) == 5, flm(x, 10) < 7),
        tir.all(tvm.runtime.convert(50) <= x, x < 57),
    )
    ck.verify(
        tir.all(fld(x, 10) == 5, flm(x, 10) <= 7),
        tir.all(tvm.runtime.convert(50) <= x, x <= 57),
    )
    ck.verify(
        tir.all(fld(x, 10) == -5, flm(x, 10) < 7),
        tir.all(tvm.runtime.convert(-50) <= x, x < -43),
    )
    ck.verify(
        tir.all(fld(x, 10) == -5, flm(x, 10) <= 7),
        tir.all(tvm.runtime.convert(-50) <= x, x <= -43),
    )

    # Merge a known floordiv and an lower bound of floormod into a value range
    ck.verify(
        tir.all(fld(x, 10) == 5, tvm.runtime.convert(7) < flm(x, 10)),
        tir.all(tvm.runtime.convert(57) < x, x < 60),
    )
    ck.verify(
        tir.all(fld(x, 10) == 5, tvm.runtime.convert(7) <= flm(x, 10)),
        tir.all(tvm.runtime.convert(57) <= x, x < 60),
    )
    ck.verify(
        tir.all(fld(x, 10) == -5, tvm.runtime.convert(7) < flm(x, 10)),
        tir.all(tvm.runtime.convert(-43) < x, x < -40),
    )
    ck.verify(
        tir.all(fld(x, 10) == -5, tvm.runtime.convert(7) <= flm(x, 10)),
        tir.all(tvm.runtime.convert(-43) <= x, x < -40),
    )

    ck.verify(tvm.te.min(x, 11) < 10, x < 10)
    ck.verify(tvm.te.min(x, 8) < 10, tvm.tir.const(1, "bool"))
    ck.verify(tvm.te.max(8, x) > 10, tvm.tir.LT(10, x))
    ck.verify(x + 1 < tvm.te.max(8, x), x < 7)

    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 10), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(-10, 0), override=True)
    ck.analyzer.update(z, tvm.arith.ConstIntBound(-5, 5), override=True)

    ck.verify(x < 11, tvm.tir.const(1, "bool"))
    ck.verify(x <= 10, tvm.tir.const(1, "bool"))
    ck.verify(z <= 5, tvm.tir.const(1, "bool"))
    ck.verify(x + y <= 10, tvm.tir.const(1, "bool"))
    ck.verify(x + y >= -10, tvm.tir.const(1, "bool"))
    ck.verify(z - 5 <= y + 10, tvm.tir.const(1, "bool"))
    ck.verify(tvm.tir.all(x > -1, z <= x + 5), tvm.tir.const(1, "bool"))
    ck.verify(x * y <= 0, tvm.tir.const(1, "bool"))
    ck.verify((x + 1) * (y - 1) < 0, tvm.tir.const(1, "bool"))
    ck.verify(y * y >= 0, tvm.tir.const(1, "bool"))
    ck.verify(x * 6 <= -3, tvm.tir.const(0, "bool"))
    ck.verify(tmod(y - 1, 3) == 0, tmod(y + (-1), 3) == 0)


def test_logical_simplify():
    ck = RewriteChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    ck.verify(tvm.tir.And(tvm.tir.EQ(x, y), tvm.tir.NE(x, y)), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(tvm.tir.NE(x, y), tvm.tir.EQ(x, y)), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(x > 1, tvm.tir.Not(x > 1)), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(x <= y, y < x), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(y < x, x <= y), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(x < 1, 0 < x), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(x < 0, 1 < x), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(x < 1, 1 <= x), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(x <= 1, 1 < x), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(1 <= x, x < 1), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(1 < x, x <= 1), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(x <= 1, 2 <= x), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(2 <= x, x <= 1), tvm.tir.const(False, "bool"))
    ck.verify(tvm.tir.And(x == 1, x != 2), x == 1)

    ck.verify(tvm.tir.Or(tvm.tir.EQ(x, y), tvm.tir.NE(x, y)), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(tvm.tir.NE(x, y), tvm.tir.EQ(x, y)), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(x > y, tvm.tir.Not(x > y)), tvm.tir.const(True, "bool"))

    ck.verify(tvm.tir.Or(x <= y, y < x), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(y < x, y >= x), tvm.tir.const(True, "bool"))

    ck.verify(tvm.tir.Or(x < 1, 0 < x), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(0 < x, x < 1), tvm.tir.const(True, "bool"))

    ck.verify(tvm.tir.Or(x < 1, 1 <= x), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(x <= 1, 1 < x), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(1 <= x, x < 1), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(1 < x, x <= 1), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(x <= 1, 2 <= x), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(2 <= x, x <= 1), tvm.tir.const(True, "bool"))
    ck.verify(tvm.tir.Or(x != 1, x == 2), x != 1)


def test_let_simplify():
    ck = RewriteChecker()
    x, y = te.var("x"), te.var("y")
    z = tvm.tir.Let(x, 1, x + 1)
    ck.verify(z + z, 4)


def test_cast_simplify():
    ck = RewriteChecker()
    x = te.var("x")

    dtypes = ["float32", "float16", "int32", "int8", "bool"]
    for dtype1 in dtypes:
        ck.verify(tvm.tir.Cast(dtype1, x - x), tvm.tir.const(0, dtype1))
        ck.verify(tvm.tir.Cast(dtype1, x == x), tvm.tir.const(1, dtype1))
        for dtype2 in dtypes:
            for i in [0, 1, 2, 3]:
                if i > 1 and (dtype1 == "bool" or dtype2 == "bool"):
                    continue
                ck.verify(tvm.tir.Cast(dtype1, tvm.tir.const(i, dtype2)), tvm.tir.const(i, dtype1))


def test_shift_left_simplify():
    ck = RewriteChecker()
    z = tvm.tir.op.call_intrin("int32", "tir.shift_left", 1, 10)
    ck.verify(z, tvm.tir.const(1 << 10, "int32"))


def test_div_zero_simplify():
    ck = RewriteChecker()
    ramp = tvm.tir.Ramp(1, 1, 2)
    broadcast = tvm.tir.Broadcast(0, 2)

    with pytest.raises(tvm.error.TVMError) as cm:
        ck.analyzer.rewrite_simplify(tvm.tir.Div(ramp, broadcast))
        assert "division by zero" in str(cm.execption)

    with pytest.raises(tvm.error.TVMError) as cm:
        ck.analyzer.rewrite_simplify(tvm.tir.Mod(ramp, broadcast))
        assert "division by zero" in str(cm.execption)

    with pytest.raises(tvm.error.TVMError) as cm:
        ck.analyzer.rewrite_simplify(tvm.tir.FloorDiv(ramp, broadcast))
        assert "division by zero" in str(cm.execption)

    with pytest.raises(tvm.error.TVMError) as cm:
        ck.analyzer.rewrite_simplify(tvm.tir.FloorMod(ramp, broadcast))
        assert "division by zero" in str(cm.execption)


def test_sub_bufferload():
    ck = RewriteChecker()
    buf = tvm.tir.decl_buffer([1], dtype="float32")
    load = tvm.tir.BufferLoad(buf, [0])
    expr = load - load
    ck.verify(expr, 0.0)


def test_if_then_else_simplify():
    ck = RewriteChecker()
    x = te.var("x", "int32")
    z = tvm.tir.if_then_else(x < 5, tvm.tir.if_then_else(x > 1, 1, 0), 0)
    ck.verify(z, tvm.tir.if_then_else(tvm.tir.And(tvm.tir.LT(x, 5), tvm.tir.LT(1, x)), 1, 0))

    z = tvm.tir.if_then_else(x > 2, tvm.tir.if_then_else(x > 1, 1, 0), 0)
    ck.verify(z, tvm.tir.if_then_else(tvm.tir.LT(2, x), 1, 0))


if __name__ == "__main__":
    pytest.main([__file__])
