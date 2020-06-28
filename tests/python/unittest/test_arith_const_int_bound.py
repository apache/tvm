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
from tvm import te

def test_dtype_bound():
    analyzer = tvm.arith.Analyzer()

    x = te.var("x", dtype="int64")
    bd = analyzer.const_int_bound(x)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF

    x = te.var("x", dtype="int8")
    bd = analyzer.const_int_bound(x)
    assert bd.min_value == -128
    assert bd.max_value == 127

    x = te.var("x", dtype="uint8")
    bd = analyzer.const_int_bound(x)
    assert bd.min_value == 0
    assert bd.max_value == 255


def test_cast_bound():
    analyzer = tvm.arith.Analyzer()
    x = te.var("x", dtype="int8")
    tmod = tvm.tir.truncmod
    bd = analyzer.const_int_bound(tmod(x, 3).astype("uint32"))
    assert bd.min_value == 0
    assert bd.max_value == 2

    bd = analyzer.const_int_bound(
        tmod(x, 3).astype("float32").astype("int32"))
    assert bd.min_value == -2
    assert bd.max_value == 2


def test_add_sub_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x", "int64"), te.var("y", "int64")
    bd = analyzer.const_int_bound(x + y)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF

    analyzer.update(x, tvm.arith.ConstIntBound(0, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(1, 10))
    bd = analyzer.const_int_bound(x + y)
    assert bd.min_value == 1
    assert bd.max_value == 14

    bd = analyzer.const_int_bound(x - y)
    assert bd.min_value == -10
    assert bd.max_value == 3

    analyzer.update(x, tvm.arith.ConstIntBound(0, bd.POS_INF), override=True)
    bd = analyzer.const_int_bound(x - y)
    assert bd.min_value == -10
    assert bd.max_value == bd.POS_INF

    bd = analyzer.const_int_bound(1 - x)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == 1


def test_mul_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-2, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(x * y + 20)
    assert bd.min_value == 0
    assert bd.max_value == 60

    analyzer.update(x, tvm.arith.ConstIntBound(-3, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-8, 2), override=True)
    bd = analyzer.const_int_bound(x * y)
    assert bd.min_value == -32
    assert bd.max_value == 24

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-8, 2), override=True)
    bd = analyzer.const_int_bound(x * y)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF


def test_truncdiv_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    tdiv = tvm.tir.truncdiv

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(tdiv(x, y))
    assert bd.min_value == -2

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-2, 0), override=True)
    bd = analyzer.const_int_bound(tdiv(x, y))
    assert bd.min_value == -4
    assert bd.max_value == 9

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-2, 1), override=True)
    bd = analyzer.const_int_bound(tdiv(x, y))
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF


def test_truncmod_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")

    tmod = tvm.tir.truncmod

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(tmod(x, y))
    assert bd.min_value == -9
    assert bd.max_value == 4

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(tmod(x, y))
    assert bd.min_value == -9
    assert bd.max_value == 9

    analyzer.update(x, tvm.arith.ConstIntBound(1, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(tmod(x, y))
    assert bd.min_value == 0
    assert bd.max_value == 9


def test_floordiv_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    fld = tvm.te.floordiv
    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(fld(x, y))
    assert bd.min_value == -9 // 4

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-2, 0), override=True)
    bd = analyzer.const_int_bound(fld(x, y))
    assert bd.min_value == -4
    assert bd.max_value == 9

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-2, 1), override=True)
    bd = analyzer.const_int_bound(fld(x, y))
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF


def test_floormod_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    flm = tvm.te.floormod

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(flm(x, y))
    assert bd.min_value == 0
    assert bd.max_value == 9

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(flm(x, y))
    assert bd.min_value == 0
    assert bd.max_value == 9

    analyzer.update(x, tvm.arith.ConstIntBound(1, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(flm(x, y))
    assert bd.min_value == 0
    assert bd.max_value == 9


def test_min_max_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 11))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(tvm.te.min(x, y))
    assert bd.min_value == -9
    assert bd.max_value == 10

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(tvm.te.min(x, y))
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == 10

    bd = analyzer.const_int_bound(tvm.te.max(x, y))
    assert bd.min_value == 4
    assert bd.max_value == bd.POS_INF

    analyzer.update(x, tvm.arith.ConstIntBound(1, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(tvm.te.max(x, y))
    assert bd.min_value == 4
    assert bd.max_value == bd.POS_INF


def test_select_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 11))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))

    bd = analyzer.const_int_bound(
        tvm.tir.Select(x > 1, (y < 0).astype("int32"), y + 1))
    assert bd.min_value == 0
    assert bd.max_value == 11


def test_shift_and_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 11))
    analyzer.update(y, tvm.arith.ConstIntBound(2, 10))

    bd = analyzer.const_int_bound(x >> y)
    assert bd.min_value == -3
    assert bd.max_value == 2

    bd = analyzer.const_int_bound(x & y)
    assert bd.min_value == 0
    assert bd.max_value == 10

    analyzer.update(x, tvm.arith.ConstIntBound(10, 11), override=True)
    bd = analyzer.const_int_bound(x & y)
    assert bd.min_value == 0
    assert bd.max_value == 10


def test_mix_index_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod

    analyzer.update(x, tvm.arith.ConstIntBound(0, 24 - 1))
    analyzer.update(y, tvm.arith.ConstIntBound(0, 3 - 1))
    bd = analyzer.const_int_bound(tmod(x, 8) + tdiv(x, 8) * 8)
    assert bd.min_value == 0
    assert bd.max_value == 24 - 1

    bd = analyzer.const_int_bound(y + x * 3)
    assert bd.min_value == 0
    assert bd.max_value == 24 * 3 - 1

    bd = analyzer.const_int_bound(tmod(x, 7) + tdiv(x, 7) * 7)
    assert bd.min_value == 0
    assert bd.max_value == (23 // 7) * 7 + 6


def test_size_var_bound():
    analyzer = tvm.arith.Analyzer()
    x = te.size_var("x")
    bd = analyzer.const_int_bound(x)
    assert bd.min_value == 0
    assert bd.max_value == bd.POS_INF


def test_let_bound():
    analyzer = tvm.arith.Analyzer()
    x = te.var("x")
    bd = analyzer.const_int_bound(tvm.tir.Let(x, 1, x + 1))
    assert bd.min_value == 2
    assert bd.max_value == 2


if __name__ == "__main__":
    test_let_bound()
    test_dtype_bound()
    test_cast_bound()
    test_add_sub_bound()
    test_mul_bound()
    test_truncdiv_bound()
    test_truncmod_bound()
    test_floordiv_bound()
    test_floormod_bound()
    test_min_max_bound()
    test_select_bound()
    test_shift_and_bound()
    test_mix_index_bound()
    test_size_var_bound()
