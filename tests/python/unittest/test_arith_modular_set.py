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
import tvm.testing
from tvm import te


def test_cast():
    analyzer = tvm.arith.Analyzer()
    x = te.var("x", dtype="int8")
    m = analyzer.modular_set((x * 3).astype("uint32"))
    assert m.coeff == 3
    assert m.base == 0
    m = analyzer.modular_set((x * 3 + 1).astype("float32").astype("int32"))
    assert m.coeff == 3
    assert m.base == 1


def test_add_sub():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x", "int64"), te.var("y", "int64")
    m = analyzer.modular_set(x * 6 + y * 4)
    assert m.coeff == 2
    assert m.base == 0

    analyzer.bind(y, x * 4 + 1)
    m = analyzer.modular_set(1 - y)
    assert m.coeff == 4
    assert m.base == 0


def test_mul():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    m = analyzer.modular_set((x * 4 + 2) * (y * 6 + 1))
    assert m.coeff == 4
    assert m.base == 2


def test_floormod():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    m = analyzer.modular_set(tvm.tir.floormod(x * 128 + y * 4, 256))
    assert m.coeff == 4
    assert m.base == 0


def test_div_shift():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    # not sure if x is non-negative
    tdiv = tvm.tir.truncdiv
    m = analyzer.modular_set(tdiv(x * 4 + 2, 2))
    assert m.coeff == 1
    assert m.base == 0
    # right shift always round down so it is fine
    m = analyzer.modular_set((x * 4 + 2) >> 1)
    assert m.coeff == 2
    assert m.base == 1
    fld = tvm.te.floordiv
    m = analyzer.modular_set(fld(x * 4 + 2, 2))
    assert m.coeff == 2
    assert m.base == 1
    # x is non-negative
    analyzer.update(x, tvm.arith.ConstIntBound(0, 100))
    m = analyzer.modular_set(tdiv(x * 4 + 2, 2))
    assert m.coeff == 2
    assert m.base == 1


def test_mod():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    tmod = tvm.tir.truncmod
    fmod = tvm.tir.floormod
    # not sure if x is non-negative
    m = analyzer.modular_set(tmod(x * 4 + 1, 4))
    assert m.coeff == 1
    assert m.base == 0
    # no need to be positive if base == 0
    m = analyzer.modular_set(tmod(x * 4, 4))
    assert m.coeff == 4
    assert m.base == 0
    # floor mod tests
    m = analyzer.modular_set(fmod(x * 4 + 3, 2))
    assert m.coeff == 2
    assert m.base == 1
    m = analyzer.modular_set(fmod(x * 4 + 3, 8))
    assert m.coeff == 4
    assert m.base == 3
    # x is non-negative
    analyzer.update(x, tvm.arith.ConstIntBound(0, 100))
    m = analyzer.modular_set(tmod(x * 4 + 3, 2))
    assert m.coeff == 2
    assert m.base == 1


def test_min_max_select():
    analyzer = tvm.arith.Analyzer()
    x, y = te.var("x"), te.var("y")
    m = analyzer.modular_set(tvm.te.min(x * 3, y * 9))
    assert m.coeff == 3
    assert m.base == 0

    m = analyzer.modular_set(tvm.te.max(x * 3 + 1, y * 9 + 4))
    assert m.coeff == 3
    assert m.base == 1

    m = analyzer.modular_set(tvm.tir.Select(x > 0, x * 3 + 1, y * 9 + 2))
    assert m.coeff == 1
    assert m.base == 0


def test_mix_index():
    a = te.var("a")
    b = te.var("b")
    analyzer = tvm.arith.Analyzer()
    tdiv = tvm.tir.truncdiv
    m = analyzer.modular_set(a * 4 + b * 6 + 7)
    assert m.coeff == 2
    assert m.base == 1

    m = analyzer.modular_set((a * 4 + 1) * (b * 8 + 3))
    assert m.coeff == 4
    assert m.base == 3

    m = analyzer.modular_set(tdiv(a * 4 + 1, b * 8 + 3))
    assert m.coeff == 1
    assert m.base == 0

    m = analyzer.modular_set((a * 4 + 1) * tdiv(b * 8, 4))
    assert m.coeff == 2
    assert m.base == 0

    m = analyzer.modular_set((a * 12 + 1) - (b * 3 * 7 + 2))
    assert m.coeff == 3
    assert m.base == 2

    m = analyzer.modular_set(a * 12 + tvm.te.min(b * 3 * 7, 2))
    assert m.coeff == 1
    assert m.base == 0


def test_constraint_scope():
    a = te.var("a")
    b = te.var("b")
    analyzer = tvm.arith.Analyzer()
    tmod = tvm.tir.truncmod

    with analyzer.constraint_scope(tmod(b, 4) == 2):
        m = analyzer.modular_set(b + 1)
        assert m.coeff == 4
        assert m.base == 3
        with analyzer.constraint_scope(tmod(a, 2) == 1):
            m = analyzer.modular_set(b + a * 2)
            assert m.coeff == 4
            assert m.base == 0
        m = analyzer.modular_set(b + a * 2)
        assert m.coeff == 2
        assert m.base == 0

    m = analyzer.modular_set(b + 1)
    assert m.coeff == 1
    assert m.base == 0


def test_intersect():
    a = te.var("a")
    analyzer = tvm.arith.Analyzer()
    tmod = tvm.tir.truncmod
    with analyzer.constraint_scope(tmod(a, 4) == 1):
        with analyzer.constraint_scope(tmod(a, 3) == 1):
            m = analyzer.modular_set(a)
            assert m.coeff == 12
            assert m.base == 1

    with analyzer.constraint_scope(tmod(a, 3) == 2):
        with analyzer.constraint_scope(tmod(a, 5) == 3):
            with analyzer.constraint_scope(tmod(a, 7) == 2):
                m = analyzer.modular_set(a)
                assert m.coeff == 105
                assert m.base == 23


def test_let():
    analyzer = tvm.arith.Analyzer()
    x = te.var("x")
    y = te.var("y")
    m = analyzer.modular_set(tvm.tir.Let(x, y * 10, x + 1))
    assert m.coeff == 10
    assert m.base == 1


def test_bitwise_and():
    analyzer = tvm.arith.Analyzer()
    x = te.var("x")
    y = te.var("y")

    # RHS of bitwise_and is 2^p - 1
    m = analyzer.modular_set((x * 16 + y * 4) & 31)
    assert m.coeff == 4
    assert m.base == 0

    # arbitrary RHS
    m = analyzer.modular_set((x * 16 + y * 4) & 17)
    assert m.coeff == 1
    assert m.base == 0


if __name__ == "__main__":
    tvm.testing.main()
