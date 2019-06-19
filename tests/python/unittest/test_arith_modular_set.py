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


def test_cast():
    analyzer = tvm.arith.Analyzer()
    x = tvm.var("x", dtype="int8")
    m = analyzer.modular_set((x * 3).astype("uint32"))
    assert m.coeff == 3
    assert m.base == 0
    m = analyzer.modular_set(
        (x * 3 + 1).astype("float32").astype("int32"))
    assert m.coeff == 3
    assert m.base == 1


def test_add_sub():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x", "int64"), tvm.var("y", "int64")
    m = analyzer.modular_set(x * 6 + y * 4)
    assert m.coeff == 2
    assert m.base == 0

    analyzer.bind(y, x * 4 + 1)
    m = analyzer.modular_set(1 - y)
    assert m.coeff == 4
    assert m.base == 0


def test_mul():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")
    m = analyzer.modular_set((x * 4 + 2) * (y * 6 + 1))
    assert m.coeff == 4
    assert m.base == 2


def test_div_shift():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")
    # not sure if x is non-negative
    m = analyzer.modular_set((x * 4 + 2) / 2)
    assert m.coeff == 1
    assert m.base == 0
    # right shift always round down so it is fine
    m = analyzer.modular_set((x * 4 + 2) >> 1)
    assert m.coeff == 2
    assert m.base == 1
    fld = tvm.floordiv
    m = analyzer.modular_set(fld(x * 4 + 2, 2))
    assert m.coeff == 2
    assert m.base == 1
    # x is non-negative
    analyzer.update(x, tvm.arith.ConstIntBound(0, 100))
    m = analyzer.modular_set((x * 4 + 2) / 2)
    assert m.coeff == 2
    assert m.base == 1


def test_min_max_select():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")
    m = analyzer.modular_set(tvm.min(x * 3, y * 9))
    assert m.coeff == 3
    assert m.base == 0

    m = analyzer.modular_set(tvm.max(x * 3 + 1, y * 9 + 4))
    assert m.coeff == 3
    assert m.base == 1

    m = analyzer.modular_set(tvm.expr.Select(x > 0, x * 3 + 1, y * 9 + 2))
    assert m.coeff == 1
    assert m.base == 0


def test_mix_index():
    a = tvm.var("a")
    b = tvm.var("b")
    analyzer = tvm.arith.Analyzer()
    m = analyzer.modular_set(a * 4 + b * 6 + 7)
    assert m.coeff == 2
    assert m.base == 1

    m = analyzer.modular_set((a * 4 + 1) * (b * 8 + 3))
    assert m.coeff == 4
    assert m.base == 3

    m = analyzer.modular_set((a * 4 + 1) / (b * 8 + 3))
    assert m.coeff == 1
    assert m.base == 0

    m = analyzer.modular_set((a * 4 + 1) * (b * 8 / 4))
    assert m.coeff == 2
    assert m.base == 0

    m = analyzer.modular_set((a * 12 + 1) - (b * 3 * 7  + 2))
    assert m.coeff == 3
    assert m.base == 2

    m = analyzer.modular_set(a * 12 + tvm.min(b * 3 * 7, 2))
    assert m.coeff == 1
    assert m.base == 0


def test_constraint_scope():
    a = tvm.var("a")
    b = tvm.var("b")
    analyzer = tvm.arith.Analyzer()
    with analyzer.constraint_scope(b % 4 == 2):
        m = analyzer.modular_set(b + 1)
        assert m.coeff == 4
        assert m.base == 3
        with analyzer.constraint_scope(a % 2 == 1):
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
    a = tvm.var("a")
    analyzer = tvm.arith.Analyzer()
    with analyzer.constraint_scope(a % 4 == 1):
        with analyzer.constraint_scope(a % 3 == 1):
            m = analyzer.modular_set(a)
            assert m.coeff == 12
            assert m.base == 1

    with analyzer.constraint_scope(a % 3 == 2):
        with analyzer.constraint_scope(a % 5 == 3):
            with analyzer.constraint_scope(a % 7 == 2):
                m = analyzer.modular_set(a)
                assert m.coeff == 105
                assert m.base == 23


if __name__ == "__main__":
    test_cast()
    test_add_sub()
    test_mul()
    test_div_shift()
    test_min_max_select()
    test_mix_index()
    test_constraint_scope()
    test_intersect()
