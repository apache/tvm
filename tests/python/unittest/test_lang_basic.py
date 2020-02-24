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
import numpy as np


def test_const():
    x = tvm.const(1, "int32")
    print(x.dtype)
    assert x.dtype == tvm.int32
    assert isinstance(x, tvm.tir.IntImm)


def test_scalar_dtype_inference():
    for data in [True, np.bool(1), np.uint8(1), np.uint16(1), np.uint32(1), np.uint64(1),
                 np.int8(1), np.int16(1), np.int32(1), np.int64(1),
                 np.float16(1), np.float32(1), np.float64(1)]:
        assert tvm.const(data).dtype == str(np.array(data).dtype)
    assert tvm.const(1).dtype == 'int32'
    assert tvm.const(1.0).dtype == 'float32'

    for data in [True, np.bool(1), np.uint8(1), np.uint16(1), np.uint32(1), np.uint64(1),
                 np.int8(1), np.int16(1), np.int32(1), np.int64(1),
                 np.float16(1), np.float32(1), np.float64(1)]:
        assert tvm.convert(data).dtype == str(np.array(data).dtype)
    assert tvm.convert(1).dtype == 'int32'
    assert tvm.convert(1.0).dtype == 'float32'


def test_make():
    x = tvm.const(1, "int32")
    y = tvm.var("x")
    z = x + y
    assert isinstance(tvm.max(x, y), tvm.tir.Max)
    assert isinstance(tvm.min(x, y), tvm.tir.Min)


def test_ir():
    x = tvm.const(1, "int32")
    y = tvm.tir.IntImm('int32', 1)
    z = x + y
    stmt = tvm.tir.Evaluate(z)
    assert isinstance(stmt, tvm.tir.Evaluate)


def test_ir2():
    x = tvm.var("n")
    a = tvm.var("array", tvm.handle)
    st = tvm.tir.Store(a, x + 1, 1)
    assert isinstance(st, tvm.tir.Store)
    assert(st.buffer_var == a)


def test_let():
    x = tvm.var('x')
    y = tvm.var('y')
    stmt = tvm.tir.LetStmt(
        x, 10, tvm.tir.Evaluate(x + 1));


def test_cast():
    x = tvm.var('x', dtype="float32")
    y = x.astype("int32")
    z = x.astype("float32x4")
    assert isinstance(y, tvm.tir.Cast)
    assert isinstance(z, tvm.tir.Broadcast)
    assert z.lanes == 4


def test_attr():
    x = tvm.var('x')
    y = tvm.var('y')
    stmt = tvm.tir.AttrStmt(
        y, "stride", 10, tvm.tir.Evaluate(x + 1));
    assert stmt.node == y

    a = tvm.convert(1)
    assert a.value == 1
    try:
        a.no_field
        assert False
    except AttributeError:
        pass


def test_basic():
    a = tvm.var('a')
    b = tvm.var('b')
    c =  a + b
    assert str(c) == '(%s + %s)' % (a.name, b.name)


def test_stmt():
    x = tvm.tir.Evaluate(0)
    tvm.tir.For(tvm.var('i'), 0, 1,
                 tvm.tir.For.Serial, 0,
                 x)


def test_dir():
    x = tvm.var('x')
    dir(x)


def test_dtype():
    x = tvm.var('x')
    assert x.dtype == 'int32'
    y = tvm.var('y')
    assert (x > y).dtype == 'bool'


def test_any():
    x = tvm.var('x')
    y = tvm.var('y')
    z = tvm.var('z')
    try:
        t = x or x
        assert False
    except ValueError:
        pass
    try:
        tvm.any()
        assert False
    except ValueError:
        pass
    assert str(tvm.any(x < y)) == '(%s < %s)' % (x.name, y.name)
    assert str(tvm.any(x < y, x > z)) == '((%s < %s) || (%s > %s))' % (
        x.name, y.name, x.name, z.name)
    assert str(tvm.any(x < y, y > z + 1, x < z * 2)) == \
        '(((%s < %s) || (%s > (%s + 1))) || (%s < (%s*2)))' % (
            x.name, y.name, y.name, z.name, x.name, z.name)


def test_all():
    x = tvm.var('x')
    y = tvm.var('y')
    z = tvm.var('z')
    try:
        t = x and x
        assert False
    except ValueError:
        pass
    try:
        tvm.all()
        assert False
    except ValueError:
        pass
    assert str(tvm.all(x < y)) == '(%s < %s)' % (x.name, y.name)
    assert str(tvm.all(x < y, x > z)) == '((%s < %s) && (%s > %s))' % (
        x.name, y.name, x.name, z.name)
    assert str(tvm.all(x < y, y > z + 1, x < z * 2)) == \
        '(((%s < %s) && (%s > (%s + 1))) && (%s < (%s*2)))' % (
            x.name, y.name, y.name, z.name, x.name, z.name)


def test_bitwise():
    x = tvm.var('x')
    y = tvm.var('y')
    assert str(x << y) == 'shift_left(x, y)'
    assert str(x >> y) == 'shift_right(x, y)'
    assert str(x & y) == 'bitwise_and(x, y)'
    assert str(x | y) == 'bitwise_or(x, y)'
    assert str(x ^ y) == 'bitwise_xor(x, y)'
    assert str(10 & x) == 'bitwise_and(10, x)'
    assert str(10 | x) == 'bitwise_or(10, x)'
    assert str(10 ^ x) == 'bitwise_xor(10, x)'
    assert str(10 >> x) == 'shift_right(10, x)'
    assert str(10 << x) == 'shift_left(10, x)'
    assert str(10 % x) == 'floormod(10, x)'
    assert str(~x) == 'bitwise_not(x)'
    assert(tvm.const(1, "int8x2") >> 1).dtype == "int8x2"
    assert(x >> tvm.const(1, "int32x2")).dtype == "int32x2"
    assert(tvm.var("z", "int8x2") << tvm.const(1, "int8x2")).dtype == "int8x2"

def test_float_bitwise():
    t = tvm.const(1.5,dtype='float32')
    for test in [lambda lhs, rhs : lhs << rhs,
                    lambda lhs, rhs : lhs >> rhs,
                    lambda lhs, rhs : lhs | rhs,
                    lambda lhs, rhs : lhs ^ rhs,
                    lambda lhs, rhs : lhs & rhs
                ]:
        try:
            test(t,10.0)
            assert False
        except tvm.TVMError:
            pass
    try:
        ~t
        assert False
    except RuntimeError:
        pass

def test_isnan():
    x = tvm.var('x', 'float32')
    assert str(tvm.isnan(x)) == 'isnan(x)'
    assert str(tvm.isnan(x).dtype) == 'bool'
    y = tvm.var('y', 'float16')
    assert str(tvm.isnan(y)) == 'isnan(float32(y))'
    z = tvm.var('z', 'int32')
    assert str(tvm.isnan(z)) == '(bool)0'
    k = tvm.var('k', 'int8x2')
    assert str(tvm.isnan(k).dtype) == 'uint1x2'


def test_equality():
    a = tvm.var('a')
    b = tvm.var('b')
    c = (a == b)
    assert not c
    d = (c != c)
    assert not d


def test_equality_string_imm():
    x = 'a'
    y = tvm.tir.StringImm(x)
    x == y.value
    x == y


if __name__ == "__main__":
    test_cast()
    test_attr()
    test_const()
    test_scalar_dtype_inference()
    test_make()
    test_ir()
    test_basic()
    test_stmt()
    test_let()
    test_dir()
    test_dtype()
    test_any()
    test_all()
    test_bitwise()
    test_float_bitwise()
    test_isnan()
    test_equality()
    test_equality_string_imm()
