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

def test_const_fold():
    def check(f, *args):
        x = f(*[tvm.const(x, "int32") for x in args])
        y = f(*args)
        if not isinstance(x, (tvm.expr.IntImm, tvm.expr.UIntImm)) or x.value != int(y):
            raise ValueError("check error: %s vs %s " % (x, y))

    check(lambda x, y: x + y, 3, 4)
    check(lambda x, y: x * y, 3, 12)
    check(lambda x, y: x * y - 10, 3, 12)
    check(lambda x, y: x - y % 10, 3, 12)
    check(lambda x, y: x // y + 10, 100, 12)
    check(lambda x, y: x & y + 10, 112, 128)
    check(lambda x, y: x > y, 112, 128)
    check(lambda x, y: x < y, 112, 128)
    check(lambda x, y: x <= y, 112, 128)
    check(lambda x, y: x >= y, 112, 128)
    check(lambda x, y: (x | y) ^ 10, 112, 128)


def test_const_fold2():
    x = tvm.var("x")
    assert (x + 0).same_as(x)
    assert (0 + x).same_as(x)
    assert (x - 0).same_as(x)
    assert (x % 1).value == 0
    assert (x * 1).same_as(x)
    assert (1 * x).same_as(x)
    assert isinstance((1 / x), tvm.expr.Div)

def test_const_fold3():
    def check_throws(f):
        try:
            f()
        except tvm.TVMError:
            pass
        else:
            raise AssertionError("Should have raised an exception but didn't.")

    # Test that using ints with logic operations is forbidden
    x = tvm.var("x")
    for val in [0, 1]:
        for func in [tvm.all, tvm.any]:
            check_throws(lambda: func(tvm.const(val, 'uint1'), x))
            check_throws(lambda: func(x, tvm.const(val, 'uint1')))

    # Test const folding when both arguments are const
    for tvm_func, py_func in [(tvm.all, lambda a, b: a and b), (tvm.any, lambda a, b: a or b)]:
        for v1 in [0, 1]:
            for v2 in [0, 1]:
                assert tvm.ir_pass.Equal(tvm_func(tvm.const(v1, 'uint1'), tvm.const(v2, 'uint1')),
                                         tvm.const(py_func(v1, v2), 'uint1'))

    x = tvm.var("x", 'uint1')
    true = tvm.const(1, 'uint1')
    false = tvm.const(0, 'uint1')

    assert tvm.all(x, true).same_as(x)
    assert tvm.all(true, x).same_as(x)
    assert tvm.any(x, false).same_as(x)
    assert tvm.any(false, x).same_as(x)

    assert tvm.all(x, false).same_as(false)
    assert tvm.all(false, x).same_as(false)
    assert tvm.any(x, true).same_as(true)
    assert tvm.any(true, x).same_as(true)


def test_const_fold4():
    x1 = tvm.const(4, "int32")
    x2 = x1 + 5
    assert isinstance(x2, tvm.expr.IntImm) and x2.value == 9
    x3 = x2 / 3
    assert isinstance(x3, tvm.expr.IntImm) and x3.value == 3
    x4 = x3 + 0.55
    assert isinstance(x4, tvm.expr.FloatImm) and abs(x4.value - 3.55) < 1e-6
    x5 = tvm.ceil(x4)
    assert isinstance(x5, tvm.expr.FloatImm) and x5.value == 4
    x6 = x5.astype('int')
    assert isinstance(x6, tvm.expr.IntImm) and x6.value == 4, "x6={}".format(x6)
    y = (tvm.round((tvm.const(6.5, 'float32') - 1) / 1.5) + 2).astype('int')
    assert isinstance(y, tvm.expr.IntImm) and y.value == 6


if __name__ == "__main__":
    test_const_fold()
    test_const_fold2()
    test_const_fold3()
    test_const_fold4()
