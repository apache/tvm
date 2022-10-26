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


def test_commutative_equal():
    x = te.var("x")
    y = te.var("y")
    m = te.var("m")
    a = te.var("a")
    b = te.var("b")
    c = te.var("c")
    int_imm = tvm.te.const(3, "int32")
    ana = tvm.arith.Analyzer()

    def func1():
        return x + y + 1

    def func2():
        return te.exp(tvm.tir.truncdiv((x + y + 1) * y, 4))

    assert tvm.tir.analysis.commutative_deep_equal(func1(), func1())
    assert tvm.tir.analysis.commutative_deep_equal(func2(), func2())
    assert not tvm.tir.analysis.commutative_deep_equal(func2(), func1())

    def func3():
        return x * m + y * c - a + b * m

    def func4():
        return c * y + m * x - a + m * b

    assert tvm.tir.analysis.commutative_deep_equal(func3(), func4())

    def func5():
        return a * b * c + x + y

    def func6():
        return b * a * c + y + x

    assert tvm.tir.analysis.commutative_deep_equal(func5(), func6())

    def func7():
        return x // int_imm * a * b * c

    def func8():
        return x // int_imm * c * a * b

    assert tvm.tir.analysis.commutative_deep_equal(func7(), func8())

    def func9():
        return x // (int_imm + y + m) * a * b * c

    def func10():
        return x // (m + int_imm + y) * c * a * b

    assert tvm.tir.analysis.commutative_deep_equal(func9(), func10())

    def func11():
        return x * y * a * b

    def func12():
        return a * y * x * b

    assert tvm.tir.analysis.commutative_deep_equal(func11(), func12())

    def func13():
        return x * m + y * (c + a) - a + b * m

    def func14():
        return (a + c) * y + m * x - a + m * b

    assert tvm.tir.analysis.commutative_deep_equal(func13(), func14())

    def func15():
        return a * (b * 3)

    def func16():
        return (a * 3) * b

    assert tvm.tir.analysis.commutative_deep_equal(func15(), func16())


if __name__ == "__main__":
    test_commutative_equal()
