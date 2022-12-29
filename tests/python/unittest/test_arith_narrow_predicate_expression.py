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

from tvm import tir
from tvm.runtime import convert


i = tir.Var("i", "int32")
j = tir.Var("j", "int32")
n = tir.Var("n", "int32")
m = tir.Var("m", "int32")
b = tir.Var("b", "bool")
buf = tir.decl_buffer(16, "int32", "buf")

tir_false = tir.IntImm("bool", False)
tir_true = tir.IntImm("bool", True)

before, expected = tvm.testing.parameters(
    # General arithmatic
    [tir_true, tir_true],
    [tir_false, tir_false],
    [b, b],
    [i > 5, i > 5],
    [i > n, i > 7],
    [i < n, i < 0],
    [i <= n, i <= 0],
    [i >= n, i >= 7],
    [n > i, convert(0) > i],
    [n < i, convert(7) < i],
    [n <= i, convert(7) <= i],
    [n >= i, convert(0) >= i],
    [i == n, tir.all(i <= 0, convert(7) <= i)],
    [n == i, tir.all(convert(7) <= i, i <= 0)],
    [i != n, tir.any(i < 0, convert(7) < i)],
    [n != i, tir.any(convert(7) < i, i < 0)],
    [i // 4 > n, i // 4 > 7],
    [n < i // 4, convert(7) < i // 4],
    [(i + n) // 4 > 0, tir.Add(i, 0) // 4 > 0],
    [(i + n) // 4 == 0, tir.all(tir.Add(i, 7) // 4 <= 0, convert(0) <= tir.Add(i, 0) // 4)],
    [i + n < 10, i + 7 < 10],
    [i - n < 10, tir.Sub(i, 0) < 10],
    [tir.Not(i < n), tir.Not(i < 7)],
    # Use of FloorMod should make the narrowing strategy bail out, as
    # it is non-monotonic.
    [i % 8 == n, tir_false],
    # Ensure that dividing by a free parameter doesn't generate a
    # divide-by-zero to be triggered later.
    [i // n == 0, tir_false],
    ### Buffer handling
    [buf.vload(0) > 0, tir_false],
    [buf.vload(0) > i, tir_false],
    [buf.vload(i) > 0, tir_false],
    [tir.And(buf.vload(i) > 0, i <= 0), tir.And(tir_false, i <= 0)],
    [tir.Or(buf.vload(i) > 0, i <= n), tir.Or(tir_false, i <= 0)],
    [tir.Or(tir.Not(buf.vload(i) > 0), i <= n), tir.Or(tir_false, i <= 0)],
)


def test_narrow_expression(before, expected):
    ranges = {n: tvm.ir.Range(0, 8)}
    after = tvm.arith._ffi_api.NarrowPredicateExpression(before, ranges)

    if expected is None:
        assert after is None
    else:
        tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm.testing.main()
