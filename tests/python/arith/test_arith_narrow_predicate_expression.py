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
# ruff: noqa: F401

import tvm
import tvm.testing
from tvm import tirx
from tvm.runtime import convert
from tvm.script import tirx as T

i = tirx.Var("i", "int32")
j = tirx.Var("j", "int32")
n = tirx.Var("n", "int32")
m = tirx.Var("m", "int32")
b = tirx.Var("b", "bool")
buf = tirx.decl_buffer(16, "int32", "buf")

tir_false = tirx.IntImm("bool", False)
tir_true = tirx.IntImm("bool", True)

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
    [n > i, T.int32(0) > i],
    [n < i, T.int32(7) < i],
    [n <= i, T.int32(7) <= i],
    [n >= i, T.int32(0) >= i],
    [i == n, tirx.all(i <= 0, T.int32(7) <= i)],
    [n == i, tirx.all(T.int32(7) <= i, i <= 0)],
    [i != n, tirx.any(i < 0, T.int32(7) < i)],
    [n != i, tirx.any(T.int32(7) < i, i < 0)],
    [i // 4 > n, i // 4 > 7],
    [n < i // 4, T.int32(7) < i // 4],
    [(i + n) // 4 > 0, tirx.Add(i, 0) // 4 > 0],
    [(i + n) // 4 == 0, tirx.all(tirx.Add(i, 7) // 4 <= 0, T.int32(0) <= tirx.Add(i, 0) // 4)],
    [i + n < 10, i + 7 < 10],
    [i - n < 10, tirx.Sub(i, 0) < 10],
    [tirx.Not(i < n), tirx.Not(i < 7)],
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
    [tirx.And(buf.vload(i) > 0, i <= 0), tirx.And(tir_false, i <= 0)],
    [tirx.Or(buf.vload(i) > 0, i <= n), tirx.Or(tir_false, i <= 0)],
    [tirx.Or(tirx.Not(buf.vload(i) > 0), i <= n), tirx.Or(tir_false, i <= 0)],
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
