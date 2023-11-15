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


def test_basic():
    a = te.var("a")
    b = te.var("b")
    c = te.var("c")
    m = tvm.arith.detect_clip_bound(tvm.tir.all(a * 1 < b * 6, a - 1 > 0), [a])
    tvm.testing.assert_prim_expr_equal(m[1], b * 6 - 1)
    assert m[0].value == 2
    m = tvm.arith.detect_clip_bound(tvm.tir.all(a * 1 < b * 6, a - 1 > 0), [a, b])
    assert len(m) == 0
    m = tvm.arith.detect_clip_bound(tvm.tir.all(a + 10 * c <= 20, b - 1 > 0), [a, b])
    tvm.testing.assert_prim_expr_equal(m[1], 20 - 10 * c)
    tvm.testing.assert_prim_expr_equal(m[2], 2)
    m = tvm.arith.detect_clip_bound(tvm.tir.all(tvm.tir.Not(a * 1 > b * 6), a - 1 > 0), [a])
    tvm.testing.assert_prim_expr_equal(m[1], b * 6)
    m = tvm.arith.detect_clip_bound(tvm.tir.all(tvm.tir.Min(a, b) > 3, a - 10 < 0), [a, b])
    tvm.testing.assert_prim_expr_equal(m[0], 4)
    tvm.testing.assert_prim_expr_equal(m[1], 9)
    tvm.testing.assert_prim_expr_equal(m[2], 4)


def test_trivial_eq():
    a = te.var("a")
    b = te.var("b")
    m = tvm.arith.detect_clip_bound(b == 3, [a, b])
    tvm.testing.assert_prim_expr_equal(m[2], 3)
    tvm.testing.assert_prim_expr_equal(m[3], 3)
    m = tvm.arith.detect_clip_bound(tvm.tir.all(a == 4, b == 3), [a, b])
    tvm.testing.assert_prim_expr_equal(m[0], 4)
    tvm.testing.assert_prim_expr_equal(m[1], 4)
    tvm.testing.assert_prim_expr_equal(m[2], 3)
    tvm.testing.assert_prim_expr_equal(m[3], 3)


if __name__ == "__main__":
    test_basic()
