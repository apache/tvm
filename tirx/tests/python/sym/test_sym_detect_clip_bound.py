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
import tvm_tirx
import tvm_tirx.testing


def test_basic():
    a = tvm_tirx.tirx.Var("a", "int32")
    b = tvm_tirx.tirx.Var("b", "int32")
    c = tvm_tirx.tirx.Var("c", "int32")
    m = tvm_tirx.sym.detect_clip_bound(tvm_tirx.tirx.all(a * 1 < b * 6, a - 1 > 0), [a])
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[1], b * 6 - 1)
    assert m[0].value == 2
    m = tvm_tirx.sym.detect_clip_bound(tvm_tirx.tirx.all(a * 1 < b * 6, a - 1 > 0), [a, b])
    assert len(m) == 0
    m = tvm_tirx.sym.detect_clip_bound(tvm_tirx.tirx.all(a + 10 * c <= 20, b - 1 > 0), [a, b])
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[1], 20 - 10 * c)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[2], 2)
    m = tvm_tirx.sym.detect_clip_bound(
        tvm_tirx.tirx.all(tvm_tirx.tirx.Not(a * 1 > b * 6), a - 1 > 0), [a]
    )
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[1], b * 6)
    m = tvm_tirx.sym.detect_clip_bound(
        tvm_tirx.tirx.all(tvm_tirx.tirx.Min(a, b) > 3, a - 10 < 0), [a, b]
    )
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[0], 4)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[1], 9)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[2], 4)


def test_trivial_eq():
    a = tvm_tirx.tirx.Var("a", "int32")
    b = tvm_tirx.tirx.Var("b", "int32")
    m = tvm_tirx.sym.detect_clip_bound(b == 3, [a, b])
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[2], 3)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[3], 3)
    m = tvm_tirx.sym.detect_clip_bound(tvm_tirx.tirx.all(a == 4, b == 3), [a, b])
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[0], 4)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[1], 4)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[2], 3)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[3], 3)


if __name__ == "__main__":
    test_basic()
