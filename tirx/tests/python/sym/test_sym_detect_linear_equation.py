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
    m = tvm_tirx.sym.detect_linear_equation(a * 4 + b * 6 + 7, [a])
    assert m[0].value == 4
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[1], b * 6 + 7)

    m = tvm_tirx.sym.detect_linear_equation(a * 4 * (a + 1) + b * 6 + 7, [a])
    assert len(m) == 0

    m = tvm_tirx.sym.detect_linear_equation(a * 4 + (a + 1) + b * 6 + 7, [a])
    assert m[0].value == 5
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[1], b * 6 + 7 + 1)

    m = tvm_tirx.sym.detect_linear_equation(a * b + 7, [a])
    assert m[0] == b

    m = tvm_tirx.sym.detect_linear_equation(b * 7, [a])
    assert m[0].value == 0

    m = tvm_tirx.sym.detect_linear_equation(b * 7, [])
    assert len(m) == 1
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[0], b * 7)

    c = tvm_tirx.tirx.Var("c", "uint32")
    m = tvm_tirx.sym.detect_linear_equation(128 - c, [c])
    assert m[0].value == -1


def test_multivariate():
    v = [tvm_tirx.tirx.Var(f"v{i}", "int32") for i in range(4)]
    b = tvm_tirx.tirx.Var("b", "int32")
    m = tvm_tirx.sym.detect_linear_equation(v[0] * (b + 4) + v[0] + v[1] * 8, v)

    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[0], b + 5)

    assert m[1].value == 8

    m = tvm_tirx.sym.detect_linear_equation(v[0] * (b + 4) + v[0] + v[1] * 8 * v[2], v)
    assert len(m) == 0

    m = tvm_tirx.sym.detect_linear_equation(v[0] * (b + 4) + v[0] + v[1] * 8 * v[1] + v[3], v)
    assert len(m) == 0

    m = tvm_tirx.sym.detect_linear_equation(((v[0] * b + v[1]) * 8 + v[2] + 1) * 2, v)
    assert m[1].value == 16
    assert m[2].value == 2
    assert m[len(m) - 1].value == 2

    m = tvm_tirx.sym.detect_linear_equation((v[0] - v[1]), [v[2]])
    assert m[0].value == 0

    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[1], v[0] - v[1])

    m = tvm_tirx.sym.detect_linear_equation((v[0] - v[1]), [])
    assert len(m) == 1
    assert tvm_tirx.sym.Analyzer().can_prove_equal(m[0], v[0] - v[1])


if __name__ == "__main__":
    test_basic()
    test_multivariate()
