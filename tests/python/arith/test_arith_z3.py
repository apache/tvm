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

import pytest

import tvm
import tvm.testing
from tvm import tirx


def _require_z3(analyzer):
    if "Z3 Prover is disabled" in analyzer.get_smtlib2():
        pytest.skip("Z3 prover is disabled in this build")


def test_z3_disabled_api_is_available():
    analyzer = tvm.arith.Analyzer()
    assert isinstance(analyzer.get_smtlib2(), str)
    assert isinstance(analyzer.get_z3_stats(), str)


def test_z3_proves_floor_division_identity():
    analyzer = tvm.arith.Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")

    with analyzer.constraint_scope(tirx.all(a > 0, b > 0, c > 0)):
        expr = ((b - a) // c) * c + a <= b
        assert analyzer.can_prove(expr)


def test_z3_bind_range():
    analyzer = tvm.arith.Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")

    analyzer.bind(a, tvm.ir.Range(1, 100000))
    analyzer.bind(b, tvm.ir.Range(1, 100000))
    analyzer.bind(c, tvm.ir.Range(1, 100000))

    expr = ((b - a) // c) * c + a <= b
    assert analyzer.can_prove(expr)


def test_z3_smtlib2_roundtrip():
    z3 = pytest.importorskip("z3")
    analyzer = tvm.arith.Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    expr = ((b - a) // c) * c + a <= b

    solver = z3.Solver()
    with analyzer.constraint_scope(tirx.all(a > 0, b > 0, c > 0)):
        solver.from_string(analyzer.get_smtlib2(expr))
    assert solver.check() == z3.unsat


def test_z3_bitwise():
    analyzer = tvm.arith.Analyzer()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    analyzer.bind(x, tvm.ir.Range(0, 256))

    assert analyzer.can_prove(tirx.bitwise_and(x, tirx.IntImm("int32", 7)) < 8)


if __name__ == "__main__":
    tvm.testing.main()
