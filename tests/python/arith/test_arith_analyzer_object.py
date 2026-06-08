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
from tvm.arith.analyzer import CompareResult, Extension
from tvm.runtime import Object


def test_analyzer_is_ffi_object_with_persistent_state():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int64")

    assert isinstance(analyzer, Object)

    analyzer.bind(x, tvm.ir.Range(0, 8))
    assert analyzer.const_int_bound_is_bound(x)
    assert analyzer.can_prove(x < 8)
    assert not analyzer.can_prove(x < 4)

    bound = analyzer.const_int_bound(x + 1)
    assert bound.min_value == 1
    assert bound.max_value == 8


def test_analyzer_object_constraint_scope_and_override_bind():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int64")

    with analyzer.constraint_scope(x % 3 == 0):
        assert analyzer.modular_set(x).coeff == 3

    assert analyzer.modular_set(x).coeff != 3

    analyzer = tvm.arith.Analyzer()
    y = tirx.Var("y", "int64")
    analyzer.bind(y, tirx.const(4, "int64"))
    tvm.ir.assert_structural_equal(analyzer.simplify(y + 1), tirx.const(5, "int64"))

    analyzer.bind(y, tirx.const(8, "int64"), allow_override=True)
    tvm.ir.assert_structural_equal(analyzer.simplify(y + 1), tirx.const(9, "int64"))


def test_analyzer_object_update_const_int_bound():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int64")

    analyzer.update(x, tvm.arith.ConstIntBound(2, 5))

    bound = analyzer.const_int_bound(x + 1)
    assert bound.min_value == 3
    assert bound.max_value == 6


def test_analyzer_object_update_modular_set():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")

    assert analyzer.modular_set(x).coeff == 1
    analyzer.update(x, tvm.arith.ModularSet(4, 0))

    result = analyzer.modular_set(x)
    assert result.coeff == 4
    assert result.base == 0


def test_analyzer_object_update_int_set():
    analyzer = tvm.arith.Analyzer()
    y = tirx.Var("y", "int32")

    analyzer.update(y, tvm.arith.IntervalSet(0, 8))

    int_set = analyzer.int_set(y)
    assert int_set.min_value.value == 0
    assert int_set.max_value.value == 8


def test_analyzer_object_update_rejects_unknown_info():
    analyzer = tvm.arith.Analyzer()
    y = tirx.Var("y", "int32")

    with pytest.raises(TypeError):
        analyzer.update(y, "not-an-info-object")


def test_analyzer_object_can_prove_comparison_predicates():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    analyzer.bind(x, tvm.ir.Range(0, 8))

    assert analyzer.can_prove(x >= 0)
    assert not analyzer.can_prove(x >= 1)
    assert analyzer.can_prove(x < 8)
    assert not analyzer.can_prove(x < 7)


def test_analyzer_object_update_const_int_bound_half_space():
    analyzer = tvm.arith.Analyzer()
    n = tirx.Var("n", "int32")

    assert not analyzer.can_prove(n >= 0)
    analyzer.update(n, tvm.arith.ConstIntBound(0, tvm.arith.ConstIntBound.POS_INF))
    assert analyzer.can_prove(n >= 0)


def test_analyzer_object_int_set_from_bound_vars():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    analyzer.bind(x, tvm.ir.Range(0, 8))

    int_set = analyzer.int_set(x + 1)
    assert int_set.min_value.value == 1
    assert int_set.max_value.value == 8


def test_analyzer_object_set_maximum_rewrite_steps():
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    expr = (x + y) * 2 - x * 2 - y * 2 + tirx.max(x, y) - tirx.min(x, y)

    capped = tvm.arith.Analyzer()
    capped.set_maximum_rewrite_steps(1)
    with pytest.raises(tvm.TVMError):
        capped.rewrite_simplify(expr)

    # A generous limit must not interfere with normal simplification.
    relaxed = tvm.arith.Analyzer()
    relaxed.set_maximum_rewrite_steps(1000)
    relaxed.rewrite_simplify(expr)


def test_analyzer_object_try_compare_transitive():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    z = tirx.Var("z", "int32")

    assert analyzer.try_compare(x, y) == CompareResult.UNKNOWN

    with analyzer.constraint_scope(x < y):
        with analyzer.constraint_scope(y < z):
            # Direct known comparison.
            assert analyzer.try_compare(x, y) == CompareResult.LT
            # Transitive chain x < y < z is found only when propagation is enabled.
            assert analyzer.try_compare(x, z) == CompareResult.LT
            assert analyzer.try_compare(x, z, propagate_inequalities=False) == CompareResult.UNKNOWN


def test_analyzer_object_enabled_extensions_round_trip():
    analyzer = tvm.arith.Analyzer()

    assert analyzer.enabled_extensions == Extension.NoExtensions

    analyzer.enabled_extensions = Extension.ComparisonOfProductAndSum
    assert analyzer.enabled_extensions == Extension.ComparisonOfProductAndSum

    analyzer.enabled_extensions = Extension.NoExtensions
    assert analyzer.enabled_extensions == Extension.NoExtensions


def test_analyzer_object_rewrite_simplify_stats():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")

    analyzer.reset_rewrite_simplify_stats()
    assert analyzer.rewrite_simplify_stats.nodes_visited == 0

    analyzer.rewrite_simplify(x + 0)
    assert analyzer.rewrite_simplify_stats.nodes_visited > 0

    analyzer.reset_rewrite_simplify_stats()
    assert analyzer.rewrite_simplify_stats.nodes_visited == 0


def test_analyzer_object_state_persists_across_ffi_calls():
    analyzer = tvm.arith.Analyzer()
    tile = tirx.Var("tile", "int32")
    i = tirx.Var("i", "int32")
    analyzer.bind(tile, tvm.tirx.const(8, "int32"))

    # The same analyzer object is borrowed by the C++ DetectIterMap entry point;
    # its binding makes the otherwise-undetectable floormod recognizable.
    result = tvm.arith.detect_iter_map([i % tile], {i: tvm.ir.Range(0, 32)}, analyzer=analyzer)
    assert len(result.indices) == 1

    # The binding still lives in the same stateful object after the FFI call.
    tvm.ir.assert_structural_equal(analyzer.simplify(tile), tvm.tirx.const(8, "int32"))


if __name__ == "__main__":
    tvm.testing.main()
