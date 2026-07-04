# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
"""Tests for TransitiveComparisonAnalyzer and the per-key index."""

import tvm
import tvm.ir
import tvm.testing
from tvm import tirx
from tvm.script import tirx as T


def test_single_bind_provability():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 100))
    assert analyzer.can_prove(x >= 0)
    assert analyzer.can_prove(x < 100)
    assert analyzer.can_prove(x <= 99)
    assert not analyzer.can_prove(x >= 1)


def test_many_binds_correctness_preserved():
    analyzer = tvm.arith.Analyzer()
    vars_ = [tirx.Var(f"v{i}", "int32") for i in range(2048)]
    for i, v in enumerate(vars_):
        analyzer.bind(v, tvm.ir.Range.from_min_extent(i, 10))
    for i in (0, len(vars_) // 2, len(vars_) - 1):
        v = vars_[i]
        assert analyzer.can_prove(v >= i)
        assert analyzer.can_prove(v < i + 10)
        assert not analyzer.can_prove(v >= i + 1)


def test_bind_override_clears_old_constraints():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 100))
    assert analyzer.can_prove(x < 100)
    analyzer.bind(x, tvm.ir.Range.from_min_extent(200, 100), allow_override=True)
    assert analyzer.can_prove(x >= 200)
    assert analyzer.can_prove(x < 300)
    assert not analyzer.can_prove(x < 100)
    assert not analyzer.can_prove(x < 200)


def test_bind_override_clears_constraints_where_var_is_rhs():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    analyzer.bind(y, tvm.ir.Range.from_min_extent(0, 10))
    analyzer.bind(x, y + 5)
    assert analyzer.can_prove(x < 15)
    analyzer.bind(y, tvm.ir.Range.from_min_extent(200, 100), allow_override=True)
    assert not analyzer.can_prove(x < 15)
    assert analyzer.can_prove(x >= 205)


def test_scoped_constraint_enter_and_exit():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 100))
    with analyzer.constraint_scope(y < x):
        assert analyzer.can_prove(y < x)
    assert not analyzer.can_prove(y < x)


def test_cross_key_lookup():
    analyzer = tvm.arith.Analyzer()
    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    analyzer.bind(a, tvm.ir.Range.from_min_extent(0, 100))
    with analyzer.constraint_scope(b > a):
        assert analyzer.can_prove(a < b)


def test_nested_constraint_scopes():
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    z = tirx.Var("z", "int32")
    analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 100))
    with analyzer.constraint_scope(y < x):
        assert analyzer.can_prove(y < x)
        with analyzer.constraint_scope(z < y):
            assert analyzer.can_prove(y < x)
            assert analyzer.can_prove(z < y)
        assert analyzer.can_prove(y < x)
        assert not analyzer.can_prove(z < y)
    assert not analyzer.can_prove(y < x)
    assert not analyzer.can_prove(z < y)


def test_unrelated_binds_do_not_match():
    analyzer = tvm.arith.Analyzer()
    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    d = tirx.Var("d", "int32")
    analyzer.bind(a, tvm.ir.Range.from_min_extent(0, 10))
    analyzer.bind(b, tvm.ir.Range.from_min_extent(0, 10))
    analyzer.bind(c, tvm.ir.Range.from_min_extent(0, 10))
    assert not analyzer.can_prove(a < b)
    assert not analyzer.can_prove(b < c)
    assert not analyzer.can_prove(c < d)


def test_scoped_then_global_bind_interaction():
    analyzer = tvm.arith.Analyzer()
    y = tirx.Var("y", "int32")
    x = tirx.Var("x", "int32")
    with analyzer.constraint_scope(y > 0):
        analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 100))
        assert analyzer.can_prove(x < 100)
        assert analyzer.can_prove(y > 0)
    assert not analyzer.can_prove(y > 0)
    assert analyzer.can_prove(x < 100)


def test_self_comparison_indexed_once():
    # `x == x` produces a Comparison with lhs_ == rhs_; IndexKnown
    # must store it once, not twice.
    analyzer = tvm.arith.Analyzer()
    x = tirx.Var("x", "int32")
    with analyzer.constraint_scope(x == x):
        assert analyzer.can_prove(x == x)
    analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 10))
    assert analyzer.can_prove(x >= 0)
    assert analyzer.can_prove(x < 10)


def test_transitively_prove_inequalities_uses_dfs_path():
    # `i < j` and `j < k` (from For ranges) compose into `i < k` only
    # when the DFS path runs (transitively_prove_inequalities=True).

    @T.prim_func
    def before(A: T.Buffer((1,), "int32")):
        for i in T.serial(0, 50):
            for j in T.serial(i + 1, 50):
                for k in T.serial(j + 1, 50):
                    if i < k:
                        A[0] = 1
                    else:
                        A[0] = 0

    @T.prim_func
    def after_dfs(A: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "before"})
        for i in T.serial(0, 50):
            for j in T.serial(i + 1, 50):
                for k in T.serial(j + 1, 50):
                    A[0] = 1

    mod = tvm.IRModule({"main": before})
    expected = tvm.IRModule({"main": after_dfs})

    with tvm.transform.PassContext(
        config={"tirx.Simplify": {"transitively_prove_inequalities": True}}
    ):
        out_with_dfs = tvm.tirx.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(out_with_dfs, expected)

    # Negative control: without the flag the if-guard must remain, so
    # the result must NOT match `expected` (proves the positive
    # assertion above actually exercises the DFS path).
    out_no_dfs = tvm.tirx.transform.Simplify()(mod)
    assert not tvm.ir.structural_equal(out_no_dfs, expected)


if __name__ == "__main__":
    tvm.testing.main()
