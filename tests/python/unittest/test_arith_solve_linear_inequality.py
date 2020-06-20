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
import random
import sys
import pytest
import tvm
from tvm import te, arith, ir, tir, testing


def check_solution(solution, vranges={}):
    """Check that solution is a bijective transformation"""
    def _check_forward(constraints1, constraints2, varmap, backvarmap):
        ana = tvm.arith.Analyzer()
        all_vranges = vranges.copy()
        all_vranges.update({v: r for v, r in constraints1.ranges.items()})

        # Check that the transformation is injective
        cond_on_vars = tir.const(1, 'bool')
        for v in constraints1.variables:
            # variable mapping is consistent
            v_back = ana.simplify(tir.stmt_functor.substitute(varmap[v], backvarmap))
            cond_on_vars = te.all(cond_on_vars, v == v_back)
        # Also we have to check that the new relations are true when old relations are true
        cond_subst = tir.stmt_functor.substitute(
            te.all(tir.const(1, 'bool'), *constraints2.relations), backvarmap)
        # We have to include relations from vranges too
        for v in constraints2.variables:
            if v in constraints2.ranges:
                r = constraints2.ranges[v]
                range_cond = te.all(v >= r.min, v < r.min + r.extent)
                range_cond = tir.stmt_functor.substitute(range_cond, backvarmap)
                cond_subst = te.all(cond_subst, range_cond)
        cond_subst = ana.simplify(cond_subst)
        testing.check_bool_expr_is_true(
            te.all(cond_subst, cond_on_vars), all_vranges,
            cond=te.all(tir.const(1, 'bool'), *constraints1.relations))

    rels = solution.dst.relations
    if len(rels) == 1 and ir.structural_equal(rels[0], False):
        # not solvable, skip
        return
    _check_forward(solution.src, solution.dst,
                   solution.src_to_dst, solution.dst_to_src)
    _check_forward(solution.dst, solution.src,
                   solution.dst_to_src, solution.src_to_dst)


def test_solve_system_of_inequalities():
    seed = random.randrange(sys.maxsize)
    print("\nThis test is intentionally non-deterministic, "
          "if it fails please report it in github issue together with this seed {}\n".format(seed))
    random.seed(seed)

    def _check(variables, formulas, coef=(-5, 5), bounds=(-20, 20)):
        vs = [te.var("x" + str(i)) for i in range(variables)]

        fs = []
        for i in range(formulas):
            s1 = sum([v*random.randint(coef[0], coef[1]) for v in vs])
            s1 += random.randint(coef[0], coef[1])
            s2 = sum([v*random.randint(coef[0], coef[1]) for v in vs])
            s2 += random.randint(coef[0], coef[1])
            op = random.choice([tir.expr.EQ, tir.expr.LE, tir.expr.LT, tir.expr.GE, tir.expr.GT])
            fs.append(op(s1, s2))

        vranges = {v: tvm.ir.expr.Range(bounds[0], bounds[1] + 1) for v in vs}
        before = te.all(tir.const(1, 'bool'), *fs)
        after = arith._ffi_api.SolveInequalitiesAsCondition(vs, vranges, fs)
        after = te.all(tir.const(1, 'bool'), *after)
        testing.check_bool_expr_is_true(before == after, vranges)

        solution = arith.solve_linear_inequalities(fs, vs, vranges, deskew_range=True)
        check_solution(solution)

    for i in range(3):
        _check(1, 1)
    for i in range(3):
        _check(1, 2)

    for i in range(3):
        _check(2, 1)
    for i in range(3):
        _check(2, 2)
    for i in range(3):
        _check(2, 3)

    # Somewhere here coefficients in the results become too large, leading to overflow,
    # so we use smaller initial coefficients
    for i in range(5):
        _check(3, 3, coef=(-2, 2))
    for i in range(5):
        _check(3, 4, coef=(-2, 2))

    for i in range(5):
        _check(4, 3, coef=(-1, 1))

    for i in range(5):
        _check(10, 2, coef=(-1, 1), bounds=(0, 4))
    for i in range(5):
        _check(10, 3, coef=(0, 1), bounds=(0, 4))


def test_dual_variable():
    x, y = te.var("x"), te.var("y")

    variables = [x, y]
    ranges = {
        x: tvm.ir.Range(-100, 100),
        y: tvm.ir.Range(0, 10),
    }
    problem = [
        tvm.tir.LE(x + y, 20),
        tvm.tir.GE(x - y, 10),
    ]

    # solution as conditions
    solution = arith._ffi_api.SolveInequalitiesAsCondition(variables, ranges, problem)
    assert len(solution) == 4
    assert ir.structural_equal(solution[0], y >= 0)
    assert ir.structural_equal(solution[1], y <= 5)
    assert ir.structural_equal(solution[2], x >= (y + 10))
    assert ir.structural_equal(solution[3], x <= (20 - y))

    # solve and get the ranges
    solution = arith.solve_linear_inequalities([
        tvm.tir.LE(x + y, 20),
        tvm.tir.GE(x - y, 10),
    ], [x, y], ranges)
    # 0 <= y <=5
    assert solution.ranges[y].min == 0
    assert solution.ranges[y].extent == 6
    # y + 10 <= x <= 20 - y
    assert ir.structural_equal(solution.ranges[x].min, y + 10)
    assert solution.ranges[x].extent == 11  # max(10 - 2y)

    # deskew the solved ranges to be starting from zero
    solution = arith.solve_linear_inequalities(problem, variables, ranges, deskew_range=True)
    [x_new, y_new] = solution.dst.variables
    [rel] = solution.dst.relations
    assert ir.structural_equal(rel, (y_new*2) + x_new <= 10)
    assert ir.structural_equal(solution.dst.ranges[x_new].min, 0)
    assert ir.structural_equal(solution.dst.ranges[x_new].extent, 11)
    assert ir.structural_equal(solution.dst.ranges[y_new].min, 0)
    assert ir.structural_equal(solution.dst.ranges[y_new].extent, 6)
    assert ir.structural_equal(solution.src_to_dst[x], x_new + (y_new + 10))
    assert ir.structural_equal(solution.src_to_dst[y], y_new)
    assert ir.structural_equal(solution.dst_to_src[x_new], x - y - 10)
    assert ir.structural_equal(solution.dst_to_src[y_new], y)


def test_equal():
    x, y = te.var("x"), te.var("y")
    problem = [
        tvm.tir.GE(x + y, 10),
        tvm.tir.GE(x - y, 2),
        tvm.tir.LE(x, 6),
    ]

    solution = arith.solve_linear_inequalities(problem, [x, y])
    assert solution.ranges[x].min == 6
    assert solution.ranges[x].extent == 1
    assert solution.ranges[y].min == 4
    assert solution.ranges[y].extent == 1

    solution = arith.solve_linear_inequalities(problem, [x, y], deskew_range=True)
    assert len(solution.dst.variables) == 0
    assert len(solution.dst.ranges) == 0
    assert len(solution.dst.relations) == 0
    assert solution.src_to_dst[x] == 6
    assert solution.src_to_dst[y] == 4


def test_multi_equal():
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    problem = [
        tvm.tir.LE(x, 6),
        tvm.tir.GE(x, 6),
        tvm.tir.GE(x - z * y, 0),
        tvm.tir.LE(x - z * y, 0),
    ]

    solution = arith.solve_linear_inequalities(problem, [x, y, z])
    assert solution.ranges[x].min == 6
    assert solution.ranges[x].extent == 1

    solution = arith.solve_linear_inequalities(problem, [x, y, z], deskew_range=True)
    assert solution.src_to_dst[y] == y
    assert solution.src_to_dst[z] == z
    assert solution.src_to_dst[x] == 6


if __name__ == "__main__":
    pytest.main([__file__])
