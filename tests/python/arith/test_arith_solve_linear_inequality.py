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


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/11458")
def test_solution_consistency():
    seed = random.randrange(sys.maxsize)
    print(
        "\nThis test is intentionally non-deterministic, "
        "if it fails please report it in github issue together with this seed {}\n".format(seed)
    )
    random.seed(seed)

    def _check(variables, formulas, coef=(-5, 5), bounds=(-20, 20)):
        vs = [te.var("x" + str(i)) for i in range(variables)]

        fs = []
        for i in range(formulas):
            s1 = sum([v * random.randint(coef[0], coef[1]) for v in vs])
            s1 += random.randint(coef[0], coef[1])
            s2 = sum([v * random.randint(coef[0], coef[1]) for v in vs])
            s2 += random.randint(coef[0], coef[1])
            op = random.choice([tir.expr.EQ, tir.expr.LE, tir.expr.LT, tir.expr.GE, tir.expr.GT])
            fs.append(op(s1, s2))

        vranges = {v: tvm.ir.expr.Range(bounds[0], bounds[1] + 1) for v in vs}
        before = te.all(tir.const(1, "bool"), *fs)
        after = arith._ffi_api.SolveInequalitiesAsCondition(vs, vranges, fs)
        after = te.all(tir.const(1, "bool"), *after)
        testing.check_bool_expr_is_true(before == after, vranges)

        solution = arith.solve_linear_inequalities(fs, vs, vranges, deskew_range=True)
        testing.check_int_constraints_trans_consistency(solution)

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
    assert ir.structural_equal(solution[0], x >= (y + 10))
    assert ir.structural_equal(solution[1], x <= (20 - y))
    assert ir.structural_equal(solution[2], y >= 0)
    assert ir.structural_equal(solution[3], y <= 5)

    # solve and get the ranges
    solution = arith.solve_linear_inequalities(problem, variables, ranges)
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
    assert ir.structural_equal(rel, (y_new * 2) + x_new <= 10)
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
    assert len(solution.relations) == 3
    assert ir.structural_equal(solution.relations[0], x == z * y)

    assert isinstance(solution.relations[1], tvm.tir.LE)
    assert solution.relations[1].b == 0
    assert isinstance(solution.relations[2], tvm.tir.LE)
    assert solution.relations[2].b == 0
    # (z*y - 6) <= 0 && (6 - z*y) <= 0
    ana = tvm.arith.Analyzer()
    assert ana.simplify(solution.relations[1].a + solution.relations[2].a) == 0
    assert ir.structural_equal(solution.relations[1].a, (z * y - 6)) or ir.structural_equal(
        solution.relations[2].a, (z * y - 6)
    )

    solution = arith.solve_linear_inequalities(problem, [x, y, z], deskew_range=True)
    assert solution.src_to_dst[y] == y
    assert solution.src_to_dst[z] == z
    assert solution.src_to_dst[x] == 6


def test_no_solution():
    x = te.var("x0")
    vranges = {x: tvm.ir.Range.from_min_extent(-20, 41)}
    problem = [-x - 4 <= -5 * x + 2, x * 4 + 5 <= x * 5]

    solution = arith.solve_linear_inequalities(problem, [x], vranges, deskew_range=True)
    assert list(solution.dst.variables) == []
    [rel] = solution.dst.relations
    assert ir.structural_equal(rel, False)
    assert len(solution.src_to_dst) == 0
    assert len(solution.dst_to_src) == 0

    solution = arith.solve_linear_inequalities(problem, [x], vranges)
    assert len(solution.variables) == 0
    assert len(solution.ranges) == 0
    [rel] = solution.relations
    assert not rel


def test_unbound_var_range():
    x = te.var("x0")
    free_var = te.var("fv")
    vranges = {x: tvm.ir.Range.from_min_extent(0, tvm.tir.Cast("int32", 1 + tvm.tir.log(free_var)))}
    problem = [x > 3]
    solution = arith.solve_linear_inequalities(
        problem,
        [x],
        vranges,
    )
    assert len(solution.variables) == 1
    assert len(solution.ranges) == 0
    assert len(solution.relations) == 3


if __name__ == "__main__":
    tvm.testing.main()
