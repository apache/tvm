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


def test_solution_consistency():
    seed = random.randrange(sys.maxsize)
    print(
        "\nThis test is intentionally non-deterministic, "
        "if it fails please report it in github issue together with this seed {}\n".format(seed)
    )
    random.seed(seed)

    def _check(num_vars, num_formulas, coef=(-5, 5), bounds=(-20, 20)):
        variables = [te.var("x" + str(i)) for i in range(num_vars)]

        relations = []
        for i in range(num_formulas):
            s1 = sum([v * random.randint(coef[0], coef[1]) for v in variables])
            s1 += random.randint(coef[0], coef[1])
            s2 = sum([v * random.randint(coef[0], coef[1]) for v in variables])
            s2 += random.randint(coef[0], coef[1])
            if random.random() < 0.7:
                op = tvm.tir.EQ
            else:
                # we also make sure it can correctly handle inequalities
                op = random.choice([tvm.tir.LE, tvm.tir.LT, tvm.tir.GE, tvm.tir.GT])
            relations.append(op(s1, s2))

        vranges = {v: tvm.ir.expr.Range(bounds[0], bounds[1] + 1) for v in variables}
        solution = arith.solve_linear_equations(relations, variables, vranges)

        testing.check_int_constraints_trans_consistency(solution)

        # leaving some variables as parameters should also be ok
        for k in [1, 2]:
            if len(variables) > k:
                solution = arith.solve_linear_equations(relations, variables[:-k], vranges)
                param_ranges = {v: vranges[v] for v in variables[-k:]}
                testing.check_int_constraints_trans_consistency(solution, param_ranges)

    for i in range(2):
        _check(num_vars=1, num_formulas=1)
    for i in range(2):
        _check(num_vars=1, num_formulas=2)

    for i in range(2):
        _check(num_vars=2, num_formulas=1)
    for i in range(2):
        _check(num_vars=2, num_formulas=2)
    for i in range(2):
        _check(num_vars=2, num_formulas=3)

    for i in range(3):
        _check(num_vars=3, num_formulas=3, coef=(-2, 2))
    for i in range(3):
        _check(num_vars=3, num_formulas=4, coef=(-2, 2))

    for i in range(3):
        _check(num_vars=4, num_formulas=3, coef=(-1, 1))

    for i in range(3):
        _check(num_vars=10, num_formulas=2, coef=(-1, 1), bounds=(0, 4))
    for i in range(3):
        _check(num_vars=10, num_formulas=3, coef=(0, 1), bounds=(0, 4))


def test_empty_var_to_solve():
    x, y = te.var("x"), te.var("y")
    equations = [
        tvm.tir.EQ(x + y, 20),
        tvm.tir.EQ(x - y, 10),
    ]
    solution = arith.solve_linear_equations(equations)
    assert len(solution.src_to_dst) == 0
    assert len(solution.dst_to_src) == 0
    assert len(solution.src.variables) == 0
    assert len(solution.src.ranges) == 0
    assert ir.structural_equal(solution.src.relations, equations)
    assert ir.structural_equal(solution.src, solution.dst)


def test_unique_solution():
    x, y = te.var("x"), te.var("y")

    solution = arith.solve_linear_equations(
        [
            tvm.tir.EQ(x + y, 20),
            tvm.tir.EQ(x - y, 10),
        ],
        [x, y],
    )
    assert list(solution.dst.variables) == []
    assert ir.structural_equal(solution.src_to_dst[x], 15)
    assert ir.structural_equal(solution.src_to_dst[y], 5)


def test_low_rank():
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    ranges = {}

    solution = arith.solve_linear_equations(
        [
            tvm.tir.EQ(x + y + z, 15),
            tvm.tir.EQ(x + y, 10),
        ],
        [x, y, z],
        ranges,
    )
    [n0] = solution.dst.variables
    assert ir.structural_equal(solution.src_to_dst[x], n0 + 10)
    assert ir.structural_equal(solution.src_to_dst[y], -n0)
    assert ir.structural_equal(solution.src_to_dst[z], 5)


def test_infer_range():
    x, y = te.var("x"), te.var("y")
    ranges = {
        x: tvm.ir.Range.from_min_extent(-5, 10),
        y: tvm.ir.Range.from_min_extent(0, 10),
    }

    solution = arith.solve_linear_equations(
        [
            tvm.tir.EQ(x + y, 0),
        ],
        [x, y],
        ranges,
    )
    [n0] = solution.dst.variables
    assert ir.structural_equal(solution.src_to_dst[x], n0)
    assert ir.structural_equal(solution.src_to_dst[y], -n0)
    # inferred from y's range
    assert ir.structural_equal(solution.dst.ranges[n0].min, -9)
    assert ir.structural_equal(solution.dst.ranges[n0].extent, 10)
    # additional inequality is added into the system for x
    [ineq] = solution.dst.relations
    assert isinstance(ineq, tvm.tir.LE)
    assert ir.structural_equal(ineq.a, -5)
    assert ir.structural_equal(ineq.b, n0)


def test_ill_formed():
    x, y = te.var("x"), te.var("y")

    solution = arith.solve_linear_equations(
        [
            tvm.tir.EQ(x + y, 0),
            tvm.tir.EQ(x - y, 0),
            tvm.tir.EQ(x, 5),
        ],
        [x, y],
        {},
    )
    assert list(solution.dst.variables) == []
    [rel] = solution.dst.relations
    assert ir.structural_equal(rel, False)
    assert len(solution.src_to_dst) == 0
    assert len(solution.dst_to_src) == 0


if __name__ == "__main__":
    tvm.testing.main()
