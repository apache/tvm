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
import numpy as np
import sys
import pytest
import tvm
from tvm import te, arith, ir, tir


def test_solve_system_of_inequalities():
    random.seed(0)

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

        print("--- before ---")
        print(fs)
        after = arith.solve_linear_inequalities(fs, vs, vranges)
        print("--- after ---")
        print(after)
        print()

        # check_bruteforce(before == after, vranges)

    _check(2, 2)

    # for i in range(3):
    #     _check(1, 1)
    # for i in range(3):
    #     _check(1, 2)
    #
    # for i in range(3):
    #     _check(2, 1)
    # for i in range(3):
    #     _check(2, 2)
    # for i in range(3):
    #     _check(2, 3)
    #
    # # Somewhere here coefficients in the results become too large, leading to overflow,
    # # so we use smaller initial coefficients
    #
    # for i in range(5):
    #     _check(3, 3, coef=(-2,2))
    # for i in range(5):
    #     _check(3, 4, coef=(-2,2))
    #
    # for i in range(5):
    #     _check(4, 3, coef=(-1,1))
    #
    # for i in range(5):
    #     _check(10, 2, coef=(-1,1), bounds=(0, 4))
    # for i in range(5):
    #     _check(10, 3, coef=(0,1), bounds=(0, 4))


def test_simple():
    x, y = te.var("x"), te.var("y")
    # TODO: following will hang forever
    # ranges = {
    #     x: tvm.ir.Range(-100, 0),
    #     y: tvm.ir.Range(0, 100),
    # }

    ranges = {
        x: tvm.ir.Range(-100, 100),
        y: tvm.ir.Range(0, 10),
    }

    solution = arith.solve_linear_inequalities([
        tvm.tir.LE(x + y, 20),
        tvm.tir.GE(x - y, 10),
    ], [x, y], ranges)

    print(solution)

    [x_new, y_new] = solution.dst.variables
    [rel] = solution.dst.relations

    assert ir.structural_equal(rel, (y_new*2) + x_new <= 10)

    assert ir.structural_equal(solution.dst.ranges[x_new].find_best_range().min, 0)
    assert ir.structural_equal(solution.dst.ranges[x_new].find_best_range().extent, 11)

    assert ir.structural_equal(solution.dst.ranges[y_new].find_best_range().min, 0)
    assert ir.structural_equal(solution.dst.ranges[y_new].find_best_range().extent, 6)

    assert ir.structural_equal(solution.src_to_dst[x], x_new + (y_new + 10))
    assert ir.structural_equal(solution.src_to_dst[y], y_new)
    assert ir.structural_equal(solution.dst_to_src[x_new], x - y - 10)
    assert ir.structural_equal(solution.dst_to_src[y_new], y)


def test_equal():
    x, y = te.var("x"), te.var("y")

    solution = arith.solve_linear_inequalities([
        tvm.tir.GE(x + y, 10),
        tvm.tir.GE(x - y, 2),
        tvm.tir.LE(x, 6),
    ], [x, y])

    print(solution)


def test_multi_equal():
    x, y = te.var("x"), te.var("y")

    solution = arith.solve_linear_inequalities([
        tvm.tir.LE(x, 6),
        tvm.tir.GE(x, 6),
        tvm.tir.GE(x - 2 * y, 0),
        tvm.tir.LE(x - 2 * y, 0),
    ], [x, y])

    print(solution)


if __name__ == "__main__":
    # test_solve_system_of_inequalities()
    test_simple()
    test_equal()
    test_multi_equal()
