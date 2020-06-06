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


def run_expr(expr, vranges):
    """ Evaluate expr for every value of free variables
    given by vranges and return the tensor of results.
    TODO(yzhliu): move to utils
    """
    def _compute_body(*us):
        vmap = {v: u + r.min for (v, r), u in zip(vranges.items(), us)}
        return tir.ir_pass.Substitute(expr, vmap)

    A = te.compute([r.extent.value for v, r in vranges.items()], _compute_body)
    args = [tvm.nd.empty(A.shape, A.dtype)]
    sch = te.create_schedule(A.op)
    mod = tvm.build(sch, [A])
    mod(*args)
    return args[0].asnumpy()


def check_bruteforce(bool_expr, vranges, cond=None):
    """ Check that bool_expr holds given the condition cond
    for every value of free variables from vranges.
    TODO(yzhliu): move to utils
    """
    if cond is not None:
        bool_expr = te.any(tir.Not(cond), bool_expr)

    res = run_expr(bool_expr, vranges)
    if not np.all(res):
        indices = list(np.argwhere(res == 0)[0])
        counterex = [(str(v), i + r.min) for (v, r), i in zip(vranges.items(), indices)]
        counterex = sorted(counterex, key=lambda x: x[0])
        counterex = ", ".join([v + " = " + str(i) for v, i in counterex])
        raise AssertionError("Expression {}\nis not true on {}\n"
                             "Counterexample: {}"
                             .format(tir.ir_pass.CanonicalSimplify(bool_expr), vranges, counterex))


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
        before = te.all(tir.const(1, 'bool'), *fs)

        print("--- before ---")
        print(fs)
        after = arith._ffi_api.SolveInequalitiesAsCondition(vs, vranges, fs)
        after = te.all(tir.const(1, 'bool'), *after)
        print("--- after ---")
        print(after)
        print()

        check_bruteforce(before == after, vranges)

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


def test_simple():
    x, y = te.var("x"), te.var("y")

    ranges = {
        x: tvm.ir.Range(-100, 100),
        y: tvm.ir.Range(0, 10),
    }

    solution = arith.solve_linear_inequalities([
        tvm.tir.LE(x + y, 20),
        tvm.tir.GE(x - y, 10),
    ], [x, y], ranges, deskew_range=True)

    print(solution)

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

    sol = arith.solve_linear_inequalities([
        tvm.tir.LE(x + y, 20),
        tvm.tir.GE(x - y, 10),
    ], [x, y], ranges)
    print(sol)
    # 0 <= y <=5
    assert sol.ranges[y].min == 0
    assert sol.ranges[y].extent == 6
    # y + 10 <= x <= 20 - y
    assert ir.structural_equal(sol.ranges[x].min, y + 10)
    assert sol.ranges[x].extent == 11  # max(10 - 2y)


def test_equal():
    x, y = te.var("x"), te.var("y")

    solution = arith.solve_linear_inequalities([
        tvm.tir.GE(x + y, 10),
        tvm.tir.GE(x - y, 2),
        tvm.tir.LE(x, 6),
    ], [x, y])

    print(solution)

    sol = arith.solve_linear_inequalities([
        tvm.tir.GE(x + y, 10),
        tvm.tir.GE(x - y, 2),
        tvm.tir.LE(x, 6),
    ], [x, y], deskew_range=True)
    print(sol)


def test_multi_equal():
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    solution = arith.solve_linear_inequalities([
        tvm.tir.LE(x, 6),
        tvm.tir.GE(x, 6),
        tvm.tir.GE(x - z * y, 0),
        tvm.tir.LE(x - z * y, 0),
    ], [x, y, z], deskew_range=True)

    print(solution)
    assert solution.src_to_dst[y] == y
    assert solution.src_to_dst[z] == z
    assert solution.src_to_dst[x] == 6

    print(arith.solve_linear_inequalities([
        tvm.tir.LE(x, 6),
        tvm.tir.GE(x, 6),
        tvm.tir.GE(x - z * y, 0),
        tvm.tir.LE(x - z * y, 0),
    ], [x, y, z]))


if __name__ == "__main__":
    # test_solve_system_of_inequalities()
    test_simple()
    # test_equal()
    # test_multi_equal()
