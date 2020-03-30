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
import tvm
from tvm import te, arith
from tvm.tir import ir_pass


def test_unique_solution():
    x, y = te.var("x"), te.var("y")
    ranges = {}

    solution = arith.solve_equations([
        tvm.tir.EQ(x + y, 20),
        tvm.tir.EQ(x - y, 10),
    ], [x, y], ranges)
    assert list(solution.dst.variables) == []
    assert ir_pass.Equal(solution.src_to_dst[x], 15)
    assert ir_pass.Equal(solution.src_to_dst[y], 5)


def test_low_rank():
    x, y, z = te.var("x"), te.var("y"), te.var("z")
    ranges = {}

    solution = arith.solve_equations([
        tvm.tir.EQ(x + y + z, 15),
        tvm.tir.EQ(x + y, 10),
    ], [x, y, z], ranges)
    [n0] = solution.dst.variables
    assert ir_pass.Equal(solution.src_to_dst[x], n0 + 10)
    assert ir_pass.Equal(solution.src_to_dst[y], -n0)
    assert ir_pass.Equal(solution.src_to_dst[z], 5)


def test_infer_range():
    x, y = te.var("x"), te.var("y")
    ranges = {
        x: tvm.ir.Range.make_by_min_extent(-5, 10),
        y: tvm.ir.Range.make_by_min_extent(0, 10),
    }

    solution = arith.solve_equations([
        tvm.tir.EQ(x + y, 0),
    ], [x, y], ranges)
    [n0] = solution.dst.variables
    assert ir_pass.Equal(solution.src_to_dst[x], n0)
    assert ir_pass.Equal(solution.src_to_dst[y], -n0)
    # inferred from y's range
    assert ir_pass.Equal(solution.dst.ranges[n0].min, -9)
    assert ir_pass.Equal(solution.dst.ranges[n0].extent, 10)
    # additional inequality is added into the system for x
    [ineq] = solution.dst.relations
    assert isinstance(ineq, tvm.tir.LE)
    assert ir_pass.Equal(ineq.a, -5)
    assert ir_pass.Equal(ineq.b, n0)


def test_ill_formed():
    x, y = te.var("x"), te.var("y")

    solution = arith.solve_equations([
        tvm.tir.EQ(x + y, 0),
        tvm.tir.EQ(x - y, 0),
        tvm.tir.EQ(x, 5),
    ], [x, y], {})
    assert list(solution.dst.variables) == []
    [rel] = solution.dst.relations
    assert ir_pass.Equal(rel, False)
    assert len(solution.src_to_dst) == 0
    assert len(solution.dst_to_src) == 0


if __name__ == "__main__":
    test_unique_solution()
    test_low_rank()
    test_infer_range()
    test_ill_formed()
