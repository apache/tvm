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
import pytest
from tvm import te


def test_exprs():
    # save load json
    x = tvm.tir.const(1, "int32")
    y = tvm.tir.const(10, "int32")
    vx = te.var("x")
    vy = te.var("y")
    vz = te.var("z")

    # test assert trigger.
    with pytest.raises(ValueError):
        tvm.ir.assert_structural_equal(x, y)

    assert not tvm.ir.structural_equal(vx, vy)
    assert tvm.ir.structural_equal(vx, vy, map_free_vars=True)
    # corner case lhs:vx == rhs:vy, but cannot map it iteslf
    assert not tvm.ir.structural_equal(vx + vx, vy + vx, map_free_vars=True)
    # corner case lhs:vx == rhs:vy, lhs:vy == rhs:vx
    assert tvm.ir.structural_equal(vx + vy, vy + vx, map_free_vars=True)
    # corner case2: rolling remap.
    assert tvm.ir.structural_equal(vx + vy + vz, vy + vz + vx, map_free_vars=True)
    assert not tvm.ir.structural_equal(vx + 1, vy + 1, map_free_vars=False)
    # Defintition remap
    assert tvm.ir.structural_equal(tvm.tir.Let(vx, 1, vx - 1),
                                   tvm.tir.Let(vy, 1, vy - 1))
    # Default same address free var remap
    assert tvm.ir.structural_equal(tvm.tir.Let(vx, 1, vx // vz),
                                   tvm.tir.Let(vy, 1, vy // vz))

    zx = vx + vx
    zy = vy + vy
    assert tvm.ir.structural_equal(zx * zx, zx * zx)
    assert tvm.ir.structural_equal(zx * zx, zy * zy, map_free_vars=True)
    assert not tvm.ir.structural_equal(zx * zx, zy * zy, map_free_vars=False)
    assert tvm.ir.structural_equal(zx * zx, (vx + vx) * (vx + vx),
                                   map_free_vars=False)


def test_prim_func():
    x = te.var('x')
    y = te.var('y')
    # counter example of same equality
    func0 = tvm.tir.PrimFunc(
        [x, y], tvm.tir.Evaluate(x + y))
    func1 = tvm.tir.PrimFunc(
        [x, y], tvm.tir.Evaluate(y + x))
    assert not tvm.ir.structural_equal(func0, func1)

    # new cases
    b = tvm.tir.decl_buffer((x,), "float32")
    stmt = tvm.tir.LetStmt(
        x, 10, tvm.tir.Evaluate(x + 1))
    func0 = tvm.tir.PrimFunc(
        [x, y, b], stmt)
    # easiest way to deep copy is via save/load
    func1 = tvm.ir.load_json(tvm.ir.save_json(func0))
    tvm.ir.assert_structural_equal(func0, func1)

    data0 = tvm.nd.array([1, 2, 3])
    data1 = tvm.nd.array([1, 2, 3])
    # attributes and ndarrays
    func0 = func0.with_attr("data", data0)
    func1 = func1.with_attr("data", data1)
    # IRModules
    mod0 = tvm.IRModule.from_expr(func0)
    mod1 = tvm.IRModule.from_expr(func1)
    tvm.ir.assert_structural_equal(mod0, mod1)


def test_attrs():
    x = tvm.ir.make_node("attrs.TestAttrs", axis=1, name="xx")
    y = tvm.ir.make_node("attrs.TestAttrs", axis=1, name="xx")
    z = tvm.ir.make_node("attrs.TestAttrs", axis=2, name="xx")
    tvm.ir.assert_structural_equal(y, x)
    assert not tvm.ir.structural_equal(y, z)



if __name__ == "__main__":
    test_exprs()
    test_prim_func()
    test_attrs()
