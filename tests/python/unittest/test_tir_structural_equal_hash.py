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
import numpy as np
import pytest
from tvm import te


def consistent_equal(x, y, map_free_vars=False):
    struct_equal0 = tvm.ir.structural_equal(x, y, map_free_vars)
    struct_equal1 = tvm.ir.structural_equal(y, x, map_free_vars)

    xhash = tvm.ir.structural_hash(x, map_free_vars)
    yhash = tvm.ir.structural_hash(y, map_free_vars)

    if struct_equal0 != struct_equal1:
        raise ValueError(
            "Non-communicative {} vs {}, sequal0={}, sequal1={}".format(
                x, y, struct_equal0, struct_equal1
            )
        )

    # NOTE: hash colision can happen but should be rare.
    # we can confirm that hash colison doesn't happen for our testcases
    if struct_equal0 != (xhash == yhash):
        raise ValueError(
            "Inconsistent {} vs {}, sequal={}, xhash={}, yhash={}".format(
                x, y, struct_equal0, xhash, yhash
            )
        )
    return struct_equal0


def test_exprs():
    # save load json
    x = tvm.tir.const(1, "int32")
    y = tvm.tir.const(10, "int32")
    vx = te.var("x")
    vy = te.var("y")
    vz = te.var("z")
    zx = vx + vx
    zy = vy + vy

    assert consistent_equal(zx * zx, (vx + vx) * (vx + vx), map_free_vars=False)

    # test assert trigger.
    with pytest.raises(ValueError):
        tvm.ir.assert_structural_equal(x, y)

    assert not consistent_equal(vx, vy)
    assert consistent_equal(vx, vy, map_free_vars=True)
    # corner case lhs:vx == rhs:vy, but cannot map it iteslf
    assert not consistent_equal(vx + vx, vy + vx, map_free_vars=True)
    # corner case lhs:vx == rhs:vy, lhs:vy == rhs:vx
    assert consistent_equal(vx + vy, vy + vx, map_free_vars=True)
    # corner case2: rolling remap.
    assert consistent_equal(vx + vy + vz, vy + vz + vx, map_free_vars=True)
    assert not consistent_equal(vx + 1, vy + 1, map_free_vars=False)
    # Defintition remap
    assert consistent_equal(tvm.tir.Let(vx, 1, vx - 1), tvm.tir.Let(vy, 1, vy - 1))
    # Default same address free var remap
    assert consistent_equal(tvm.tir.Let(vx, 1, vx // vz), tvm.tir.Let(vy, 1, vy // vz))

    assert consistent_equal(zx * zx, zx * zx)
    assert consistent_equal(zx * zx, zy * zy, map_free_vars=True)
    assert not consistent_equal(zx * zx, zy * zy, map_free_vars=False)


def test_prim_func():
    x = te.var("x")
    y = te.var("y")
    # counter example of same equality
    func0 = tvm.tir.PrimFunc([x, y], tvm.tir.Evaluate(x + y))
    func1 = tvm.tir.PrimFunc([x, y], tvm.tir.Evaluate(y + x))
    assert not consistent_equal(func0, func1)

    # new cases
    b = tvm.tir.decl_buffer((x,), "float32")
    stmt = tvm.tir.LetStmt(x, 10, tvm.tir.Evaluate(x + 1))
    func0 = tvm.tir.PrimFunc([x, y, b], stmt)
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


def test_array():
    x = np.arange(10)
    nx = tvm.nd.array(x)
    ny = tvm.nd.array(x)
    nz = tvm.nd.array(x.reshape(2, 5))
    assert consistent_equal(nx, ny)
    assert not consistent_equal(nx, nz)


def test_env_func():
    @tvm.register_func("test.sequal.env_func")
    def test(x):
        return x + 1

    x = tvm.ir.EnvFunc.get("test.sequal.env_func")
    y = tvm.ir.EnvFunc.get("test.sequal.env_func")
    assert consistent_equal(y, x)


def test_attrs():
    x = tvm.ir.make_node("attrs.TestAttrs", axis=1, name="xx")
    y = tvm.ir.make_node("attrs.TestAttrs", axis=1, name="xx")
    z = tvm.ir.make_node("attrs.TestAttrs", axis=2, name="xx")
    tvm.ir.assert_structural_equal(y, x)
    assert not consistent_equal(y, z)

    x = tvm.runtime.convert({"x": [1, 2, 3], "y": 2})
    y = tvm.runtime.convert({"y": 2, "x": [1, 2, 3]})
    z = tvm.runtime.convert({"y": 2, "x": [1, 2, 3, 4]})
    assert consistent_equal(y, x)
    assert not consistent_equal(y, z)


def test_stmt():
    x = te.var("x")
    y = te.var("y")
    n = 128
    A = te.placeholder((n, n), name="A")
    B = te.placeholder((n, n), name="B")
    ii = te.var("i")
    jj = te.var("j")

    Ab = tvm.tir.decl_buffer((n,), name="A")
    n = te.var("n")

    def func2():
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(Ab)
        with ib.for_range(0, n, name="i") as i:
            A[i] = A[i] + 1
            with ib.for_range(0, 10, name="j") as j:
                A[j] = A[j] + 2
                A[j] = A[j] + 2
        return ib.get()

    assert consistent_equal(func2(), func2())


def test_buffer_load_store():
    b = tvm.tir.decl_buffer((10, 10), "float32")
    x = tvm.tir.BufferLoad(b, [0, 1])
    y = tvm.tir.BufferLoad(b, [0, 1])
    z = tvm.tir.BufferLoad(b, [1, 2])
    assert consistent_equal(y, x)
    assert not consistent_equal(y, z)

    i = tvm.tir.Var("x", "int32")
    sx = tvm.tir.BufferStore(b, 0.1, [0, i])
    sy = tvm.tir.BufferStore(b, 0.1, [0, i])
    sz = tvm.tir.BufferStore(b, 0.1, [1, i])
    assert consistent_equal(sy, sx)
    assert not consistent_equal(sy, sz)


if __name__ == "__main__":
    test_exprs()
    test_prim_func()
    test_attrs()
    test_array()
    test_env_func()
    test_stmt()
    test_buffer_load_store()
