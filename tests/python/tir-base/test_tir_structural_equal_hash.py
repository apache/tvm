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
from tvm.runtime import ObjectPath
from tvm.script import tir as T, ir as I


def consistent_equal(x, y, map_free_vars=False):
    struct_equal0 = tvm.ir.structural_equal(x, y, map_free_vars)
    struct_equal1 = tvm.ir.structural_equal(y, x, map_free_vars)

    xhash = tvm.ir.structural_hash(x, map_free_vars)
    yhash = tvm.ir.structural_hash(y, map_free_vars)

    if struct_equal0 != struct_equal1:
        raise ValueError(
            "Non-commutative {} vs {}, sequal0={}, sequal1={}".format(
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


def get_sequal_mismatch(x, y, map_free_vars=False):
    mismatch_0 = tvm.ir.base.get_first_structural_mismatch(x, y, map_free_vars)
    mismatch_1 = tvm.ir.base.get_first_structural_mismatch(y, x, map_free_vars)

    if mismatch_0 is None and mismatch_1 is None:
        return None

    if (
        mismatch_0 is None
        or mismatch_1 is None
        or mismatch_0[0] != mismatch_1[1]
        or mismatch_0[1] != mismatch_1[0]
    ):
        raise ValueError(
            "Non-commutative {} vs {}, mismatch_0={}, mismatch_1={}".format(
                x, y, mismatch_0, mismatch_1
            )
        )

    return mismatch_0


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


def test_prim_func_param_count_mismatch():
    x = te.var("x")
    y = te.var("y")
    z = te.var("z")
    # counter example of same equality
    func0 = tvm.tir.PrimFunc([x, y], tvm.tir.Evaluate(x))
    func1 = tvm.tir.PrimFunc([x, y, z], tvm.tir.Evaluate(x))
    lhs_path, rhs_path = get_sequal_mismatch(func0, func1)
    expected_lhs_path = ObjectPath.root().attr("params").missing_array_element(2)
    expected_rhs_path = ObjectPath.root().attr("params").array_index(2)
    assert lhs_path == expected_lhs_path
    assert rhs_path == expected_rhs_path


def test_prim_func_param_dtype_mismatch():
    x = te.var("x")
    y_0 = te.var("y", dtype="int32")
    y_1 = te.var("z", dtype="float32")
    # counter example of same equality
    func0 = tvm.tir.PrimFunc([x, y_0], tvm.tir.Evaluate(x))
    func1 = tvm.tir.PrimFunc([x, y_1], tvm.tir.Evaluate(x))
    lhs_path, rhs_path = get_sequal_mismatch(func0, func1)
    expected_path = ObjectPath.root().attr("params").array_index(1).attr("dtype")
    assert lhs_path == expected_path
    assert rhs_path == expected_path


def test_prim_func_body_mismatch():
    x_0 = te.var("x")
    y_0 = te.var("y")
    x_1 = te.var("x")
    y_1 = te.var("y")
    # counter example of same equality
    func0 = tvm.tir.PrimFunc([x_0, y_0], tvm.tir.Evaluate(x_0 + x_0))
    func1 = tvm.tir.PrimFunc([x_1, y_1], tvm.tir.Evaluate(x_1 + y_1))
    lhs_path, rhs_path = get_sequal_mismatch(func0, func1)
    expected_path = ObjectPath.root().attr("body").attr("value").attr("b")
    assert lhs_path == expected_path
    assert rhs_path == expected_path


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


def test_buffer_storage_scope():
    x = te.var("x", dtype="handle")

    buffer_local_0 = tvm.tir.decl_buffer((10, 10), "float32", scope="local")
    buffer_local_1 = tvm.tir.decl_buffer((10, 10), "float32", scope="local")
    buffer_global = tvm.tir.decl_buffer((10, 10), "float32")
    buffer_empty = tvm.tir.decl_buffer((10, 10), "float32", scope="")

    func0 = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_local_0})
    func1 = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_local_1})
    func2 = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_global})
    func3 = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_empty})

    assert consistent_equal(func0, func1)
    assert consistent_equal(func2, func3)
    assert not consistent_equal(func0, func2)


def test_buffer_map_mismatch():
    x = te.var("x")
    buffer_0 = tvm.tir.decl_buffer((10, 10))
    buffer_0_clone = tvm.tir.decl_buffer((10, 10))
    buffer_1 = tvm.tir.decl_buffer((10, 20))

    func_0 = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_0})
    func_0_clone = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_0_clone})
    func_1 = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_1})

    lhs_path, rhs_path = get_sequal_mismatch(func_0, func_1)
    expected_path = (
        ObjectPath.root().attr("buffer_map").map_value(x).attr("shape").array_index(1).attr("value")
    )
    assert lhs_path == expected_path
    assert rhs_path == expected_path

    assert get_sequal_mismatch(func_0, func_0_clone) is None


def test_buffer_map_length_mismatch():
    x = te.var("x")
    y = te.var("x")

    buffer_0 = tvm.tir.decl_buffer((10, 10))
    buffer_1 = tvm.tir.decl_buffer((10, 20))

    func_0 = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_0})
    func_1 = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x), buffer_map={x: buffer_0, y: buffer_1})

    lhs_path, rhs_path = get_sequal_mismatch(func_0, func_1)

    expected_lhs_path = ObjectPath.root().attr("buffer_map").missing_map_entry()
    assert lhs_path == expected_lhs_path
    expected_rhs_path = ObjectPath.root().attr("buffer_map").map_value(y)
    assert rhs_path == expected_rhs_path


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


def test_while():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    wx = tvm.tir.While(x > 0, tvm.tir.Evaluate(x))
    wy = tvm.tir.While(y > 0, tvm.tir.Evaluate(y))
    assert not consistent_equal(wx, wy)
    assert consistent_equal(wx, wy, map_free_vars=True)


def test_while_condition_mismatch():
    x = tvm.tir.Var("x", "int32")
    w_0 = tvm.tir.While(x > 0, tvm.tir.Evaluate(x))
    w_1 = tvm.tir.While(x < 0, tvm.tir.Evaluate(x))
    lhs_path, rhs_path = get_sequal_mismatch(w_0, w_1)
    expected_path = ObjectPath.root().attr("condition")
    assert lhs_path == expected_path
    assert rhs_path == expected_path


def test_while_body_mismatch():
    x = tvm.tir.Var("x", "int32")
    w_0 = tvm.tir.While(x > 0, tvm.tir.Evaluate(x))
    w_1 = tvm.tir.While(x > 0, tvm.tir.Evaluate(x + 1))
    lhs_path, rhs_path = get_sequal_mismatch(w_0, w_1)
    expected_path = ObjectPath.root().attr("body").attr("value")
    assert lhs_path == expected_path
    assert rhs_path == expected_path


def test_seq_mismatch():
    x = tvm.tir.Var("x", "int32")
    seq_0 = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(x),
            tvm.tir.Evaluate(x + 1),
            tvm.tir.Evaluate(x + 2),
            tvm.tir.Evaluate(x + 3),
        ]
    )
    seq_1 = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(x),
            tvm.tir.Evaluate(x + 1),
            tvm.tir.Evaluate(x + 99),
            tvm.tir.Evaluate(x + 3),
        ]
    )
    lhs_path, rhs_path = get_sequal_mismatch(seq_0, seq_1)
    expected_path = (
        ObjectPath.root().attr("seq").array_index(2).attr("value").attr("b").attr("value")
    )
    assert lhs_path == expected_path
    assert rhs_path == expected_path


def test_seq_mismatch_different_lengths():
    # Make sure we report a difference inside the array first, rather than the difference in length
    x = tvm.tir.Var("x", "int32")
    seq_0 = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(x),
            tvm.tir.Evaluate(x + 1),
            tvm.tir.Evaluate(x + 2),
            tvm.tir.Evaluate(x + 3),
        ]
    )
    seq_1 = tvm.tir.SeqStmt([tvm.tir.Evaluate(x), tvm.tir.Evaluate(x + 1), tvm.tir.Evaluate(x + 3)])
    lhs_path, rhs_path = get_sequal_mismatch(seq_0, seq_1)
    expected_path = (
        ObjectPath.root().attr("seq").array_index(2).attr("value").attr("b").attr("value")
    )
    assert lhs_path == expected_path
    assert rhs_path == expected_path


def test_seq_length_mismatch():
    x = tvm.tir.Var("x", "int32")
    seq_0 = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(x),
            tvm.tir.Evaluate(x + 1),
            tvm.tir.Evaluate(x + 2),
            tvm.tir.Evaluate(x + 3),
        ]
    )
    seq_1 = tvm.tir.SeqStmt([tvm.tir.Evaluate(x), tvm.tir.Evaluate(x + 1), tvm.tir.Evaluate(x + 2)])
    lhs_path, rhs_path = get_sequal_mismatch(seq_0, seq_1)
    expected_lhs_path = ObjectPath.root().attr("seq").array_index(3)
    expected_rhs_path = ObjectPath.root().attr("seq").missing_array_element(3)
    assert lhs_path == expected_lhs_path
    assert rhs_path == expected_rhs_path


def test_ir_module_equal():
    def generate(n: int):
        @I.ir_module
        class module:
            @T.prim_func
            def func(A: T.Buffer(1, "int32")):
                for i in range(n):
                    A[0] = A[0] + 1

        return module

    # Equivalent IRModules should compare as equivalent, even though
    # they have distinct GlobalVars, and GlobalVars usually compare by
    # reference equality.
    tvm.ir.assert_structural_equal(generate(16), generate(16))

    # When there is a difference, the location should include the
    # function name that caused the failure.
    with pytest.raises(ValueError) as err:
        tvm.ir.assert_structural_equal(generate(16), generate(32))

    assert '<root>.functions[I.GlobalVar("func")].body.extent.value' in err.value.args[0]


def test_nan_values_are_equivalent():
    """Structural equality treats two NaN values as equivalent.

    By IEEE, a check of `NaN == NaN` returns false, as does
    `abs(NaN - NaN) < tolerance`.  However, for the purpose of
    comparing IR representations, both NaN values are equivalent.

    """

    @T.prim_func(private=True)
    def func_1():
        return T.float32("nan")

    @T.prim_func(private=True)
    def func_2():
        return T.float32("nan")

    tvm.ir.assert_structural_equal(func_1, func_2)
    assert tvm.ir.structural_hash(func_1) == tvm.ir.structural_hash(func_2)


def test_all_nan_values_are_equivalent():
    """Structural equality treats two NaN values as equivalent.

    IEEE defines NaN as any value that has all exponent bits set,
    and has a non-zero mantissa.  For the purposes of comparing IR
    representations, all NaN values are considered equivalent.

    """

    # A NaN with the first payload bit set.
    nan_all_zeros = np.int32(0x7FC00000).view("float32")

    # A NaN with the last payload bit set.
    nan_with_payload = np.int32(0x7F800001).view("float32")

    float_1 = T.float32(nan_all_zeros)
    float_2 = T.float32(nan_with_payload)

    tvm.ir.assert_structural_equal(float_1, float_2)
    assert tvm.ir.structural_hash(float_1) == tvm.ir.structural_hash(float_2)


if __name__ == "__main__":
    tvm.testing.main()
