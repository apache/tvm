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
import numpy as np
import pytest

import tvm_tirx
from tvm_tirx import ir


def test_const():
    x = tvm_tirx.tirx.const(1, "int32")
    assert x.ty.dtype == "int32"
    assert isinstance(x, tvm_tirx.ir.IntImm)


def test_te_const():
    x = tvm_tirx.tirx.const(1, "int32")
    assert x.ty.dtype == "int32"
    assert isinstance(x, tvm_tirx.ir.IntImm)


def test_tir_const_dtype_inference():
    for data in [
        True,
        bool(1),
        np.uint8(1),
        np.uint16(1),
        np.uint32(1),
        np.uint64(1),
        np.int8(1),
        np.int16(1),
        np.int32(1),
        np.int64(1),
        np.float16(1),
        np.float32(1),
        np.float64(1),
    ]:
        expected = {"b": "bool", "i": "int32", "u": "int32", "f": "float32"}[
            np.array(data).dtype.kind
        ]
        assert tvm_tirx.tirx.const(data).ty.dtype == expected

    assert tvm_tirx.tirx.const(True).ty.dtype == "bool"
    assert tvm_tirx.tirx.const(1).ty.dtype == "int32"
    assert tvm_tirx.tirx.const(1.0).ty.dtype == "float32"


def test_make():
    x = tvm_tirx.tirx.const(1, "int32")
    y = tvm_tirx.tirx.Var("x", "int32")
    z = x + y
    assert isinstance(tvm_tirx.tirx.max(x, y), tvm_tirx.tirx.Max)
    assert isinstance(tvm_tirx.tirx.min(x, y), tvm_tirx.tirx.Min)


def test_ir():
    x = tvm_tirx.tirx.const(1, "int32")
    y = tvm_tirx.ir.IntImm("int32", 1)
    z = x + y
    stmt = tvm_tirx.ir.Evaluate(z)
    assert isinstance(stmt, tvm_tirx.ir.Evaluate)


def test_ir2():
    buf_size = tvm_tirx.tirx.Var("size", "int32")
    x = tvm_tirx.tirx.Var("n", "int32")

    storage_type = ir.PrimType("int32")
    handle_type = ir.PtrType(storage_type)
    array = tvm_tirx.tirx.Var("array", handle_type)
    buf = tvm_tirx.tirx.decl_tensor([buf_size], "int32", data=array)

    st = tvm_tirx.ir.TensorStore(buf, [1], x + 1)
    assert isinstance(st, tvm_tirx.ir.TensorStore)
    assert st.dest == buf
    assert st.dest.data.args[0].same_as(buf)
    assert st.dest.data.ty == array.ty


def test_let():
    x = tvm_tirx.tirx.Var("x", "int32")
    y = tvm_tirx.tirx.Var("y", "int32")
    stmt = tvm_tirx.ir.Bind(x, 10)


def test_cast():
    x = tvm_tirx.tirx.Var("x", "float32")
    y = x.astype("int32")
    z = x.astype("float32x4")
    assert isinstance(y, tvm_tirx.tirx.Cast)
    assert isinstance(z, tvm_tirx.ir.prim.vector_expr.Broadcast)
    assert z.lanes == 4

    s = tvm_tirx.ir.StringImm("s")
    with pytest.raises(
        TypeError, match="Operator overloading is not supported for expression type"
    ):
        s.astype("int")


def test_attr():
    a = tvm_tirx.runtime.convert(1)
    assert a == 1
    try:
        a.no_field
        assert False
    except AttributeError:
        pass


def test_basic():
    a = tvm_tirx.tirx.Var("a", "int32")
    b = tvm_tirx.tirx.Var("b", "int32")
    c = a + b
    assert str(c) == f"{a.name} + {b.name}"


def test_stmt():
    x = tvm_tirx.ir.Evaluate(0)
    tvm_tirx.ir.For(tvm_tirx.tirx.Var("i", "int32"), 0, 1, tvm_tirx.ir.ForKind.DEFAULT, x)
    tvm_tirx.ir.For(tvm_tirx.tirx.Var("i", "int32"), 0, 1, tvm_tirx.ir.ForKind.UNROLLED, x, step=2)


def test_dir():
    x = tvm_tirx.tirx.Var("x", "int32")
    dir(x)


def test_dtype():
    x = tvm_tirx.tirx.Var("x", "int32")
    assert x.ty.dtype == "int32"
    y = tvm_tirx.tirx.Var("y", "int32")
    assert (x > y).ty.dtype == "bool"


def test_any():
    x = tvm_tirx.tirx.Var("x", "int32")
    y = tvm_tirx.tirx.Var("y", "int32")
    z = tvm_tirx.tirx.Var("z", "int32")
    try:
        t = x or x
        assert False
    except ValueError:
        pass
    try:
        tvm_tirx.tirx.any()
        assert False
    except ValueError:
        pass
    assert str(tvm_tirx.tirx.any(x < y)) == f"{x.name} < {y.name}"
    assert str(tvm_tirx.tirx.any(x < y, x > z)) == (f"{x.name} < {y.name} or {x.name} > {z.name}")
    assert (
        str(tvm_tirx.tirx.any(x < y, y > z + 1, x < z * 2))
        == f"{x.name} < {y.name} or {y.name} > {z.name} + 1 or "
        f"{x.name} < {z.name} * 2"
    )


def test_all():
    x = tvm_tirx.tirx.Var("x", "int32")
    y = tvm_tirx.tirx.Var("y", "int32")
    z = tvm_tirx.tirx.Var("z", "int32")
    try:
        t = x and x
        assert False
    except ValueError:
        pass
    try:
        tvm_tirx.tirx.all()
        assert False
    except ValueError:
        pass
    assert str(tvm_tirx.tirx.all(x < y)) == f"{x.name} < {y.name}"
    assert str(tvm_tirx.tirx.all(x < y, x > z)) == (f"{x.name} < {y.name} and {x.name} > {z.name}")
    assert (
        str(tvm_tirx.tirx.all(x < y, y > z + 1, x < z * 2))
        == f"{x.name} < {y.name} and {y.name} > {z.name} + 1 and "
        f"{x.name} < {z.name} * 2"
    )


def test_bitwise():
    x = tvm_tirx.tirx.Var("x", "int32")
    y = tvm_tirx.tirx.Var("y", "int32")
    assert str(x << y) == "x << y"
    assert str(x >> y) == "x >> y"
    assert str(x & y) == "x & y"
    assert str(x | y) == "x | y"
    assert str(x ^ y) == "x ^ y"
    assert str(10 & x) == "10 & x"
    assert str(10 | x) == "10 | x"
    assert str(10 ^ x) == "10 ^ x"
    assert str(10 >> x) == "10 >> x"
    assert str(10 << x) == "10 << x"
    assert str(10 % x) == "10 % x"

    assert str(~x) == "~x"
    assert (tvm_tirx.tirx.const(1, "int8x2") >> 1).ty.dtype == "int8x2"
    assert (x >> tvm_tirx.tirx.const(1, "int32x2")).ty.dtype == "int32x2"
    assert (
        tvm_tirx.tirx.Var("z", "int8x2") << tvm_tirx.tirx.const(1, "int8x2")
    ).ty.dtype == "int8x2"


def test_float_bitwise():
    t = tvm_tirx.tirx.const(1.5, dtype="float32")
    for test in [
        lambda lhs, rhs: lhs << rhs,
        lambda lhs, rhs: lhs >> rhs,
        lambda lhs, rhs: lhs | rhs,
        lambda lhs, rhs: lhs ^ rhs,
        lambda lhs, rhs: lhs & rhs,
    ]:
        try:
            test(t, 10.0)
            assert False
        except RuntimeError:
            pass
    try:
        ~t
        assert False
    except RuntimeError:
        pass


def test_shift_bounds():
    x = tvm_tirx.tirx.Var("x", "int32")
    for test in [lambda lhs, rhs: lhs << rhs, lambda lhs, rhs: lhs >> rhs]:
        # negative case
        for testcase in [(x, -1), (x, 32)]:
            try:
                test(*testcase)
                assert False
            except RuntimeError:
                pass

        # positive case
        for testcase in [(x, 0), (x, 16), (x, 31)]:
            test(*testcase)


def test_divide_by_zero():
    for test in [
        lambda lhs, rhs: tvm_tirx.tirx.floormod(lhs, rhs),
        lambda lhs, rhs: tvm_tirx.tirx.floordiv(lhs, rhs),
        lambda lhs, rhs: tvm_tirx.tirx.truncmod(lhs, rhs),
        lambda lhs, rhs: tvm_tirx.tirx.truncdiv(lhs, rhs),
        lambda lhs, rhs: tvm_tirx.tirx.div(lhs, rhs),
    ]:
        try:
            test(tvm_tirx.tirx.const(5, "int32"), tvm_tirx.tirx.const(0, "int32"))
            assert False
        except RuntimeError:
            pass


def test_infinity():
    assert str(tvm_tirx.tirx.infinity(ty="float16")) == 'T.float16("inf")'
    assert str(tvm_tirx.tirx.infinity(ty="float32")) == 'T.float32("inf")'
    assert str(tvm_tirx.tirx.infinity(ty="float64")) == 'T.float64("inf")'


def test_isnan():
    x = tvm_tirx.tirx.Var("x", "float32")
    assert str(tvm_tirx.tirx.isnan(x)) == "T.isnan(x)"
    assert str(tvm_tirx.tirx.isnan(x).ty.dtype) == "bool"
    y = tvm_tirx.tirx.Var("y", "float16")
    assert str(tvm_tirx.tirx.isnan(y)) == "T.isnan(y)"
    z = tvm_tirx.tirx.Var("z", "int32")
    assert str(tvm_tirx.tirx.isnan(z)) == "T.isnan(z)"
    k = tvm_tirx.tirx.Var("k", "int8x2")
    assert str(tvm_tirx.tirx.isnan(k).ty.dtype) == "boolx2"


def test_equality():
    a = tvm_tirx.tirx.Var("a", "int32")
    b = tvm_tirx.tirx.Var("b", "int32")
    c = a == b
    assert not c
    d = c != c
    assert not d


def test_equality_string_imm():
    x = "a"
    y = tvm_tirx.ir.StringImm(x)
    x == y.value
    x == y


def test_function():
    x = tvm_tirx.tirx.Var("x", "int32")
    y = tvm_tirx.tirx.Var("y", "int32")
    b = tvm_tirx.tirx.decl_tensor((x,), "float32")
    stmt = tvm_tirx.ir.SeqStmt([tvm_tirx.ir.Bind(x, 10), tvm_tirx.ir.Evaluate(x + 1)])

    func = tvm_tirx.tirx.Function([x, y, b], stmt)
    # make sure we can print
    assert func.params[2].same_as(b)
    assert not hasattr(func, "tensor_map")

    assert sum(tvm_tirx.tirx.is_tensor_var(param) for param in func.params) == 1
    assert func.with_body(tvm_tirx.ir.Evaluate(0)).params[2].same_as(b)
    f2 = func.with_attr({"calling_conv": 1, "tirx.noalias": True})
    assert f2.attrs["calling_conv"] == 1
    assert not func.attrs


def test_vars():
    x = tvm_tirx.tirx.Var("xyz", "int8")
    assert x.ty.dtype == "int8"
    ptype = tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float"))
    x = tvm_tirx.tirx.Var("xyz", ptype)
    assert x.ty == ptype
    assert isinstance(ptype.element_type, tvm_tirx.ir.PrimType)


def test_scoped_storage_vars():
    dtype = "float"
    storage_scope = "global.texture"
    ptype = tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType(dtype), storage_scope)
    x = tvm_tirx.tirx.Var("xyz", ptype)
    assert x.ty == ptype
    assert x.ty.scope == tvm_tirx.runtime.MemoryScope(storage_scope)
    assert isinstance(ptype.element_type, tvm_tirx.ir.PrimType)


def test_tensor_load_store():
    b = tvm_tirx.tirx.decl_tensor((10,), "float32")
    x = tvm_tirx.tirx.TensorLoad(b, [0])
    assert isinstance(x, tvm_tirx.ir.TensorLoad)
    assert callable(tvm_tirx.tirx.TensorLoad)
    assert x.ty.dtype == "float32"
    assert x.source == b
    assert not hasattr(x, "tensor")
    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("tensor_load"), 1, 2, 1, 3)
    core_load = tvm_tirx.ir.TensorLoad(b, [0], loc)
    dialect_load = tvm_tirx.tirx.TensorLoad(b, [0], loc)
    tvm_tirx.ir.assert_structural_equal(core_load, dialect_load)
    assert core_load.loc.same_as(loc)
    restored = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(core_load))
    assert isinstance(restored.ty, tvm_tirx.ir.PrimType)
    tvm_tirx.ir.assert_structural_equal(core_load, restored, map_free_vars=True)
    s = tvm_tirx.ir.TensorStore(b, [0], 0.1)
    assert isinstance(s, tvm_tirx.ir.TensorStore)

    tensor = tvm_tirx.tirx.decl_tensor((4, 8), "float32", elem_offset=100)
    tvm_tirx.ir.assert_structural_equal(
        tensor[2, 3].indices,
        [tvm_tirx.ir.IntImm("int32", 2), tvm_tirx.ir.IntImm("int32", 3)],
    )


@pytest.mark.parametrize("constructor", [tvm_tirx.ir.TensorLoad, tvm_tirx.tirx.TensorLoad])
def test_tensor_load_validation(constructor):
    tensor = tvm_tirx.tirx.decl_tensor((8, 8), "float32")
    with pytest.raises(RuntimeError, match="dimensional"):
        constructor(tensor, [0])
    with pytest.raises(RuntimeError, match="Only the last index"):
        constructor(tensor, [tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 4), 0])
    with pytest.raises(TypeError, match="does not support TensorLoad"):
        constructor(tvm_tirx.tirx.Var("scalar", "float32"), [0])


@pytest.mark.parametrize(
    "element_dtype,index_lanes,result_dtype",
    [
        ("float32x2", 4, "float32x8"),
        ("float32x2", "scalable", "float32xvscalex8"),
        ("float32xvscalex2", 4, "float32xvscalex8"),
    ],
)
def test_tensor_load_vector_elements(element_dtype, index_lanes, result_dtype):
    tensor = tvm_tirx.tirx.decl_tensor((32,), element_dtype)
    lanes = 4 * tvm_tirx.tirx.vscale() if index_lanes == "scalable" else index_lanes
    load = tvm_tirx.ir.TensorLoad(tensor, [tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, lanes)])
    assert load.ty.dtype == result_dtype


def test_tensor_load_rejects_two_scalable_types():
    tensor = tvm_tirx.tirx.decl_tensor((32,), "float32xvscalex2")
    with pytest.raises(RuntimeError, match="can't both be scalable"):
        tvm_tirx.ir.TensorLoad(
            tensor, [tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 4 * tvm_tirx.tirx.vscale())]
        )


def test_intimm_cond():
    x = tvm_tirx.runtime.convert(1)
    y = tvm_tirx.runtime.convert(1)
    s = {x}
    assert y in s
    assert x == y
    assert x < 20
    assert not (x >= 20)
    assert x < 10 and y < 10
    assert not tvm_tirx.runtime.convert(x != 1)
    assert x == 1


def _create_ramp(lanes):
    return tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, lanes)


def _create_broadcast(lanes):
    return tvm_tirx.ir.prim.vector_expr.Broadcast(0, lanes)


@pytest.mark.parametrize("lanes", [tvm_tirx.ir.IntImm(ty="int64", value=11)])
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_lane_types(lanes, node_func):
    def _check_dtype(node):
        assert node.lanes.ty.dtype == "int32"
        assert node.lanes == 11

    _check_dtype(node_func(lanes))


@pytest.mark.parametrize("lanes", [(11 * tvm_tirx.tirx.vscale()), (tvm_tirx.tirx.vscale() * 11)])
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_scalable_vec(lanes, node_func):
    def _check_dtype(node):
        assert node.lanes.a.equal(tvm_tirx.tirx.vscale())
        assert node.lanes.b == 11

    _check_dtype(node_func(lanes))


@pytest.mark.parametrize(
    "lanes",
    [(tvm_tirx.tirx.vscale()), (tvm_tirx.tirx.vscale() + 3), (tvm_tirx.tirx.vscale() * 2 + 5)],
)
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_scalable_vec_error(lanes, node_func):
    with pytest.raises(RuntimeError):
        node_func(lanes)


def test_broadcast_to_scalable_vec():
    vec = tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 4 * tvm_tirx.tirx.vscale()) + 3
    broadcast = vec.b

    assert isinstance(broadcast, tvm_tirx.ir.prim.vector_expr.Broadcast)
    assert broadcast.value == 3
    assert broadcast.lanes.a.equal(tvm_tirx.tirx.vscale())
    assert broadcast.lanes.b == 4


def test_tensor_load_scalable_vec():
    buf = tvm_tirx.tirx.decl_tensor((24,), "float32")
    index = tvm_tirx.ir.prim.vector_expr.Ramp(1, 1, 8 * tvm_tirx.tirx.vscale())
    load = tvm_tirx.tirx.TensorLoad(buf, [index])

    assert isinstance(load, tvm_tirx.ir.TensorLoad)
    assert load.ty.dtype == "float32xvscalex8"


def test_tensor_store_scalable_vec():
    b = tvm_tirx.tirx.decl_tensor((24,), "int32")
    value = tvm_tirx.ir.prim.vector_expr.Broadcast(1, 4 * tvm_tirx.tirx.vscale())
    index = tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 4 * tvm_tirx.tirx.vscale())
    store = tvm_tirx.ir.TensorStore(b, [index], value)

    assert isinstance(store, tvm_tirx.ir.TensorStore)
    assert store.value.ty.dtype == "int32xvscalex4"


def test_scalable_vec_cast():
    b = tvm_tirx.tirx.decl_tensor((24,), "float32")
    value = tvm_tirx.ir.prim.vector_expr.Broadcast(1, 12 * tvm_tirx.tirx.vscale()).astype(
        "float32xvscalex12"
    )
    index = tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 12 * tvm_tirx.tirx.vscale())

    store = tvm_tirx.ir.TensorStore(b, [index], value)

    assert isinstance(store.value.value, tvm_tirx.ir.FloatImm)


@pytest.mark.parametrize("dtype", ["float16", "float16x2", "float32x4", "int32", "int32x2"])
def test_isnan_preserves_canonical_operand(dtype):
    value = tvm_tirx.tirx.Var("value", dtype)
    result = tvm_tirx.tirx.isnan(value)
    assert result.op.name == "prim.isnan"
    assert result.args[0].same_as(value)
    assert result.ty.dtype.lanes == value.ty.dtype.lanes
    expected_dtype = "bool" + (dtype[dtype.index("x") :] if "x" in dtype else "")
    assert result.ty.dtype == expected_dtype
    for constant in (tvm_tirx.tirx.const(0, "int32"), tvm_tirx.tirx.const(float("nan"), "float16")):
        assert tvm_tirx.tirx.isnan(constant).args[0].same_as(constant)


if __name__ == "__main__":
    tvm_tirx.testing.main()
