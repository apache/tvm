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
# ruff: noqa: F811, F841
import numpy as np
import pytest

import tvm
from tvm import ir


def test_const():
    x = tvm.tirx.const(1, "int32")
    assert x.dtype == "int32"
    assert isinstance(x, tvm.tirx.IntImm)


def test_te_const():
    x = tvm.tirx.const(1, "int32")
    assert x.dtype == "int32"
    assert isinstance(x, tvm.tirx.IntImm)


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
        assert tvm.tirx.const(data).dtype == str(np.array(data).dtype)

    assert tvm.tirx.const(True).dtype == "bool"
    assert tvm.tirx.const(1).dtype == "int32"
    assert tvm.tirx.const(1.0).dtype == "float32"


def test_make():
    x = tvm.tirx.const(1, "int32")
    y = tvm.tirx.Var("x", "int32")
    z = x + y
    assert isinstance(tvm.tirx.max(x, y), tvm.tirx.Max)
    assert isinstance(tvm.tirx.min(x, y), tvm.tirx.Min)


def test_ir():
    x = tvm.tirx.const(1, "int32")
    y = tvm.tirx.IntImm("int32", 1)
    z = x + y
    stmt = tvm.tirx.Evaluate(z)
    assert isinstance(stmt, tvm.tirx.Evaluate)


def test_ir2():
    buf_size = tvm.tirx.Var("size", "int32")
    x = tvm.tirx.Var("n", "int32")

    storage_type = ir.PrimType("int32")
    handle_type = ir.PointerType(storage_type)
    array = tvm.tirx.Var("array", handle_type)
    buf = tvm.tirx.decl_buffer([buf_size], "int32", data=array)

    st = tvm.tirx.BufferStore(buf, x + 1, [1])
    assert isinstance(st, tvm.tirx.BufferStore)
    assert st.buffer == buf
    assert st.buffer.data == array


def test_let():
    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Var("y", "int32")
    stmt = tvm.tirx.Bind(x, 10)


def test_cast():
    x = tvm.tirx.Var("x", "float32")
    y = x.astype("int32")
    z = x.astype("float32x4")
    assert isinstance(y, tvm.tirx.Cast)
    assert isinstance(z, tvm.tirx.Broadcast)
    assert z.lanes == 4

    s = tvm.tirx.StringImm("s")
    with pytest.raises(tvm.error.TVMError):
        try:
            s.astype("int")
        except Exception as e:
            assert "Can't cast a handle to other types" in str(e)
            raise


def test_attr():
    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Var("y", "int32")
    stmt = tvm.tirx.AttrStmt(y, "stride", 10, tvm.tirx.Evaluate(x + 1))
    assert stmt.node == y

    a = tvm.runtime.convert(1)
    assert a == 1
    try:
        a.no_field
        assert False
    except AttributeError:
        pass


def test_basic():
    a = tvm.tirx.Var("a", "int32")
    b = tvm.tirx.Var("b", "int32")
    c = a + b
    assert str(c) == f"{a.name} + {b.name}"


def test_stmt():
    x = tvm.tirx.Evaluate(0)
    tvm.tirx.For(tvm.tirx.Var("i", "int32"), 0, 1, tvm.tirx.ForKind.SERIAL, x)
    tvm.tirx.For(tvm.tirx.Var("i", "int32"), 0, 1, tvm.tirx.ForKind.UNROLLED, x, step=2)


def test_dir():
    x = tvm.tirx.Var("x", "int32")
    dir(x)


def test_dtype():
    x = tvm.tirx.Var("x", "int32")
    assert x.dtype == "int32"
    y = tvm.tirx.Var("y", "int32")
    assert (x > y).dtype == "bool"


def test_any():
    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Var("y", "int32")
    z = tvm.tirx.Var("z", "int32")
    try:
        t = x or x
        assert False
    except ValueError:
        pass
    try:
        tvm.tirx.any()
        assert False
    except ValueError:
        pass
    assert str(tvm.tirx.any(x < y)) == f"{x.name} < {y.name}"
    assert str(tvm.tirx.any(x < y, x > z)) == f"{x.name} < {y.name} or {x.name} > {z.name}"
    assert (
        str(tvm.tirx.any(x < y, y > z + 1, x < z * 2))
        == f"{x.name} < {y.name} or {y.name} > {z.name} + 1 or {x.name} < {z.name} * 2"
    )


def test_all():
    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Var("y", "int32")
    z = tvm.tirx.Var("z", "int32")
    try:
        t = x and x
        assert False
    except ValueError:
        pass
    try:
        tvm.tirx.all()
        assert False
    except ValueError:
        pass
    assert str(tvm.tirx.all(x < y)) == f"{x.name} < {y.name}"
    assert str(tvm.tirx.all(x < y, x > z)) == f"{x.name} < {y.name} and {x.name} > {z.name}"
    assert (
        str(tvm.tirx.all(x < y, y > z + 1, x < z * 2))
        == f"{x.name} < {y.name} and {y.name} > {z.name} + 1 and {x.name} < {z.name} * 2"
    )


def test_bitwise():
    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Var("y", "int32")
    assert str(x << y) == "T.shift_left(x, y)"
    assert str(x >> y) == "T.shift_right(x, y)"
    assert str(x & y) == "T.bitwise_and(x, y)"
    assert str(x | y) == "T.bitwise_or(x, y)"
    assert str(x ^ y) == "T.bitwise_xor(x, y)"
    assert str(10 & x) == "T.bitwise_and(10, x)"
    assert str(10 | x) == "T.bitwise_or(10, x)"
    assert str(10 ^ x) == "T.bitwise_xor(10, x)"
    assert str(10 >> x) == "T.shift_right(10, x)"
    assert str(10 << x) == "T.shift_left(10, x)"
    assert str(10 % x) == "10 % x"

    assert str(~x) == "T.bitwise_not(x)"
    assert (tvm.tirx.const(1, "int8x2") >> 1).dtype == "int8x2"
    assert (x >> tvm.tirx.const(1, "int32x2")).dtype == "int32x2"
    assert (tvm.tirx.Var("z", "int8x2") << tvm.tirx.const(1, "int8x2")).dtype == "int8x2"


def test_float_bitwise():
    t = tvm.tirx.const(1.5, dtype="float32")
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
        except tvm.TVMError:
            pass
    try:
        ~t
        assert False
    except RuntimeError:
        pass


def test_shift_bounds():
    x = tvm.tirx.Var("x", "int32")
    for test in [lambda lhs, rhs: lhs << rhs, lambda lhs, rhs: lhs >> rhs]:
        # negative case
        for testcase in [(x, -1), (x, 32)]:
            try:
                test(*testcase)
                assert False
            except tvm.TVMError:
                pass

        # positive case
        for testcase in [(x, 0), (x, 16), (x, 31)]:
            test(*testcase)


def test_divide_by_zero():
    for test in [
        lambda lhs, rhs: tvm.tirx.floormod(lhs, rhs),
        lambda lhs, rhs: tvm.tirx.floordiv(lhs, rhs),
        lambda lhs, rhs: tvm.tirx.truncmod(lhs, rhs),
        lambda lhs, rhs: tvm.tirx.truncdiv(lhs, rhs),
        lambda lhs, rhs: tvm.tirx.div(lhs, rhs),
    ]:
        try:
            test(tvm.tirx.const(5, "int32"), tvm.tirx.const(0, "int32"))
            assert False
        except tvm.TVMError:
            pass


def test_infinity():
    assert str(tvm.tirx.infinity("float16")) == 'T.float16("inf")'
    assert str(tvm.tirx.infinity("float32")) == 'T.float32("inf")'
    assert str(tvm.tirx.infinity("float64")) == 'T.float64("inf")'


def test_isnan():
    x = tvm.tirx.Var("x", "float32")
    assert str(tvm.tirx.isnan(x)) == "T.isnan(x)"
    assert str(tvm.tirx.isnan(x).dtype) == "bool"
    y = tvm.tirx.Var("y", "float16")
    assert str(tvm.tirx.isnan(y)) == 'T.isnan(T.Cast("float32", y))'
    z = tvm.tirx.Var("z", "int32")
    assert str(tvm.tirx.isnan(z)) == "T.bool(False)"
    k = tvm.tirx.Var("k", "int8x2")
    assert str(tvm.tirx.isnan(k).dtype) == "boolx2"


def test_equality():
    a = tvm.tirx.Var("a", "int32")
    b = tvm.tirx.Var("b", "int32")
    c = a == b
    assert not c
    d = c != c
    assert not d


def test_equality_string_imm():
    x = "a"
    y = tvm.tirx.StringImm(x)
    x == y.value
    x == y


def test_prim_func():
    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Var("y", "int32")
    b = tvm.tirx.decl_buffer((x,), "float32")
    stmt = tvm.tirx.SeqStmt([tvm.tirx.Bind(x, 10), tvm.tirx.Evaluate(x + 1)])

    func = tvm.tirx.PrimFunc([x, y, b], stmt)
    # make sure we can print
    assert func.buffer_map[func.params[2]].same_as(b)

    assert len(func.buffer_map) == 1
    f2 = func.with_attr({"calling_conv": 1, "tirx.noalias": True})
    assert f2.attrs["calling_conv"] == 1
    assert not func.attrs


def test_vars():
    x = tvm.tirx.Var("xyz", "int8")
    assert x.dtype == "int8"
    ptype = tvm.ir.PointerType(tvm.ir.PrimType("float"))
    x = tvm.tirx.Var("xyz", ptype)
    assert x.dtype == "handle"
    assert x.type_annotation == ptype
    assert isinstance(ptype.element_type, tvm.ir.PrimType)


def test_scoped_storage_vars():
    dtype = "float"
    storage_scope = "global.texture"
    ptype = tvm.ir.PointerType(tvm.ir.PrimType(dtype), storage_scope)
    x = tvm.tirx.Var("xyz", ptype)
    assert x.dtype == "handle"
    assert x.type_annotation == ptype
    assert x.type_annotation.storage_scope == storage_scope
    assert isinstance(ptype.element_type, tvm.ir.PrimType)


def test_buffer_load_store():
    b = tvm.tirx.decl_buffer((10,), "float32")
    x = tvm.tirx.BufferLoad(b, [0])
    assert isinstance(x, tvm.tirx.BufferLoad)
    assert x.dtype == "float32"
    assert x.buffer == b
    s = tvm.tirx.BufferStore(b, 0.1, [0])
    assert isinstance(s, tvm.tirx.BufferStore)


def test_intimm_cond():
    x = tvm.runtime.convert(1)
    y = tvm.runtime.convert(1)
    s = {x}
    assert y in s
    assert x == y
    assert x < 20
    assert not (x >= 20)
    assert x < 10 and y < 10
    assert not tvm.runtime.convert(x != 1)
    assert x == 1


def _create_ramp(lanes):
    return tvm.tirx.Ramp(0, 1, lanes)


def _create_broadcast(lanes):
    return tvm.tirx.Broadcast(0, lanes)


@pytest.mark.parametrize("lanes", [tvm.tirx.IntImm(dtype="int64", value=11)])
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_lane_types(lanes, node_func):
    def _check_dtype(node):
        assert node.lanes.dtype == "int32"
        assert node.lanes == 11

    _check_dtype(node_func(lanes))


@pytest.mark.parametrize("lanes", [(11 * tvm.tirx.vscale()), (tvm.tirx.vscale() * 11)])
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_scalable_vec(lanes, node_func):
    def _check_dtype(node):
        assert node.lanes.a.equal(tvm.tirx.vscale())
        assert node.lanes.b == 11

    _check_dtype(node_func(lanes))


@pytest.mark.parametrize(
    "lanes", [(tvm.tirx.vscale()), (tvm.tirx.vscale() + 3), (tvm.tirx.vscale() * 2 + 5)]
)
@pytest.mark.parametrize("node_func", [_create_ramp, _create_broadcast])
def test_scalable_vec_error(lanes, node_func):
    with pytest.raises(tvm.error.TVMError):
        node_func(lanes)


def test_broadcast_to_scalable_vec():
    vec = tvm.tirx.expr.Ramp(0, 1, 4 * tvm.tirx.vscale()) + 3
    broadcast = vec.b

    assert isinstance(broadcast, tvm.tirx.expr.Broadcast)
    assert broadcast.value == 3
    assert broadcast.lanes.a.equal(tvm.tirx.vscale())
    assert broadcast.lanes.b == 4


def test_buffer_load_scalable_vec():
    buf = tvm.tirx.decl_buffer((24,), "float32")
    index = tvm.tirx.expr.Ramp(1, 1, 8 * tvm.tirx.vscale())
    load = tvm.tirx.BufferLoad(buf, [index])

    assert isinstance(load, tvm.tirx.BufferLoad)
    assert load.dtype == "float32xvscalex8"


def test_buffer_store_scalable_vec():
    b = tvm.tirx.decl_buffer((24,), "int32")
    value = tvm.tirx.expr.Broadcast(1, 4 * tvm.tirx.vscale())
    index = tvm.tirx.expr.Ramp(0, 1, 4 * tvm.tirx.vscale())
    store = tvm.tirx.BufferStore(b, value, [index])

    assert isinstance(store, tvm.tirx.BufferStore)
    assert store.value.dtype == "int32xvscalex4"


def test_buffer_store_predicate_invalid_scalability():
    b = tvm.tirx.decl_buffer((24,), "int32")
    value = tvm.tirx.expr.Broadcast(1, 4 * tvm.tirx.vscale())
    index = tvm.tirx.expr.Ramp(0, 1, 4 * tvm.tirx.vscale())
    predicate = tvm.tirx.expr.Broadcast(tvm.tirx.IntImm("int1", 1), 4)

    err_msg = "Predicate mask dtype and value dtype must both be scalable."
    with pytest.raises(tvm.TVMError, match=err_msg):
        tvm.tirx.BufferStore(b, value, [index], predicate)


def test_buffer_store_predicate_invalid_lanes():
    b = tvm.tirx.decl_buffer((24,), "int32")
    value = tvm.tirx.expr.Broadcast(1, 4 * tvm.tirx.vscale())
    index = tvm.tirx.expr.Ramp(0, 1, 4 * tvm.tirx.vscale())
    predicate = tvm.tirx.expr.Broadcast(tvm.tirx.IntImm("int1", 1), 8 * tvm.tirx.vscale())

    err_msg = (
        "Got a predicate mask with 8 lanes, but trying to store a "
        "value with 4 lanes. The number of lanes must match."
    )
    with pytest.raises(tvm.TVMError, match=err_msg):
        tvm.tirx.BufferStore(b, value, [index], predicate)


def test_buffer_store_predicate_elements_invalid_type():
    b = tvm.tirx.decl_buffer((24,), "int32")
    value = tvm.tirx.expr.Broadcast(1, 4 * tvm.tirx.vscale())
    index = tvm.tirx.expr.Ramp(0, 1, 4 * tvm.tirx.vscale())
    predicate = tvm.tirx.expr.Broadcast(1, 4 * tvm.tirx.vscale())

    err_msg = "Predicate mask elements must be boolean values, but got int32."
    with pytest.raises(tvm.TVMError, match=err_msg):
        tvm.tirx.BufferStore(b, value, [index], predicate)


def test_buffer_load_predicate_elements_invalid_type():
    b = tvm.tirx.decl_buffer((24,), "int32")
    index = tvm.tirx.expr.Ramp(0, 1, 4 * tvm.tirx.vscale())
    predicate = tvm.tirx.expr.Broadcast(1, 4 * tvm.tirx.vscale())

    err_msg = "Predicate mask elements must be boolean values, but got int32."
    with pytest.raises(tvm.TVMError, match=err_msg):
        tvm.tirx.BufferLoad(b, [index], predicate)


def test_buffer_store_predicate_invalid_scalability():
    b = tvm.tirx.decl_buffer((24,), "int32")
    index = tvm.tirx.expr.Ramp(0, 1, 4 * tvm.tirx.vscale())
    predicate = tvm.tirx.expr.Broadcast(tvm.tirx.IntImm("int1", 1), 4)

    err_msg = "Predicate mask dtype and load indices must both be scalable."
    with pytest.raises(tvm.TVMError, match=err_msg):
        tvm.tirx.BufferLoad(b, [index], predicate)


def test_buffer_store_predicate_invalid_lanes():
    b = tvm.tirx.decl_buffer((24,), "int32")
    index = tvm.tirx.expr.Ramp(0, 1, 4 * tvm.tirx.vscale())
    predicate = tvm.tirx.expr.Broadcast(tvm.tirx.IntImm("int1", 1), 8 * tvm.tirx.vscale())

    err_msg = (
        "Got a predicate mask with 8 lanes, but trying to load a "
        "vector with 4 lanes. The number of lanes must match."
    )
    with pytest.raises(tvm.TVMError, match=err_msg):
        tvm.tirx.BufferLoad(b, [index], predicate)


def test_scalable_vec_cast():
    b = tvm.tirx.decl_buffer((24,), "float32")
    value = tvm.tirx.expr.Broadcast(1, 12 * tvm.tirx.vscale()).astype("float32xvscalex12")
    index = tvm.tirx.expr.Ramp(0, 1, 12 * tvm.tirx.vscale())

    store = tvm.tirx.BufferStore(b, value, [index])

    assert isinstance(store.value.value, tvm.tirx.expr.FloatImm)


if __name__ == "__main__":
    tvm.testing.main()
