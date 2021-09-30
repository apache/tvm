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
import pytest
import tvm
from tvm import te
import numpy as np


def test_const():
    x = tvm.tir.const(1, "int32")
    print(x.dtype)
    assert x.dtype == "int32"
    assert isinstance(x, tvm.tir.IntImm)


def test_scalar_dtype_inference():
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
        assert tvm.tir.const(data).dtype == str(np.array(data).dtype)
    assert tvm.tir.const(1).dtype == "int32"
    assert tvm.tir.const(1.0).dtype == "float32"

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
        assert tvm.runtime.convert(data).dtype == str(np.array(data).dtype)
    assert tvm.runtime.convert(1).dtype == "int32"
    assert tvm.runtime.convert(1.0).dtype == "float32"


def test_make():
    x = tvm.tir.const(1, "int32")
    y = te.var("x")
    z = x + y
    assert isinstance(tvm.tir.max(x, y), tvm.tir.Max)
    assert isinstance(tvm.tir.min(x, y), tvm.tir.Min)


def test_ir():
    x = tvm.tir.const(1, "int32")
    y = tvm.tir.IntImm("int32", 1)
    z = x + y
    stmt = tvm.tir.Evaluate(z)
    assert isinstance(stmt, tvm.tir.Evaluate)


def test_ir2():
    x = te.var("n")
    a = te.var("array", "handle")
    st = tvm.tir.Store(a, x + 1, 1)
    assert isinstance(st, tvm.tir.Store)
    assert st.buffer_var == a


def test_let():
    x = te.var("x")
    y = te.var("y")
    stmt = tvm.tir.LetStmt(x, 10, tvm.tir.Evaluate(x + 1))


def test_cast():
    x = te.var("x", dtype="float32")
    y = x.astype("int32")
    z = x.astype("float32x4")
    assert isinstance(y, tvm.tir.Cast)
    assert isinstance(z, tvm.tir.Broadcast)
    assert z.lanes == 4

    s = tvm.tir.StringImm("s")
    with pytest.raises(tvm.error.TVMError) as cm:
        s.astype("int")
        assert "Can't cast a handle to other types" in str(cm.execption)


def test_attr():
    x = te.var("x")
    y = te.var("y")
    stmt = tvm.tir.AttrStmt(y, "stride", 10, tvm.tir.Evaluate(x + 1))
    assert stmt.node == y

    a = tvm.runtime.convert(1)
    assert a.value == 1
    try:
        a.no_field
        assert False
    except AttributeError:
        pass


def test_basic():
    a = te.var("a")
    b = te.var("b")
    c = a + b
    assert str(c) == "(%s: int32 + %s: int32)" % (a.name, b.name)


def test_stmt():
    x = tvm.tir.Evaluate(0)
    tvm.tir.For(te.var("i"), 0, 1, tvm.tir.ForKind.SERIAL, x)


def test_dir():
    x = te.var("x")
    dir(x)


def test_dtype():
    x = te.var("x")
    assert x.dtype == "int32"
    y = te.var("y")
    assert (x > y).dtype == "bool"


def test_any():
    x = te.var("x")
    y = te.var("y")
    z = te.var("z")
    try:
        t = x or x
        assert False
    except ValueError:
        pass
    try:
        tvm.tir.any()
        assert False
    except ValueError:
        pass
    assert str(tvm.tir.any(x < y)) == "(%s: int32 < %s: int32)" % (x.name, y.name)
    assert str(tvm.tir.any(x < y, x > z)) == "((%s: int32 < %s: int32) || (%s > %s: int32))" % (
        x.name,
        y.name,
        x.name,
        z.name,
    )
    assert str(
        tvm.tir.any(x < y, y > z + 1, x < z * 2)
    ) == "(((%s: int32 < %s: int32) || (%s > (%s: int32 + 1))) || (%s < (%s*2)))" % (
        x.name,
        y.name,
        y.name,
        z.name,
        x.name,
        z.name,
    )


def test_all():
    x = te.var("x")
    y = te.var("y")
    z = te.var("z")
    try:
        t = x and x
        assert False
    except ValueError:
        pass
    try:
        tvm.tir.all()
        assert False
    except ValueError:
        pass
    assert str(tvm.tir.all(x < y)) == "(%s: int32 < %s: int32)" % (x.name, y.name)
    assert str(tvm.tir.all(x < y, x > z)) == "((%s: int32 < %s: int32) && (%s > %s: int32))" % (
        x.name,
        y.name,
        x.name,
        z.name,
    )
    assert str(
        tvm.tir.all(x < y, y > z + 1, x < z * 2)
    ) == "(((%s: int32 < %s: int32) && (%s > (%s: int32 + 1))) && (%s < (%s*2)))" % (
        x.name,
        y.name,
        y.name,
        z.name,
        x.name,
        z.name,
    )


def test_bitwise():
    x = te.var("x")
    y = te.var("y")
    assert str(x << y) == "@tir.shift_left(x: int32, y: int32, dtype=int32)"
    assert str(x >> y) == "@tir.shift_right(x: int32, y: int32, dtype=int32)"
    assert str(x & y) == "@tir.bitwise_and(x: int32, y: int32, dtype=int32)"
    assert str(x | y) == "@tir.bitwise_or(x: int32, y: int32, dtype=int32)"
    assert str(x ^ y) == "@tir.bitwise_xor(x: int32, y: int32, dtype=int32)"
    assert str(10 & x) == "@tir.bitwise_and(10, x: int32, dtype=int32)"
    assert str(10 | x) == "@tir.bitwise_or(10, x: int32, dtype=int32)"
    assert str(10 ^ x) == "@tir.bitwise_xor(10, x: int32, dtype=int32)"
    assert str(10 >> x) == "@tir.shift_right(10, x: int32, dtype=int32)"
    assert str(10 << x) == "@tir.shift_left(10, x: int32, dtype=int32)"
    assert str(10 % x) == "floormod(10, x: int32)"

    assert str(~x) == "@tir.bitwise_not(x: int32, dtype=int32)"
    assert (tvm.tir.const(1, "int8x2") >> 1).dtype == "int8x2"
    assert (x >> tvm.tir.const(1, "int32x2")).dtype == "int32x2"
    assert (te.var("z", "int8x2") << tvm.tir.const(1, "int8x2")).dtype == "int8x2"


def test_float_bitwise():
    t = tvm.tir.const(1.5, dtype="float32")
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
    x = te.var("x")
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
        lambda lhs, rhs: tvm.tir.floormod(lhs, rhs),
        lambda lhs, rhs: tvm.tir.floordiv(lhs, rhs),
        lambda lhs, rhs: tvm.tir.truncmod(lhs, rhs),
        lambda lhs, rhs: tvm.tir.truncdiv(lhs, rhs),
        lambda lhs, rhs: tvm.tir.div(lhs, rhs),
    ]:
        try:
            test(tvm.tir.const(5, "int32"), tvm.tir.const(0, "int32"))
            assert False
        except tvm.TVMError:
            pass


def test_isnan():
    x = te.var("x", "float32")
    assert str(tvm.tir.isnan(x)) == "@tir.isnan(x: float32, dtype=bool)"
    assert str(tvm.tir.isnan(x).dtype) == "bool"
    y = te.var("y", "float16")
    assert str(tvm.tir.isnan(y)) == "@tir.isnan(cast(float32, y: float16), dtype=bool)"
    z = te.var("z", "int32")
    assert str(tvm.tir.isnan(z)) == "False"
    k = te.var("k", "int8x2")
    assert str(tvm.tir.isnan(k).dtype) == "uint1x2"


def test_equality():
    a = te.var("a")
    b = te.var("b")
    c = a == b
    assert not c
    d = c != c
    assert not d


def test_equality_string_imm():
    x = "a"
    y = tvm.tir.StringImm(x)
    x == y.value
    x == y


def test_prim_func():
    x = te.var("x")
    y = te.var("y")
    b = tvm.tir.decl_buffer((x,), "float32")
    stmt = tvm.tir.LetStmt(x, 10, tvm.tir.Evaluate(x + 1))

    func = tvm.tir.PrimFunc([x, y, b], stmt)
    # make sure we can print
    func.astext()
    assert func.buffer_map[func.params[2]].same_as(b)

    assert len(func.buffer_map) == 1
    f2 = func.with_attr({"calling_conv": 1, "tir.noalias": True})
    assert f2.attrs["calling_conv"].value == 1
    assert func.attrs is None


def test_vars():
    x = tvm.tir.Var("xyz", "int8")
    assert x.dtype == "int8"
    ptype = tvm.ir.PointerType(tvm.ir.PrimType("float"))
    x = tvm.tir.Var("xyz", ptype)
    assert x.dtype == "handle"
    assert x.type_annotation == ptype
    assert isinstance(ptype.element_type, tvm.ir.PrimType)


def test_scoped_storage_vars():
    dtype = "float"
    storage_scope = "global.texture"
    ptype = tvm.ir.PointerType(tvm.ir.PrimType(dtype), storage_scope)
    x = tvm.tir.Var("xyz", ptype)
    assert x.dtype == "handle"
    assert x.type_annotation == ptype
    assert x.type_annotation.storage_scope == storage_scope
    assert isinstance(ptype.element_type, tvm.ir.PrimType)


def test_buffer_load_store():
    b = tvm.tir.decl_buffer((10,), "float32")
    x = tvm.tir.BufferLoad(b, [0])
    assert isinstance(x, tvm.tir.BufferLoad)
    assert x.dtype == "float32"
    assert x.buffer == b
    s = tvm.tir.BufferStore(b, 0.1, [0])
    assert isinstance(s, tvm.tir.BufferStore)

    s = tvm.tir.BufferRealize(b, [tvm.ir.Range(0, 1)], True, tvm.tir.Evaluate(0))
    assert isinstance(s, tvm.tir.BufferRealize)


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


def test_block_blockrealize():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    vx = tvm.tir.IterVar((16, 16), "vx", 0)
    vx_var = vx.var
    vy = tvm.tir.IterVar((16, 16), "vy", 2)
    vy_var = vy.var
    A = tvm.tir.decl_buffer((16), "float32")
    B = tvm.tir.decl_buffer((16, 16), "float32")
    alloc_buffer = tvm.tir.decl_buffer((16, 16), "float32")
    match_buffer = tvm.tir.decl_buffer((16, 16), "float32")
    init_body = tvm.tir.BufferStore(A, 0.0, [vx_var])
    body = tvm.tir.BufferStore(
        A,
        tvm.tir.BufferLoad(A, [vx_var]) + tvm.tir.BufferLoad(B, [vx_var, vy_var]),
        [vx_var],
    )
    reads = [
        tvm.tir.BufferRegion(
            B, [tvm.ir.Range.from_min_extent(vx_var, 1), tvm.ir.Range.from_min_extent(vy_var, 1)]
        )
    ]
    writes = [tvm.tir.BufferRegion(A, [tvm.ir.Range.from_min_extent(vx_var, 1)])]
    block_match_buffer = tvm.tir.MatchBufferRegion(
        match_buffer, tvm.tir.BufferRegion(B, [tvm.ir.Range(0, 16), tvm.ir.Range(0, 16)])
    )

    block = tvm.tir.Block(
        [vx, vy],
        reads,
        writes,
        "block",
        body,
        init=init_body,
        alloc_buffers=[alloc_buffer],
        match_buffers=[block_match_buffer],
        annotations={"attr_key": "attr_value"},
    )

    # Checking Block
    assert isinstance(block, tvm.tir.Block)
    # Checking iter_vars
    assert block.iter_vars[0] == vx
    assert block.iter_vars[1] == vy
    # Checking reads/writes region
    assert isinstance(block.reads[0], tvm.tir.BufferRegion)
    assert block.reads[0].buffer == B
    assert block.reads[0].region[0].min == vx_var
    assert block.reads[0].region[1].min == vy_var
    assert isinstance(block.writes[0], tvm.tir.BufferRegion)
    assert block.writes[0].buffer == A
    assert block.writes[0].region[0].min == vx_var
    assert block.writes[0].region[0].extent == 1
    # Checking name_hint
    assert block.name_hint == "block"
    # Checking body
    assert block.body == body
    # Checking init
    assert block.init == init_body
    # Checking alloc_buffers
    assert block.alloc_buffers[0] == alloc_buffer
    # Checking match_buffers
    assert block.match_buffers[0].buffer == match_buffer
    assert isinstance(block.match_buffers[0].source, tvm.tir.BufferRegion)
    assert block.match_buffers[0].source.buffer == B
    assert block.match_buffers[0].source.region[0].min == 0
    assert block.match_buffers[0].source.region[0].extent == 16

    # Checking BlockRealize
    block_realize = tvm.tir.BlockRealize([x, y], tvm.tir.const(True, "bool"), block)
    assert isinstance(block_realize, tvm.tir.BlockRealize)
    assert block_realize.iter_values[0] == x
    assert block_realize.iter_values[1] == y
    assert block_realize.predicate == tvm.tir.const(True, "bool")
    assert block_realize.block == block

    # make sure we can print using ReprPrinter
    str(block)
    str(block_realize)
    # make sure we can print using TIRTextPrinter
    func = tvm.tir.PrimFunc([], block_realize)
    output = func.astext()
    assert output.find("meta[tir.BlockRealise]") == -1
    assert output.find("bind") != -1
    assert output.find("reads") != -1
    assert output.find("writes") != -1
    assert output.find("alloc_buffer") != -1
    assert output.find("match_buffer") != -1
    assert output.find("attr") != -1
    assert output.find("with init()") != -1


def test_tir_allocate():
    dtype = "int8"
    storage_scope = "global"
    ptype = tvm.ir.PointerType(tvm.ir.PrimType(dtype), storage_scope)
    a = te.var("buffer", ptype)
    allocate = tvm.tir.Allocate(
        buffer_var=a,
        dtype=dtype,
        extents=[2, 2],
        condition=tvm.get_global_func("tir.const_true")(dtype, None),
        body=tvm.tir.Evaluate(2 + 1),
        annotations={
            "attr1": "foo",
            "attr2": "bar",
        },
    )
    assert allocate.buffer_var == a
    assert allocate.dtype == "int8"
    assert list(allocate.extents) == [2, 2]
    assert allocate.annotations["attr1"] == "foo"
    assert allocate.annotations["attr2"] == "bar"


if __name__ == "__main__":
    pytest.main([__file__])
