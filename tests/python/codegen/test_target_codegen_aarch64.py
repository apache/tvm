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

"""
Codegen tests for AArch64
"""

import re
import pytest

import tvm
from tvm import te
from tvm.script import tir as T
from tvm.target.codegen import llvm_version_major


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_mul(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: A[i] * B[i], name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and mul instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"mul\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_add(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: A[i] + B[i], name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and add instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"add\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_sub(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: A[i] - B[i], name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and sub instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"sub\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_muladd(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.placeholder(m, dtype=type, name="C")
        D = te.compute((m), lambda i: A[i] * B[i] + C[i], name="D")
        f = tvm.tir.build(te.create_prim_func([A, B, C, D]), target=target)

        # Verify we see SVE load instructions and either mad or mla instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"mad|mla\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_max(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: tvm.te.max(A[i], B[i]))
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and cmgt + sel instructions or a max instruction, all using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        compare = re.findall(
            r"cmgt\tp[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )
        select = re.findall("sel\tz[0-9].[shdb], p[0-9], z[0-9].[shdb], z[0-9].[shdb]", assembly)
        max = re.findall(
            r"max\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert (len(compare) > 1 and len(select) == len(compare)) or len(max) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_min(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: tvm.te.min(A[i], B[i]))
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and cmgt + sel instructions or a min instruction, all using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        compare = re.findall(
            r"cmgt\tp[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )
        select = re.findall("sel\tz[0-9].[shdb], p[0-9], z[0-9].[shdb], z[0-9].[shdb]", assembly)
        min = re.findall(
            r"min\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert (len(compare) > 1 and len(select) == len(compare)) or len(min) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_div(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: tvm.te.div(A[i], B[i]))
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and div instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"div\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) >= 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_mod(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: tvm.te.floormod(A[i], B[i]), name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and mls instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"mls\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 0

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_eq(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: A[i] == B[i], name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and cmpeq or cmeq instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"cm(p)?eq\tp[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_neq(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: A[i] != B[i], name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and cmpgt, cmgt, cmpne or cmne instructions, all using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"cm(p)?(gt|ne)\tp[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_or(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: A[i] | B[i], name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and orr instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"orr\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_and(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: A[i] & B[i], name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see SVE load instructions and and instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"and\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_not(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        C = te.compute((m), lambda i: ~A[i], name="C")
        f = tvm.tir.build(te.create_prim_func([A, C]), target=target)

        # Verify we see SVE load instructions and eor instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"eor\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
        )

        assert len(loads) > 1
        assert len(matches) > 1

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.xfail(
    reason="Awaiting llvm support for gathered loads",
    strict=True,
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_memcpy(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype="int32", name="B")
        C = te.compute((m), lambda i: A[B[i]], name="C")
        f = tvm.tir.build(te.create_prim_func([A, B, C]), target=target)

        # Verify we see gather instructions in the assembly
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)

        assert len(loads) > 0

    check_correct_assembly(type=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 11, reason="Vscale is not supported in earlier versions of LLVM"
)
def test_codegen_vscale():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    vscale = tvm.tir.vscale()

    @T.prim_func
    def main(A: T.Buffer((5,), "int32")):
        for i in range(5):
            A[i] = 2 * vscale

    build_mod = tvm.tir.build(main, target=target)
    llvm = build_mod.get_source()

    assert re.findall(r"llvm.vscale.i32", llvm), "No vscale in generated LLVM."


@pytest.mark.skipif(
    llvm_version_major() < 11, reason="Vscale is not supported in earlier versions of LLVM"
)
def test_scalable_buffer_load_store():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    @T.prim_func
    def my_func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (128,), "float32")
        B = T.match_buffer(b, (128,), "float32")
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        B[T.ramp(0, 1, 4 * T.vscale())] = A[T.ramp(0, 1, 4 * T.vscale())]

    mod = tvm.tir.build(my_func, target=target)
    llvm = mod.get_source("ll")

    assert re.findall(r"load <vscale x 4 x float>", llvm), "No scalable load in generated LLVM."
    assert re.findall(r" store <vscale x 4 x float>", llvm), "No scalable store in generated LLVM."


@pytest.mark.skipif(
    llvm_version_major() < 11, reason="Vscale is not supported in earlier versions of LLVM"
)
def test_scalable_broadcast():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    @T.prim_func
    def my_func(a: T.handle):
        A = T.match_buffer(a, (128,), "float32")
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        A[T.ramp(0, 1, 4 * T.vscale())] = T.broadcast(1, 4 * T.vscale())

    mod = tvm.tir.build(my_func, target=target)
    llvm = mod.get_source("ll")

    assert re.findall(
        r"shufflevector \(<vscale x 4 x float> insertelement \(<vscale x 4 x float>", llvm
    ), "No scalable broadcast in generated LLVM."
    assert re.findall(r" store <vscale x 4 x float>", llvm), "No scalable store in generated LLVM."


@pytest.mark.skipif(
    llvm_version_major() < 13,
    reason="Function attribute vscale_range() is not supported in earlier versions of LLVM",
)
@pytest.mark.parametrize(
    "mattr,expect_attr",
    [
        ("+neon", False),
        ("+sve", True),
        ("+v9a", True),
        ("+sme", True),
    ],
)
def test_vscale_range_function_attribute(mattr, expect_attr):
    target = f"llvm -mtriple=aarch64-linux-gnu -mattr={mattr}"

    m = te.var("m")
    A = te.placeholder(m, dtype="float32", name="A")
    C = te.compute((m), lambda i: A[i] + 1, name="C")
    f = tvm.tir.build(te.create_prim_func([A, C]), target=target)

    # Check if the vscale_range() attribute exists
    ll = f.get_source("ll")
    attr = re.findall(rf".*vscale_range\(\d+,\d+\)*.", ll)

    if expect_attr:
        assert (
            len(attr) > 0
        ), f"Function attribute vscale_range() was not found in generated LLVM IR"
    else:
        assert (
            len(attr) == 0
        ), f"Unexpected function attribute vscale_range() was found in generated LLVM IR"


@pytest.mark.skip(
    reason="Vscale and get.active.lane.mask are not supported in earlier versions of LLVM",
)
def test_get_active_lane_mask():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    @T.prim_func
    def before(a: T.handle):
        A = T.match_buffer(a, (30,), "int1")
        for i in range(T.ceildiv(30, T.vscale() * 4)):
            A[i : i + T.vscale() * 4] = T.get_active_lane_mask("uint1xvscalex4", i, 30)

    with tvm.target.Target(target):
        out = tvm.tir.build(before)

    ll = out.get_source("ll")
    assert "get.active.lane.mask" in ll


@pytest.mark.skip(
    reason="Vscale and get.active.lane.mask are not supported in earlier versions of LLVM",
)
def test_predicated_scalable_buffer():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    @T.prim_func
    def before(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0 in T.serial(T.ceildiv(16, 4 * T.vscale())):
            for i_1 in T.vectorized(4 * T.vscale()):
                if i_0 * 4 * T.vscale() + i_1 < 14:
                    B[i_0 * 4 * T.vscale() + i_1] = A[i_0 * 4 * T.vscale() + i_1] + 1.0

    with tvm.target.Target(target):
        out = tvm.tir.build(before)

    ll = out.get_source("ll")
    assert "get.active.lane.mask" in ll
    assert "llvm.masked.load" in ll
    assert "llvm.masked.store" in ll


if __name__ == "__main__":
    tvm.testing.main()
