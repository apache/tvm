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
from tvm.topi.arm_cpu.pstate_attributes import SMEAttributes
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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([D.op])

        f = tvm.build(s, [A, B, C, D], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

        # Verify we see SVE load instructions and div instructions using z registers
        assembly = f.get_source("asm")
        loads = re.findall("ld1[whdb]	{ z", assembly)
        matches = re.findall(
            r"div\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
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
def test_mod(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(type):
        m = te.var("m")
        A = te.placeholder(m, dtype=type, name="A")
        B = te.placeholder(m, dtype=type, name="B")
        C = te.compute((m), lambda i: tvm.te.floormod(A[i], B[i]), name="C")
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, C], target)

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
        s = te.create_schedule([C.op])

        f = tvm.build(s, [A, B, C], target)

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

    build_mod = tvm.build(main, target=target)
    llvm = build_mod.get_source()

    assert re.findall(r"llvm.vscale.i32", llvm), "No vscale in generated LLVM."


@pytest.mark.skipif(
    llvm_version_major() < 16, reason="SME is not supported in earlier versions of LLVM"
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_matmul_sme(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+v9a,+sme"

    def check_correct_assembly(dtype):
        A = te.placeholder((32, 32), dtype=dtype, name="A")
        B = te.placeholder((32, 32), dtype=dtype, name="B")

        with tvm.target.Target(target):
            C = tvm.topi.arm_cpu.matmul.compute_matmul_sme(
                A, B, None, "float32", False, dtype == "float16"
            )
            prim_func = te.create_prim_func([A, B, C])

            sch = tvm.tir.Schedule(prim_func)
            tvm.topi.arm_cpu.matmul.tir_schedule_matmul_sme(sch)
            prim_func = sch.mod

            f = tvm.build(prim_func, target=target)

        assembly = f.get_source("asm")
        smstart = re.findall(r"smstart\t(sm|za)", assembly)
        loads = re.findall(r"ld1[whdb]\t{\s?za", assembly)
        mopa = re.findall(
            r"fmopa\tza[0-9].[shdb],( p[0-9]/[zm],)?( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]",
            assembly,
        )
        stores = re.findall(r"st1[whdb]\t{\s?za", assembly)
        smstop = re.findall(r"smstop\t(sm|za)", assembly)
        whilelo = re.findall(r"whilelo\tp[0-9].[shdb]", assembly)

        assert len(smstart) > 0
        assert len(loads) > 0
        assert len(mopa) > 0
        assert len(stores) > 0
        assert len(smstop) > 0
        assert len(whilelo) > 0

    check_correct_assembly(dtype=dtype)


def test_matmul_sme_no_reduction_block():
    @T.prim_func
    def prim_func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (4,))
        B = T.match_buffer(b, (4,))
        for i in range(3):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    sch = tvm.tir.Schedule(prim_func)
    with pytest.raises(AssertionError, match="Expected a single gemm reduction block."):
        tvm.topi.arm_cpu.matmul.tir_schedule_matmul_sme(sch)


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

    mod = tvm.build(my_func, target=target)
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

    mod = tvm.build(my_func, target=target)
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
    s = te.create_schedule([C.op])

    with tvm.target.Target(target) as target:
        f = tvm.build(s, [A, C], target)

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


@pytest.mark.skipif(
    llvm_version_major() < 16, reason="Test requires an LLVM version of at least 16 to target SME"
)
@pytest.mark.parametrize(
    "attr_key,attr_value,expected",
    [
        (
            SMEAttributes.STREAMING_MODE,
            SMEAttributes.StreamingModeValues.ENABLED,
            "aarch64_pstate_sm_enabled",
        ),
        (
            SMEAttributes.STREAMING_MODE,
            SMEAttributes.StreamingModeValues.COMPATIBLE,
            "aarch64_pstate_sm_compatible",
        ),
        (SMEAttributes.ZA_STORAGE, SMEAttributes.ZAStorageValues.NEW, "aarch64_pstate_za_new"),
        (
            SMEAttributes.ZA_STORAGE,
            SMEAttributes.ZAStorageValues.SHARED,
            "aarch64_pstate_za_shared",
        ),
    ],
)
def test_function_attributes(attr_key, attr_value, expected):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sme"

    @T.prim_func
    def prim_func(a: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        A = T.match_buffer(a, (16,), "float32")
        C = T.match_buffer(c, (1,), "float32")

        with T.block("extern"):
            T.block_attr({attr_key: attr_value})
            for i in range(16):
                C[0] += A[i]

    func = tvm.build(prim_func, target=target)
    ll = func.get_source("ll")

    # Check that the attribute exists
    attr = re.findall(rf".*{expected}*.", ll)
    assert attr, f"Function attribute {expected} was not found in generated LLVM IR"

    # Check this attribute is used on the "compute" function
    func_attr_label = attr[0].split(" ")[1]
    found_compute_func = False
    for match in re.findall(rf".*{func_attr_label}*.", ll):
        if "_compute_" in match:
            found_compute_func = True

    assert found_compute_func, (
        f"The attribute {expected} was found to be under the label {func_attr_label}, "
        "but it was not used by the 'compute' scope function."
    )


def test_unsupported_function_attribute_type():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sme"

    @T.prim_func
    def prim_func(a: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        A = T.match_buffer(a, (16,), "float32")
        C = T.match_buffer(c, (1,), "float32")

        with T.block("extern"):
            T.block_attr({SMEAttributes.STREAMING_MODE: True})
            with T.block("root"):
                for i in range(16):
                    C[0] += A[i]

    err_msg = f"Expect {SMEAttributes.STREAMING_MODE} to have a String value but was IntImm"
    with pytest.raises(tvm.error.TVMError, match=err_msg):
        tvm.build(prim_func, target=target)


@pytest.mark.parametrize(
    "attr_key,attr_value",
    [
        (SMEAttributes.STREAMING_MODE, SMEAttributes.StreamingModeValues.ENABLED),
        (SMEAttributes.ZA_STORAGE, SMEAttributes.ZAStorageValues.NEW),
    ],
)
def test_unsupported_multiple_function_attributes(attr_key, attr_value):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sme"

    @T.prim_func
    def prim_func(a: T.handle, c: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        C = T.match_buffer(c, (1,), "float32")

        with T.block("root"):
            with T.block("extern"):
                T.block_attr({attr_key: attr_value})
                for i in range(16):
                    C[0] += A[i] * 2
            with T.block("extern2"):
                T.block_attr({attr_key: attr_value})
                for i in range(16):
                    C[0] += A[i] * 3

    err_msg = f"Multiple definitions of {attr_key} attribute found in the function default_function_compute_"
    with pytest.raises(tvm.error.TVMError, match=err_msg):
        tvm.build(prim_func, target=target)


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "conv2d_impl",
    [
        (
            tvm.topi.arm_cpu.compute_conv2d_NHWC_hybrid_SVE,
            tvm.topi.arm_cpu.schedule_conv2d_NHWC_hybrid_SVE,
            False,
        ),
        (
            tvm.topi.arm_cpu.compute_conv2d_NHWC_hybrid_SVE,
            tvm.topi.arm_cpu.schedule_conv2d_NHWC_hybrid_TIR,
            True,
        ),
    ],
)
def test_conv2d_sve(dtype, conv2d_impl):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"

    def check_correct_assembly(dtype, compute, schedule, use_tir_schedule):
        A = te.placeholder((1, 32, 32, 3), dtype=dtype, name="A")
        W = te.placeholder((3, 3, 3, 8), dtype=dtype, name="B")
        stride = padding = dilation = 1
        B = compute(A, W, stride, padding, dilation, dtype)
        if use_tir_schedule:
            func = te.create_prim_func([A, W, B])
            sch = schedule(tvm.tir.Schedule(func))
            f = tvm.build(sch.mod["main"], target)
        else:
            s = schedule([B])
            f = tvm.build(s, [A, W, B], target)
        assembly = f.get_source("asm")

        loads = re.findall(r"ld1[r]?[q]?[whdb]\t{\s?z", assembly)
        compute_ops = re.findall(
            r"fm(la|ad)\tz\d+.[shdb], (p\d+\/[zm], )?z\d+.[shdb], z\d+.[shdb]",
            assembly,
        )
        stores = re.findall(r"st1[whdb]\t{\s?z", assembly)

        assert len(loads) > 0
        assert len(compute_ops) > 0
        assert len(stores) > 0

    with tvm.target.Target(target):
        check_correct_assembly(dtype, *conv2d_impl)


@pytest.mark.skipif(
    llvm_version_major() < 16, reason="Test requires an LLVM version of at least 16 to target SME"
)
@pytest.mark.parametrize("dtype", ["float32"])
def test_conv2d_sme(dtype):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+v9a,+sme"

    def check_correct_assembly(dtype):
        A = te.placeholder((1, 32, 32, 3), dtype=dtype, name="A")
        W = te.placeholder((3, 3, 3, 8), dtype=dtype, name="B")
        stride = padding = dilation = 1

        B = tvm.topi.arm_cpu.compute_conv2d_NHWC_hybrid_SME(A, W, stride, padding, dilation, dtype)
        func = te.create_prim_func([A, W, B])
        sch = tvm.topi.arm_cpu.schedule_conv2d_NHWC_hybrid_TIR(tvm.tir.Schedule(func))
        f = tvm.build(sch.mod["main"], target)

        assembly = f.get_source("asm")
        smstart = re.findall(r"smstart\t(sm|za)", assembly)
        loads = re.findall(r"ld1[whdb]\t{\s?za", assembly)
        mopa = re.findall(
            r"fmopa\tza[0-9].[shdb],( p[0-9]/[zm],)?( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]",
            assembly,
        )
        stores = re.findall(r"st1[whdb]\t{\s?za", assembly)
        smstop = re.findall(r"smstop\t(sm|za)", assembly)
        whilelo = re.findall(r"whilelo\tp[0-9].[shdb]", assembly)

        assert len(smstart) > 0
        assert len(loads) > 0
        assert len(mopa) > 0
        assert len(stores) > 0
        assert len(smstop) > 0
        assert len(whilelo) > 0

    with tvm.target.Target(target):
        check_correct_assembly(dtype=dtype)


@pytest.mark.skipif(
    llvm_version_major() < 11,
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
        out = tvm.build(before)

    ll = out.get_source("ll")
    assert "get.active.lane.mask" in ll


@pytest.mark.skipif(
    llvm_version_major() < 11,
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
        out = tvm.build(before)

    ll = out.get_source("ll")
    assert "get.active.lane.mask" in ll
    assert "llvm.masked.load" in ll
    assert "llvm.masked.store" in ll


if __name__ == "__main__":
    tvm.testing.main()
