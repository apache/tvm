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
from tvm import te
from tvm.script import tir as T
import re
import pytest

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


if __name__ == "__main__":
    tvm.testing.main()
