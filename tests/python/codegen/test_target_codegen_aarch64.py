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
from tvm.script import tir as T, ir as I
from tvm.target.codegen import llvm_version_major


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_mul(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] * B[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"mul\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_add(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] + B[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"add\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_sub(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] - B[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"sub\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_muladd(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle, var_D: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            D = T.match_buffer(var_D, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("D"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i], C[v_i])
                    T.writes(D[v_i])
                    D[v_i] = A[v_i] * B[v_i] + C[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"mad|mla\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_max(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = T.max(A[v_i], B[v_i])

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    compare = re.findall(
        r"cmgt\tp[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )
    select = re.findall("sel\tz[0-9].[shdb], p[0-9], z[0-9].[shdb], z[0-9].[shdb]", assembly)
    max_instr = re.findall(
        r"max\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert (len(compare) > 1 and len(select) == len(compare)) or len(max_instr) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_min(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = T.min(A[v_i], B[v_i])

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    compare = re.findall(
        r"cmgt\tp[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )
    select = re.findall("sel\tz[0-9].[shdb], p[0-9], z[0-9].[shdb], z[0-9].[shdb]", assembly)
    min_instr = re.findall(
        r"min\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert (len(compare) > 1 and len(select) == len(compare)) or len(min_instr) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_div(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = tvm.tir.div(A[v_i], B[v_i])

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"div\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) >= 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_mod(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = T.floormod(A[v_i], B[v_i])

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"mls\tz[0-9].[shdb],( p[0-9]/[m],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 0


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_eq(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), "bool")
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] == B[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"cm(p)?eq\tp[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype",
    ["float", "float16", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
)
def test_neq(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), "bool")
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] != B[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"cm(p)?(gt|ne)\tp[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_or(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] | B[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"orr\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_and(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] & B[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"and\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


@pytest.mark.skipif(
    llvm_version_major() < 15, reason="Test requires an LLVM version of at least 15 to target SVE"
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
)
def test_not(dtype):
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i])
                    T.writes(C[v_i])
                    C[v_i] = ~A[v_i]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)
    matches = re.findall(
        r"eor\tz[0-9].[shdb],( p[0-9]/[zm],)? z[0-9].[shdb], z[0-9].[shdb]", assembly
    )

    assert len(loads) > 1
    assert len(matches) > 1


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
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,), dtype=dtype)
            B = T.match_buffer(var_B, (m,), "int32")
            C = T.match_buffer(var_C, (m,), dtype=dtype)
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(B[v_i], A[B[v_i]])
                    T.writes(C[v_i])
                    C[v_i] = A[B[v_i]]

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    assembly = f.inspect_source("asm")
    loads = re.findall("ld1[whdb]	{ z", assembly)

    assert len(loads) > 0


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
    target = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": [mattr]}

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_C: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(var_A, (m,))
            C = T.match_buffer(var_C, (m,))
            for i in range(m):
                with T.sblock("C"):
                    v_i = T.axis.spatial(m, i)
                    T.reads(A[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] + T.float32(1)

    with tvm.target.Target(target):
        f = tvm.tir.build(Module)

    # Check if the vscale_range() attribute exists
    ll = f.inspect_source("ll")
    attr = re.findall(rf".*vscale_range\(\d+,\d+\)*.", ll)

    if expect_attr:
        assert (
            len(attr) > 0
        ), f"Function attribute vscale_range() was not found in generated LLVM IR"
    else:
        assert (
            len(attr) == 0
        ), f"Unexpected function attribute vscale_range() was found in generated LLVM IR"


if __name__ == "__main__":
    tvm.testing.main()
