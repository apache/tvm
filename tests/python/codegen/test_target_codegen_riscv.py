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
# ruff: noqa: E501, F841

import re
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.target.codegen import target_has_features
from tvm.testing import env


@pytest.mark.skipif(not env.has_llvm_min_version(14), reason="need llvm >= 14")
@tvm.testing.parametrize_targets(
    {
        "kind": "llvm",
        "device": "riscv_cpu",
        "mtriple": "riscv32-linux-gnu",
        "mcpu": "generic-rv32",
        "mattr": ["+i", "+m"],
    },
    {
        "kind": "llvm",
        "device": "riscv_cpu",
        "mtriple": "riscv32-linux-gnu",
        "mcpu": "generic-rv32",
        "mattr": ["+i", "+m", "+v"],
    },
    {
        "kind": "llvm",
        "device": "riscv_cpu",
        "mtriple": "riscv64-linux-gnu",
        "mcpu": "generic-rv64",
        "mattr": ["+64bit", "+a", "+c", "+d", "+f", "+m"],
    },
    {
        "kind": "llvm",
        "device": "riscv_cpu",
        "mtriple": "riscv64-linux-gnu",
        "mcpu": "generic-rv64",
        "mattr": ["+64bit", "+a", "+c", "+d", "+f", "+m", "+v"],
    },
)
def test_rvv(target):
    def check_rvv_presence(N, extent):
        @T.prim_func(s_tir=True)
        def load_vec(A: T.Buffer((N,), "int8")):
            for j in T.vectorized(0, extent):
                A[j] = 1

        f = tvm.tirx.build(load_vec, target)
        # Check RVV `vsetvli` prensence
        assembly = f.inspect_source("asm")
        if target_has_features("v"):
            assert "vsetvli" in assembly
        else:
            assert "vsetvli" not in assembly

    with tvm.target.Target(target):
        check_rvv_presence(16, 32)


@pytest.mark.skipif(not env.has_llvm_min_version(14), reason="need llvm >= 14")
@tvm.testing.parametrize_targets(
    {
        "kind": "llvm",
        "device": "riscv_cpu",
        "mtriple": "riscv32-linux-gnu",
        "mcpu": "generic-rv32",
        "mattr": ["+i", "+m", "+v"],
    },
    {
        "kind": "llvm",
        "device": "riscv_cpu",
        "mtriple": "riscv64-linux-gnu",
        "mcpu": "generic-rv64",
        "mattr": ["+64bit", "+a", "+c", "+d", "+f", "+m", "+v"],
    },
)
def test_rvv_vscale_llvm_dbginfo(target):
    # fmt: off
    @T.prim_func(s_tir=True)
    def rvv_with_vscale(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
        A = T.match_buffer(A_handle, (8,), dtype="float32", align=4, offset_factor=1)
        B = T.match_buffer(B_handle, (4, 8), dtype="float32", align=4, offset_factor=1, strides=[8, 1])
        C = T.match_buffer(C_handle, (4,), dtype="float32", align=4, offset_factor=1)
        with T.sblock("root"):
            T.reads(A[0:8], B[0:4, 0:8])
            zero = T.call_llvm_intrin("float32xvscalex2", "llvm.riscv.vfmv.v.f", T.Broadcast(T.float32(0.0), T.vscale() * 2), C[0], T.uint64(1))
            vec_A = T.call_llvm_intrin("float32xvscalex4", "llvm.riscv.vle", T.Broadcast(T.float32(0.0), T.vscale() * 4), T.tvm_access_ptr(T.type_annotation("float32"), A.data, 0, 8, 1), T.int64(8))
            vec_B = T.call_llvm_intrin("float32xvscalex4", "llvm.riscv.vle", T.Broadcast(T.float32(0.0), T.vscale() * 4), T.tvm_access_ptr(T.type_annotation("float32"), B.data, 0 * 8, 8, 1), T.int64(8))
            prod = T.call_llvm_intrin("float32xvscalex4", "llvm.riscv.vfmul", T.Broadcast(T.float32(0.0), T.vscale() * 4), vec_A, vec_B, T.uint64(7), T.uint64(8))
            redsum = T.call_llvm_intrin("float32xvscalex2", "llvm.riscv.vfredusum", T.Broadcast(T.float32(0.0), T.vscale() * 2), prod, zero, T.uint64(7), T.uint64(8))
    # fmt: on

    # tvm.error.InternalError: Can't fetch the lanes of a scalable vector at a compile time.
    with tvm.target.Target(target):
        f = tvm.tirx.build(rvv_with_vscale, target)


@pytest.mark.skipif(not env.has_llvm_min_version(14), reason="need llvm >= 14")
def test_rvv_fixed_width_vectorized_loop_uses_scalable_chunks():
    @T.prim_func(s_tir=True)
    def fixed16_negative(
        A: T.Buffer((14, 23, 67, 99), "float32"),
        B: T.Buffer((14, 23, 67, 99), "float32"),
    ):
        for n, c, h, wo in T.grid(14, 23, 67, 7):
            for wi in T.vectorized(0, 16):
                if wo * 16 + wi < 99:
                    B[n, c, h, wo * 16 + wi] = T.float32(0) - A[n, c, h, wo * 16 + wi]

    @T.prim_func(s_tir=True)
    def fixed16_negative_int64(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        for wi in T.vectorized(T.int64(0), T.int64(16)):
            B[wi] = T.float32(0) - A[wi]

    target = tvm.target.Target(
        {
            "kind": "llvm",
            "device": "riscv_cpu",
            "mtriple": "riscv64-linux-gnu",
            "mcpu": "generic-rv64",
            "mattr": ["+64bit", "+a", "+c", "+d", "+f", "+m", "+v"],
        }
    )

    def check_codegen(func):
        with target:
            f = tvm.tirx.build(func, target)

        assembly = f.inspect_source("asm")
        assert "vle32.v" in assembly
        assert "vse32.v" in assembly
        assert not re.search(r"\bflw\b", assembly)
        assert not re.search(r"\bfsub\.s\b", assembly)
        assert not re.search(r"\bfsw\b", assembly)

    check_codegen(fixed16_negative)
    check_codegen(fixed16_negative_int64)


if __name__ == "__main__":
    tvm.testing.main()
