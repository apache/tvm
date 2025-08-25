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
import tvm.testing
from tvm.script import tir as T
from tvm.target.codegen import target_has_features


@tvm.testing.requires_llvm_minimum_version(14)
@tvm.testing.parametrize_targets(
    "llvm -device=riscv_cpu -mtriple=riscv32-linux-gnu -mcpu=generic-rv32 -mattr=+i,+m",
    "llvm -device=riscv_cpu -mtriple=riscv32-linux-gnu -mcpu=generic-rv32 -mattr=+i,+m,+v",
    "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m",
    "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v",
)
def test_rvv(target):
    def check_rvv_presence(N, extent):
        @T.prim_func
        def load_vec(A: T.Buffer((N,), "int8")):
            for j in T.vectorized(0, extent):
                A[j] = 1

        f = tvm.tir.build(load_vec, target)
        # Check RVV `vsetvli` prensence
        assembly = f.inspect_source("asm")
        if target_has_features("v"):
            assert "vsetvli" in assembly
        else:
            assert "vsetvli" not in assembly

    with tvm.target.Target(target):
        check_rvv_presence(16, 32)


@tvm.testing.requires_llvm_minimum_version(14)
@tvm.testing.parametrize_targets(
    "llvm -device=riscv_cpu -mtriple=riscv32-linux-gnu -mcpu=generic-rv32 -mattr=+i,+m,+v",
    "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v",
)
def test_rvv_vscale_llvm_dbginfo(target):
    # fmt: off
    @T.prim_func
    def rvv_with_vscale(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
        A = T.match_buffer(A_handle, (8,), dtype="float32", align=4, offset_factor=1)
        B = T.match_buffer(B_handle, (4, 8), dtype="float32", align=4, offset_factor=1, strides=[8, 1])
        C = T.match_buffer(C_handle, (4,), dtype="float32", align=4, offset_factor=1)
        with T.block("root"):
            T.reads(A[0:8], B[0:4, 0:8])
            zero = T.call_llvm_intrin("float32xvscalex2", "llvm.riscv.vfmv.v.f", T.Broadcast(T.float32(0.0), T.vscale() * 2), C[0], T.uint64(1))
            vec_A = T.call_llvm_intrin("float32xvscalex4", "llvm.riscv.vle", T.Broadcast(T.float32(0.0), T.vscale() * 4), T.tvm_access_ptr(T.type_annotation("float32"), A.data, 0, 8, 1), T.int64(8))
            vec_B = T.call_llvm_intrin("float32xvscalex4", "llvm.riscv.vle", T.Broadcast(T.float32(0.0), T.vscale() * 4), T.tvm_access_ptr(T.type_annotation("float32"), B.data, 0 * 8, 8, 1), T.int64(8))
            prod = T.call_llvm_intrin("float32xvscalex4", "llvm.riscv.vfmul", T.Broadcast(T.float32(0.0), T.vscale() * 4), vec_A, vec_B, T.uint64(7), T.uint64(8))
            redsum = T.call_llvm_intrin("float32xvscalex2", "llvm.riscv.vfredusum", T.Broadcast(T.float32(0.0), T.vscale() * 2), prod, zero, T.uint64(7), T.uint64(8))
    # fmt: on

    # tvm.error.InternalError: Can't fetch the lanes of a scalable vector at a compile time.
    with tvm.target.Target(target):
        f = tvm.tir.build(rvv_with_vscale, target)


if __name__ == "__main__":
    tvm.testing.main()
