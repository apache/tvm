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
        assembly = f.get_source("asm")
        if target_has_features("v"):
            assert "vsetvli" in assembly
        else:
            assert "vsetvli" not in assembly

    with tvm.target.Target(target):
        check_rvv_presence(16, 32)


if __name__ == "__main__":
    tvm.testing.main()
