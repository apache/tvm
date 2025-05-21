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
from tvm.target import _ffi_api, codegen, Target
from tvm.target.codegen import target_has_features, llvm_get_vector_width

LLVM_VERSION = codegen.llvm_version_major()

# fmt: off
min_llvm_version, tvm_target, vec_width = tvm.testing.parameters(
    # generic, no vector -> (default 128)
    (-1, "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+i,+m", 128),
    (-1, "llvm -device=riscv_cpu -mtriple=riscv32-linux-gnu -mcpu=generic-rv32 -mattr=+64bit,+a,+c,+d,+f,+m", 128),
    # generic, with vector -> (default zvl128b)
    (-1, "llvm -device=riscv_cpu -mtriple=riscv32-linux-gnu -mcpu=generic-rv32 -mattr=+i,+m,+v", 128),
    (-1, "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v", 128),
    # explicit +zvlXXXb
    (14, "llvm -device=riscv_cpu -mtriple=riscv32-linux-gnu -mcpu=generic-rv32 -mattr=+i,+m,+v,+zvl64b", 128),
    (14, "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v,+zvl256b", 256),
    (14, "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v,+zvl512b", 512),
    # vendor CPU
    (17, "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=sifive-x280", 512),
    (18, "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=sifive-p670", 128),
    (19, "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=spacemit-x60", 256),
)  # fmt: on


def test_riscv_rvv_features(min_llvm_version, tvm_target, vec_width):
    """Test RVV features support for different targets.

    Parameters
    ----------
    min_llvm_version : int
        Minimal LLVM version.
    tvm_target : str
        TVM target.
    vec_width : bool
        Expected vector width.
    """

    # skip test on llvm_version
    if LLVM_VERSION < min_llvm_version:
        return

    with Target(tvm_target):
        assert llvm_get_vector_width() == vec_width
