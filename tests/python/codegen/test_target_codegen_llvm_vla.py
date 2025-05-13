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
Codegen tests for VLA extensions
"""

import re
import pytest

import tvm
from tvm import te
from tvm.script import tir as T
from tvm.target.codegen import llvm_version_major


@pytest.mark.skipif(
    llvm_version_major() < 11, reason="Vscale is not supported in earlier versions of LLVM"
)
@tvm.testing.parametrize_targets(
    "llvm -mtriple=aarch64-linux-gnu -mattr=+sve",
    "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v",
)
def test_codegen_vscale(target):
    vscale = tvm.tir.vscale()

    @T.prim_func
    def main(A: T.Buffer((5,), "int32")):
        for i in range(5):
            A[i] = 2 * vscale

    with tvm.target.Target(target):
        build_mod = tvm.tir.build(main)

    llvm = build_mod.get_source()
    assert re.findall(r"llvm.vscale.i32", llvm), "No vscale in generated LLVM."


@pytest.mark.skipif(
    llvm_version_major() < 11, reason="Vscale is not supported in earlier versions of LLVM"
)
@tvm.testing.parametrize_targets(
    "llvm -mtriple=aarch64-linux-gnu -mattr=+sve",
    "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v",
)
def test_scalable_buffer_load_store(target):
    @T.prim_func
    def my_func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (128,), "float32")
        B = T.match_buffer(b, (128,), "float32")
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        B[T.ramp(0, 1, 4 * T.vscale())] = A[T.ramp(0, 1, 4 * T.vscale())]

    with tvm.target.Target(target):
        mod = tvm.tir.build(my_func)

    llvm = mod.get_source("ll")
    assert re.findall(r"load <vscale x 4 x float>", llvm), "No scalable load in generated LLVM."
    assert re.findall(r" store <vscale x 4 x float>", llvm), "No scalable store in generated LLVM."


@pytest.mark.skipif(
    llvm_version_major() < 11, reason="Vscale is not supported in earlier versions of LLVM"
)
@tvm.testing.parametrize_targets(
    "llvm -mtriple=aarch64-linux-gnu -mattr=+sve",
    "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v",
)
def test_scalable_broadcast(target):
    @T.prim_func
    def my_func(a: T.handle):
        A = T.match_buffer(a, (128,), "float32")
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        A[T.ramp(0, 1, 4 * T.vscale())] = T.broadcast(1, 4 * T.vscale())

    with tvm.target.Target(target):
        mod = tvm.tir.build(my_func)

    llvm = mod.get_source("ll")
    assert re.findall(
        r"shufflevector \(<vscale x 4 x float> insertelement \(<vscale x 4 x float>", llvm
    ), "No scalable broadcast in generated LLVM."
    assert re.findall(r" store <vscale x 4 x float>", llvm), "No scalable store in generated LLVM."


@pytest.mark.skip(
    reason="Vscale and get.active.lane.mask are not supported in earlier versions of LLVM",
)
@tvm.testing.parametrize_targets(
    "llvm -mtriple=aarch64-linux-gnu -mattr=+sve",
    "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v",
)
def test_get_active_lane_mask(target):
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
@tvm.testing.parametrize_targets(
    "llvm -mtriple=aarch64-linux-gnu -mattr=+sve",
    "llvm -device=riscv_cpu -mtriple=riscv64-linux-gnu -mcpu=generic-rv64 -mattr=+64bit,+a,+c,+d,+f,+m,+v",
)
def test_predicated_scalable_buffer(target):
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
