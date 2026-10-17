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
"""Cross-compilation and object export without a remote runtime."""

import struct

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.codegen import llvm
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@I.ir_module
class AddModule:
    @T.function
    def main(
        A: T.Tensor((1024,), "float32"),
        B: T.Tensor((1024,), "float32"),
        C: T.Tensor((1024,), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        for i0_0 in T.parallel(256):
            for i0_1 in T.vectorized(4):
                C[i0_0 * 4 + i0_1] = A[i0_0 * 4 + i0_1] + B[i0_0 * 4 + i0_1]


def verify_elf(path, machine, byte_order):
    header = path.read_bytes()[:20]
    assert header[:4] == b"\x7fELF"
    assert header[5] == (1 if byte_order == "<" else 2)
    assert struct.unpack(byte_order + "H", header[18:20])[0] == machine


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_arm_object_and_assembly(tmp_path):
    if "arm" not in llvm.get_targets():
        pytest.skip("LLVM ARM backend is not enabled")
    target = {"kind": "llvm", "mtriple": "armv7-none-linux-gnueabihf"}
    module = tvm_tirx.tirx.compile(AddModule, target=target).mod
    object_path = tmp_path / "add.o"
    module.write_to_file(str(object_path))
    verify_elf(object_path, machine=0x28, byte_order="<")

    assembly_path = tmp_path / "add.asm"
    module.write_to_file(str(assembly_path))
    assembly = assembly_path.read_text()
    assert ".text" in assembly
    assert "main" in assembly


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int8"])
def test_powerpc_big_endian_object(tmp_path, dtype):
    if "powerpc" not in llvm.get_targets():
        pytest.skip("LLVM PowerPC backend is not enabled")

    @T.function
    def main(A: T.Tensor((10,), dtype), B: T.Tensor((10,), dtype)):
        for i in range(10):
            B[i] = A[i] + tvm_tirx.tirx.const(1, dtype)

    target = {"kind": "llvm", "mtriple": "powerpc-linux-gnu"}
    module = tvm_tirx.tirx.compile(main, target=target).mod
    object_path = tmp_path / "add_one.o"
    module.write_to_file(str(object_path))
    verify_elf(object_path, machine=0x14, byte_order=">")


if __name__ == "__main__":
    tvm_tirx.testing.main()
