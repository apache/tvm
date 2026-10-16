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
# ruff: noqa: F401, F841
"""Test cross compilation"""

import os
import struct

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import rpc
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import cc, utils
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


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_add_pipeline():
    nn = 1024

    def verify_elf(path, e_machine):
        with open(path, "rb") as fi:
            arr = fi.read(20)
            assert struct.unpack("ccc", arr[1:4]) == (b"E", b"L", b"F")
            endian = struct.unpack("b", arr[0x5:0x6])[0]
            endian = "<" if endian == 1 else ">"
            assert struct.unpack(endian + "h", arr[0x12:0x14])[0] == e_machine

    def build_arm():
        target = {"kind": "llvm", "mtriple": "armv7-none-linux-gnueabihf"}
        if not tvm_tirx.runtime.enabled("llvm"):
            print(f"Skip because {target} is not enabled..")
            return
        temp = utils.tempdir()
        f = tvm_tirx.tirx.compile(AddModule, target=target).mod
        path = temp.relpath("myadd.o")
        f.write_to_file(path)
        verify_elf(path, 0x28)
        asm_path = temp.relpath("myadd.asm")
        f.write_to_file(asm_path)
        # Do a RPC verification, launch kernel on Arm Board if available.
        host = os.environ.get("TVM_RPC_ARM_HOST", None)
        remote = None
        if host:
            port = int(os.environ["TVM_RPC_ARM_PORT"])
            try:
                remote = rpc.connect(host, port)
            except RuntimeError as e:
                pass

        if remote:
            remote.upload(path)
            farm = remote.load_module("myadd.o")
            dev = remote.cpu(0)
            n = nn
            a = dev.tensor(np.random.uniform(size=n).astype("float32"))
            b = dev.tensor(np.random.uniform(size=n).astype("float32"))
            c = dev.tensor(np.zeros(n, dtype="float32"))
            farm(a, b, c)
            tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
            print("Verification finish on remote..")

    build_arm()


@pytest.mark.skipif(
    tvm_tirx.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_bigendian_rpc():
    """Test big endian rpc when there is a PowerPC RPC server available"""
    host = os.environ.get("TVM_POWERPC_TEST_HOST", None)
    port = os.environ.get("TVM_POWERPC_TEST_PORT", 9090)
    if host is None:
        pytest.skip("PowerPC RPC host is not configured")

    def verify_rpc(remote, target, shape, dtype):
        @T.function
        def main(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)):
            for i in range(shape[0]):
                B[i] = A[i] + tvm_tirx.tirx.const(1, dtype)

        f = tvm_tirx.tirx.compile(main, target=target).mod

        dev = remote.cpu(0)
        a = dev.tensor(np.random.randint(0, 256, size=shape).astype(dtype))
        b = dev.tensor(np.zeros(shape).astype(dtype))
        temp = utils.tempdir()
        path_dso = temp.relpath("dev_lib.o")
        f.write_to_file(path_dso)
        remote.upload(path_dso)
        f = remote.load_module("dev_lib.o")
        f(a, b)
        tvm_tirx.testing.assert_allclose(a.numpy() + 1, b.numpy())

    print("Test RPC connection to PowerPC...")
    remote = rpc.connect(host, port)
    target = {"kind": "llvm", "mtriple": "powerpc-linux-gnu"}
    for dtype in ["float32", "float64", "int32", "int8"]:
        verify_rpc(remote, target, (10,), dtype)


if __name__ == "__main__":
    test_llvm_add_pipeline()
