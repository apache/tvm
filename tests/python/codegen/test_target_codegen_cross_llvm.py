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
"""Test cross compilation"""

import os
import struct

import numpy as np

import tvm
import tvm.testing
from tvm import rpc
from tvm.contrib import cc, utils
from tvm.script import ir as I
from tvm.script import tir as T


@I.ir_module
class AddModule:
    @T.prim_func
    def main(
        A: T.Buffer((1024,), "float32"),
        B: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for i0_0 in T.parallel(256):
            for i0_1 in T.vectorized(4):
                with T.sblock("C"):
                    v_i0 = T.axis.spatial(1024, i0_0 * 4 + i0_1)
                    T.reads(A[v_i0], B[v_i0])
                    T.writes(C[v_i0])
                    C[v_i0] = A[v_i0] + B[v_i0]


@tvm.testing.requires_llvm
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
        if not tvm.runtime.enabled("llvm"):
            print("Skip because %s is not enabled.." % target)
            return
        temp = utils.tempdir()
        f = tvm.tir.build(AddModule, target=target)
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
            except tvm.error.TVMError as e:
                pass

        if remote:
            remote.upload(path)
            farm = remote.load_module("myadd.o")
            dev = remote.cpu(0)
            n = nn
            a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
            b = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
            c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
            farm(a, b, c)
            tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
            print("Verification finish on remote..")

    build_arm()


if __name__ == "__main__":
    test_llvm_add_pipeline()
