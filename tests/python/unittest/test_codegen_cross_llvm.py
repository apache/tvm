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
import tvm
import os
import struct
from tvm import rpc
from tvm.contrib import util, cc
import numpy as np

def test_llvm_add_pipeline():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    s[C].parallel(xo)
    s[C].vectorize(xi)

    def verify_elf(path, e_machine):
        with open(path, "rb") as fi:
            arr = fi.read(20)
            assert struct.unpack('ccc', arr[1:4]) == (b'E',b'L',b'F')
            endian = struct.unpack('b', arr[0x5:0x6])[0]
            endian = '<' if endian == 1 else '>'
            assert struct.unpack(endian + 'h', arr[0x12:0x14])[0] == e_machine

    def build_i386():
        if not tvm.runtime.enabled("llvm"):
            print("Skip because llvm is not enabled..")
            return
        temp = util.tempdir()
        target = "llvm -target=i386-pc-linux-gnu"
        f = tvm.build(s, [A, B, C], target)
        path = temp.relpath("myadd.o")
        f.save(path)
        verify_elf(path, 0x03)

    def build_arm():
        target = "llvm -target=armv7-none-linux-gnueabihf"
        if not tvm.runtime.enabled(target):
            print("Skip because %s is not enabled.." % target)
            return
        temp = util.tempdir()
        f = tvm.build(s, [A, B, C], target)
        path = temp.relpath("myadd.o")
        f.save(path)
        verify_elf(path, 0x28)
        asm_path = temp.relpath("myadd.asm")
        f.save(asm_path)
        # Do a RPC verification, launch kernel on Arm Board if available.
        host = os.environ.get('TVM_RPC_ARM_HOST', None)
        remote = None
        if host:
            port = int(os.environ['TVM_RPC_ARM_PORT'])
            try:
                remote = rpc.connect(host, port)
            except tvm.error.TVMError as e:
                pass

        if remote:
            remote.upload(path)
            farm = remote.load_module("myadd.o")
            ctx = remote.cpu(0)
            n = nn
            a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
            b = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
            c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
            farm(a, b, c)
            tvm.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy())
            print("Verification finish on remote..")

    build_i386()
    build_arm()

if __name__ == "__main__":
    test_llvm_add_pipeline()
