"""Test cross compilation"""
import tvm
import os
import struct
from tvm.contrib import util, cc, rpc
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
        if not tvm.module.enabled("llvm"):
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
        if not tvm.module.enabled(target):
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
            except tvm.TVMError as e:
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
            np.testing.assert_allclose(
                c.asnumpy(), a.asnumpy() + b.asnumpy())
            print("Verification finish on remote..")

    build_i386()
    build_arm()

if __name__ == "__main__":
    test_llvm_add_pipeline()
