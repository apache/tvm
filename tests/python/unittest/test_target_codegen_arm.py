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
from tvm import te
import re
import os
import ctypes


def test_llvm_flip_pipeline_sve():
    target = "llvm -device=arm_cpu -mtriple=aarch64-gnu-linux -mattr=v8.2a,+sve"

    def check_llvm(nn, base):
        n = tvm.runtime.convert(nn)
        A = te.placeholder((n + base), name="A")
        C = te.compute((n,), lambda i: A(nn + base - i - 1), name="C")
        s = te.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], factor=4)
        s[C].parallel(xo)
        s[C].vectorize_scalable(xi)

        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)

    #         ctx = remote.context(target)
    #         # launch the kernel.
    #         n = nn
    #         a = tvm.nd.array(np.random.uniform(size=(n + base)).astype(A.dtype), ctx)
    #         c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    #         f(a, c)
    #         tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy()[::-1][:n])

    check_llvm(4, 0)
    check_llvm(128, 8)
    check_llvm(3, 0)
    check_llvm(128, 1)


def test_llvm_vadd_pipeline_sve():
    target = "llvm -device=arm_cpu -mtriple=aarch64-gnu-linux -mattr=v8.2a,+sve"

    def check_llvm(n, lanes):
        A = te.placeholder((n,), name="A", dtype="float32x%d" % lanes)
        B = te.compute((n,), lambda i: A[i], name="B")
        C = te.compute((n,), lambda i: B[i] + tvm.tir.const(1, A.dtype), name="C")
        s = te.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], nparts=2)
        _, xi = s[C].split(xi, factor=2)
        s[C].parallel(xo)
        s[C].vectorize_scalable(xi)
        s[B].compute_at(s[C], xo)
        xo, xi = s[B].split(B.op.axis[0], factor=2)
        s[B].vectorize_scalable(xi)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)

    #         ctx = remote.context(target)
    #         # launch the kernel.
    #         a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np.random.uniform(size=(n, lanes)))
    #         c = tvm.nd.empty((n,), C.dtype, ctx)
    #         f(a, c)
    #         tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)

    check_llvm(64, 2)
    check_llvm(512, 2)


def test_llvm_madd_pipeline_sve():
    target = "llvm -device=arm_cpu -mtriple=aarch64-gnu-linux -mattr=v8.2a,+sve"

    def check_llvm(nn, base, stride):
        n = tvm.runtime.convert(nn)
        A = te.placeholder((n + base, stride), name="A")
        C = te.compute((n, stride), lambda i, j: A(base + i, j) + 1, name="C")
        s = te.create_schedule(C.op)
        xo, xi = s[C].split(C.op.axis[0], factor=4)
        s[C].parallel(xo)
        s[C].vectorize_scalable(xi)
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)

    #         ctx = remote.context(target)
    #         # launch the kernel.
    #         n = nn
    #         a = tvm.nd.array(np.random.uniform(size=(n + base, stride)).astype(A.dtype), ctx)
    #         c = tvm.nd.array(np.zeros((n, stride), dtype=C.dtype), ctx)
    #         f(a, c)
    #         tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy()[base:] + 1)

    check_llvm(64, 0, 2)
    check_llvm(4, 0, 1)

    with tvm.transform.PassContext(config={"tir.noalias": False}):
        check_llvm(4, 0, 3)


def test_popcount():
    target = "llvm -mtriple=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"

    def check_correct_assembly(type, elements, counts):
        n = tvm.runtime.convert(elements)
        A = te.placeholder(n, dtype=type, name="A")
        B = te.compute(A.shape, lambda i: tvm.tir.popcount(A[i]), name="B")
        s = te.create_schedule(B.op)
        s[B].vectorize(s[B].op.axis[0])
        f = tvm.build(s, [A, B], target)

        # Verify we see the correct number of vpaddl and vcnt instructions in the assembly
        assembly = f.get_source("asm")
        matches = re.findall("vpaddl", assembly)
        assert len(matches) == counts
        matches = re.findall("vcnt", assembly)
        assert len(matches) == 1

    check_correct_assembly("uint16", 8, 1)
    check_correct_assembly("uint16", 4, 1)
    check_correct_assembly("uint32", 4, 2)
    check_correct_assembly("uint32", 2, 2)
    check_correct_assembly("uint64", 2, 3)


def test_vmlal_s16():
    target = "llvm -mtriple=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"

    def check_correct_assembly(N):
        K = te.size_var("K")
        A = te.placeholder((K, N), dtype="int8", name="A")
        B = te.placeholder((K, N), dtype="int8", name="B")
        k = te.reduce_axis((0, K))
        C = te.compute(
            (N,),
            lambda n: te.sum(A[k, n].astype("int32") * B[k, n].astype("int32"), axis=[k]),
            name="C",
        )
        s = te.create_schedule(C.op)
        s[C].vectorize(s[C].op.axis[0])
        f = tvm.build(s, [A, B, C], target)

        # Verify we see the correct number of vmlal.s16 instructions
        assembly = f.get_source("asm")
        matches = re.findall("vmlal.s16", assembly)
        assert len(matches) == N // 4

    check_correct_assembly(8)
    check_correct_assembly(16)
    check_correct_assembly(32)
    check_correct_assembly(64)

    def check_broadcast_correct_assembly(N):
        K = te.size_var("K")
        A = te.placeholder((K, N), dtype="int8", name="A")
        B = te.placeholder((K,), dtype="int8", name="B")
        k = te.reduce_axis((0, K))
        C = te.compute(
            (N,),
            lambda n: te.sum(A[k, n].astype("int32") * B[k].astype("int32"), axis=[k]),
            name="C",
        )
        s = te.create_schedule(C.op)
        s[C].vectorize(s[C].op.axis[0])
        f = tvm.build(s, [A, B, C], target)

        # Verify we see the correct number of vmlal.s16 instructions
        assembly = f.get_source("asm")
        matches = re.findall("vmlal.s16", assembly)
        assert len(matches) == N // 4

    check_broadcast_correct_assembly(8)
    check_broadcast_correct_assembly(16)
    check_broadcast_correct_assembly(32)
    check_broadcast_correct_assembly(64)


if __name__ == "__main__":
    test_popcount()
    test_vmlal_s16()
