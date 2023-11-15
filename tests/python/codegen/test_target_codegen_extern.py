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
import numpy as np
import tvm.testing


@tvm.testing.uses_gpu
def test_add_pipeline():
    nn = 64
    max_threads = 4
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")

    def extern_generator(ins, outs):
        """Manually write the IR for the extern function, add pipeline"""
        ib = tvm.tir.ir_builder.create()
        with ib.for_range(0, (n + 1) // 2) as i:
            ib.emit(
                outs[0].vstore(
                    i * 2, ins[0].vload(i * 2, "float32x2") + tvm.tir.const(1, "float32x2")
                )
            )
        return ib.get()

    def extern_generator_gpu(ins, outs):
        """Manually write the IR for the extern function, add pipeline"""
        ib = tvm.tir.ir_builder.create()
        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(bx, "thread_extent", (nn + max_threads - 1) // max_threads)
        ib.scope_attr(tx, "thread_extent", max_threads)
        idx = bx.var * max_threads + tx.var
        with ib.if_scope(ib.likely(idx < n)):
            ib.emit(
                outs[0].vstore(
                    idx * 2, ins[0].vload(idx * 2, "float32x2") + tvm.tir.const(1, "float32x2")
                )
            )
        return ib.get()

    C_cpu = te.extern(A.shape, [A], extern_generator, name="C")
    C_gpu = te.extern(A.shape, [A], extern_generator_gpu, name="C")
    s_cpu = te.create_schedule(C_cpu.op)
    s_gpu = te.create_schedule(C_gpu.op)
    print(tvm.lower(s_cpu, [A, C_cpu], simple_mode=True))
    print(tvm.lower(s_gpu, [A, C_gpu], simple_mode=True))

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return
        s = s_gpu if target in ["opencl", "cuda"] else s_cpu
        C = C_gpu if target in ["opencl", "cuda"] else C_cpu
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)
        dev = tvm.device(target, 0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        f(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)

    check_target("llvm")
    check_target("opencl")
    check_target("cuda")


def test_pack_buffer_simple():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")

    def extern_generator(ins, outs):
        """Manually write the IR for the extern function, add pipeline."""
        return tvm.tir.call_packed("my_extern_array_func1", ins[0], outs[0])

    C = te.extern(A.shape, [A], extern_generator, name="C")
    s = te.create_schedule(C.op)

    @tvm.register_func
    def my_extern_array_func1(aa, bb):
        aa.copyto(bb)

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

        f(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy())

    check_target("stackvm")
    check_target("llvm")


def test_pack_buffer_intermediate():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.compute((n,), lambda i: A[i] + 1, name="B")

    def extern_generator(ins, outs):
        """Manually write the IR for the extern function, add pipeline."""
        return tvm.tir.call_packed("my_extern_array_func2", ins[0], outs[0])

    C = te.extern(B.shape, [B], extern_generator, name="C")
    s = te.create_schedule(C.op)

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return
        # build and invoke the kernel.
        f = tvm.build(s, [A, C], target)
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

        @tvm.register_func
        def my_extern_array_func2(aa, bb):
            assert aa.shape == a.shape
            tvm.testing.assert_allclose(aa.numpy(), a.numpy() + 1)
            aa.copyto(bb)

        f(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)

    check_target("llvm")


if __name__ == "__main__":
    test_pack_buffer_simple()
    test_pack_buffer_intermediate()
    test_add_pipeline()
