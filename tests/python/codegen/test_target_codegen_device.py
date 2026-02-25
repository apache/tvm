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
import numpy as np

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T


@tvm.testing.requires_gpu
def test_large_uint_imm():
    value = (1 << 63) + 123
    value_const = tvm.tir.const(value, "uint64")

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((12,), "uint64")):
            T.func_attr({"tir.noalias": True})
            for i0_0 in T.thread_binding(6, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(2, thread="threadIdx.x"):
                    with T.sblock("A"):
                        v_i0 = T.axis.spatial(12, i0_0 * 2 + i0_1)
                        T.reads()
                        T.writes(A[v_i0])
                        A[v_i0] = value_const + T.uint64(3)

    def check_target(target):
        target_kind = target["kind"] if isinstance(target, dict) else target
        if not tvm.testing.device_enabled(target_kind):
            return
        dev = tvm.device(target_kind, 0)
        f = tvm.compile(Module, target=target)
        # launch the kernel.
        a = tvm.runtime.empty((12,), dtype="uint64", device=dev)
        f(a)
        assert a.numpy()[0] == value + 3

    check_target("cuda")
    check_target({"kind": "vulkan", "from_device": 0})


@tvm.testing.requires_gpu
def test_add_pipeline():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, B: T.Buffer((), "float32"), var_D: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int32(is_size_var=True)
            A = T.match_buffer(var_A, (n,))
            D = T.match_buffer(var_D, (n,))
            C = T.alloc_buffer((n,))
            for i0_0 in T.thread_binding((n + 255) // 256, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(256, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i0 = T.axis.spatial(n, i0_0 * 256 + i0_1)
                        T.where(i0_0 * 256 + i0_1 < n)
                        T.reads(A[v_i0], B[()])
                        T.writes(C[v_i0])
                        C[v_i0] = A[v_i0] + B[()]
            for i0_0 in T.thread_binding((n + 255) // 256, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(256, thread="threadIdx.x"):
                    with T.sblock("D"):
                        v_i0 = T.axis.spatial(n, i0_0 * 256 + i0_1)
                        T.where(i0_0 * 256 + i0_1 < n)
                        T.reads(C[v_i0])
                        T.writes(D[v_i0])
                        D[v_i0] = C[v_i0] + T.float32(1.0)

    def check_target(device, host):
        if not tvm.testing.device_enabled(device) or not tvm.testing.device_enabled(host):
            return
        dev = tvm.device(device, 0)
        target = tvm.target.Target(device, host)
        mhost = tvm.tir.build(Module, target=target)
        f = mhost.main
        # launch the kernel.
        n = 1027
        a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
        b = tvm.runtime.tensor(np.random.uniform(size=()).astype("float32"), dev)
        d = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        f(a, b, d)
        tvm.testing.assert_allclose(d.numpy(), a.numpy() + b.numpy() + 1)

    check_target("cuda", host="llvm")
    check_target("nvptx", host="llvm")
    check_target("vulkan", host="llvm")
    check_target("rocm", host="llvm")


if __name__ == "__main__":
    test_large_uint_imm()
    test_add_pipeline()
