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
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_large_uint_imm():
    value = (1 << 63) + 123
    value_const = tvm.tirx.const(value, "uint64")

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((12,), "uint64")):
            T.func_attr({"tirx.noalias": True})
            for i0_0 in T.thread_binding(6, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(2, thread="threadIdx.x"):
                    A[i0_0 * 2 + i0_1] = value_const + T.uint64(3)

    def check_target(target):
        target_kind = target["kind"] if isinstance(target, dict) else target
        if not tvm.testing.device_enabled(target_kind):
            return
        f = tvm.compile(Module, target=target)

        def run_and_check():
            dev = tvm.device(target_kind, 0)
            a = tvm.runtime.empty((12,), dtype="uint64", device=dev)
            f(a)
            assert a.numpy()[0] == value + 3

        tvm.testing.run_with_gpu_lock(run_and_check)

    check_target("cuda")
    check_target({"kind": "vulkan", "from_device": 0})


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_add_pipeline():
    n = T.dynamic("n", "int32")

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((n,)), B: T.Buffer((), "float32"), D: T.Buffer((n,))):
            T.func_attr({"tirx.noalias": True})

            C = T.alloc_buffer((n,))
            for i0_0 in T.thread_binding((n + 255) // 256, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(256, thread="threadIdx.x"):
                    if i0_0 * 256 + i0_1 < n:
                        C[i0_0 * 256 + i0_1] = A[i0_0 * 256 + i0_1] + B[()]
            for i0_0 in T.thread_binding((n + 255) // 256, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(256, thread="threadIdx.x"):
                    if i0_0 * 256 + i0_1 < n:
                        D[i0_0 * 256 + i0_1] = C[i0_0 * 256 + i0_1] + T.float32(1.0)

    def check_target(device, host):
        if not tvm.testing.device_enabled(device) or not tvm.testing.device_enabled(host):
            return
        target = tvm.target.Target(device, host)
        mhost = tvm.tirx.build(Module, target=target)
        f = mhost.main
        n = 1027

        def run_and_check():
            dev = tvm.device(device, 0)
            a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
            b = tvm.runtime.tensor(np.random.uniform(size=()).astype("float32"), dev)
            d = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
            f(a, b, d)
            tvm.testing.assert_allclose(d.numpy(), a.numpy() + b.numpy() + 1)

        tvm.testing.run_with_gpu_lock(run_and_check)

    check_target("cuda", host="llvm")
    # check_target("nvptx", host="llvm")  # nvptx kernel entry-point lookup not wired here
    check_target("vulkan", host="llvm")
    check_target("rocm", host="llvm")


if __name__ == "__main__":
    test_large_uint_imm()
    test_add_pipeline()
