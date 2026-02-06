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
import tvm.testing
import numpy as np
from tvm.script import tir as T, ir as I


@tvm.testing.uses_gpu
def test_add_pipeline():
    """Test extern-style add pipeline with vectorized operations."""
    nn = 64
    max_threads = 4

    # CPU version: serial loop with vectorized operations
    @I.ir_module
    class ModuleCPU:
        @T.prim_func
        def main(A: T.Buffer((64,), "float32"), C: T.Buffer((64,), "float32")):
            for i in T.serial((64 + 1) // 2):
                C[T.Ramp(i * 2, 1, 2)] = A[T.Ramp(i * 2, 1, 2)] + T.Broadcast(T.float32(1), 2)

    # GPU version: thread bindings with vectorized operations
    @I.ir_module
    class ModuleGPU:
        @T.prim_func
        def main(A: T.Buffer((64,), "float32"), C: T.Buffer((64,), "float32")):
            bx = T.launch_thread("blockIdx.x", (64 + 4 - 1) // 4)
            tx = T.launch_thread("threadIdx.x", 4)
            idx = bx * 4 + tx
            if T.likely(idx < 64):
                C[T.Ramp(idx * 2, 1, 2)] = A[T.Ramp(idx * 2, 1, 2)] + T.Broadcast(T.float32(1), 2)

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return
        mod = ModuleGPU if target in ["opencl", "cuda"] else ModuleCPU
        # build and invoke the kernel.
        f = tvm.compile(mod, target=target)
        dev = tvm.device(target, 0)
        # launch the kernel.
        n = nn
        a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
        c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        f(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)

    check_target("llvm")
    check_target("opencl")
    check_target("cuda")


def test_pack_buffer_simple():
    """Test call_packed with buffer arguments."""
    nn = 1024

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((1024,), "float32"), C: T.Buffer((1024,), "float32")):
            T.evaluate(T.call_packed("my_extern_array_func1", A, C))

    @tvm.register_global_func
    def my_extern_array_func1(aa, bb):
        aa.copyto(bb)

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return
        # build and invoke the kernel.
        f = tvm.compile(Module, target=target)
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
        c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)

        f(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy())

    check_target("llvm")


if __name__ == "__main__":
    tvm.testing.main()
