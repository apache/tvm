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
"""codegen related to bool types"""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.mark.gpu
@pytest.mark.parametrize("target", ["llvm", "cuda", "rocm", "vulkan", "metal", "opencl"])
def test_cmp_load_store(target):
    if not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @I.ir_module
    class GPUModule:
        @T.function
        def main(
            A: T.Tensor((32,), "float32"),
            B: T.Tensor((32,), "float32"),
            D: T.Tensor((32,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 8) as bx:
                with T.launch_thread("threadIdx.x", 4) as tx:
                    C = T.alloc_tensor(ty_args=[T.Tensor((1,), "bool", scope="local")])
                    C[0] = B[bx * 4 + tx] < A[bx * 4 + tx]
                    D[bx * 4 + tx] = T.Cast(C[0] and T.float32(1.0) < A[bx * 4 + tx], "float32")

    @I.ir_module
    class CPUModule:
        @T.function
        def main(
            A: T.Tensor((32,), "float32"),
            B: T.Tensor((32,), "float32"),
            D: T.Tensor((32,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            C = T.alloc_tensor(ty_args=[T.Tensor((32,), "bool")])
            for i0 in range(32):
                C[i0] = B[i0] < A[i0]
            for i0 in range(32):
                D[i0] = T.Cast(C[i0] and T.float32(1.0) < A[i0], "float32")

    arr_size = 32
    is_gpu = tvm_tirx.target.Target(target).kind.name != "llvm"
    mod = GPUModule if is_gpu else CPUModule

    f = tvm_tirx.tirx.compile(mod, target=target)

    a_np = np.random.uniform(size=arr_size).astype("float32")
    b_np = np.random.uniform(size=arr_size).astype("float32")

    def run_and_check():
        dev = tvm_tirx.device_from_target(target)
        a = tvm_tirx.runtime.tensor(a_np, dev)
        b = tvm_tirx.runtime.tensor(b_np, dev)
        d = tvm_tirx.runtime.tensor(np.zeros(arr_size, dtype="float32"), dev)
        f(a, b, d)
        np.testing.assert_equal(
            d.numpy(),
            np.logical_and(a_np > b_np, a_np > 1).astype("float32"),
        )

    if is_gpu:
        tvm_tirx.testing.run_with_gpu_lock(run_and_check)
    else:
        run_and_check()


def test_bitwise_not_c(tmp_path):
    @T.function
    def complement(values: T.Tensor((5,), "int32"), output: T.Tensor((5,), "bool")):
        for i in range(5):
            output[i] = T.bitwise_not(values[i] != 0)

    built = tvm_tirx.tirx.compile(complement, target="c")
    library = str(tmp_path / "complement.so")
    built.export_library(library)
    built = tvm_tirx.runtime.load_module(library)

    values = np.array([-3, -1, 0, 1, 2], dtype="int32")
    output = tvm_tirx.runtime.tensor(np.zeros(5, dtype="bool"))
    built(tvm_tirx.runtime.tensor(values), output)
    np.testing.assert_array_equal(output.numpy(), np.logical_not(values))


if __name__ == "__main__":
    tvm_tirx.testing.main()
