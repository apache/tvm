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

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tir as T


@tvm.testing.uses_gpu
def test_cmp_load_store(target, dev):
    @I.ir_module
    class GPUModule:
        @T.prim_func
        def main(
            A: T.Buffer((32,), "float32"),
            B: T.Buffer((32,), "float32"),
            D: T.Buffer((32,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            C = T.alloc_buffer((32,), "bool")
            for i0_0 in T.thread_binding(8, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(4, thread="blockIdx.x"):
                    with T.sblock("C"):
                        v_i0 = T.axis.spatial(32, i0_0 * 4 + i0_1)
                        T.reads(B[v_i0], A[v_i0])
                        T.writes(C[v_i0])
                        C[v_i0] = B[v_i0] < A[v_i0]
            for i0_0 in T.thread_binding(8, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(4, thread="blockIdx.x"):
                    with T.sblock("D"):
                        v_i0 = T.axis.spatial(32, i0_0 * 4 + i0_1)
                        T.reads(C[v_i0], A[v_i0])
                        T.writes(D[v_i0])
                        D[v_i0] = T.Cast("float32", C[v_i0] and T.float32(1.0) < A[v_i0])

    @I.ir_module
    class CPUModule:
        @T.prim_func
        def main(
            A: T.Buffer((32,), "float32"),
            B: T.Buffer((32,), "float32"),
            D: T.Buffer((32,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            C = T.alloc_buffer((32,), "bool")
            for i0 in range(32):
                with T.sblock("C"):
                    v_i0 = T.axis.spatial(32, i0)
                    T.reads(B[v_i0], A[v_i0])
                    T.writes(C[v_i0])
                    C[v_i0] = B[v_i0] < A[v_i0]
            for i0 in range(32):
                with T.sblock("D"):
                    v_i0 = T.axis.spatial(32, i0)
                    T.reads(C[v_i0], A[v_i0])
                    T.writes(D[v_i0])
                    D[v_i0] = T.Cast("float32", C[v_i0] and T.float32(1.0) < A[v_i0])

    arr_size = 32
    is_gpu = tvm.target.Target(target).kind.name != "llvm"
    mod = GPUModule if is_gpu else CPUModule

    f = tvm.compile(mod, target=target)

    a_np = np.random.uniform(size=arr_size).astype("float32")
    b_np = np.random.uniform(size=arr_size).astype("float32")
    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    d = tvm.runtime.tensor(np.zeros(arr_size, dtype="float32"), dev)
    f(a, b, d)
    np.testing.assert_equal(
        d.numpy(),
        np.logical_and(a_np > b_np, a_np > 1).astype("float32"),
    )


if __name__ == "__main__":
    tvm.testing.main()
