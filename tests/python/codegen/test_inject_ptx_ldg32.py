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
from tvm.script import tir as T
import numpy as np
import tvm.testing


@T.prim_func
def vector_add(A: T.Buffer((16), "float32"), B: T.Buffer((32), "float32")) -> None:
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_local = T.alloc_buffer((32), "float32", scope="local")

        with T.block():
            T.reads(A[0:16])
            T.writes(A_local[0:32])
            A_local[tx] = T.if_then_else(tx % 2 == 0, A[tx // 2], T.float32(0), dtype="float32")
            B[tx] = A_local[tx] + 1.0


@tvm.testing.requires_cuda
def test_inject_ptx_intrin():
    f = vector_add
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, _ = tvm.contrib.nvcc.parse_compute_version(arch)
    if major < 8:
        # Require at least SM80
        return
    with tvm.transform.PassContext(config={"tir.ptx_ldg32": True}):
        mod = tvm.build(f, target="cuda")
    A_np = np.random.rand(16).astype("float32")
    B_np = np.zeros((32)).astype("float32")
    dev = tvm.cuda(0)
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)

    C_np = np.zeros((32)).astype("float32")

    for i in range(32):
        if i % 2 == 0:
            C_np[i] = A_np[i // 2]
        C_np[i] += 1.0

    tvm.testing.assert_allclose(B_nd.numpy(), C_np)


if __name__ == "__main__":
    test_inject_ptx_intrin()
