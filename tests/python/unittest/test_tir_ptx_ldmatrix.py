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
def ptx_ldmatrix(
    A: T.Buffer((16, 16), "float16"), B: T.Buffer((16, 16), "float16"), num: T.int32, trans: T.uint8
) -> None:
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([16, 16], "float16", scope="shared")
        A_local = T.alloc_buffer([8], "float16", scope="local")

        for i in range(8):
            A_shared[i * 2 + tx // 16, tx % 16] = A[i * 2 + tx // 16, tx % 16]

        T.evaluate(
            T.ptx_ldmatrix(
                trans,
                num,
                ".b16",
                A_local.data,
                0,
                A_shared.data,
                16 * (tx % 16) + 8 * (tx // 16),
                dtype="float16",
            )
        )

        for k in range(2):
            for j in range(2):
                for i in range(2):
                    B[8 * j + tx // 4, 8 * k + (tx % 4) * 2 + i] = A_local[4 * k + 2 * j + i]


@tvm.testing.requires_cuda_compute_version(7, 5)
def test_ptx_ldmatrix():
    f = ptx_ldmatrix
    _, _, param_num, param_trans = f.params

    for num in [1, 2, 4]:
        for trans in [False, True]:
            mod = tvm.build(f.specialize({param_num: num, param_trans: trans}), target="cuda")
            A_np = np.random.rand(16, 16).astype("float16")
            A_mask_np = np.zeros_like(A_np)
            if num == 1:
                if trans:
                    A_mask_np[:8, :8] = A_np[:8, :8].T
                else:
                    A_mask_np[:8, :8] = A_np[:8, :8]
            elif num == 2:
                if trans:
                    A_mask_np[:8, :8] = A_np[:8, :8].T
                    A_mask_np[8:16, :8] = A_np[8:16, :8].T
                else:
                    A_mask_np[:16, :8] = A_np[:16, :8]
            else:  # num == 4
                if trans:
                    A_mask_np[:8, :8] = A_np[:8, :8].T
                    A_mask_np[8:16, :8] = A_np[8:16, :8].T
                    A_mask_np[:8, 8:16] = A_np[:8, 8:16].T
                    A_mask_np[8:16, 8:16] = A_np[8:16, 8:16].T
                else:
                    A_mask_np[:16, :16] = A_np[:16, :16]
            B_np = np.zeros((16, 16)).astype("float16")
            dev = tvm.cuda(0)
            A_nd = tvm.nd.array(A_np, device=dev)
            B_nd = tvm.nd.array(B_np, device=dev)
            mod(A_nd, B_nd)
            tvm.testing.assert_allclose(B_nd.numpy(), A_mask_np)


if __name__ == "__main__":
    test_ptx_ldmatrix()
