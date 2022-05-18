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
def ptx_cp_async(
    A: T.Buffer[(32, 128), "float16"], B: T.Buffer[(16, 128), "float16"]
) -> None:
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float16", scope="shared")

        for i in range(16):
            T.evaluate(
                T.ptx_cp_async(A_shared.data, tx * 128 + 8 * i, A.data, tx * 128 + 8 * i, 16, dtype="float16")
            )

        T.ptx_wait_group(0)

        for i in range(128):
            B[tx, i] = A_shared[tx, i]


@tvm.testing.requires_cuda
def test_ptx_cp_async():
    f = ptx_cp_async
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, minor = tvm.contrib.nvcc.parse_compute_version(arch)
    if major * 10 + minor < 80:
        # Require at least SM80
        return

if __name__ == "__main__":
    test_ptx_cp_async()
