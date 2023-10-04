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
def ptx_cp_async(A: T.Buffer((32, 128), "float16"), B: T.Buffer((32, 128), "float16")) -> None:
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float16", scope="shared")
        T.reads(A[0:32, 0:128])
        T.writes(B[0:32, 0:128])

        for i in range(16):
            T.evaluate(
                T.ptx_cp_async(
                    A_shared.data, tx * 128 + 8 * i, A.data, tx * 128 + 8 * i, 16, dtype="float16"
                )
            )

        # TODO(masahi): Remove dtype requirement from TVMScript parser
        T.evaluate(T.ptx_commit_group(dtype=""))
        T.evaluate(T.ptx_wait_group(0, dtype=""))

        for i in range(128):
            B[tx, i] = A_shared[tx, i]


@tvm.testing.requires_cuda_compute_version(8)
def test_ptx_cp_async():
    f = ptx_cp_async

    mod = tvm.build(f, target="cuda")
    A_np = np.random.rand(32, 128).astype("float16")
    B_np = np.zeros((32, 128)).astype("float16")
    dev = tvm.cuda(0)
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)
    tvm.testing.assert_allclose(B_nd.numpy(), A_np)


@T.prim_func
def ptx_cp_async_barrier(
    A: T.Buffer((32, 128), "float16"), B: T.Buffer((32, 128), "float16")
) -> None:
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float16", scope="shared")

        T.reads(A[0:32, 0:128])
        T.writes(B[0:32, 0:128])

        T.evaluate(T.create_barriers(1, dtype=""))
        T.evaluate(T.ptx_init_barrier_thread_count(0, 32, dtype=""))

        for i in range(16):
            T.evaluate(
                T.ptx_cp_async(
                    A_shared.data, tx * 128 + 8 * i, A.data, tx * 128 + 8 * i, 16, dtype="float16"
                )
            )

        T.evaluate(T.ptx_cp_async_barrier(0, dtype=""))
        T.evaluate(T.ptx_arrive_barrier(0, dtype=""))
        T.evaluate(T.ptx_wait_barrier(0, dtype=""))

        for i in range(128):
            B[tx, i] = A_shared[tx, i]


@tvm.testing.requires_cuda_compute_version(8)
def test_ptx_cp_async_barrier():
    f = ptx_cp_async_barrier

    mod = tvm.build(f, target="cuda")
    A_np = np.random.rand(32, 128).astype("float16")
    B_np = np.zeros((32, 128)).astype("float16")
    dev = tvm.cuda(0)
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)
    tvm.testing.assert_allclose(B_nd.numpy(), A_np)


@T.prim_func
def ptx_cp_async_bulk(A: T.Buffer((32, 128), "float16"), B: T.Buffer((32, 128), "float16")) -> None:
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float16", scope="shared")

        T.reads(A[0:32, 0:128])
        T.writes(B[0:32, 0:128])

        T.evaluate(T.create_barriers(1, dtype=""))
        T.evaluate(T.ptx_init_barrier_thread_count(0, 32, dtype=""))

        T.evaluate(
            T.ptx_cp_async_bulk(A_shared.data, tx * 128, A.data, tx * 128, 256, 0, dtype="float16")
        )

        T.evaluate(T.ptx_arrive_barrier_expect_tx(0, 256, dtype=""))
        T.evaluate(T.ptx_wait_barrier(0, dtype=""))

        for i in range(128):
            B[tx, i] = A_shared[tx, i]


@tvm.testing.requires_cuda_compute_version(9)
def test_ptx_cp_async_bulk():
    f = ptx_cp_async_bulk

    mod = tvm.build(f, target="cuda")
    A_np = np.random.rand(32, 128).astype("float16")
    B_np = np.zeros((32, 128)).astype("float16")
    dev = tvm.cuda(0)
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)
    tvm.testing.assert_allclose(B_nd.numpy(), A_np)


if __name__ == "__main__":
    test_ptx_cp_async()
    test_ptx_cp_async_barrier()
    test_ptx_cp_async_bulk()
