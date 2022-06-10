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


def count_cp_async(stmt):
    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Call) and str(n.op) == "tir.ptx_cp_async":
            num_alloc[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    return num_alloc[0]


def generate_global_to_shared_vectorized_copy(dtype, vector_size):
    num_iters = 128 // vector_size
    vector_size_expr = tvm.runtime.convert(vector_size)

    @T.prim_func
    def ptx_global_to_shared_copy(
        A: T.Buffer[(32, 128), dtype], B: T.Buffer[(32, 128), dtype]
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        bx = T.env_thread("blockIdx.x")
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(bx, 1)
        T.launch_thread(tx, 32)
        with T.block():
            A_shared = T.alloc_buffer([32, 128], dtype, scope="shared")
            T.reads(A[0:32, 0:128])
            T.writes(B[0:32, 0:128])

            T.attr("default", "async_scope", 1)
            for i in T.serial(num_iters):
                for j in T.vectorized(vector_size):
                    A_shared[tx, i * vector_size_expr + j] = A[tx, i * vector_size_expr + j]

            T.evaluate(T.ptx_commit_group(dtype=""))
            T.evaluate(T.ptx_wait_group(0, dtype=""))

            for i in range(128):
                B[tx, i] = A_shared[tx, i]

    return ptx_global_to_shared_copy


@T.prim_func
def ptx_global_to_shared_copy_fp32x1(
    A: T.Buffer[(32, 128), "float32"], B: T.Buffer[(32, 128), "float32"]
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float32", scope="shared")
        T.reads(A[0:32, 0:128])
        T.writes(B[0:32, 0:128])

        T.attr("default", "async_scope", 1)
        for i in T.serial(128):
            A_shared[tx, i] = A[tx, i]

        T.evaluate(T.ptx_commit_group(dtype=""))
        T.evaluate(T.ptx_wait_group(0, dtype=""))

        for i in range(128):
            B[tx, i] = A_shared[tx, i]


@T.prim_func
def ptx_global_to_shared_dyn_copy_fp16x8(
    A: T.Buffer[(32, 128), "float16"],
    B: T.Buffer[(32, 128), "float16"],
    C: T.Buffer[(32, 128), "float16"],
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.block():
        A_shared = T.alloc_buffer([32, 128], "float16", scope="shared.dyn")
        B_shared = T.alloc_buffer([32, 128], "float16", scope="shared.dyn")
        T.reads(A[0:32, 0:128], B[0:32, 0:128])
        T.writes(C[0:32, 0:128])

        T.attr("default", "async_scope", 1)
        for i in T.serial(16):
            for j in T.vectorized(8):
                A_shared[tx, i * 8 + j] = A[tx, i * 8 + j]
                B_shared[tx, i * 8 + j] = B[tx, i * 8 + j]

        T.evaluate(T.ptx_commit_group(dtype=""))
        T.evaluate(T.ptx_wait_group(0, dtype=""))

        for i in range(128):
            C[tx, i] = A_shared[tx, i] + B_shared[tx, i]


@tvm.testing.requires_cuda
def test_inject_async_copy():
    for dtype, vec_size in [("float16", 8), ("float16", 4), ("float32", 4), ("float32", 1)]:
        if vec_size == 1:
            f = ptx_global_to_shared_copy_fp32x1
        else:
            f = generate_global_to_shared_vectorized_copy(dtype, vec_size)

        mod = tvm.IRModule.from_expr(f)
        mod = tvm.tir.transform.FlattenBuffer()(mod)
        if vec_size > 1:
            mod = tvm.tir.transform.VectorizeLoop()(mod)
        mod = tvm.tir.transform.InjectPTXAsyncCopy()(mod)

        assert count_cp_async(mod["main"].body) == 1

        if not tvm.testing.is_ampere_or_newer():
            continue

        with tvm.transform.PassContext(config={"tir.use_ptx_async_copy": 1}):
            mod = tvm.build(tvm.IRModule.from_expr(f), target="cuda")

        A_np = np.random.rand(32, 128).astype(dtype)
        B_np = np.zeros((32, 128)).astype(dtype)
        dev = tvm.cuda(0)
        A_nd = tvm.nd.array(A_np, device=dev)
        B_nd = tvm.nd.array(B_np, device=dev)
        mod(A_nd, B_nd)
        tvm.testing.assert_allclose(B_nd.numpy(), A_np)


@tvm.testing.requires_cuda
def test_inject_async_copy_shared_dyn():
    f = ptx_global_to_shared_dyn_copy_fp16x8

    mod = tvm.IRModule.from_expr(f)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    mod = tvm.tir.transform.VectorizeLoop()(mod)
    mod = tvm.tir.transform.MergeDynamicSharedMemoryAllocations()(mod)
    mod = tvm.tir.transform.InjectPTXAsyncCopy()(mod)

    assert count_cp_async(mod["main"].body) == 2

    if not tvm.testing.is_ampere_or_newer():
        return

    with tvm.transform.PassContext(config={"tir.use_ptx_async_copy": 1}):
        mod = tvm.build(tvm.IRModule.from_expr(f), target="cuda")

    A_np = np.random.rand(32, 128).astype("float16")
    B_np = np.random.rand(32, 128).astype("float16")
    C_np = np.zeros((32, 128)).astype("float16")
    dev = tvm.cuda(0)
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    C_nd = tvm.nd.array(C_np, device=dev)
    mod(A_nd, B_nd, C_nd)
    tvm.testing.assert_allclose(C_nd.numpy(), A_np + B_np)


if __name__ == "__main__":
    test_inject_async_copy()
    test_inject_async_copy_shared_dyn()
