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
# pylint: disable=invalid-name, too-many-locals, too-many-statements, unused-argument
"""Test workload for lowering and build"""
import tvm
from tvm import tir
from tvm.script import tir as T
import tvm.testing
import numpy as np


@T.prim_func
def tensorcore_gemm(a: T.handle, b: T.handle, c: T.handle) -> None:
    # match buffer
    A = T.match_buffer(a, [1024, 1024], "float16")
    B = T.match_buffer(b, [1024, 1024], "float16")
    C = T.match_buffer(c, [1024, 1024], "float32")

    # body
    for blockIdx_x in T.thread_binding(0, 16, "blockIdx.x"):
        for blockIdx_y in T.thread_binding(0, 8, "blockIdx.y"):
            with T.block():
                bx, by = T.axis.remap("SS", [blockIdx_x, blockIdx_y])
                shared_A = T.alloc_buffer([1024, 1024], "float16", scope="shared")
                shared_B = T.alloc_buffer([1024, 1024], "float16", scope="shared")
                wmma_A = T.alloc_buffer([1024, 1024], "float16", scope="wmma.matrix_a")
                wmma_B = T.alloc_buffer([1024, 1024], "float16", scope="wmma.matrix_b")
                wmma_C = T.alloc_buffer([1024, 1024], "float32", scope="wmma.accumulator")
                for ty in T.thread_binding(0, 2, "threadIdx.y"):
                    for tz in T.thread_binding(0, 2, "threadIdx.z"):
                        for i, j in T.grid(2, 4):
                            with T.block():
                                vi = T.axis.S(64, bx * 4 + ty * 2 + i)
                                vj = T.axis.S(64, by * 8 + tz * 4 + j)
                                T.reads([])
                                T.writes(wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                                C0 = T.match_buffer(
                                    wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                                    (16, 16),
                                    "float32",
                                    strides=[16 * 4, 1],
                                    scope="wmma.accumulator",
                                    offset_factor=1,
                                )
                                T.evaluate(
                                    T.tvm_fill_fragment(
                                        C0.data,
                                        16,
                                        16,
                                        16,
                                        i * 4 + j,
                                        T.float32(0),
                                        dtype="handle",
                                    )
                                )

                        for ko in range(0, 32):
                            # copy data from global to shared
                            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                                for i0, j0 in T.grid(1, 4):
                                    for j1 in T.vectorized(0, 4):
                                        with T.block():
                                            vi = T.axis.S(1024, bx * 64 + ty * 32 + tx + i0)
                                            vj = T.axis.S(1024, ko * 32 + tz * 16 + j0 * 4 + j1)
                                            shared_A[vi, vj + 8] = A[vi, vj]

                                for i0, j0 in T.grid(2, 4):
                                    for j1 in T.vectorized(0, 4):
                                        with T.block():
                                            vi = T.axis.S(1024, by * 128 + ty * 64 + tx * 2 + i0)
                                            vj = T.axis.S(1024, ko * 32 + tz * 16 + j0 * 4 + j1)
                                            shared_B[vi, vj + 8] = B[vi, vj]

                            for ki in range(0, 2):
                                for i in range(0, 2):
                                    with T.block():
                                        vi = T.axis.S(64, bx * 4 + ty * 2 + i)
                                        vk = T.axis.S(64, ko * 2 + ki)
                                        T.reads(
                                            shared_A[
                                                vi * 16 : vi * 16 + 16,
                                                vk * 16 : vk * 16 + 16 + 8,
                                            ]
                                        )
                                        T.writes(
                                            wmma_A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16]
                                        )
                                        s0 = T.var("int32")
                                        s1 = T.var("int32")
                                        A0 = T.match_buffer(
                                            shared_A[
                                                vi * 16 : vi * 16 + 16,
                                                vk * 16 : vk * 16 + 16 + 8,
                                            ],
                                            (16, 16 + 8),
                                            "float16",
                                            strides=[s0, s1],
                                            scope="shared",
                                            offset_factor=1,
                                        )
                                        wmma_A0 = T.match_buffer(
                                            wmma_A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="wmma.matrix_a",
                                            offset_factor=1,
                                        )
                                        T.evaluate(
                                            T.tvm_load_matrix_sync(
                                                wmma_A0.data,
                                                16,
                                                16,
                                                16,
                                                i,
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype="float16"),
                                                    A0.data,
                                                    A0.elem_offset + 8,
                                                    A0.strides[0],
                                                    1,
                                                    dtype="handle",
                                                ),
                                                A0.strides[0],
                                                "row_major",
                                                dtype="handle",
                                            )
                                        )
                                for j in range(0, 4):
                                    with T.block():
                                        vj = T.axis.S(64, by * 8 + tz * 4 + j)
                                        vk = T.axis.S(64, ko * 2 + ki)
                                        T.reads(
                                            shared_B[
                                                vj * 16 : vj * 16 + 16,
                                                vk * 16 : vk * 16 + 16 + 8,
                                            ]
                                        )
                                        T.writes(
                                            wmma_B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16]
                                        )
                                        s0 = T.var("int32")
                                        s1 = T.var("int32")
                                        B0 = T.match_buffer(
                                            shared_B[
                                                vj * 16 : vj * 16 + 16,
                                                vk * 16 : vk * 16 + 16 + 8,
                                            ],
                                            (16, 16 + 8),
                                            "float16",
                                            strides=[s0, s1],
                                            scope="shared",
                                            offset_factor=1,
                                        )
                                        wmma_B0 = T.match_buffer(
                                            wmma_B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="wmma.matrix_b",
                                            offset_factor=1,
                                        )
                                        T.evaluate(
                                            T.tvm_load_matrix_sync(
                                                wmma_B0.data,
                                                16,
                                                16,
                                                16,
                                                j,
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype="float16"),
                                                    B0.data,
                                                    B0.elem_offset + 8,
                                                    B0.strides[0],
                                                    1,
                                                    dtype="handle",
                                                ),
                                                B0.strides[0],
                                                "col_major",
                                                dtype="handle",
                                            )
                                        )
                                for i, j in T.grid(2, 4):
                                    with T.block():
                                        vi = T.axis.S(64, bx * 4 + ty * 2 + i)
                                        vj = T.axis.S(64, by * 8 + tz * 4 + j)
                                        vk = T.axis.R(64, ko * 2 + ki)
                                        T.reads(
                                            [
                                                wmma_A[
                                                    vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16
                                                ],
                                                wmma_B[
                                                    vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16
                                                ],
                                                wmma_C[
                                                    vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16
                                                ],
                                            ]
                                        )
                                        T.writes(
                                            wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16]
                                        )
                                        wmma_A1 = T.match_buffer(
                                            wmma_A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="wmma.matrix_a",
                                            offset_factor=1,
                                        )
                                        wmma_B1 = T.match_buffer(
                                            wmma_B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="wmma.matrix_b",
                                            offset_factor=1,
                                        )
                                        wmma_C1 = T.match_buffer(
                                            wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                                            (16, 16),
                                            "float32",
                                            strides=[16 * 4, 1],
                                            scope="wmma.accumulator",
                                            offset_factor=1,
                                        )
                                        T.evaluate(
                                            T.tvm_mma_sync(
                                                wmma_C1.data,
                                                i * 4 + j,
                                                wmma_A1.data,
                                                i,
                                                wmma_B1.data,
                                                j,
                                                wmma_C1.data,
                                                i * 4 + j,
                                                dtype="handle",
                                            )
                                        )
                        for i, j in T.grid(2, 4):
                            with T.block():
                                vi = T.axis.S(64, bx * 4 + ty * 2 + i)
                                vj = T.axis.S(64, by * 8 + tz * 4 + j)
                                T.reads(wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                                T.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                                s0 = T.var("int32")
                                s1 = T.var("int32")
                                wmma_C2 = T.match_buffer(
                                    wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                                    (16, 16),
                                    "float32",
                                    strides=[16 * 4, 1],
                                    scope="wmma.accumulator",
                                    offset_factor=1,
                                )
                                C1 = T.match_buffer(
                                    C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                                    (16, 16),
                                    "float32",
                                    strides=[s0, s1],
                                    offset_factor=1,
                                )
                                T.evaluate(
                                    T.tvm_store_matrix_sync(
                                        wmma_C2.data,
                                        16,
                                        16,
                                        16,
                                        i * 4 + j,
                                        T.tvm_access_ptr(
                                            T.type_annotation(dtype="float32"),
                                            C1.data,
                                            C1.elem_offset,
                                            C1.strides[0],
                                            1,
                                            dtype="handle",
                                        ),
                                        C1.strides[0],
                                        "row_major",
                                        dtype="handle",
                                    )
                                )


@tvm.testing.requires_cuda
def test_gemm_tensorcore():
    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    b_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    c_np = np.dot(a_np.astype("float32"), b_np.T.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
    f = tvm.build(tensorcore_gemm, target="cuda", name="dense")
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=100)
    t = evaluator(a, b, c).mean
    num_flops = 2 * 1024 * 1024 * 1024
    gflops = num_flops / (t * 1e3) / 1e6
    print("gemm with tensor core: %f ms" % (t * 1e3))
    print("GFLOPS: %f" % gflops)


if __name__ == "__main__":
    test_gemm_tensorcore()
