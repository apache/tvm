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
"""Test workload for lowering and build."""
import numpy as np

import tvm
import tvm.testing
from tvm.script import tir as T


@T.prim_func
def tensorcore_gemm(handle_a: T.handle, handle_b: T.handle, handle_c: T.handle) -> None:
    # pylint: disable=missing-function-docstring
    # match buffer
    match_buffer_a = T.match_buffer(handle_a, [1024, 1024], "float16")
    match_buffer_b = T.match_buffer(handle_b, [1024, 1024], "float16")
    match_buffer_c = T.match_buffer(handle_c, [1024, 1024], "float32")

    # body
    for block_idx_x in T.thread_binding(0, 16, "blockIdx.x"):
        for block_idx_y in T.thread_binding(0, 8, "blockIdx.y"):
            with T.block():
                axis_bx, axis_by = T.axis.remap("SS", [block_idx_x, block_idx_y])
                shared_a = T.alloc_buffer([1024, 1024], "float16", scope="shared")
                shared_b = T.alloc_buffer([1024, 1024], "float16", scope="shared")
                wmma_a = T.alloc_buffer([1024, 1024], "float16", scope="wmma.matrix_a")
                wmma_b = T.alloc_buffer([1024, 1024], "float16", scope="wmma.matrix_b")
                wmma_c = T.alloc_buffer([1024, 1024], "float32", scope="wmma.accumulator")

                # pylint: disable=too-many-nested-blocks
                for thread_ty in T.thread_binding(0, 2, "threadIdx.y"):
                    for thread_tz in T.thread_binding(0, 2, "threadIdx.z"):
                        for index_i, index_jj in T.grid(2, 4):
                            with T.block():
                                new_axis_vi = T.axis.S(64, axis_bx * 4 + thread_ty * 2 + index_i)
                                new_axis_vj = T.axis.S(64, axis_by * 8 + thread_tz * 4 + index_jj)
                                T.reads([])
                                T.writes(
                                    wmma_c[
                                        new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                        new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                    ]
                                )
                                match_buffer_c0 = T.match_buffer(
                                    wmma_c[
                                        new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                        new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                    ],
                                    (16, 16),
                                    "float32",
                                    strides=[16 * 4, 1],
                                    scope="wmma.accumulator",
                                    offset_factor=1,
                                )
                                T.evaluate(
                                    T.tvm_fill_fragment(
                                        match_buffer_c0.data,
                                        16,
                                        16,
                                        16,
                                        index_i * 4 + index_jj,
                                        T.float32(0),  # pylint: disable=not-callable
                                        dtype="handle",
                                    )
                                )

                        for k_o in range(0, 32):
                            # copy data from global to shared
                            for thread_tx in T.thread_binding(0, 32, "threadIdx.x"):
                                for index_i0, index_j0 in T.grid(1, 4):
                                    for index_j1 in T.vectorized(0, 4):
                                        with T.block():
                                            new_axis_vi = T.axis.S(
                                                1024,
                                                axis_bx * 64
                                                + thread_ty * 32
                                                + thread_tx
                                                + index_i0,
                                            )
                                            new_axis_vj = T.axis.S(
                                                1024,
                                                k_o * 32 + thread_tz * 16 + index_j0 * 4 + index_j1,
                                            )
                                            shared_a[new_axis_vi, new_axis_vj + 8] = match_buffer_a[
                                                new_axis_vi, new_axis_vj
                                            ]

                                for index_i0, index_j0 in T.grid(2, 4):
                                    for index_j1 in T.vectorized(0, 4):
                                        with T.block():
                                            new_axis_vi = T.axis.S(
                                                1024,
                                                axis_by * 128
                                                + thread_ty * 64
                                                + thread_tx * 2
                                                + index_i0,
                                            )
                                            new_axis_vj = T.axis.S(
                                                1024,
                                                k_o * 32 + thread_tz * 16 + index_j0 * 4 + index_j1,
                                            )
                                            shared_b[new_axis_vi, new_axis_vj + 8] = match_buffer_b[
                                                new_axis_vi, new_axis_vj
                                            ]

                            for k_i in range(0, 2):
                                for index_i in range(0, 2):
                                    with T.block():
                                        new_axis_vi = T.axis.S(
                                            64, axis_bx * 4 + thread_ty * 2 + index_i
                                        )
                                        axis_vk = T.axis.S(64, k_o * 2 + k_i)
                                        T.reads(
                                            shared_a[
                                                new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16 + 8,
                                            ]
                                        )
                                        T.writes(
                                            wmma_a[
                                                new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16,
                                            ]
                                        )
                                        stride0 = T.int32()
                                        stride1 = T.int32()
                                        match_buffer_a0 = T.match_buffer(
                                            shared_a[
                                                new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16 + 8,
                                            ],
                                            (16, 16 + 8),
                                            "float16",
                                            strides=[stride0, stride1],
                                            scope="shared",
                                            offset_factor=1,
                                        )
                                        wmma_a0 = T.match_buffer(
                                            wmma_a[
                                                new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="wmma.matrix_a",
                                            offset_factor=1,
                                        )
                                        T.evaluate(
                                            T.tvm_load_matrix_sync(
                                                wmma_a0.data,
                                                16,
                                                16,
                                                16,
                                                index_i,
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype="float16"),
                                                    match_buffer_a0.data,
                                                    match_buffer_a0.elem_offset + 8,
                                                    match_buffer_a0.strides[0],
                                                    1,
                                                    dtype="handle",
                                                ),
                                                match_buffer_a0.strides[0],
                                                "row_major",
                                                dtype="handle",
                                            )
                                        )
                                for index_jj in range(0, 4):
                                    with T.block():
                                        new_axis_vj = T.axis.S(
                                            64, axis_by * 8 + thread_tz * 4 + index_jj
                                        )
                                        axis_vk = T.axis.S(64, k_o * 2 + k_i)
                                        T.reads(
                                            shared_b[
                                                new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16 + 8,
                                            ]
                                        )
                                        T.writes(
                                            wmma_b[
                                                new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16,
                                            ]
                                        )
                                        stride0 = T.int32()
                                        stride1 = T.int32()
                                        match_buffer_b0 = T.match_buffer(
                                            shared_b[
                                                new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16 + 8,
                                            ],
                                            (16, 16 + 8),
                                            "float16",
                                            strides=[stride0, stride1],
                                            scope="shared",
                                            offset_factor=1,
                                        )
                                        wmma_b0 = T.match_buffer(
                                            wmma_b[
                                                new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="wmma.matrix_b",
                                            offset_factor=1,
                                        )
                                        T.evaluate(
                                            T.tvm_load_matrix_sync(
                                                wmma_b0.data,
                                                16,
                                                16,
                                                16,
                                                index_jj,
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype="float16"),
                                                    match_buffer_b0.data,
                                                    match_buffer_b0.elem_offset + 8,
                                                    match_buffer_b0.strides[0],
                                                    1,
                                                    dtype="handle",
                                                ),
                                                match_buffer_b0.strides[0],
                                                "col_major",
                                                dtype="handle",
                                            )
                                        )
                                for index_i, index_jj in T.grid(2, 4):
                                    with T.block():
                                        new_axis_vi = T.axis.S(
                                            64, axis_bx * 4 + thread_ty * 2 + index_i
                                        )
                                        new_axis_vj = T.axis.S(
                                            64, axis_by * 8 + thread_tz * 4 + index_jj
                                        )
                                        axis_vk = T.axis.R(64, k_o * 2 + k_i)
                                        T.reads(
                                            [
                                                wmma_a[
                                                    new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                    axis_vk * 16 : axis_vk * 16 + 16,
                                                ],
                                                wmma_b[
                                                    new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                                    axis_vk * 16 : axis_vk * 16 + 16,
                                                ],
                                                wmma_c[
                                                    new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                    new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                                ],
                                            ]
                                        )
                                        T.writes(
                                            wmma_c[
                                                new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                            ]
                                        )
                                        wmma_a1 = T.match_buffer(
                                            wmma_a[
                                                new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="wmma.matrix_a",
                                            offset_factor=1,
                                        )
                                        wmma_b1 = T.match_buffer(
                                            wmma_b[
                                                new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                                axis_vk * 16 : axis_vk * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="wmma.matrix_b",
                                            offset_factor=1,
                                        )
                                        wmma_c1 = T.match_buffer(
                                            wmma_c[
                                                new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                                new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float32",
                                            strides=[16 * 4, 1],
                                            scope="wmma.accumulator",
                                            offset_factor=1,
                                        )
                                        T.evaluate(
                                            T.tvm_mma_sync(
                                                wmma_c1.data,
                                                index_i * 4 + index_jj,
                                                wmma_a1.data,
                                                index_i,
                                                wmma_b1.data,
                                                index_jj,
                                                wmma_c1.data,
                                                index_i * 4 + index_jj,
                                                dtype="handle",
                                            )
                                        )
                        for index_i, index_jj in T.grid(2, 4):
                            with T.block():
                                new_axis_vi = T.axis.S(64, axis_bx * 4 + thread_ty * 2 + index_i)
                                new_axis_vj = T.axis.S(64, axis_by * 8 + thread_tz * 4 + index_jj)
                                T.reads(
                                    wmma_c[
                                        new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                        new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                    ]
                                )
                                T.writes(
                                    match_buffer_c[
                                        new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                        new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                    ]
                                )
                                stride0 = T.int32()
                                stride1 = T.int32()
                                wmma_c2 = T.match_buffer(
                                    wmma_c[
                                        new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                        new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                    ],
                                    (16, 16),
                                    "float32",
                                    strides=[16 * 4, 1],
                                    scope="wmma.accumulator",
                                    offset_factor=1,
                                )
                                match_buffer_c1 = T.match_buffer(
                                    match_buffer_c[
                                        new_axis_vi * 16 : new_axis_vi * 16 + 16,
                                        new_axis_vj * 16 : new_axis_vj * 16 + 16,
                                    ],
                                    (16, 16),
                                    "float32",
                                    strides=[stride0, stride1],
                                    offset_factor=1,
                                )
                                T.evaluate(
                                    T.tvm_store_matrix_sync(
                                        wmma_c2.data,
                                        16,
                                        16,
                                        16,
                                        index_i * 4 + index_jj,
                                        T.tvm_access_ptr(
                                            T.type_annotation(dtype="float32"),
                                            match_buffer_c1.data,
                                            match_buffer_c1.elem_offset,
                                            match_buffer_c1.strides[0],
                                            1,
                                            dtype="handle",
                                        ),
                                        match_buffer_c1.strides[0],
                                        "row_major",
                                        dtype="handle",
                                    )
                                )


@tvm.testing.requires_cuda
def test_gemm_tensorcore():
    """Test running gemm on tensorcore."""
    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    b_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    c_np = np.dot(a_np.astype("float32"), b_np.T.astype("float32"))
    buff_a = tvm.nd.array(a_np, dev)
    buff_b = tvm.nd.array(b_np, dev)
    buff_c = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
    myfunc = tvm.build(tensorcore_gemm, target="cuda", name="dense")
    myfunc(buff_a, buff_b, buff_c)
    tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)

    evaluator = myfunc.time_evaluator(myfunc.entry_name, dev, number=100)
    time_elapsed = evaluator(buff_a, buff_b, buff_c).mean
    num_flops = 2 * 1024 * 1024 * 1024
    gflops = num_flops / (time_elapsed * 1e3) / 1e6
    print("gemm with tensor core: %f ms" % (time_elapsed * 1e3))
    print("GFLOPS: %f" % gflops)


if __name__ == "__main__":
    test_gemm_tensorcore()
