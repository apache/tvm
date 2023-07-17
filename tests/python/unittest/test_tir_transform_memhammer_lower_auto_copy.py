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
from tvm import te
from tvm.script import tir as T
import sys
import pytest


@tvm.script.ir_module
class Transpose:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for ty in T.thread_binding(8, thread="threadIdx.y"):
                with T.block():
                    A_shared_dyn = T.alloc_buffer([16, 128], dtype="float32", scope="shared.dyn")
                    with T.block("A_shared"):
                        T.block_attr({"auto_copy": 1})
                        for ax0, ax1 in T.grid(128, 16):
                            A_shared_dyn[ax1, ax0] = A[ax0, ax1]
                    with T.block("B"):
                        T.block_attr({"auto_copy": 1})
                        for ax1, ax0 in T.grid(16, 128):
                            B[ax1, ax0] = A_shared_dyn[ax1, ax0]


@tvm.script.ir_module
class GlobalToShared:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="shared.dyn"
                            )
                            with T.block("A_shared"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax0, ax1] = A[bx * 128 + ax0, by * 128 + ax1]
                            with T.block("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm.script.ir_module
class SharedToGlobal:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="shared.dyn"
                            )
                            with T.block("A_shared"):
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax1, ax0] = A[bx * 128 + ax0, by * 128 + ax1]
                            with T.block("B"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for ax1, ax0 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax1, ax0]


@tvm.script.ir_module
class GlobalToSharedWithLocalStage:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="shared.dyn"
                            )
                            with T.block("A_shared"):
                                T.block_attr(
                                    {"auto_copy": 1, "vector_bytes": 16, "local_stage": True}
                                )
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax0, ax1] = A[bx * 128 + ax0, by * 128 + ax1]
                            with T.block("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm.script.ir_module
class SharedToWmma:
    @T.prim_func
    def main() -> None:
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer(
                                [128, 128], dtype="float16", scope="shared.dyn"
                            )
                            A_wmma = T.alloc_buffer(
                                [128, 128], dtype="float16", scope="wmma.matrix_a"
                            )
                            with T.block("A_wmma"):
                                T.block_attr({"auto_copy": 1})
                                for ax0, ax1 in T.grid(128, 128):
                                    A_wmma[ax0, ax1] = A_shared_dyn[ax0, ax1]


@tvm.script.ir_module
class WmmaToShared:
    @T.prim_func
    def main() -> None:
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            C_shared = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="shared.dyn"
                            )
                            with T.block("C_shared"):
                                T.block_attr({"auto_copy": 1})
                                for ax0, ax1 in T.grid(128, 128):
                                    C_shared[ax0, ax1] = C_accum[ax0, ax1]


@tvm.script.ir_module
class WmmaToGlobal:
    @T.prim_func
    def main(c: T.handle) -> None:
        C = T.match_buffer(c, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            with T.block("C_global"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    C[bx * 128 + ax0, by * 128 + ax1] = C_accum[ax0, ax1]


@tvm.script.ir_module
class WmmaToGlobalWithFusion:
    @T.prim_func
    def main(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [1024])
        C = T.match_buffer(c, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            with T.block("C_global"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    C[bx * 128 + ax0, by * 128 + ax1] = (
                                        C_accum[ax0, ax1] + A[bx * 128 + ax0]
                                    )


@tvm.script.ir_module
class MmaToGlobal:
    @T.prim_func
    def main(c: T.handle) -> None:
        C = T.match_buffer(c, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="m16n8k8.matrixC"
                            )
                            with T.block("C_global"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    C[bx * 128 + ax0, by * 128 + ax1] = C_accum[ax0, ax1]


@tvm.script.ir_module
class TransformedGlobalToShared:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer(
                                [128, 128], dtype="float32", strides=[128, 1], scope="shared.dyn"
                            )
                            with T.block("A_shared"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for outer in T.serial(16):
                                    for ty_1 in T.thread_binding(8, thread="threadIdx.y"):
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            for vec in T.vectorized(4):
                                                A_shared_dyn[
                                                    (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    // 128
                                                    % 128,
                                                    (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    % 128,
                                                ] = A[
                                                    bx * 128
                                                    + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    // 128
                                                    % 128,
                                                    by * 128
                                                    + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    % 128,
                                                ]
                            with T.block("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm.script.ir_module
class TransformedSharedToGlobal:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer(
                                [128, 128], dtype="float32", strides=[129, 1], scope="shared.dyn"
                            )
                            with T.block("A_shared"):
                                T.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.writes(A_shared_dyn[0:128, 0:128])
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax1, ax0] = A[bx * 128 + ax0, by * 128 + ax1]
                            with T.block("B"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for outer in T.serial(16):
                                    for ty_1 in T.thread_binding(8, thread="threadIdx.y"):
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            for vec in T.vectorized(4):
                                                B[
                                                    bx * 128
                                                    + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    // 128
                                                    % 128,
                                                    by * 128
                                                    + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    % 128,
                                                ] = A_shared_dyn[
                                                    (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    % 128,
                                                    (((outer * 8 + ty_1) * 32 + tx) * 4 + vec)
                                                    // 128
                                                    % 128,
                                                ]


@tvm.script.ir_module
class TransformedGlobalToSharedWithLocalStage:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (1024, 1024))
        B = T.match_buffer(b, (1024, 1024))
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block(""):
                            T.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            T.writes(B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            A_shared_dyn = T.alloc_buffer(
                                (128, 128), strides=(128, 1), scope="shared.dyn"
                            )
                            with T.block("A_shared"):
                                T.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.writes(A_shared_dyn[0:128, 0:128])
                                T.block_attr(
                                    {"auto_copy": 1, "local_stage": True, "vector_bytes": 16}
                                )
                                A_shared_dyn_local = T.alloc_buffer((16, 4), scope="local")
                                for ax0_ax1_fused_1 in T.thread_binding(8, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(
                                        32, thread="threadIdx.x"
                                    ):
                                        for ax0_ax1_fused_0_cache in range(16):
                                            for ax0_ax1_fused_3_cache in T.vectorized(4):
                                                A_shared_dyn_local[
                                                    ax0_ax1_fused_0_cache
                                                    * 8
                                                    * 32
                                                    * 4
                                                    // 128
                                                    % 128
                                                    // 8,
                                                    ax0_ax1_fused_3_cache % 128,
                                                ] = A[
                                                    bx * 128
                                                    + (
                                                        (
                                                            (
                                                                ax0_ax1_fused_0_cache * 8
                                                                + ax0_ax1_fused_1
                                                            )
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3_cache
                                                    )
                                                    // 128
                                                    % 128,
                                                    by * 128
                                                    + (
                                                        (
                                                            (
                                                                ax0_ax1_fused_0_cache * 8
                                                                + ax0_ax1_fused_1
                                                            )
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3_cache
                                                    )
                                                    % 128,
                                                ]
                                        for ax0_ax1_fused_0 in range(16):
                                            for ax0_ax1_fused_3 in T.vectorized(4):
                                                A_shared_dyn[
                                                    (
                                                        (
                                                            (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1)
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3
                                                    )
                                                    // 128
                                                    % 128,
                                                    (
                                                        (
                                                            (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1)
                                                            * 32
                                                            + ax0_ax1_fused_2
                                                        )
                                                        * 4
                                                        + ax0_ax1_fused_3
                                                    )
                                                    % 128,
                                                ] = A_shared_dyn_local[
                                                    ax0_ax1_fused_0 * 8 * 32 * 4 // 128 % 128 // 8,
                                                    ax0_ax1_fused_3 % 128,
                                                ]
                            with T.block("B"):
                                T.reads(A_shared_dyn[0:128, 0:128])
                                T.writes(B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                for ax0 in range(128):
                                    for ax1 in range(128):
                                        B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm.script.ir_module
class TransformedSharedToWmma:
    @T.prim_func
    def main() -> None:
        s0 = T.int32()
        s1 = T.int32()
        # body
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer(
                                [128, 128], dtype="float16", strides=[136, 1], scope="shared.dyn"
                            )
                            A_wmma = T.alloc_buffer(
                                [128, 128], dtype="float16", scope="wmma.matrix_a"
                            )
                            with T.block("C_shared"):
                                T.reads(A_shared_dyn[0:128, 0:128])
                                T.writes(A_wmma[0:128, 0:128])
                                T.block_attr({"auto_copy": 1})
                                for ax00, ax10 in T.grid(8, 8):
                                    with T.block("wmma_load"):
                                        T.reads(
                                            A_shared_dyn[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ]
                                        )
                                        T.writes(
                                            A_wmma[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ]
                                        )
                                        src = T.match_buffer(
                                            A_shared_dyn[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            strides=[s1, s0],
                                            scope="shared.dyn",
                                            offset_factor=16,
                                        )
                                        tgt = T.match_buffer(
                                            A_wmma[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            scope="wmma.matrix_a",
                                            offset_factor=16,
                                        )
                                        T.evaluate(
                                            T.tvm_load_matrix_sync(
                                                tgt.data,
                                                16,
                                                16,
                                                16,
                                                tgt.elem_offset // 256
                                                + tgt.elem_offset % 256 // 16,
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype="float16"),
                                                    src.data,
                                                    src.elem_offset,
                                                    s1 * 16,
                                                    1,
                                                    dtype="handle",
                                                ),
                                                s1,
                                                "row_major",
                                                dtype="handle",
                                            )
                                        )


@tvm.script.ir_module
class TransformedWmmaToShared:
    @T.prim_func
    def main() -> None:
        s0 = T.int32()
        s1 = T.int32()
        # body
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            C_shared = T.alloc_buffer(
                                [128, 128], dtype="float32", strides=[136, 1], scope="shared.dyn"
                            )
                            with T.block("A_wmma"):
                                T.reads(C_accum[0:128, 0:128])
                                T.writes(C_shared[0:128, 0:128])
                                T.block_attr({"auto_copy": 1})
                                for ax00, ax10 in T.grid(8, 8):
                                    with T.block("wmma_store"):
                                        T.reads(
                                            C_accum[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ]
                                        )
                                        T.writes(
                                            C_shared[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ]
                                        )
                                        src = T.match_buffer(
                                            C_accum[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float32",
                                            scope="wmma.accumulator",
                                            offset_factor=16,
                                        )
                                        tgt = T.match_buffer(
                                            C_shared[
                                                ax00 * 16 : ax00 * 16 + 16,
                                                ax10 * 16 : ax10 * 16 + 16,
                                            ],
                                            [16, 16],
                                            dtype="float32",
                                            strides=[s1, s0],
                                            scope="shared.dyn",
                                            offset_factor=16,
                                        )
                                        T.evaluate(
                                            T.tvm_store_matrix_sync(
                                                src.data,
                                                16,
                                                16,
                                                16,
                                                src.elem_offset // 256
                                                + src.elem_offset % 256 // 16,
                                                T.tvm_access_ptr(
                                                    T.type_annotation(dtype="float32"),
                                                    tgt.data,
                                                    tgt.elem_offset,
                                                    s1 * 16,
                                                    2,
                                                    dtype="handle",
                                                ),
                                                s1,
                                                "row_major",
                                                dtype="handle",
                                            )
                                        )


@tvm.script.ir_module
class TransformedWmmaToGlobal:
    @T.prim_func
    def main(C: T.Buffer((1024, 1024), "float32")):
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block(""):
                            T.reads()
                            T.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            C_accum = T.alloc_buffer((128, 128), scope="wmma.accumulator")
                            with T.block("C_global"):
                                T.reads(C_accum[0:128, 0:128])
                                T.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                C_accum_shared_dyn = T.alloc_buffer(
                                    (8, 8, 16, 16), strides=(2048, 256, 16, 1), scope="shared.dyn"
                                )
                                for ax0_0 in range(8):
                                    for ax1_0 in range(8):
                                        with T.block("wmma_store"):
                                            T.reads(
                                                C_accum[
                                                    ax0_0 * 16 : ax0_0 * 16 + 16,
                                                    ax1_0 * 16 : ax1_0 * 16 + 16,
                                                ]
                                            )
                                            T.writes(C_accum_shared_dyn[ty, ax1_0, 0:16, 0:16])
                                            src = T.match_buffer(
                                                C_accum[
                                                    ax0_0 * 16 : ax0_0 * 16 + 16,
                                                    ax1_0 * 16 : ax1_0 * 16 + 16,
                                                ],
                                                (16, 16),
                                                scope="wmma.accumulator",
                                                offset_factor=16,
                                            )
                                            s1 = T.int32()
                                            s0 = T.int32()
                                            tgt = T.match_buffer(
                                                C_accum_shared_dyn[ty, ax1_0, 0:16, 0:16],
                                                (16, 16),
                                                strides=(s1, s0),
                                                scope="shared.dyn",
                                                offset_factor=16,
                                            )
                                            T.tvm_store_matrix_sync(
                                                src.data,
                                                16,
                                                16,
                                                16,
                                                src.elem_offset // 256
                                                + src.elem_offset % 256 // 16,
                                                T.tvm_access_ptr(
                                                    T.type_annotation("float32"),
                                                    tgt.data,
                                                    tgt.elem_offset,
                                                    s1 * 16,
                                                    2,
                                                ),
                                                s1,
                                                "row_major",
                                            )
                                    for (
                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                    ) in range(16):
                                        for (
                                            ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                        ) in T.thread_binding(8, thread="threadIdx.y"):
                                            for (
                                                ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                            ) in T.thread_binding(32, thread="threadIdx.x"):
                                                for ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3 in (
                                                    T.vectorized(4)
                                                ):
                                                    C[
                                                        bx * 128
                                                        + (
                                                            ax0_0 * 16
                                                            + (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            % 16
                                                        ),
                                                        by * 128
                                                        + (
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            // 16
                                                            % 8
                                                            * 16
                                                            + (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            % 16
                                                        ),
                                                    ] = C_accum_shared_dyn[
                                                        (
                                                            (
                                                                (
                                                                    ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                    * 8
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                )
                                                                * 32
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                            )
                                                            * 4
                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                        )
                                                        // 16
                                                        // 16
                                                        // 8
                                                        % 8,
                                                        (
                                                            (
                                                                (
                                                                    ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                    * 8
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                )
                                                                * 32
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                            )
                                                            * 4
                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                        )
                                                        // 16
                                                        // 16
                                                        % 8,
                                                        (
                                                            (
                                                                (
                                                                    ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                    * 8
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                )
                                                                * 32
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                            )
                                                            * 4
                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                        )
                                                        // 16
                                                        % 16,
                                                        (
                                                            (
                                                                (
                                                                    ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                    * 8
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                )
                                                                * 32
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                            )
                                                            * 4
                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                        )
                                                        % 16,
                                                    ]


@tvm.script.ir_module
class TransformedWmmaToGlobalWithFusion:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"), C: T.Buffer((1024, 1024), "float32")) -> None:
        s0 = T.int32()
        s1 = T.int32()
        # body
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            T.reads(A[bx * 128 : bx * 128 + 128])
                            T.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            C_accum = T.alloc_buffer(
                                [128, 128], dtype="float32", scope="wmma.accumulator"
                            )
                            with T.block("C_global"):
                                T.reads(C_accum[0:128, 0:128], A[bx * 128 : bx * 128 + 128])
                                T.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                C_accum_shared_dyn = T.alloc_buffer(
                                    (8, 8, 16, 16), strides=(2048, 256, 16, 1), scope="shared.dyn"
                                )
                                for ax0_0 in range(8):
                                    for ax1_0 in range(8):
                                        with T.block("wmma_store"):
                                            T.reads(
                                                C_accum[
                                                    ax0_0 * 16 : ax0_0 * 16 + 16,
                                                    ax1_0 * 16 : ax1_0 * 16 + 16,
                                                ]
                                            )
                                            T.writes(C_accum_shared_dyn[ty, ax1_0, 0:16, 0:16])
                                            src = T.match_buffer(
                                                C_accum[
                                                    ax0_0 * 16 : ax0_0 * 16 + 16,
                                                    ax1_0 * 16 : ax1_0 * 16 + 16,
                                                ],
                                                (16, 16),
                                                scope="wmma.accumulator",
                                                offset_factor=16,
                                            )
                                            s1 = T.int32()
                                            s0 = T.int32()
                                            tgt = T.match_buffer(
                                                C_accum_shared_dyn[ty, ax1_0, 0:16, 0:16],
                                                (16, 16),
                                                strides=(s1, s0),
                                                scope="shared.dyn",
                                                offset_factor=16,
                                            )
                                            T.tvm_store_matrix_sync(
                                                src.data,
                                                16,
                                                16,
                                                16,
                                                src.elem_offset // 256
                                                + src.elem_offset % 256 // 16,
                                                T.tvm_access_ptr(
                                                    T.type_annotation("float32"),
                                                    tgt.data,
                                                    tgt.elem_offset,
                                                    s1 * 16,
                                                    2,
                                                ),
                                                s1,
                                                "row_major",
                                            )
                                    for (
                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                    ) in range(16):
                                        for (
                                            ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                        ) in T.thread_binding(8, thread="threadIdx.y"):
                                            for (
                                                ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                            ) in T.thread_binding(32, thread="threadIdx.x"):
                                                for ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3 in (
                                                    T.vectorized(4)
                                                ):
                                                    C[
                                                        bx * 128
                                                        + (
                                                            ax0_0 * 16
                                                            + (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            % 16
                                                        ),
                                                        by * 128
                                                        + (
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            // 16
                                                            % 8
                                                            * 16
                                                            + (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            % 16
                                                        ),
                                                    ] = (
                                                        C_accum_shared_dyn[
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            // 16
                                                            // 8
                                                            % 8,
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            // 16
                                                            % 8,
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            // 16
                                                            % 16,
                                                            (
                                                                (
                                                                    (
                                                                        ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                        * 8
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                    )
                                                                    * 32
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                )
                                                                * 4
                                                                + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                            )
                                                            % 16,
                                                        ]
                                                        + A[
                                                            bx * 128
                                                            + (
                                                                ax0_0 * 16
                                                                + (
                                                                    (
                                                                        (
                                                                            ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0
                                                                            * 8
                                                                            + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1
                                                                        )
                                                                        * 32
                                                                        + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2
                                                                    )
                                                                    * 4
                                                                    + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3
                                                                )
                                                                // 16
                                                                % 16
                                                            )
                                                        ]
                                                    )


@tvm.script.ir_module
class TransformedMmaToGlobal:
    @T.prim_func
    def main(C: T.Buffer((1024, 1024), "float32")):
        with T.block("root"):
            T.block_attr({"warp_execution": T.bool(True)})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block(""):
                            T.reads()
                            T.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                            C_accum = T.alloc_buffer((128, 128), scope="m16n8k8.matrixC")
                            with T.block("C_global"):
                                T.reads(C_accum[0:128, 0:128])
                                T.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                C_accum_shared_dyn = T.alloc_buffer(
                                    (8, 16, 8, 8), strides=(1152, 72, 8, 1), scope="shared.dyn"
                                )
                                for ax0_0 in range(16):
                                    for ax1_0 in range(16):
                                        with T.block("mma_store"):
                                            T.reads(
                                                C_accum[
                                                    ax0_0 * 8 : ax0_0 * 8 + 8,
                                                    ax1_0 * 8 : ax1_0 * 8 + 8,
                                                ]
                                            )
                                            T.writes(C_accum_shared_dyn[ty, ax1_0, 0:8, 0:8])
                                            src = T.match_buffer(
                                                C_accum[
                                                    ax0_0 * 8 : ax0_0 * 8 + 8,
                                                    ax1_0 * 8 : ax1_0 * 8 + 8,
                                                ],
                                                (8, 8),
                                                scope="m16n8k8.matrixC",
                                                offset_factor=8,
                                            )
                                            tgt = T.match_buffer(
                                                C_accum_shared_dyn[ty, ax1_0, 0:8, 0:8],
                                                (8, 8),
                                                strides=("s1", "s0"),
                                                scope="shared.dyn",
                                                offset_factor=8,
                                            )
                                            tx = T.launch_thread("threadIdx.x", 32)
                                            for vec in T.vectorized(2):
                                                tgt[tx // 4, tx % 4 * 2 + vec] = src[
                                                    tx // 4, tx % 4 * 2 + vec
                                                ]
                                    for ax1_1 in range(8):
                                        for ty_0 in T.thread_binding(8, thread="threadIdx.y"):
                                            for tx_0 in T.thread_binding(32, thread="threadIdx.x"):
                                                for v in T.vectorized(4):
                                                    C[
                                                        bx * 128
                                                        + (
                                                            ax0_0 * 8
                                                            + (
                                                                ((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4
                                                                + v
                                                            )
                                                            // 8
                                                            % 8
                                                        ),
                                                        by * 128
                                                        + (
                                                            (
                                                                ((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4
                                                                + v
                                                            )
                                                            // 8
                                                            // 8
                                                            % 16
                                                            * 8
                                                            + (
                                                                ((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4
                                                                + v
                                                            )
                                                            % 8
                                                        ),
                                                    ] = C_accum_shared_dyn[
                                                        (((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4 + v)
                                                        // 8
                                                        // 8
                                                        // 16
                                                        % 8,
                                                        (((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4 + v)
                                                        // 8
                                                        // 8
                                                        % 16,
                                                        (((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4 + v)
                                                        // 8
                                                        % 8,
                                                        (((ax1_1 * 8 + ty_0) * 32 + tx_0) * 4 + v)
                                                        % 8,
                                                    ]


def _check(original, transformed):
    mod = tvm.tir.transform.LowerAutoCopy()(original)
    tvm.ir.assert_structural_equal(mod, transformed, True)


def test_coalesce_vectorize():
    _check(GlobalToShared, TransformedGlobalToShared)


def test_inverse():
    _check(SharedToGlobal, TransformedSharedToGlobal)


def test_local_stage():
    _check(GlobalToSharedWithLocalStage, TransformedGlobalToSharedWithLocalStage)


def test_rewrite_shared_to_wmma():
    _check(SharedToWmma, TransformedSharedToWmma)


def test_rewrite_wmma_to_shared():
    _check(WmmaToShared, TransformedWmmaToShared)


def test_rewrite_wmma_to_global():
    _check(WmmaToGlobal, TransformedWmmaToGlobal)


def verify_single_allocation(stmt, alloc_size=None):
    num_alloc = [0]
    alloc_extents = []

    def verify(n):
        if (
            isinstance(n, tvm.tir.Block)
            and n.alloc_buffers is not None
            and (True in ((buf.scope() == "shared.dyn") for buf in n.alloc_buffers))
        ):
            num_alloc[0] += len(n.alloc_buffers)
            for buf in n.alloc_buffers:
                alloc_extents.append(buf.shape)

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 1

    if alloc_size:

        def prod(arr):
            ret = 1
            for element in arr:
                ret *= element
            return ret

        assert prod(alloc_extents[0]) == alloc_size


def test_auto_padding():
    mod = tvm.tir.transform.LowerAutoCopy()(Transpose)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    verify_single_allocation(mod["main"].body, 16 * 130)


def test_rewrite_wmma_to_global_fusion():
    _check(WmmaToGlobalWithFusion, TransformedWmmaToGlobalWithFusion)


def test_rewrite_mma_to_global():
    _check(MmaToGlobal, TransformedMmaToGlobal)


if __name__ == "__main__":
    test_coalesce_vectorize()
    test_inverse()
    test_local_stage()
    test_rewrite_shared_to_wmma()
    test_rewrite_wmma_to_shared()
    test_rewrite_wmma_to_global()
    test_auto_padding()
    test_rewrite_wmma_to_global_fusion()
    test_rewrite_mma_to_global()
