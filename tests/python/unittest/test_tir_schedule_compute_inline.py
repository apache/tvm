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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_multi_producer_consumer(a: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    D = T.match_buffer(d, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0  # B has two consumers
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block("D"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + 2.0 + C[vi, vj]  # D has two producers


@T.prim_func
def elementwise_multi_consumer_inlined(a: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    D = T.match_buffer(d, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0
    for i, j in T.grid(128, 128):
        with T.block("D"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = A[vi, vj] * 2.0 + 2.0 + C[vi, vj]


@T.prim_func
def elementwise_standalone(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + 1.0


@T.prim_func
def elementwise_standalone_dce(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + 1.0


@T.prim_func
def elementwise_under_loop(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    for i in T.serial(0, 128):
        for j in T.serial(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for j in T.serial(0, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_inlined(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@T.prim_func
def fail_multi_reader_writer(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.alloc_buffer((128, 128))
    D = T.match_buffer(d, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
            C[vi, vj] = A[vi, vj] + 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + C[vi, vj]


@T.prim_func
def elementwise_multi_reverse_loads(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = (B[vi, vj] + 1.0) * (B[vi, vj] * 2.0) + 3.0


@T.prim_func
def elementwise_multi_reverse_loads_inlined(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = (A[vi, vj] * 2.0 + 1.0) * (A[vi, vj] * 2.0 * 2.0) + 3.0


@T.prim_func
def elementwise_reverse_affine_load(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(8, 32, 8, 8), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j, k, l in T.grid(8, 32, 8, 8):
        with T.block("C"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            C[vi, vj, vk, vl] = B[
                ((((vi * 32) + vj) * 8 + vk) * 8 + vl) // 128,
                ((((vi * 32) + vj) * 8 + vk) * 8 + vl) % 128,
            ]


@T.prim_func
def elementwise_reverse_affine_load_inlined(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(8, 32, 8, 8), "float32"]
) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[
                (vj + vi * 128) // 2048,
                (vj + vi * 128) // 64 % 32,
                ((vj + vi * 128) // 8) % 8,
                (vj + vi * 128) % 8,
            ] = (
                A[vi, vj] * 2.0
            )


@T.prim_func
def elementwise_reverse_affine_load_unit_iter(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(8, 16, 1), "float32"],
    D: T.Buffer[(1, 8, 16, 128), "float32"],
) -> None:
    C = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0
    for i, j, k, l in T.grid(1, 8, 16, 128):
        with T.block("C"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            D[vi, vj, vk, vl] = C[vj * 16 + vk, vl] + B[vj, vk, vi]


@T.prim_func
def elementwise_reverse_affine_load_unit_iter_inlined(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(8, 16, 1), "float32"],
    D: T.Buffer[(1, 8, 16, 128), "float32"],
) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[0, vi // 16, vi % 16, vj] = A[vi, vj] * 2.0 + B[vi // 16, vi % 16, 0]


@T.prim_func
def elementwise_reverse_affine_load_unit_iter_simplified(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(8, 16, 1), "float32"],
    D: T.Buffer[(1, 8, 16, 128), "float32"],
) -> None:
    C = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0
    for i, j, k in T.grid(8, 16, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            D[0, vi, vj, vk] = C[vi * 16 + vj, vk] + B[vi, vj, 0]


@T.prim_func
def elementwise_reverse_affine_load_unit_iter_simplified_inlined(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(8, 16, 1), "float32"],
    D: T.Buffer[(1, 8, 16, 128), "float32"],
) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[0, vi // 16, vi % 16, vj] = A[vi, vj] * 2.0 + B[vi // 16, vi % 16, 0]


@T.prim_func
def elementwise_reverse_affine_chain(
    A: T.Buffer[(128, 128), "float32"], D: T.Buffer[(1, 8, 16, 128), "float32"]
):
    B = T.alloc_buffer((128, 128))
    C = T.alloc_buffer((8, 16, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j, k in T.grid(8, 16, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            C[vi, vj, vk] = B[vi * 16 + vj, vk] + 1.0
    for i, j, k, l in T.grid(1, 8, 16, 128):
        with T.block("D"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            D[vi, vj, vk, vl] = C[vj, vk, vl]


@T.prim_func
def elementwise_reverse_affine_chain_inlined(
    A: T.Buffer[(128, 128), "float32"], D: T.Buffer[(1, 8, 16, 128), "float32"]
) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[0, vi // 16, vi % 16, vj] = A[vi, vj] * 2.0 + 1.0


@T.prim_func
def elementwise_multi_reverse_affine_load(
    A: T.Buffer[(128, 128), "float32"],
    C: T.Buffer[(8, 16, 128), "float32"],
) -> None:
    B = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j, k in T.grid(8, 16, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            C[vi, vj, vk] = B[vi * 16 + vj, vk] + B[vi * 16 + vj, vk]


@T.prim_func
def elementwise_multi_reverse_affine_load_inlined(
    A: T.Buffer[(128, 128), "float32"],
    C: T.Buffer[(8, 16, 128), "float32"],
) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi // 16, vi % 16, vj] = A[vi, vj] * 2.0 + A[vi, vj] * 2.0


@T.prim_func
def elementwise_reverse_non_affine_load(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(8, 16, 128), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j, k in T.grid(8, 16, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            C[vi, vj, vk] = B[vi * 16 + vj, vi * 16 + vj]


@T.prim_func
def opaque_access_load(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[0:128, 0:128])
            T.writes(C[0:128, 0:128])
            T.evaluate(B.access_ptr("r", extent=128))
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def opaque_access_store(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[0:128, 0:128])
            T.writes(C[0:128, 0:128])
            T.evaluate(B.access_ptr("r", extent=128))
            T.evaluate(C.access_ptr("w", extent=128))
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def buffer_matched(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            Bb = T.match_buffer(B[vi : vi + 1, vj], (1, 1))
            C[vi, vj] = Bb[0, 0] + 1.0


@T.prim_func
def elementwise_predicate(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.where(B[i, j] < 10.0)
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_predicate_inlined(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.where(A[i, j] * 2.0 < 10.0)
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@T.prim_func
def elementwise_multi_loads(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + B[vi, vj + 1] + B[vi, vj + 2]


@T.prim_func
def elementwise_multi_loads_inlined(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0 + A[vi, vj + 1] * 2.0 + A[vi, vj + 2] * 2.0


@T.prim_func
def access_opaque_ptr_then_elemwise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [1024])
    B = T.match_buffer(b, [1024])
    A_cache = T.alloc_buffer([1024])
    BB = T.alloc_buffer([1024])
    with T.block("opaque"):
        # annotated opaque partial access
        T.reads(A[0:512])
        T.writes(A_cache[0:512])
        T.evaluate(A.access_ptr("r", extent=512))
        T.evaluate(A_cache.access_ptr("w", extent=512))
    for i in range(512):
        with T.block("BB"):
            vi = T.axis.remap("S", [i])
            BB[vi] = A_cache[vi] * 2.0
    for i in range(512):
        with T.block("B"):
            vi = T.axis.remap("S", [i])
            B[vi] = BB[vi] + 1.0


@T.prim_func
def access_opaque_ptr_then_elemwise_inline(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [1024], dtype="float32")
    B = T.match_buffer(b, [1024], dtype="float32")
    A_cache = T.alloc_buffer([1024], dtype="float32")
    with T.block("opaque"):
        # annotated opaque partial access should be kept
        T.reads(A[0:512])
        T.writes([A_cache[0:512]])
        T.evaluate(A.access_ptr("r", extent=512))
        T.evaluate(A_cache.access_ptr("w", extent=512))
    for i in T.serial(0, 512):
        with T.block("B"):
            vi = T.axis.spatial(512, i)
            T.reads([A_cache[vi]])
            T.writes([B[vi]])
            B[vi] = A_cache[vi] * 2.0 + 1.0


@T.prim_func
def matmul_relu(var_A: T.handle, var_B: T.handle, var_compute: T.handle) -> None:
    A = T.match_buffer(var_A, [512, 512], dtype="float32")
    B = T.match_buffer(var_B, [512, 512], dtype="float32")
    compute = T.match_buffer(var_compute, [512, 512], dtype="float32")
    C = T.alloc_buffer([512, 512], dtype="float32")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with T.block("C"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads([C[i, j], A[i, k], B[k, j]])
            T.writes([C[i, j]])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]
    for i0, i1 in T.grid(512, 512):
        with T.block("compute"):
            i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
            T.reads([C[i0_1, i1_1]])
            T.writes([compute[i0_1, i1_1]])
            compute[i0_1, i1_1] = T.max(C[i0_1, i1_1], T.float32(0))


@T.prim_func
def inline_block_with_init(
    A: T.Buffer[(1, 512, 7, 7), "float32"],
    B: T.Buffer[(1, 512, 1, 1), "float32"],
) -> None:
    B_rf = T.alloc_buffer([1, 512, 1, 1, 49], dtype="float32")
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 512, 1, 1, 49, 1):
        with T.block("tensor_rf"):
            vi4 = T.axis.spatial(49, i4)
            ax0 = T.axis.spatial(1, 0)
            ax1 = T.axis.spatial(512, i1)
            ax2 = T.axis.spatial(1, 0)
            ax3 = T.axis.spatial(1, 0)
            with T.init():
                B_rf[ax0, ax1, ax2, ax3, vi4] = T.float32(0)
            B_rf[ax0, ax1, ax2, ax3, vi4] = (
                B_rf[ax0, ax1, ax2, ax3, vi4]
                + A[
                    ax0,
                    ax1,
                    ax2 * 7 + vi4 // 7,
                    ax3 * 7 + vi4 % 7,
                ]
            )
    for i0, i1 in T.grid(1, 512):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(49, 1, 1, 1, 1):
            with T.block("tensor"):
                vi4, ax0_1 = T.axis.remap("RS", [ax0, ax1])
                ax1_1 = T.axis.spatial(512, i1 + ax2)
                ax2_1, ax3_1 = T.axis.remap("SS", [ax3, ax4])
                with T.init():
                    B[ax0_1, ax1_1, ax2_1, ax3_1] = T.float32(0)
                B[ax0_1, ax1_1, ax2_1, ax3_1] = (
                    B[ax0_1, ax1_1, ax2_1, ax3_1] + B_rf[ax0_1, ax1_1, ax2_1, ax3_1, vi4]
                )


@T.prim_func
def exp_exp_opaque_access_with_tvm_access_ptr(
    lookup_table: T.Buffer[(1024,), "int8"],
    x: T.Buffer[(16,), "float16"],
    compute: T.Buffer[(16,), "float16"],
) -> None:
    compute_1 = T.alloc_buffer([16], dtype="float16")
    for i0 in T.serial(16):
        with T.block("compute"):
            i0_1 = T.axis.spatial(16, i0)
            T.reads(x[i0_1])
            T.writes(compute_1[i0_1])
            compute_1[i0_1] = T.exp(x[i0_1], dtype="float16")
    for i0 in T.serial(16):
        with T.block("compute_1"):
            i0_2 = T.axis.spatial(16, i0)
            T.reads(lookup_table[0:1024], compute_1[i0_2])
            T.writes(compute[i0_2])
            T.evaluate(lookup_table.access_ptr("r"))
            compute[i0_2] = T.exp(
                compute_1[i0_2],
                dtype="float16",
            )


@T.prim_func
def exp_exp_opaque_access_with_tvm_access_ptr_inlined(
    lookup_table: T.Buffer[(1024,), "int8"],
    x: T.Buffer[(16,), "float16"],
    compute: T.Buffer[(16,), "float16"],
) -> None:
    for i0 in T.serial(16):
        with T.block("compute_1"):
            i0_1 = T.axis.spatial(16, i0)
            # Do not put the opaque access to new write region when opaque access
            # wrapped with a tvm_access_ptr and the access mask set to "read only"
            T.reads(lookup_table[0:1024], x[i0_1])
            T.writes(compute[i0_1])
            T.evaluate(lookup_table.access_ptr("r"))
            compute[i0_1] = T.exp(
                T.exp(x[i0_1], dtype="float16"),
                dtype="float16",
            )


@T.prim_func
def elementwise_overcomputed_producer(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(127, 127), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(127, 127):
        with T.block("C"):
            cvi, cvj = T.axis.remap("SS", [i, j])
            C[cvi, cvj] = B[cvi, cvj] + 1.0


@T.prim_func
def elementwise_overcomputed_producer_reverse_inlined(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(127, 127), "float32"]
) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.where(i < 127 and j < 127)
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@T.prim_func
def elementwise_producer_not_cover_consumer(
    A: T.Buffer[(128, 128), "float32"], D: T.Buffer[(256, 128), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(256, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = T.if_then_else(vi >= 128, B[vi - 128, vj], T.float32(0), dtype="float32")


@T.prim_func
def elementwise_predicate_producer(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((127, 128))
    C = T.match_buffer(c, (127, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.where(i < 127)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(127, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_predicate_producer_inlined(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (127, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            T.where(i < 127)
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = A[vi, vj] * T.float32(2) + T.float32(1)


# fmt: off
@tvm.script.ir_module
class Conv2dInt8_TensorCore_with_predicate:
    @T.prim_func
    def main(p0: T.Buffer[(16, 56, 56, 64), "int8"], p1: T.Buffer[(256, 1, 1, 64), "int8"], p2: T.Buffer[(1, 1, 1, 256), "int32"], p3: T.Buffer[(1, 1, 1, 256), "int32"], p4: T.Buffer[256, "int32"], p5: T.Buffer[256, "int32"], p6: T.Buffer[256, "int32"], p7: T.Buffer[(), "int32"], p8: T.Buffer[1, "int32"], p9: T.Buffer[(16, 56, 56, 256), "int32"], compute: T.Buffer[(16, 56, 56, 256), "int32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit":1024})
            compute_3 = T.alloc_buffer([16, 56, 56, 256], dtype="int32")
            conv2d_nhwc_reindex_shared = T.alloc_buffer([50176, 256], dtype="int32", scope="shared")
            conv2d_nhwc_reindex_shared_wmma_accumulator = T.alloc_buffer([50176, 256], dtype="int32", scope="wmma.accumulator")
            pad_temp_reindex_shared = T.alloc_buffer([50176, 64], dtype="int8", scope="shared")
            p1_reindex_shared = T.alloc_buffer([1, 1, 256, 64], dtype="int8", scope="shared")
            pad_temp_reindex_shared_wmma_matrix_a = T.alloc_buffer([50176, 64], dtype="int8", scope="wmma.matrix_a")
            p1_reindex_shared_wmma_matrix_b = T.alloc_buffer([1, 1, 256, 64], dtype="int8", scope="wmma.matrix_b")
            for ax2_0_0_ax3_0_0_fused in T.thread_binding(32, thread="blockIdx.y"):
                for ax2_0_1_ax3_0_1_fused in T.thread_binding(196, thread="blockIdx.x"):
                    for ax2_0_2_ax3_0_2_fused in T.thread_binding(4, thread="threadIdx.y"):
                        for ax0_0, ax1_0, ax4_0_0 in T.grid(1, 1, 2):
                            for ax0_ax1_fused in T.serial(1024):
                                with T.block("pad_temp_reindex_shared"):
                                    v0 = T.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0_ax1_fused // 32)
                                    v1 = T.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_fused % 32)
                                    T.reads(p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                                    T.writes(pad_temp_reindex_shared[v0, v1])
                                    T.block_attr({"buffer_dim_align":[[0, 0, 32, 16]], "meta_schedule.cooperative_fetch":4})
                                    pad_temp_reindex_shared[v0, v1] = p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1]
                            for ax0_ax1_ax2_ax3_fused in T.serial(2048):
                                with T.block("p1_reindex_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(1, 0)
                                    v2 = T.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + ax0_ax1_ax2_ax3_fused // 32)
                                    v3 = T.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    T.reads(p1[v2, v0, v1, v3])
                                    T.writes(p1_reindex_shared[v0, v1, v2, v3])
                                    T.block_attr({"buffer_dim_align":[[0, 2, 32, 16]], "meta_schedule.cooperative_fetch":3})
                                    p1_reindex_shared[v0, v1, v2, v3] = p1[v2, v0, v1, v3]
                            for ax0_1, ax1_1, ax4_0_1 in T.grid(1, 1, 2):
                                for ax0_0_1, ax1_0_1 in T.grid(1, 1):
                                    with T.block("pad_temp_reindex_shared_wmma.matrix_a_o"):
                                        v0_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2)
                                        v1_o = T.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1)
                                        T.reads(pad_temp_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                        T.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_s8_a"})
                                        for ax0_1_1, ax1_1_1 in T.grid(16, 16):
                                            with T.block("pad_temp_reindex_shared_wmma.matrix_a"):
                                                v0_i, v1_i = T.axis.remap("SS", [ax0_1_1, ax1_1_1])
                                                T.reads(pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                T.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                                for ax0, ax1, ax2_0, ax3_0 in T.grid(1, 1, 2, 1):
                                    with T.block("p1_reindex_shared_wmma.matrix_b_o"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(1, 0)
                                        v2_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax2_0)
                                        v3_o = T.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1)
                                        T.reads(p1_reindex_shared[v0, v1, v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                        T.writes(p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_16x16x16_s8_b_trans"})
                                        for ax2_1, ax3_1 in T.grid(16, 16):
                                            with T.block("p1_reindex_shared_wmma.matrix_b"):
                                                v2_i, v3_i = T.axis.remap("SS", [ax2_1, ax3_1])
                                                T.reads(p1_reindex_shared[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                T.writes(p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                p1_reindex_shared_wmma_matrix_b[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i] = p1_reindex_shared[v0, v1, v2_o * 16 + v2_i, v3_o * 16 + v3_i]
                                for ax2_0_3, ax3_0_3, ax0_2, ax1_2, ax4_0_2, ax2_0_4, ax3_0_4 in T.grid(1, 1, 1, 1, 1, 1, 2):
                                    with T.block("conv2d_nhwc_o"):
                                        v0 = T.axis.reduce(1, 0)
                                        v1 = T.axis.reduce(1, 0)
                                        v2_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax2_0_3 + ax2_0_4)
                                        v3_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax3_0_3 * 2 + ax3_0_4)
                                        v4_o = T.axis.reduce(4, ax4_0_0 * 2 + ax4_0_1 + ax4_0_2)
                                        T.reads(pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 : v2_o * 16 + 16, v4_o * 16 : v4_o * 16 + 16], p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16 : v3_o * 16 + 16, v4_o * 16 : v4_o * 16 + 16])
                                        T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 : v2_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync_16x16x16_s8s8s32_trans", "meta_schedule.auto_tensorize_init":"wmma_fill_16x16x16_s32", "meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "warp_execution":1})
                                        with T.init():
                                            for ax2_1, ax3_1 in T.grid(16, 16):
                                                with T.block("conv2d_nhwc_init"):
                                                    v2_i_init, v3_i_init = T.axis.remap("SS", [ax2_1, ax3_1])
                                                    T.reads()
                                                    T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init])
                                                    conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init] = 0
                                        for ax2_1, ax3_1, ax4_1 in T.grid(16, 16, 16):
                                            with T.block("conv2d_nhwc"):
                                                v2_i, v3_i, v4_i = T.axis.remap("SSR", [ax2_1, ax3_1, ax4_1])
                                                T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i], pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16 + v3_i, v4_o * 16 + v4_i])
                                                T.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] + T.cast(pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], "int32") * T.cast(p1_reindex_shared_wmma_matrix_b[v0, v1, v3_o * 16 + v3_i, v4_o * 16 + v4_i], "int32")
                        for ax0_0, ax1_0 in T.grid(1, 2):
                            with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2)
                                v1_o = T.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax1_0)
                                T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                T.writes(conv2d_nhwc_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                T.block_attr({"meta_schedule.auto_tensorize":"wmma_store_16x16x16_s32_shared"})
                                for ax0_1, ax1_1 in T.grid(16, 16):
                                    with T.block("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                        v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                        T.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        T.writes(conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                    for ax0, ax1_0, ax1_1, ax1_2, ax1_3 in T.grid(32, 1, 4, 32, 2):
                        with T.block("conv2d_nhwc_reindex_shared"):
                            T.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 2 + ax1_3 < 64)
                            v0 = T.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0)
                            v1 = T.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + (ax1_0 * 256 + ax1_1 * 64 + ax1_2 * 2 + ax1_3))
                            T.reads(p7[()], conv2d_nhwc_reindex_shared[v0, v1], p2[0, 0, 0, v1], p3[0, 0, 0, v1], p4[v1], p5[v1], p6[v1], p8[0])
                            T.writes(compute_3[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            compute_3[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1] = T.q_multiply_shift(T.max(T.min(p7[()] + T.q_multiply_shift_per_axis(conv2d_nhwc_reindex_shared[v0, v1] - p2[0, 0, 0, v1] + p3[0, 0, 0, v1], p4[v1], p5[v1], p6[v1], 31, False, True, dtype="int32"), 255), 0) - p8[0], 1457846997, 31, 0, dtype="int32")
            for i0_12, i1_12, i2_12, i3_12 in T.grid(16, 56, 56, 256):
                with T.block("compute_4"):
                    i0_13, i1_13, i2_13, i3_13 = T.axis.remap("SSSS", [i0_12, i1_12, i2_12, i3_12])
                    T.reads(compute_3[i0_13, i1_13, i2_13, i3_13], p9[i0_13, i1_13, i2_13, i3_13])
                    T.writes(compute[i0_13, i1_13, i2_13, i3_13])
                    compute[i0_13, i1_13, i2_13, i3_13] = T.max(T.min(compute_3[i0_13, i1_13, i2_13, i3_13] + T.q_multiply_shift(p9[i0_13, i1_13, i2_13, i3_13], 2101000910, 31, 0, dtype="int32"), 255), 0)
# fmt: on

# pylint: enable=no-member,invalid-name,unused-variable

use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})


def test_compute_inline_elementwise(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_c = sch.get_block("C")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_compute_inline_under_loop(use_block_name):
    sch = tir.Schedule(elementwise_under_loop, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_c = sch.get_block("C")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"
    verify_trace_roundtrip(sch=sch, mod=elementwise_under_loop)


def test_compute_inline_as_dce(use_block_name):
    sch = tir.Schedule(elementwise_standalone, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_c = sch.get_block("C")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_standalone_dce, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"
    verify_trace_roundtrip(sch=sch, mod=elementwise_standalone)


def test_compute_inline_multi_consumer(use_block_name):
    sch = tir.Schedule(elementwise_multi_producer_consumer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    block_c = sch.get_block("C")
    block_d = sch.get_block("D")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_multi_consumer_inlined, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"
    assert sch.get(block_d).name_hint == "D"
    verify_trace_roundtrip(sch=sch, mod=elementwise_multi_producer_consumer)


def test_compute_inline_fail_multi_writer(use_block_name):
    sch = tir.Schedule(fail_multi_reader_writer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block_b)


def test_reverse_compute_inline_elementwise(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_b).name_hint == "B"
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reverse_compute_inline_under_loop(use_block_name):
    sch = tir.Schedule(elementwise_under_loop, debug_mask="all")
    block_b = sch.get_block("B")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_b).name_hint == "B"
    verify_trace_roundtrip(sch=sch, mod=elementwise_under_loop)


def test_reverse_compute_inline_fail_as_dce(use_block_name):
    sch = tir.Schedule(elementwise_standalone, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_b)


def test_reverse_compute_inline_fail_multi_producer(use_block_name):
    sch = tir.Schedule(elementwise_multi_producer_consumer, debug_mask="all")
    block_d = "D" if use_block_name else sch.get_block("D")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_d)


def test_reverse_compute_inline_fail_multi_reader(use_block_name):
    sch = tir.Schedule(fail_multi_reader_writer, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_c)


def test_reverse_compute_multi_reverse_loads(use_block_name):
    sch = tir.Schedule(elementwise_multi_reverse_loads, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(elementwise_multi_reverse_loads_inlined, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_multi_reverse_loads)


def test_reverse_compute_inline_affine_load(use_block_name):
    sch = tir.Schedule(elementwise_reverse_affine_load, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(elementwise_reverse_affine_load_inlined, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_reverse_affine_load)


def test_reverse_compute_inline_multi_affine_load(use_block_name):
    sch = tir.Schedule(elementwise_multi_reverse_affine_load, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(elementwise_multi_reverse_affine_load_inlined, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_multi_reverse_affine_load)


def test_reverse_compute_inline_affine_load_unit_iter(use_block_name):
    sch = tir.Schedule(elementwise_reverse_affine_load_unit_iter, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(
        elementwise_reverse_affine_load_unit_iter_inlined, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_reverse_affine_load_unit_iter)


def test_reverse_compute_inline_affine_load_unit_iter_simplified(use_block_name):
    sch = tir.Schedule(elementwise_reverse_affine_load_unit_iter_simplified, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(
        elementwise_reverse_affine_load_unit_iter_simplified_inlined, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_reverse_affine_load_unit_iter_simplified)


@pytest.mark.parametrize("reverse_order", [True, False])
def test_reverse_compute_inline_affine_chain(use_block_name, reverse_order):
    sch = tir.Schedule(elementwise_reverse_affine_chain, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    block_d = "D" if use_block_name else sch.get_block("D")
    if reverse_order:
        sch.reverse_compute_inline(block_d)
        sch.reverse_compute_inline(block_c)
    else:
        sch.reverse_compute_inline(block_c)
        sch.reverse_compute_inline(block_d)
    tvm.ir.assert_structural_equal(elementwise_reverse_affine_chain_inlined, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_reverse_affine_chain)


def test_reverse_compute_fail_non_affine_load(use_block_name):
    sch = tir.Schedule(elementwise_reverse_non_affine_load, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_c)


def test_reverse_compute_fail_multi_reverse_loads(use_block_name):
    sch = tir.Schedule(elementwise_multi_loads, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_c)


def test_opaque_access_load(use_block_name):
    sch = tir.Schedule(opaque_access_load, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block_b)


def test_opaque_access_store(use_block_name):
    sch = tir.Schedule(opaque_access_store, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block_b)


def test_buffer_matched(use_block_name):
    sch = tir.Schedule(buffer_matched, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block_b)


def test_output_block(use_block_name):
    sch = tir.Schedule(matmul_relu, debug_mask="all")
    block = sch.get_block("compute")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block)


def test_compute_inline_predicate(use_block_name):
    sch = tir.Schedule(elementwise_predicate, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_predicate_inlined, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_predicate)


def test_compute_inline_multi_loads(use_block_name):
    sch = tir.Schedule(elementwise_multi_loads, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_block("B")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_multi_loads_inlined, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_multi_loads)


def test_compute_inline_with_opaque_access(use_block_name):
    """Test not rewrite opaque reads/writes after irrelavant compute inline"""
    sch = tir.Schedule(access_opaque_ptr_then_elemwise, debug_mask="all")
    BB = "BB" if use_block_name else sch.get_block("BB")
    sch.compute_inline(BB)
    tvm.ir.assert_structural_equal(access_opaque_ptr_then_elemwise_inline, sch.mod["main"])


def test_inline_block_with_init():
    sch = tir.Schedule(inline_block_with_init, debug_mask="all")
    block = sch.get_block(name="tensor_rf", func_name="main")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block=block)


def test_compute_inline_opaque_access_with_tvm_access_ptr(use_block_name):
    """Test opaque access with tvm_access_ptr after compute inline"""
    sch = tir.Schedule(exp_exp_opaque_access_with_tvm_access_ptr, debug_mask="all")
    compute = "compute" if use_block_name else sch.get_block("compute")
    sch.compute_inline(compute)
    tvm.ir.assert_structural_equal(
        exp_exp_opaque_access_with_tvm_access_ptr_inlined, sch.mod["main"]
    )


def test_reverse_compute_inline_overcomputed_producer(use_block_name):
    """Test reverse compute inline overcomputed producer"""
    sch = tir.Schedule(elementwise_overcomputed_producer, debug_mask="all")
    compute = "C" if use_block_name else sch.get_block("C")
    sch.reverse_compute_inline(compute)
    tvm.ir.assert_structural_equal(
        elementwise_overcomputed_producer_reverse_inlined, sch.mod["main"]
    )


def test_reverse_compute_inline_error_producer_not_cover_consumer(use_block_name):
    """Test reverse compute inline failure when the inlined block iter domains are not covered by
    its producer
    """
    sch = tir.Schedule(elementwise_producer_not_cover_consumer, debug_mask="all")
    compute = "C" if use_block_name else sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(compute)


def test_reverse_compute_inline_producer_predicate_allowed():
    """Test a case where reverse compute inline is allowed even though the producer has a
    non-trivial predicate.
    """

    sch = tir.Schedule(elementwise_predicate_producer, debug_mask="all")
    sch.reverse_compute_inline(sch.get_block("C"))
    tvm.ir.assert_structural_equal(elementwise_predicate_producer_inlined, sch.mod["main"])


def test_reverse_compute_inline_producer_predicate_disallowed():
    """Test reverse compute inline failure when the producer has a non-trivial predicate that cannot be
    implied by the synthesized predicate of the new inlined block.
    """

    sch = tir.Schedule(Conv2dInt8_TensorCore_with_predicate, debug_mask="all")

    with pytest.raises(tvm.tir.ScheduleError) as e:
        sch.reverse_compute_inline(sch.get_block("compute_4"))

    assert (
        "that cannot be implied by the synthesized predicate True of the new inlined block"
        in str(e)
    )


if __name__ == "__main__":
    tvm.testing.main()
