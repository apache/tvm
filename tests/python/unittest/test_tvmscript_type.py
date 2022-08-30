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
# pylint: disable=missing-function-docstring,missing-module-docstring,invalid-name,pointless-string-statement
from tvm.script import tir as T

"""
This prim func include necessary buffer types that need to be checked
e.g. reads/writes, match_buffer/alloc_buffer, serial/block etc.
"""


@T.prim_func
def element_wise_storage_align(a: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=64, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=64, offset_factor=1)
    # body
    with T.block("root"):
        T.reads([])
        T.writes([])
        B = T.alloc_buffer([128, 128], elem_offset=0, align=64, offset_factor=1)
        for i0 in T.serial(0, 128):
            for ax1 in T.serial(0, 128):
                with T.block("B"):
                    vi = T.axis.S(128, i0)
                    vj = T.axis.S(128, ax1)
                    T.reads([A[vi, vj]])
                    T.writes([B[vi, vj]])
                    T.block_attr({"buffer_dim_align": [[0, 0, 128, 127]]})
                    B[vi, vj] = A[vi, vj] * T.float32(2)
            for i1 in T.serial(0, 128):
                with T.block("C"):
                    vi_1, vj_1 = T.axis.remap("SS", [i0, i1])
                    T.reads([B[vi_1, vj_1]])
                    T.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = B[vi_1, vj_1] + T.float32(1)


"""
This prim func include necessary thread types that need to be checked
e.g. env_thread, launch_thread, thread_binding etc.
"""


@T.prim_func
def element_wise_env_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    j1_0 = T.env_thread("threadIdx.x")
    j0_0 = T.env_thread("threadIdx.x")
    i = T.env_thread("blockIdx.x")
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    T.launch_thread(i, 128)
    T.launch_thread(j0_0, 4)
    T.launch_thread(j1_0, 4)

    for blockIdx_x in T.thread_binding(0, 128, "blockIdx.x"):
        for threadIdx_x in T.thread_binding(0, 4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[blockIdx_x, threadIdx_x * 32 + j0_1] = (
                        A[blockIdx_x, threadIdx_x * 32 + j0_1] * 2.0
                    )
            for j1_1 in T.serial(0, 32):
                with T.block(""):
                    C[blockIdx_x, threadIdx_x * 32 + j1_1] = (
                        B[blockIdx_x, threadIdx_x * 32 + j1_1] + 1.0
                    )


"""
This test case is added to test T.grid
"""


@T.prim_func
def loop_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i, ko in T.grid(128, 4):
        for ki in T.thread_binding(0, 32, thread="threadIdx.x"):
            with T.block("B"):
                vi = T.axis.S(128, i)
                vk = T.axis.R(128, ko * 32 + ki)
                T.reads([B[vi], A[vi, vk]])
                T.writes([B[vi]])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vk]


"""
This test case is added to test T.comm_reducer, T.reinterpret, T.tvm_thread_allreduce
"""


@T.prim_func
def lowered_loop_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    normal_reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i in T.serial(0, 128):
        for ki in T.thread_binding(0, 32, thread="threadIdx.x"):
            normal_reduce_temp0[0] = T.float32(0)
            for ko in T.serial(0, 4):
                with T.block("B_normal_reduction"):
                    vi = T.axis.S(128, i)
                    vk = T.axis.R(128, ko * 32 + ki)
                    T.reads([A[vi, vk], normal_reduce_temp0[0]])
                    T.writes([normal_reduce_temp0[0]])
                    normal_reduce_temp0[0] = normal_reduce_temp0[0] + A[vi, vk]
            with T.block("B_cross_thread_reduction"):
                T.reads([normal_reduce_temp0[0]])
                T.writes([reduce_temp0[0]])
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp0[0],
                        True,
                        reduce_temp0.data,
                        ki,
                        dtype="handle",
                    )
                )
            with T.block("B_write_back"):
                vi = T.axis.S(128, i)
                T.reads([reduce_temp0[0]])
                T.writes([B[vi]])
                B[vi] = reduce_temp0[0]


"""
This test case is added to test T.Buffer with slice as argument and T.exp
"""


@T.prim_func
def different_access_indices(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128, 128], dtype="float32")
    B = T.match_buffer(b, [128, 128], dtype="float32")
    for i, j in T.grid(128, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads([B[vi, vj], A[vi, vj, vk]])
                T.writes(
                    [
                        B[
                            T.min(vj, vi) : T.min(vj, vi)  # type: ignore[misc]
                            + (T.max(vj, vi) + 1 - T.min(vj, vi)),
                            T.min(vi, vj) : T.min(vi, vj)  # type: ignore[misc]
                            + (T.max(vi, vj) + 1 - T.min(vi, vj)),
                        ]
                    ]
                )
                with T.init():
                    B[vj, vi] = T.exp(B[vj, vi], dtype="float32")
                B[vi, vj] = B[vi, vj] + A[vi, vj, vk]


# Not running any test as we only want to type-check here
if __name__ == "__main__":
    pass
