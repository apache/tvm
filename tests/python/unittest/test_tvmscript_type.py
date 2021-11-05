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


@T.prim_func
def element_wise_storage_align(a: T.handle, c: T.handle) -> None:
    """
    This prim func include necessary buffer types that need to be checked
    e.g. reads/writes, match_buffer/alloc_buffer, serial/block etc.
    """
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with T.block("root"):
        T.reads([])
        T.writes([])
        B = T.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
        for i0 in T.serial(0, 128):
            for ax1 in T.serial(0, 128):
                with T.block("B"):
                    vi, vj = T.axis.remap("SS", [i0, ax1])
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


@T.prim_func
def element_wise_env_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    """
    This prim func include necessary thread types that need to be checked
    e.g. env_thread, launch_thread, thread_binding etc.
    """
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


# Not running any test as we only want to type-check here
if __name__ == "__main__":
    pass
