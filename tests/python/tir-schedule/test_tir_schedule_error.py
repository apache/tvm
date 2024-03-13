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
import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j in T.grid(128, 128):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.float32(0)
        for k in range(128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def two_kernels(var_A: T.handle, var_B: T.handle, seq_len: T.int32):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (1, seq_len * 8), "int32")
    B = T.match_buffer(var_B, (1, seq_len * 8), "int32", align=8)
    with T.block("exclusive_scan"):
        T.reads()
        T.writes()
        s8: T.int32 = seq_len * 8
        if s8 == 0:
            blockIdx_x = T.launch_thread("blockIdx.x", 1)
        else:
            with T.launch_thread("threadIdx.x", 1024) as threadIdx_x:
                blockIdx_x = T.launch_thread("blockIdx.x", T.ceildiv(s8, 1024))
                i: T.int32 = blockIdx_x * 1024 + threadIdx_x
                if i < s8:
                    B[i // s8, i % s8] = A[i // s8, i % s8]


# pylint: enable=no-member,invalid-name,unused-variable


def test_tir_schedule_error_detail():
    sch = tir.Schedule(matmul, debug_mask="all", error_render_level="detail")
    with pytest.raises(tir.ScheduleError) as excinfo:
        sch.get_block("wrong_name")
    (msg,) = excinfo.value.args
    assert "Cannot find a block with the name: wrong_name" in msg


def test_tir_schedule_error_fast():
    sch = tir.Schedule(matmul, debug_mask="all", error_render_level="fast")
    with pytest.raises(tir.ScheduleError) as excinfo:
        sch.get_block("wrong_name")
    (msg,) = excinfo.value.args
    assert "Cannot find a block with the specified name" in msg


def test_tir_schedule_error_none():
    sch = tir.Schedule(matmul, debug_mask="all", error_render_level="none")
    with pytest.raises(tir.ScheduleError) as excinfo:
        sch.get_block("wrong_name")
    (msg,) = excinfo.value.args
    assert "(not rendered)" in msg


def test_tir_schedule_attribute_error():
    sch = tir.Schedule(matmul)
    with pytest.raises(AttributeError):
        sch.non_existent_field()


def test_tir_schedule_two_kernels():
    tir.Schedule(two_kernels)


if __name__ == "__main__":
    tvm.testing.main()
