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
from tvm import tir
from tvm.script import ty


# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)
        for k in range(0, 128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


# pylint: enable=no-member,invalid-name,unused-variable


def test_tir_schedule_error_detail():
    sch = tir.Schedule(matmul, debug_mode=True, error_render_level="detail")
    with pytest.raises(tir.ScheduleError) as excinfo:
        sch.get_block("wrong_name")
    (msg,) = excinfo.value.args
    assert "Cannot find a block with the name: wrong_name" in msg


def test_tir_schedule_error_fast():
    sch = tir.Schedule(matmul, debug_mode=True, error_render_level="fast")
    with pytest.raises(tir.ScheduleError) as excinfo:
        sch.get_block("wrong_name")
    (msg,) = excinfo.value.args
    assert "Cannot find a block with the specified name" in msg


def test_tir_schedule_error_none():
    sch = tir.Schedule(matmul, debug_mode=True, error_render_level="none")
    with pytest.raises(tir.ScheduleError) as excinfo:
        sch.get_block("wrong_name")
    (msg,) = excinfo.value.args
    assert "(not rendered)" in msg


if __name__ == "__main__":
    test_tir_schedule_error_detail()
    test_tir_schedule_error_fast()
    test_tir_schedule_error_none()
