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
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name

@T.prim_func
def element_wise(a: T.handle, c: T.handle) -> None:
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
                    vi, vj = T.axis.remap("SS", [i0, ax1])
                    T.reads([A[vi, vj]])
                    T.writes([B[vi, vj]])
                    B[vi, vj] = (A[vi, vj]*T.float32(2))
            for i1 in T.serial(0, 128):
                with T.block("C"):
                    vi_1, vj_1 = T.axis.remap("SS", [i0, i1])
                    T.reads([B[vi_1, vj_1]])
                    T.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = (B[vi_1, vj_1] + T.float32(1))


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
                    vi, vj = T.axis.remap("SS", [i0, ax1])
                    T.reads([A[vi, vj]])
                    T.writes([B[vi, vj]])
                    T.block_attr({"buffer_dim_align":[[0, 0, 128, 127]]})
                    B[vi, vj] = (A[vi, vj]*T.float32(2))
            for i1 in T.serial(0, 128):
                with T.block("C"):
                    vi_1, vj_1 = T.axis.remap("SS", [i0, i1])
                    T.reads([B[vi_1, vj_1]])
                    T.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = (B[vi_1, vj_1] + T.float32(1))


@T.prim_func
def element_wise_invalid_annotation(a: T.handle, c: T.handle) -> None:
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
                    T.block_attr({"buffer_dim_align": [0]})
                    vi, vj = T.axis.remap("SS", [i0, ax1])
                    T.reads([A[vi, vj]])
                    T.writes([B[vi, vj]])
                    B[vi, vj] = (A[vi, vj]*T.float32(2))
            for i1 in T.serial(0, 128):
                with T.block("C"):
                    vi_1, vj_1 = T.axis.remap("SS", [i0, i1])
                    T.reads([B[vi_1, vj_1]])
                    T.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = (B[vi_1, vj_1] + T.float32(1))


use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})

def test_storage_align(use_block_name):
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    B = 'B' if use_block_name else s.get_block("B")
    s.storage_align(B, 0, axis=0, factor=128, offset=127)
    assert_structural_equal_ignore_global_symbol(element_wise_storage_align, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_storage_align_update():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    B = s.get_block("B")
    s.storage_align(B, 0, axis=0, factor=128, offset=0)
    s.storage_align(B, 0, axis=0, factor=128, offset=127)
    assert_structural_equal_ignore_global_symbol(element_wise_storage_align, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_storage_align_invalid_factor1():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    B = s.get_block("B")
    with pytest.raises(tir.ScheduleError):
        s.storage_align(B, 0, axis=0, factor=0, offset=127)


def test_storage_align_invalid_factor2():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    B = s.get_block("B")
    with pytest.raises(tir.ScheduleError):
        s.storage_align(B, 0, axis=0, factor=-1, offset=127)


def test_storage_align_invalid_buffer():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    C = s.get_block("C")
    with pytest.raises(tir.ScheduleError):
        s.storage_align(C, 0, axis=0, factor=128, offset=127)


def test_storage_align_invalid_buffer_index():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    B = s.get_block("B")
    with pytest.raises(tir.ScheduleError):
        s.storage_align(B, 2, axis=0, factor=128, offset=127)


def test_storage_align_invalid_axis():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    B = s.get_block("B")
    with pytest.raises(tir.ScheduleError):
        s.storage_align(B, 0, axis=2, factor=128, offset=127)


def test_storage_align_invalid_annotation():
    func = element_wise_invalid_annotation
    s = tir.Schedule(func, debug_mask='all')
    B = s.get_block("B")
    with pytest.raises(tir.ScheduleError):
        s.storage_align(B, 0, axis=2, factor=128, offset=127)


if __name__ == "__main__":
    test_storage_align()
    test_storage_align_update()
    test_storage_align_invalid_factor1()
    test_storage_align_invalid_factor2()
    test_storage_align_invalid_buffer()
    test_storage_align_invalid_buffer_index()
    test_storage_align_invalid_axis()
    test_storage_align_invalid_annotation()
