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
    verify_trace_roundtrip,
    assert_structural_equal_ignore_global_symbol,
)

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

def test_annotate_read_region():
    @T.prim_func
    def before(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    @T.prim_func
    def expected(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi - 1:vi - 1 + 2, vj - 1:vj - 1 + 2])
                T.writes(B[vi, vj])
                T.block_attr({"explicit_read_region": 0})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0    
    
    sch = tir.Schedule(before, debug_mask="all")
    block = sch.get_block("B")
    sch.annotate_read_region(block, 0, lambda vi, vj: ((vi - 1, vi + 1), (vj - 1, vj + 1)))
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)

def test_annotate_read_region_for_resize():
    @T.prim_func
    def resize_before(x: T.Buffer((1, 1, 32, 32), "float16"), resize: T.Buffer((1, 1, 16, 16), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        for i0, i1, i2, i3 in T.grid(1, 1, 16, 16):
            with T.block("resize"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, 0:32, 0:32])
                T.writes(resize[v_i0, v_i1, v_i2, v_i3])
                resize[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T.Cast("float32", x[v_i0, v_i1, T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i2) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0), T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i3) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0)]))

    @T.prim_func
    def resize_expected(x: T.Buffer((1, 1, 32, 32), "float16"), resize: T.Buffer((1, 1, 16, 16), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        for i0, i1, i2, i3 in T.grid(1, 1, 16, 16):
            with T.block("resize"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, v_i2 * 2 - 1:v_i2 * 2 + 1, v_i3 * 2 - 1:v_i3 * 2 + 1])
                T.writes(resize[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"explicit_read_region": 0})
                resize[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T.Cast("float32", x[v_i0, v_i1, T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i2) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0), T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i3) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0)]))

    sch = tir.Schedule(resize_before, debug_mask="all")
    block = sch.get_block("resize")
    sch.annotate_read_region(block, 0, gen_new_ranges = lambda v_i0, v_i1, v_i2, v_i3: [
        v_i0, v_i1, 
        (v_i2 * 2 - 1, v_i2 * 2 + 1),
        (v_i3 * 2 - 1, v_i3 * 2 + 1)
    ])
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], resize_expected)
    verify_trace_roundtrip(sch=sch, mod=resize_before)

if __name__ == "__main__":
    tvm.testing.main()
