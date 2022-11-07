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
from tvm.tir.schedule.schedule import ScheduleError
from tvm.tir.schedule.testing import verify_trace_roundtrip


@T.prim_func
def transpose_elementwise(
    A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]
) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vj, vi] * 2.0


@T.prim_func
def transpose_elementwise_reindex_read(
    A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]
) -> None:
    A_reindex = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("A_reindex"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_reindex[vi, vj] = A[vj, vi]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A_reindex[vi, vj] * 2.0


@T.prim_func
def conv2d_nhwc(
    Input: T.Buffer[(1, 224, 224, 3), "float32"],
    Weight: T.Buffer[(7, 7, 3, 64), "float32"],
    Conv2d_nhwc: T.Buffer[(1, 112, 112, 64), "float32"],
) -> None:
    PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with T.block("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                ((((i1_1 >= 3) and (i1_1 < 227)) and (i2_1 >= 3)) and (i2_1 < 227)),
                Input[i0_1, (i1_1 - 3), (i2_1 - 3), i3_1],
                T.float32(0),
                dtype="float32",
            )
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with T.block("conv2d_nhwc"):
            n, h, w, co, rh, rw, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            with T.init():
                Conv2d_nhwc[n, h, w, co] = T.float32(0)
            Conv2d_nhwc[n, h, w, co] = Conv2d_nhwc[n, h, w, co] + (
                PadInput[n, ((h * 2) + rh), ((w * 2) + rw), ((T.floordiv(co, 64) * 3) + rc)]
                * Weight[rh, rw, rc, co]
            )


@T.prim_func
def conv2d_nhwc_reindex_data(
    Input: T.Buffer[(1, 224, 224, 3), "float32"],
    Weight: T.Buffer[(7, 7, 3, 64), "float32"],
    Conv2d_nhwc: T.Buffer[(1, 112, 112, 64), "float32"],
) -> None:
    PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
    ReindexInput = T.alloc_buffer([1, 112, 112, 7, 7, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with T.block("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                ((((i1_1 >= 3) and (i1_1 < 227)) and (i2_1 >= 3)) and (i2_1 < 227)),
                Input[i0_1, (i1_1 - 3), (i2_1 - 3), i3_1],
                T.float32(0),
                dtype="float32",
            )
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 112, 112, 7, 7, 3):
        with T.block("ReindexInput"):
            n, h, w, rh, rw, rc = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
            ReindexInput[n, h, w, rh, rw, rc] = PadInput[n, ((h * 2) + rh), ((w * 2) + rw), rc]
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with T.block("conv2d_nhwc"):
            n, h, w, co, rh, rw, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            with T.init():
                Conv2d_nhwc[n, h, w, co] = T.float32(0)
            Conv2d_nhwc[n, h, w, co] = Conv2d_nhwc[n, h, w, co] + (
                ReindexInput[n, h, w, rh, rw, rc] * Weight[rh, rw, rc, co]
            )


@T.prim_func
def conv2d_nhwc_reindex_weight(
    var_inputs: T.handle, var_weight: T.handle, var_conv2d_nhwc: T.handle
) -> None:
    inputs = T.match_buffer(var_inputs, [1, 224, 224, 3], dtype="float32")
    weight = T.match_buffer(var_weight, [7, 7, 3, 64], dtype="float32")
    conv2d_nhwc = T.match_buffer(var_conv2d_nhwc, [1, 112, 112, 64], dtype="float32")
    PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
    weight_reindex = T.alloc_buffer([64, 7, 7, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with T.block("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1])
            T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                i1_1 >= 3 and i1_1 < 227 and i2_1 >= 3 and i2_1 < 227,
                inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1],
                T.float32(0),
                dtype="float32",
            )
    for ax3, ax4, ax5, ax6 in T.grid(64, 7, 7, 3):
        with T.block("weight_reindex"):
            v3, v4, v5, v6 = T.axis.remap("SSSS", [ax3, ax4, ax5, ax6])
            T.reads(weight[v4, v5, v6, v3])
            T.writes(weight_reindex[v3, v4, v5, v6])
            weight_reindex[v3, v4, v5, v6] = weight[v4, v5, v6, v3]
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with T.block("conv2d_nhwc"):
            n, h, w, co, rh, rw, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            T.reads(
                PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc],
                weight_reindex[co, rh, rw, rc],
            )
            T.writes(conv2d_nhwc[n, h, w, co])
            with T.init():
                conv2d_nhwc[n, h, w, co] = T.float32(0)
            conv2d_nhwc[n, h, w, co] = (
                conv2d_nhwc[n, h, w, co]
                + PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc]
                * weight_reindex[co, rh, rw, rc]
            )


@T.prim_func
def matmul(
    A: T.Buffer[(512, 512), "float32"],
    B: T.Buffer[(512, 512), "float32"],
    C: T.Buffer[(512, 512), "float32"],
) -> None:
    for i0, i1, i2 in T.grid(512, 512, 512):
        with T.block("matmul"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C[i, j], A[i, k], B[k, j])
            T.writes(C[i, j])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


@T.prim_func
def matmul_reindex_write(
    A: T.Buffer[(512, 512), "float32"],
    B: T.Buffer[(512, 512), "float32"],
    C: T.Buffer[(512, 512), "float32"],
) -> None:
    C_reindex = T.alloc_buffer([512, 512], dtype="float32")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with T.block("matmul"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C_reindex[i, j], A[i, k], B[k, j])
            T.writes(C_reindex[i, j])
            with T.init():
                C_reindex[i, j] = T.float32(0)
            C_reindex[i, j] = C_reindex[i, j] + A[i, k] * B[k, j]
    for i0, i1 in T.grid(512, 512):
        with T.block("C_reindex"):
            v0, v1 = T.axis.remap("SS", [i0, i1])
            T.reads(C_reindex[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_reindex[v0, v1]


@T.prim_func
def multiple_read(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vj, vi] + A[vi, vj]


@T.prim_func
def mixed_dtype(
    p0: T.Buffer[(T.int64(2), 1280), "float16"],
    p1: T.Buffer[(1280, 1280), "float16"],
    T_matmul_NT: T.Buffer[(T.int64(2), 1280), "float16"],
) -> None:
    for i0, i1, i2 in T.grid(T.int64(2), 1280, 1280):
        with T.block("T_matmul_NT"):
            i = T.axis.spatial(T.int64(2), i0)
            j, k = T.axis.remap("SR", [i1, i2])
            T.reads(p0[i, k], p1[j, k])
            T.writes(T_matmul_NT[i, j])
            with T.init():
                T_matmul_NT[i, j] = T.float16(0)
            T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1[j, k]


@T.prim_func
def mixed_dtype_reindex_write(
    p0: T.Buffer[(T.int64(2), 1280), "float16"],
    p1: T.Buffer[(1280, 1280), "float16"],
    T_matmul_NT: T.Buffer[(T.int64(2), 1280), "float16"],
) -> None:
    T_matmul_NT_reindex = T.alloc_buffer([T.int64(2), 1280], dtype="float16")
    for i0, i1, i2 in T.grid(T.int64(2), 1280, 1280):
        with T.block("T_matmul_NT"):
            i = T.axis.spatial(T.int64(2), i0)
            j, k = T.axis.remap("SR", [i1, i2])
            T.reads(p0[i, k], p1[j, k])
            T.writes(T_matmul_NT_reindex[i, j])
            with T.init():
                T_matmul_NT_reindex[i, j] = T.float16(0)
            T_matmul_NT_reindex[i, j] = T_matmul_NT_reindex[i, j] + p0[i, k] * p1[j, k]
    for ax0, ax1 in T.grid(T.int64(2), 1280):
        with T.block("T_matmul_NT_reindex"):
            v0 = T.axis.spatial(T.int64(2), ax0)
            (v1,) = T.axis.remap("S", [ax1])
            T.reads(T_matmul_NT_reindex[v0, v1])
            T.writes(T_matmul_NT[v0, v1])
            T_matmul_NT[v0, v1] = T_matmul_NT_reindex[v0, v1]


@T.prim_func
def matmul_unit_dim(
    A: T.Buffer[(1, 512), "float32"],
    B: T.Buffer[(512, 1), "float32"],
    C: T.Buffer[(1, 1), "float32"],
) -> None:
    for i0, i1, i2 in T.grid(1, 1, 512):
        with T.block("matmul"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C[i, j], A[i, k], B[k, j])
            T.writes(C[i, j])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


@T.prim_func
def matmul_unit_dim_reindex_write(
    A: T.Buffer[(1, 512), "float32"],
    B: T.Buffer[(512, 1), "float32"],
    C: T.Buffer[(1, 1), "float32"],
) -> None:
    C_reindex = T.alloc_buffer([1, 1], dtype="float32")
    for i0, i1, i2 in T.grid(1, 1, 512):
        with T.block("matmul"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C_reindex[i, j], A[i, k], B[k, j])
            T.writes(C_reindex[i, j])
            with T.init():
                C_reindex[i, j] = T.float32(0)
            C_reindex[i, j] = C_reindex[i, j] + A[i, k] * B[k, j]
    for i0, i1 in T.grid(1, 1):
        with T.block("C_reindex"):
            v0, v1 = T.axis.remap("SS", [i0, i1])
            T.reads(C_reindex[v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_reindex[v0, v1]


use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})
use_buffer_name = tvm.testing.parameter(by_dict={"buffer_index": False, "buffer_name": True})


def test_reindex_read_basic(use_block_name, use_buffer_name):
    sch = tir.Schedule(transpose_elementwise)
    block = "B" if use_block_name else sch.get_block("B")
    buf = "A" if use_buffer_name else ("read", 0)
    sch.reindex(block, buf)
    tvm.ir.assert_structural_equal(transpose_elementwise_reindex_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=transpose_elementwise)


def test_conv2d_reindex_weight(use_block_name, use_buffer_name):
    sch = tir.Schedule(conv2d_nhwc)
    block = "conv2d_nhwc" if use_block_name else sch.get_block("conv2d_nhwc")
    buf = "Weight" if use_buffer_name else ("read", 1)
    sch.reindex(block, buf)
    tvm.ir.assert_structural_equal(conv2d_nhwc_reindex_weight, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=conv2d_nhwc)


def test_conv2d_reindex_data(use_block_name, use_buffer_name):
    sch = tir.Schedule(conv2d_nhwc)
    block = "conv2d_nhwc" if use_block_name else sch.get_block("conv2d_nhwc")
    buf = "PadInput" if use_buffer_name else ("read", 0)
    sch.reindex(block, buf)
    tvm.ir.assert_structural_equal(conv2d_nhwc_reindex_data, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=conv2d_nhwc)


def test_matmul_reindex_write(use_block_name, use_buffer_name):
    sch = tir.Schedule(matmul)
    block = "matmul" if use_block_name else sch.get_block("matmul")
    buf = "C" if use_buffer_name else ("write", 0)
    sch.reindex(block, buf)
    tvm.ir.assert_structural_equal(matmul_reindex_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=matmul)


def test_reindex_fail_multiple_read(use_block_name, use_buffer_name):
    sch = tir.Schedule(multiple_read)
    block = "B" if use_block_name else sch.get_block("B")
    buf = "A" if use_buffer_name else ("read", 0)
    with pytest.raises(ScheduleError):
        sch.reindex(block, buf)


def test_reindex_mixed_dtype(use_block_name, use_buffer_name):
    sch = tir.Schedule(mixed_dtype)
    block = "T_matmul_NT" if use_block_name else sch.get_block("T_matmul_NT")
    buf = "T_matmul_NT" if use_buffer_name else ("write", 0)
    sch.reindex(block, buf)
    tvm.ir.assert_structural_equal(mixed_dtype_reindex_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=mixed_dtype)


def test_matmul_unit_dim_reindex_write(use_block_name, use_buffer_name):
    sch = tir.Schedule(matmul_unit_dim)
    block = "matmul" if use_block_name else sch.get_block("matmul")
    buf = "C" if use_buffer_name else ("write", 0)
    sch.reindex(block, buf)
    tvm.ir.assert_structural_equal(matmul_unit_dim_reindex_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=matmul_unit_dim)


if __name__ == "__main__":
    tvm.testing.main()
