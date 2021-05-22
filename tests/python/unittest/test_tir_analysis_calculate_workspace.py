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
import numpy as np
import pytest

import tvm
from tvm import tir, script
from tvm.ir import Range
from tvm.script import ty


# fmt: off
@tvm.script.tir
def primfunc_global_allocates(placeholder_144: ty.handle, placeholder_145: ty.handle, placeholder_146: ty.handle, T_cast_48: ty.handle) -> None:
    # function attr dict
    tir.func_attr({"global_symbol": "fused_nn_conv2d_add_cast_fixed_point_multiply_clip_cast_cast_13", "tir.noalias": True})
    placeholder_147 = tir.match_buffer(placeholder_144, [1, 14, 14, 512], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    placeholder_148 = tir.match_buffer(placeholder_145, [3, 3, 512, 1], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    placeholder_149 = tir.match_buffer(placeholder_146, [1, 1, 1, 512], dtype="int32", elem_offset=0, align=128, offset_factor=1)
    T_cast_49 = tir.match_buffer(T_cast_48, [1, 14, 14, 512], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    # body
    PaddedInput_22 = tir.allocate([131072], "int16", "global")
    DepthwiseConv2d_9 = tir.allocate([100352], "int32", "global")
    for i1_29, i2_39, i3_40 in tir.grid(16, 16, 512):
        PaddedInput_22[(((i1_29*8192) + (i2_39*512)) + i3_40)] = tir.if_then_else(((((1 <= i1_29) and (i1_29 < 15)) and (1 <= i2_39)) and (i2_39 < 15)), tir.load("int16", placeholder_147.data, ((((i1_29*7168) + (i2_39*512)) + i3_40) - 7680)), tir.int16(0), dtype="int16")
    for i_9, j_9, c_9 in tir.grid(14, 14, 512):
        DepthwiseConv2d_9[(((i_9*7168) + (j_9*512)) + c_9)] = 0
        for di_9, dj_9 in tir.grid(3, 3):
            DepthwiseConv2d_9[(((i_9*7168) + (j_9*512)) + c_9)] = (tir.load("int32", DepthwiseConv2d_9, (((i_9*7168) + (j_9*512)) + c_9)) + (tir.load("int16", PaddedInput_22, (((((i_9*8192) + (di_9*8192)) + (j_9*512)) + (dj_9*512)) + c_9)).astype("int32")*tir.load("int16", placeholder_148.data, (((di_9*1536) + (dj_9*512)) + c_9)).astype("int32")))
    for ax1_27, ax2_28, ax3_30 in tir.grid(14, 14, 512):
        DepthwiseConv2d_9[(((ax1_27*7168) + (ax2_28*512)) + ax3_30)] = (tir.load("int32", DepthwiseConv2d_9, (((ax1_27*7168) + (ax2_28*512)) + ax3_30)) + tir.load("int32", placeholder_149.data, ax3_30))
    for i1_30, i2_40, i3_41 in tir.grid(14, 14, 512):
        DepthwiseConv2d_9[(((i1_30*7168) + (i2_40*512)) + i3_41)] = tir.q_multiply_shift(tir.load("int32", DepthwiseConv2d_9, (((i1_30*7168) + (i2_40*512)) + i3_41)), 1269068532, 31, -4, dtype="int32")
    for i1_31, i2_41, i3_42 in tir.grid(14, 14, 512):
        DepthwiseConv2d_9[(((i1_31*7168) + (i2_41*512)) + i3_42)] = tir.max(tir.max(tir.load("int32", DepthwiseConv2d_9, (((i1_31*7168) + (i2_41*512)) + i3_42)), 255), 0)
    for ax1_28, ax2_29, ax3_31 in tir.grid(14, 14, 512):
        PaddedInput_22[(((ax1_28*7168) + (ax2_29*512)) + ax3_31)] = tir.load("int32", DepthwiseConv2d_9, (((ax1_28*7168) + (ax2_29*512)) + ax3_31)).astype("uint8")
    for ax1_29, ax2_30, ax3_32 in tir.grid(14, 14, 512):
        T_cast_49.data[(((ax1_29*7168) + (ax2_30*512)) + ax3_32)] = tir.load("uint8", PaddedInput_22, (((ax1_29*7168) + (ax2_30*512)) + ax3_32)).astype("int16")
# fmt: on


# fmt: off
@tvm.script.tir
def primfunc_local_allocates(placeholder_162: ty.handle, placeholder_163: ty.handle, placeholder_164: ty.handle, T_cast_76: ty.handle) -> None:
    # function attr dict
    tir.func_attr({"global_symbol": "fused_nn_conv2d_add_cast_fixed_point_multiply_clip_cast_cast_9", "tir.noalias": True})
    placeholder_165 = tir.match_buffer(placeholder_162, [1, 14, 14, 512], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    placeholder_166 = tir.match_buffer(placeholder_163, [3, 3, 512, 1], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    placeholder_167 = tir.match_buffer(placeholder_164, [1, 1, 1, 512], dtype="int32", elem_offset=0, align=128, offset_factor=1)
    T_cast_77 = tir.match_buffer(T_cast_76, [1, 14, 14, 512], dtype="int16", elem_offset=0, align=128, offset_factor=1)
    # body
    PaddedInput_25 = tir.allocate([1, 16, 16, 512], "int16", "global")
    for i1_35, i2_46, i3_47 in tir.grid(16, 16, 512):
        PaddedInput_25[(((i1_35*8192) + (i2_46*512)) + i3_47)] = tir.if_then_else(((((1 <= i1_35) and (i1_35 < 15)) and (1 <= i2_46)) and (i2_46 < 15)), tir.load("int16", placeholder_165.data, ((((i1_35*7168) + (i2_46*512)) + i3_47) - 7680)), tir.int16(0), dtype="int16")
    T_add_11 = tir.allocate([1, 14, 14, 512], "int32", "global")
    with tir.allocate([1, 14, 14, 512], "int32", "global") as DepthwiseConv2d_11:
        for i_11, j_11, c_11 in tir.grid(14, 14, 512):
            DepthwiseConv2d_11[(((i_11*7168) + (j_11*512)) + c_11)] = 0
            for di_11, dj_11 in tir.grid(3, 3):
                DepthwiseConv2d_11[(((i_11*7168) + (j_11*512)) + c_11)] = (tir.load("int32", DepthwiseConv2d_11, (((i_11*7168) + (j_11*512)) + c_11)) + (tir.load("int16", PaddedInput_25, (((((i_11*8192) + (di_11*8192)) + (j_11*512)) + (dj_11*512)) + c_11)).astype("int32")*tir.load("int16", placeholder_166.data, (((di_11*1536) + (dj_11*512)) + c_11)).astype("int32")))
        for ax1_44, ax2_45, ax3_47 in tir.grid(14, 14, 512):
            T_add_11[(((ax1_44*7168) + (ax2_45*512)) + ax3_47)] = (tir.load("int32", DepthwiseConv2d_11, (((ax1_44*7168) + (ax2_45*512)) + ax3_47)) + tir.load("int32", placeholder_167.data, ax3_47))
    compute_22 = tir.allocate([1, 14, 14, 512], "int32", "global")
    with tir.allocate([1, 14, 14, 512], "int32", "global") as T_cast_78:
        for ax1_45, ax2_46, ax3_48 in tir.grid(14, 14, 512):
            T_cast_78[(((ax1_45*7168) + (ax2_46*512)) + ax3_48)] = tir.load("int32", T_add_11, (((ax1_45*7168) + (ax2_46*512)) + ax3_48))
        for i1_36, i2_47, i3_48 in tir.grid(14, 14, 512):
            compute_22[(((i1_36*7168) + (i2_47*512)) + i3_48)] = tir.q_multiply_shift(tir.load("int32", T_cast_78, (((i1_36*7168) + (i2_47*512)) + i3_48)), 1948805937, 31, -5, dtype="int32")
    T_cast_79 = tir.allocate([1, 14, 14, 512], "uint8", "global")
    with tir.allocate([1, 14, 14, 512], "int32", "global") as compute_23:
        for i1_37, i2_48, i3_49 in tir.grid(14, 14, 512):
            compute_23[(((i1_37*7168) + (i2_48*512)) + i3_49)] = tir.max(tir.max(tir.load("int32", compute_22, (((i1_37*7168) + (i2_48*512)) + i3_49)), 255), 0)
        for ax1_46, ax2_47, ax3_49 in tir.grid(14, 14, 512):
            T_cast_79[(((ax1_46*7168) + (ax2_47*512)) + ax3_49)] = tir.load("int32", compute_23, (((ax1_46*7168) + (ax2_47*512)) + ax3_49)).astype("uint8")
    for ax1_47, ax2_48, ax3_50 in tir.grid(14, 14, 512):
        T_cast_77.data[(((ax1_47*7168) + (ax2_48*512)) + ax3_50)] = tir.load("uint8", T_cast_79, (((ax1_47*7168) + (ax2_48*512)) + ax3_50)).astype("int16")
# fmt: on


@pytest.mark.parametrize("alignment_and_size", [(1, 663552), (10, 663560)])
def test_global_allocates(alignment_and_size):
    alignment = alignment_and_size[0]
    size = alignment_and_size[1]
    primfunc = primfunc_global_allocates
    assert tvm.tir.analysis.calculate_workspace_bytes(primfunc, alignment) == size


@pytest.mark.parametrize("alignment_and_size", [(1, 1566720), (100, 1567100)])
def test_local_allocates(alignment_and_size):
    alignment = alignment_and_size[0]
    size = alignment_and_size[1]
    primfunc = primfunc_local_allocates
    assert tvm.tir.analysis.calculate_workspace_bytes(primfunc, alignment) == size


if __name__ == "__main__":
    test_global_allocates()
    test_local_allocates()
