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
import pytest

import tvm
from tvm import tir
from tvm.script import tir as T


# fmt: off
@T.prim_func
def primfunc_global_allocates(placeholder_144: T.handle, placeholder_145: T.handle, placeholder_146: T.handle, T_cast_48: T.handle) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "fused_nn_conv2d_add_cast_fixed_point_multiply_clip_cast_cast_13", "tir.noalias": True})
    placeholder_147 = T.match_buffer(placeholder_144, [100352], dtype="int16", elem_offset=0, align=64, offset_factor=1)
    placeholder_148 = T.match_buffer(placeholder_145, [4608], dtype="int16", elem_offset=0, align=64, offset_factor=1)
    placeholder_149 = T.match_buffer(placeholder_146, [512], dtype="int32", elem_offset=0, align=64, offset_factor=1)
    T_cast_49 = T.match_buffer(T_cast_48, [100352], dtype="int16", elem_offset=0, align=64, offset_factor=1)
    # body
    PaddedInput_22 = T.decl_buffer([131072], "int16")
    DepthwiseConv2d_9 = T.decl_buffer([100352], "int32")
    for i1_29, i2_39, i3_40 in T.grid(16, 16, 512):
        PaddedInput_22[(((i1_29*8192) + (i2_39*512)) + i3_40)] = T.if_then_else(((((1 <= i1_29) and (i1_29 < 15)) and (1 <= i2_39)) and (i2_39 < 15)), placeholder_147[((((i1_29*7168) + (i2_39*512)) + i3_40) - 7680)], T.int16(0), dtype="int16")
    for i_9, j_9, c_9 in T.grid(14, 14, 512):
        DepthwiseConv2d_9[(((i_9*7168) + (j_9*512)) + c_9)] = 0
        for di_9, dj_9 in T.grid(3, 3):
            DepthwiseConv2d_9[(((i_9*7168) + (j_9*512)) + c_9)] = (DepthwiseConv2d_9[(((i_9*7168) + (j_9*512)) + c_9)] + (PaddedInput_22[(((((i_9*8192) + (di_9*8192)) + (j_9*512)) + (dj_9*512)) + c_9)].astype("int32")*placeholder_148[(((di_9*1536) + (dj_9*512)) + c_9)].astype("int32")))
    for ax1_27, ax2_28, ax3_30 in T.grid(14, 14, 512):
        DepthwiseConv2d_9[(((ax1_27*7168) + (ax2_28*512)) + ax3_30)] = (DepthwiseConv2d_9[(((ax1_27*7168) + (ax2_28*512)) + ax3_30)] + placeholder_149[ax3_30])
    for i1_30, i2_40, i3_41 in T.grid(14, 14, 512):
        DepthwiseConv2d_9[(((i1_30*7168) + (i2_40*512)) + i3_41)] = T.q_multiply_shift(DepthwiseConv2d_9[(((i1_30*7168) + (i2_40*512)) + i3_41)], 1269068532, 31, -4, dtype="int32")
    for i1_31, i2_41, i3_42 in T.grid(14, 14, 512):
        DepthwiseConv2d_9[(((i1_31*7168) + (i2_41*512)) + i3_42)] = T.max(T.max(DepthwiseConv2d_9[(((i1_31*7168) + (i2_41*512)) + i3_42)], 255), 0)
    for ax1_28, ax2_29, ax3_31 in T.grid(14, 14, 512):
        PaddedInput_22[(((ax1_28*7168) + (ax2_29*512)) + ax3_31)] = DepthwiseConv2d_9[(((ax1_28*7168) + (ax2_29*512)) + ax3_31)].astype("uint8")
    for ax1_29, ax2_30, ax3_32 in T.grid(14, 14, 512):
        T_cast_49[(((ax1_29*7168) + (ax2_30*512)) + ax3_32)] = PaddedInput_22[(((ax1_29*7168) + (ax2_30*512)) + ax3_32)].astype("int16")
# fmt: on


# fmt: off
@T.prim_func
def primfunc_local_allocates(placeholder_162: T.handle, placeholder_163: T.handle, placeholder_164: T.handle, T_cast_76: T.handle) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "fused_nn_conv2d_add_cast_fixed_point_multiply_clip_cast_cast_9", "tir.noalias": True})
    placeholder_165 = T.match_buffer(placeholder_162, [100352], dtype="int16", elem_offset=0, align=64, offset_factor=1)
    placeholder_166 = T.match_buffer(placeholder_163, [4608], dtype="int16", elem_offset=0, align=64, offset_factor=1)
    placeholder_167 = T.match_buffer(placeholder_164, [512], dtype="int32", elem_offset=0, align=64, offset_factor=1)
    T_cast_77 = T.match_buffer(T_cast_76, [100352], dtype="int16", elem_offset=0, align=64, offset_factor=1)
    sid_21 = T.allocate_const([0,1,2,3,4,5,6,7], "int8", [8])
    # body
    PaddedInput_25 = T.decl_buffer([131072], "int16")
    for i1_35, i2_46, i3_47 in T.grid(16, 16, 512):
        PaddedInput_25[(((i1_35*8192) + (i2_46*512)) + i3_47)] = T.if_then_else(((((1 <= i1_35) and (i1_35 < 15)) and (1 <= i2_46)) and (i2_46 < 15)), placeholder_165[((((i1_35*7168) + (i2_46*512)) + i3_47) - 7680)], T.int16(0), dtype="int16")
    T_add_11 = T.decl_buffer([100352], "int32")
    with T.decl_buffer([100352], "int32") as DepthwiseConv2d_11:
        for i_11, j_11, c_11 in T.grid(14, 14, 512):
            DepthwiseConv2d_11[(((i_11*7168) + (j_11*512)) + c_11)] = 0
            for di_11, dj_11 in T.grid(3, 3):
                DepthwiseConv2d_11[(((i_11*7168) + (j_11*512)) + c_11)] = (DepthwiseConv2d_11[(((i_11*7168) + (j_11*512)) + c_11)] + (PaddedInput_25[(((((i_11*8192) + (di_11*8192)) + (j_11*512)) + (dj_11*512)) + c_11)].astype("int32")*placeholder_166[(((di_11*1536) + (dj_11*512)) + c_11)].astype("int32")))
        for ax1_44, ax2_45, ax3_47 in T.grid(14, 14, 512):
            T_add_11[(((ax1_44*7168) + (ax2_45*512)) + ax3_47)] = (DepthwiseConv2d_11[(((ax1_44*7168) + (ax2_45*512)) + ax3_47)] + placeholder_167[ax3_47])
    compute_22 = T.decl_buffer([100352], "int32")
    with T.decl_buffer([100352], "int32") as T_cast_78:
        for ax1_45, ax2_46, ax3_48 in T.grid(14, 14, 512):
            T_cast_78[(((ax1_45*7168) + (ax2_46*512)) + ax3_48)] = T_add_11[(((ax1_45*7168) + (ax2_46*512)) + ax3_48)]
        for i1_36, i2_47, i3_48 in T.grid(14, 14, 512):
            compute_22[(((i1_36*7168) + (i2_47*512)) + i3_48)] = T.q_multiply_shift(T_cast_78[(((i1_36*7168) + (i2_47*512)) + i3_48)], 1948805937, 31, -5, dtype="int32")
    T_cast_79 = T.decl_buffer([100352], "uint8")
    with T.decl_buffer([100352], "int32") as compute_23:
        for i1_37, i2_48, i3_49 in T.grid(14, 14, 512):
            compute_23[(((i1_37*7168) + (i2_48*512)) + i3_49)] = T.max(T.max(compute_22[(((i1_37*7168) + (i2_48*512)) + i3_49)], 255), 0)
        for ax1_46, ax2_47, ax3_49 in T.grid(14, 14, 512):
            T_cast_79[(((ax1_46*7168) + (ax2_47*512)) + ax3_49)] = compute_23[(((ax1_46*7168) + (ax2_47*512)) + ax3_49)].astype("uint8")
    for ax1_47, ax2_48, ax3_50 in T.grid(14, 14, 512):
        T_cast_77[(((ax1_47*7168) + (ax2_48*512)) + ax3_50)] = T_cast_79[(((ax1_47*7168) + (ax2_48*512)) + ax3_50)].astype("int16")
# fmt: on


@pytest.mark.parametrize("alignment,size,consts", [(1, 663552, 0), (10, 663560, 0)])
def test_global_allocates(alignment, size, consts):
    primfunc = primfunc_global_allocates
    assert tvm.tir.analysis.calculate_constant_bytes(primfunc, alignment) == consts
    assert tvm.tir.analysis.calculate_workspace_bytes(primfunc, alignment) == size


@pytest.mark.parametrize("alignment,size,consts", [(1, 1566720, 8), (100, 1567100, 100)])
def test_local_allocates(alignment, size, consts):
    primfunc = primfunc_local_allocates
    assert tvm.tir.analysis.calculate_constant_bytes(primfunc, alignment) == consts
    assert tvm.tir.analysis.calculate_workspace_bytes(primfunc, alignment) == size


if __name__ == "__main__":
    test_global_allocates()
    test_local_allocates()
