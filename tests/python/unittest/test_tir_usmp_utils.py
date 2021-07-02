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
from tvm import tir, script
from tvm.script import ty
from tvm.tir import stmt_functor


# fmt: off
@tvm.script.tir
class LinearStructure:
    def tvmgen_default_fused_cast_subtract(placeholder_2: ty.handle, placeholder_3: ty.handle, T_subtract: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = tir.match_buffer(placeholder_2, [1, 224, 224, 3], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = tir.match_buffer(placeholder_3, [], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        T_subtract_1 = tir.match_buffer(T_subtract, [1, 224, 224, 3], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in tir.serial(0, 224):
            for ax2_1, ax3_inner_1 in tir.grid(224, 3):
                tir.store(T_subtract_1.data, (((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1), (tir.cast(tir.load("uint8", placeholder_4.data, (((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)), "int16") - tir.load("int16", placeholder_5.data, 0)), True)

    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: ty.handle, placeholder_63: ty.handle, placeholder_64: ty.handle, T_cast_20: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", "tir.noalias": True})
        placeholder_65 = tir.match_buffer(placeholder_62, [1, 224, 224, 3], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_66 = tir.match_buffer(placeholder_63, [7, 7, 3, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_67 = tir.match_buffer(placeholder_64, [1, 1, 1, 64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_21 = tir.match_buffer(T_cast_20, [1, 112, 112, 64], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_7 = tir.allocate([157323], "int16", "global")
        for i0_i1_fused_7 in tir.serial(0, 229):
            for i2_7, i3_7 in tir.grid(229, 3):
                tir.store(PaddedInput_7, (((i0_i1_fused_7*687) + (i2_7*3)) + i3_7), tir.if_then_else(((((2 <= i0_i1_fused_7) and (i0_i1_fused_7 < 226)) and (2 <= i2_7)) and (i2_7 < 226)), tir.load("int16", placeholder_65.data, ((((i0_i1_fused_7*672) + (i2_7*3)) + i3_7) - 1350)), tir.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_7 in tir.serial(0, 12544):
            Conv2dOutput_7 = tir.allocate([64], "int32", "global")
            for ff_3 in tir.serial(0, 64):
                tir.store(Conv2dOutput_7, ff_3, 0, True)
                for ry_2, rx_2, rc_7 in tir.grid(7, 7, 3):
                    tir.store(Conv2dOutput_7, ff_3, (tir.load("int32", Conv2dOutput_7, ff_3) + (tir.cast(tir.load("int16", PaddedInput_7, (((((tir.floordiv(ax0_ax1_fused_ax2_fused_7, 112)*1374) + (ry_2*687)) + (tir.floormod(ax0_ax1_fused_ax2_fused_7, 112)*6)) + (rx_2*3)) + rc_7)), "int32")*tir.cast(tir.load("int16", placeholder_66.data, ((((ry_2*1344) + (rx_2*192)) + (rc_7*64)) + ff_3)), "int32"))), True)
            for ax3_inner_7 in tir.serial(0, 64):
                tir.store(T_cast_21.data, ((ax0_ax1_fused_ax2_fused_7*64) + ax3_inner_7), tir.cast(tir.max(tir.min(tir.q_multiply_shift((tir.load("int32", Conv2dOutput_7, ax3_inner_7) + tir.load("int32", placeholder_67.data, ax3_inner_7)), 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8"), True)

    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: ty.handle, T_cast_6: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = tir.match_buffer(placeholder_28, [1, 112, 112, 64], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_7 = tir.match_buffer(T_cast_6, [1, 56, 56, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_2 = tir.allocate([200704], "uint8", "global")
        for ax0_ax1_fused_4 in tir.serial(0, 56):
            for ax2_4 in tir.serial(0, 56):
                for ax3_init in tir.serial(0, 64):
                    tir.store(tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init), tir.uint8(0), True)
                for rv0_rv1_fused_1, ax3_2 in tir.grid(9, 64):
                    tir.store(tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2), tir.max(tir.load("uint8", tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)), tir.if_then_else(((((ax0_ax1_fused_4*2) + tir.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + tir.floormod(rv0_rv1_fused_1, 3)) < 112)), tir.load("uint8", placeholder_29.data, (((((ax0_ax1_fused_4*14336) + (tir.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (tir.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)), tir.uint8(0), dtype="uint8")), True)
        for ax0_ax1_fused_5 in tir.serial(0, 56):
            for ax2_5, ax3_3 in tir.grid(56, 64):
                tir.store(T_cast_7.data, (((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3), tir.cast(tir.load("uint8", tensor_2, (((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)), "int16"), True)

    def tvmgen_default_run_model(input: ty.handle, output: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "tvmgen_default_run_model", "runner_function": True})
        # body
        tir.attr("default", "device_id", 0)
        tir.attr("default", "device_type", 1)
        sid_9 = tir.allocate([301056], "int8", "global")
        sid_8 = tir.allocate([802816], "int8", "global")
        tir.evaluate(tir.call_extern("tvmgen_default_fused_cast_subtract", input, tir.lookup_param("p0", dtype="handle"), sid_9, dtype="int32"))
        tir.evaluate(tir.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", sid_9, tir.lookup_param("p1", dtype="handle"), tir.lookup_param("p2", dtype="handle"), sid_8, dtype="int32"))
        tir.evaluate(tir.call_extern("tvmgen_default_fused_nn_max_pool2d_cast", sid_8, output, dtype="int32"))
    __tvm_meta__ = None
# fmt: on


def test_create_buffer_info():
    buffer_info_obj = tvm.tir.usmp.BufferInfo("buf1", 256)
    assert buffer_info_obj.name_hint == "buf1"
    assert buffer_info_obj.size_bytes == 256
    # default workspace alignment
    assert buffer_info_obj.alignment == 1

    buffer_info_obj = tvm.tir.usmp.BufferInfo("buf2", 512, 8)
    assert buffer_info_obj.name_hint == "buf2"
    assert buffer_info_obj.size_bytes == 512
    assert buffer_info_obj.alignment == 8


def test_create_array_buffer_info():
    fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
    tir_mod = LinearStructure()
    main_func = tir_mod["tvmgen_default_run_model"]
    buffer_info_map = tvm.tir.usmp.analysis.extract_buffer_info(main_func, tir_mod)
    buffer_info_array = fcreate_array_bi(buffer_info_map)

    current_offset = 0
    offsets = []
    for bi in buffer_info_array:
        bi.set_pool_offsets("global", current_offset)
        offsets.append(current_offset)
        current_offset += bi.size_bytes

    bi_idx = 0
    for _, bi in buffer_info_map.items():
        assert bi.pool_name == "global"
        assert bi.pool_offset == offsets[bi_idx]
        bi_idx += 1


if __name__ == "__main__":
    pytest.main([__file__])
