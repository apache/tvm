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
from tvm.ir import Range
from tvm.script import tir as T
from tvm.tir import stmt_functor
from tvm.tir import PrimFunc
from tvm.tir.usmp import utils as usmp_utils
from tvm.target import Target


def _replace_stmt_with_buf_var_names(buffer_info_map):
    """helper to replace tir.allocates with buffer names"""
    new_buffer_info_map = dict()
    for k, v in buffer_info_map.items():
        new_buffer_info_map[v.buffer_var.name] = k
    return new_buffer_info_map


def _verify_conflicts(main_buf_name, conflicting_buf_names, buffer_info_map):
    """helper to check expected liveness conflicts"""
    buf_info = buffer_info_map[main_buf_name]
    for conflict in buf_info.conflicts:
        assert conflict.name_hint in conflicting_buf_names


def _get_allocates(primfunc):
    """helper to extract all allocate nodes by name"""
    allocates = dict()

    def get_allocate(stmt):
        if isinstance(stmt, tvm.tir.Allocate):
            allocates[str(stmt.buffer_var.name)] = stmt

    stmt_functor.post_order_visit(primfunc.body, get_allocate)
    return allocates


def _assign_poolinfos_to_allocates_in_primfunc(primfunc, pool_infos):
    """helper to assing poolinfos to allocate nodes in a tir.PrimFunc"""

    def set_poolinfos(stmt):
        if isinstance(stmt, tvm.tir.Allocate):
            return tvm.tir.Allocate(
                buffer_var=stmt.buffer_var,
                dtype=stmt.dtype,
                extents=stmt.extents,
                condition=stmt.condition,
                body=stmt.body,
                annotations={tvm.tir.usmp.utils.CANDIDATE_MEMORY_POOL_ATTR: pool_infos},
            )

    return primfunc.with_body(stmt_functor.ir_transform(primfunc.body, None, set_poolinfos))


def _assign_poolinfos_to_allocates_in_irmodule(mod, pool_infos):
    """helper to assing poolinfos to allocate nodes in a IRModule"""
    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, tvm.tir.PrimFunc):
            ret[global_var] = _assign_poolinfos_to_allocates_in_primfunc(basefunc, pool_infos)
    return ret


# fmt: off
@tvm.script.ir_module
class LinearStructure:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [1, 224, 224, 3], dTpe="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [1, 224, 224, 3], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T.store(T_subtract_1.data, (((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1), (T.cast(T.load("uint8", placeholder_4.data, (((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)), "int16") - T.load("int16", placeholder_5.data, 0)), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", "tir.noalias": True})
        placeholder_65 = T.match_buffer(placeholder_62, [1, 224, 224, 3], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_66 = T.match_buffer(placeholder_63, [7, 7, 3, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_67 = T.match_buffer(placeholder_64, [1, 1, 1, 64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_21 = T.match_buffer(T_cast_20, [1, 112, 112, 64], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_7 = T.allocate([157323], "int16", "global")
        for i0_i1_fused_7 in T.serial(0, 229):
            for i2_7, i3_7 in T.grid(229, 3):
                T.store(PaddedInput_7, (((i0_i1_fused_7*687) + (i2_7*3)) + i3_7), T.if_then_else(((((2 <= i0_i1_fused_7) and (i0_i1_fused_7 < 226)) and (2 <= i2_7)) and (i2_7 < 226)), T.load("int16", placeholder_65.data, ((((i0_i1_fused_7*672) + (i2_7*3)) + i3_7) - 1350)), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_7 in T.serial(0, 12544):
            Conv2dOutput_7 = T.allocate([64], "int32", "global")
            for ff_3 in T.serial(0, 64):
                T.store(Conv2dOutput_7, ff_3, 0, True)
                for ry_2, rx_2, rc_7 in T.grid(7, 7, 3):
                    T.store(Conv2dOutput_7, ff_3, (T.load("int32", Conv2dOutput_7, ff_3) + (T.cast(T.load("int16", PaddedInput_7, (((((T.floordiv(ax0_ax1_fused_ax2_fused_7, 112)*1374) + (ry_2*687)) + (T.floormod(ax0_ax1_fused_ax2_fused_7, 112)*6)) + (rx_2*3)) + rc_7)), "int32")*T.cast(T.load("int16", placeholder_66.data, ((((ry_2*1344) + (rx_2*192)) + (rc_7*64)) + ff_3)), "int32"))), True)
            for ax3_inner_7 in T.serial(0, 64):
                T.store(T_cast_21.data, ((ax0_ax1_fused_ax2_fused_7*64) + ax3_inner_7), T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_7, ax3_inner_7) + T.load("int32", placeholder_67.data, ax3_inner_7)), 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_28, [1, 112, 112, 64], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_7 = T.match_buffer(T_cast_6, [1, 56, 56, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_2 = T.allocate([200704], "uint8", "global")
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    T.store(tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init), T.uint8(0), True)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    T.store(tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2), T.max(T.load("uint8", tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)), T.if_then_else(((((ax0_ax1_fused_4*2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)), T.load("uint8", placeholder_29.data, (((((ax0_ax1_fused_4*14336) + (T.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (T.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)), T.uint8(0), dtype="uint8")), True)
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T.store(T_cast_7.data, (((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3), T.cast(T.load("uint8", tensor_2, (((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)), "int16"), True)

    @T.prim_func
    def tvmgen_default_run_model(input: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_run_model", "runner_function": True})
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        sid_9 = T.allocate([301056], "int8", "global")
        sid_8 = T.allocate([802816], "int8", "global")
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input, T.lookup_param("p0", dtype="handle"), sid_9, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", sid_9, T.lookup_param("p1", dtype="handle"), T.lookup_param("p2", dtype="handle"), sid_8, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_max_pool2d_cast", sid_8, output, dtype="int32"))
    __tvm_meta__ = None
# fmt: on


def test_linear():
    fast_memory_pool = usmp_utils.PoolInfo(
        pool_name="fast_memory", target_access={Target("c"): usmp_utils.PoolInfo.READ_WRITE_ACCESS}
    )
    slow_memory_pool = usmp_utils.PoolInfo(
        pool_name="slow_memory", target_access={Target("c"): usmp_utils.PoolInfo.READ_WRITE_ACCESS}
    )
    tir_mod = LinearStructure
    tir_mod = _assign_poolinfos_to_allocates_in_irmodule(
        tir_mod, [fast_memory_pool, slow_memory_pool]
    )
    buffer_info_map = tvm.tir.usmp.analysis.extract_buffer_info(
        tir_mod["tvmgen_default_run_model"], tir_mod
    )
    buffer_info_map = _replace_stmt_with_buf_var_names(buffer_info_map)

    # check conflicts
    _verify_conflicts("sid_8", ["Conv2dOutput_7", "tensor_2"], buffer_info_map)
    _verify_conflicts("Conv2dOutput_7", ["PaddedInput_7", "sid_8"], buffer_info_map)
    _verify_conflicts("PaddedInput_7", ["sid_9", "Conv2dOutput_7"], buffer_info_map)
    _verify_conflicts("tensor_2", ["sid_8"], buffer_info_map)
    _verify_conflicts("sid_9", ["PaddedInput_7"], buffer_info_map)

    # check sizes
    assert buffer_info_map["sid_8"].size_bytes == 802816
    assert buffer_info_map["Conv2dOutput_7"].size_bytes == 256
    assert buffer_info_map["PaddedInput_7"].size_bytes == 314646
    assert buffer_info_map["tensor_2"].size_bytes == 200704
    assert buffer_info_map["sid_9"].size_bytes == 301056

    # check_pool_candidates
    assert [
        pool_info.pool_name for pool_info in list(buffer_info_map["sid_8"].pool_candidates)
    ] == ["fast_memory", "slow_memory"]


# fmt: off
@tvm.script.ir_module
class ParallelSerialMixedForLoops:
    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1(placeholder_68: T.handle, placeholder_69: T.handle, placeholder_70: T.handle, T_cast_22: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", "tir.noalias": True})
        placeholder_71 = T.match_buffer(placeholder_68, [1, 56, 56, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_72 = T.match_buffer(placeholder_69, [3, 3, 64, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_73 = T.match_buffer(placeholder_70, [1, 1, 1, 192], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_23 = T.match_buffer(T_cast_22, [1, 56, 56, 192], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_8 = T.allocate([215296], "int16", "global")
        for i0_i1_fused_8 in T.serial(0, 58):
            for i2_8, i3_8 in T.grid(58, 64):
                T.store(PaddedInput_8, (((i0_i1_fused_8*3712) + (i2_8*64)) + i3_8), T.if_then_else(((((1 <= i0_i1_fused_8) and (i0_i1_fused_8 < 57)) and (1 <= i2_8)) and (i2_8 < 57)), T.load("int16", placeholder_71.data, ((((i0_i1_fused_8*3584) + (i2_8*64)) + i3_8) - 3648)), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_8 in T.parallel(0, 3136):
            dummy_allocate = T.allocate([1], "int32", "global")
            for ax3_outer_4 in T.serial(0, 3):
                Conv2dOutput_8 = T.allocate([64], "int32", "global")
                for ff_4 in T.serial(0, 64):
                    T.store(Conv2dOutput_8, ff_4, 0, True)
                    for ry_3, rx_3, rc_8 in T.grid(3, 3, 64):
                        T.store(Conv2dOutput_8, ff_4, (T.load("int32", Conv2dOutput_8, ff_4) + (T.cast(T.load("int16", PaddedInput_8, (((((T.floordiv(ax0_ax1_fused_ax2_fused_8, 56)*3712) + (ry_3*3712)) + (rx_3*64)) + (T.floormod(ax0_ax1_fused_ax2_fused_8, 56)*64)) + rc_8)), "int32")*T.cast(T.load("int16", placeholder_72.data, (((((ry_3*36864) + (rx_3*12288)) + (rc_8*192)) + (ax3_outer_4*64)) + ff_4)), "int32"))), True)
                for ax3_inner_8 in T.serial(0, 64):
                    T.store(T_cast_23.data, (((ax0_ax1_fused_ax2_fused_8*192) + (ax3_outer_4*64)) + ax3_inner_8), T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_8, ax3_inner_8) + T.load("int32", placeholder_73.data, ((ax3_outer_4*64) + ax3_inner_8))), 1139793473, 31, -6, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_run_model(input: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_run_model", "runner_function": True})
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", input, T.lookup_param("p5", dtype="handle"), T.lookup_param("p6", dtype="handle"), output, dtype="int32"))


__tvm_meta__ = None
# fmt: on


# fmt: off
@tvm.script.ir_module
class AllSerialForLoops:
    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1(placeholder_68: T.handle, placeholder_69: T.handle, placeholder_70: T.handle, T_cast_22: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", "tir.noalias": True})
        placeholder_71 = T.match_buffer(placeholder_68, [1, 56, 56, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_72 = T.match_buffer(placeholder_69, [3, 3, 64, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_73 = T.match_buffer(placeholder_70, [1, 1, 1, 192], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_23 = T.match_buffer(T_cast_22, [1, 56, 56, 192], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_8 = T.allocate([215296], "int16", "global")
        for i0_i1_fused_8 in T.serial(0, 58):
            for i2_8, i3_8 in T.grid(58, 64):
                T.store(PaddedInput_8, (((i0_i1_fused_8*3712) + (i2_8*64)) + i3_8), T.if_then_else(((((1 <= i0_i1_fused_8) and (i0_i1_fused_8 < 57)) and (1 <= i2_8)) and (i2_8 < 57)), T.load("int16", placeholder_71.data, ((((i0_i1_fused_8*3584) + (i2_8*64)) + i3_8) - 3648)), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_8 in T.serial(0, 3136):
            dummy_allocate = T.allocate([1], "int32", "global")
            for ax3_outer_4 in T.serial(0, 3):
                Conv2dOutput_8 = T.allocate([64], "int32", "global")
                for ff_4 in T.serial(0, 64):
                    T.store(Conv2dOutput_8, ff_4, 0, True)
                    for ry_3, rx_3, rc_8 in T.grid(3, 3, 64):
                        T.store(Conv2dOutput_8, ff_4, (T.load("int32", Conv2dOutput_8, ff_4) + (T.cast(T.load("int16", PaddedInput_8, (((((T.floordiv(ax0_ax1_fused_ax2_fused_8, 56)*3712) + (ry_3*3712)) + (rx_3*64)) + (T.floormod(ax0_ax1_fused_ax2_fused_8, 56)*64)) + rc_8)), "int32")*T.cast(T.load("int16", placeholder_72.data, (((((ry_3*36864) + (rx_3*12288)) + (rc_8*192)) + (ax3_outer_4*64)) + ff_4)), "int32"))), True)
                for ax3_inner_8 in T.serial(0, 64):
                    T.store(T_cast_23.data, (((ax0_ax1_fused_ax2_fused_8*192) + (ax3_outer_4*64)) + ax3_inner_8), T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_8, ax3_inner_8) + T.load("int32", placeholder_73.data, ((ax3_outer_4*64) + ax3_inner_8))), 1139793473, 31, -6, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_run_model(input: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_run_model", "runner_function": True})
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", input, T.lookup_param("p5", dtype="handle"), T.lookup_param("p6", dtype="handle"), output, dtype="int32"))


__tvm_meta__ = None
# fmt: on


def test_parallel_serial_mixed_for_loops():
    global_ws_pool = usmp_utils.PoolInfo(
        pool_name="global_workspace",
        target_access={Target("c"): usmp_utils.PoolInfo.READ_WRITE_ACCESS},
    )
    all_serial_tir_mod = AllSerialForLoops
    all_serial_tir_mod = _assign_poolinfos_to_allocates_in_irmodule(
        all_serial_tir_mod, [global_ws_pool]
    )
    main_func = all_serial_tir_mod["tvmgen_default_run_model"]
    buffer_info_map = tvm.tir.usmp.analysis.extract_buffer_info(main_func, all_serial_tir_mod)
    buffer_info_map = _replace_stmt_with_buf_var_names(buffer_info_map)

    # When all loops are serial all allocates are touched by USMP
    assert len(buffer_info_map) == 3
    for name, _ in buffer_info_map.items():
        assert name in ["dummy_allocate", "Conv2dOutput_8", "PaddedInput_8"]

    parallel_serial_mixed_tir_mod = ParallelSerialMixedForLoops
    parallel_serial_mixed_tir_mod = _assign_poolinfos_to_allocates_in_irmodule(
        parallel_serial_mixed_tir_mod, [global_ws_pool]
    )
    main_func = parallel_serial_mixed_tir_mod["tvmgen_default_run_model"]
    buffer_info_map = tvm.tir.usmp.analysis.extract_buffer_info(
        main_func, parallel_serial_mixed_tir_mod
    )
    buffer_info_map = _replace_stmt_with_buf_var_names(buffer_info_map)

    # USMP will not touch (yet) the allocates inside parallel for loops
    assert len(buffer_info_map) == 2
    for name, _ in buffer_info_map.items():
        assert name in ["Conv2dOutput_8", "PaddedInput_8"]


# fmt: off
@tvm.script.ir_module
class InceptionStructure:
    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d(placeholder: T.handle, tensor: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d", "tir.noalias": True})
        placeholder_1 = T.match_buffer(placeholder, [1, 56, 56, 192], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        tensor_1 = T.match_buffer(tensor, [1, 28, 28, 192], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused in T.serial(0, 28):
            for ax2 in T.serial(0, 28):
                for ax3_outer_init, ax3_inner_init in T.grid(3, 64):
                    T.store(tensor_1.data, ((((ax0_ax1_fused*5376) + (ax2*192)) + (ax3_outer_init*64)) + ax3_inner_init), T.uint8(0), True)
                for rv0_rv1_fused, ax3_outer, ax3_inner in T.grid(9, 3, 64):
                    T.store(tensor_1.data, ((((ax0_ax1_fused*5376) + (ax2*192)) + (ax3_outer*64)) + ax3_inner), T.max(T.load("uint8", tensor_1.data, ((((ax0_ax1_fused*5376) + (ax2*192)) + (ax3_outer*64)) + ax3_inner)), T.if_then_else(((((ax0_ax1_fused*2) + T.floordiv(rv0_rv1_fused, 3)) < 56) and (((ax2*2) + T.floormod(rv0_rv1_fused, 3)) < 56)), T.load("uint8", placeholder_1.data, ((((((ax0_ax1_fused*21504) + (T.floordiv(rv0_rv1_fused, 3)*10752)) + (ax2*384)) + (T.floormod(rv0_rv1_fused, 3)*192)) + (ax3_outer*64)) + ax3_inner)), T.uint8(0), dtype="uint8")), True)

    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [1, 224, 224, 3], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [1, 224, 224, 3], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T.store(T_subtract_1.data, (((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1), (T.cast(T.load("uint8", placeholder_4.data, (((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)), "int16") - T.load("int16", placeholder_5.data, 0)), True)

    @T.prim_func
    def tvmgen_default_fused_cast(placeholder_6: T.handle, T_cast: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast", "tir.noalias": True})
        placeholder_7 = T.match_buffer(placeholder_6, [1, 28, 28, 192], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_1 = T.match_buffer(T_cast, [1, 28, 28, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_2 in T.serial(0, 28):
            for ax2_2, ax3_outer_1, ax3_inner_2 in T.grid(28, 12, 16):
                T.store(T_cast_1.data, ((((ax0_ax1_fused_2*5376) + (ax2_2*192)) + (ax3_outer_1*16)) + ax3_inner_2), T.cast(T.load("uint8", placeholder_7.data, ((((ax0_ax1_fused_2*5376) + (ax2_2*192)) + (ax3_outer_1*16)) + ax3_inner_2)), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_concatenate(placeholder_8: T.handle, placeholder_9: T.handle, placeholder_10: T.handle, placeholder_11: T.handle, T_concat: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_concatenate", "tir.noalias": True})
        placeholder_12 = T.match_buffer(placeholder_8, [1, 28, 28, 64], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_concat_1 = T.match_buffer(T_concat, [1, 28, 28, 256], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_13 = T.match_buffer(placeholder_9, [1, 28, 28, 128], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_14 = T.match_buffer(placeholder_11, [1, 28, 28, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_15 = T.match_buffer(placeholder_10, [1, 28, 28, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        for ax0_ax1_fused_3 in T.serial(0, 28):
            for ax2_3, ax3 in T.grid(28, 256):
                T.store(T_concat_1.data, (((ax0_ax1_fused_3*7168) + (ax2_3*256)) + ax3), T.if_then_else((224 <= ax3), T.load("uint8", placeholder_14.data, ((((ax0_ax1_fused_3*896) + (ax2_3*32)) + ax3) - 224)), T.if_then_else((192 <= ax3), T.load("uint8", placeholder_15.data, ((((ax0_ax1_fused_3*896) + (ax2_3*32)) + ax3) - 192)), T.if_then_else((64 <= ax3), T.load("uint8", placeholder_13.data, ((((ax0_ax1_fused_3*3584) + (ax2_3*128)) + ax3) - 64)), T.load("uint8", placeholder_12.data, (((ax0_ax1_fused_3*1792) + (ax2_3*64)) + ax3)), dtype="uint8"), dtype="uint8"), dtype="uint8"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast(placeholder_16: T.handle, placeholder_17: T.handle, placeholder_18: T.handle, T_cast_2: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", "tir.noalias": True})
        placeholder_19 = T.match_buffer(placeholder_16, [1, 56, 56, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_20 = T.match_buffer(placeholder_17, [1, 1, 64, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_21 = T.match_buffer(placeholder_18, [1, 1, 1, 64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_3 = T.match_buffer(T_cast_2, [1, 56, 56, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput = T.allocate([200704], "int16", "global")
        for i0_i1_fused in T.serial(0, 56):
            for i2, i3 in T.grid(56, 64):
                T.store(PaddedInput, (((i0_i1_fused*3584) + (i2*64)) + i3), T.load("int16", placeholder_19.data, (((i0_i1_fused*3584) + (i2*64)) + i3)), True)
        for ax0_ax1_fused_ax2_fused in T.serial(0, 3136):
            Conv2dOutput = T.allocate([64], "int32", "global")
            for ff in T.serial(0, 64):
                T.store(Conv2dOutput, ff, 0, True)
                for rc in T.serial(0, 64):
                    T.store(Conv2dOutput, ff, (T.load("int32", Conv2dOutput, ff) + (T.cast(T.load("int16", PaddedInput, ((ax0_ax1_fused_ax2_fused*64) + rc)), "int32")*T.cast(T.load("int16", placeholder_20.data, ((rc*64) + ff)), "int32"))), True)
            for ax3_inner_3 in T.serial(0, 64):
                T.store(T_cast_3.data, ((ax0_ax1_fused_ax2_fused*64) + ax3_inner_3), T.cast(T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput, ax3_inner_3) + T.load("int32", placeholder_21.data, ax3_inner_3)), 1191576922, 31, -4, dtype="int32"), 255), 0), "uint8"), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1(placeholder_22: T.handle, placeholder_23: T.handle, placeholder_24: T.handle, T_cast_4: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", "tir.noalias": True})
        placeholder_25 = T.match_buffer(placeholder_22, [1, 28, 28, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_26 = T.match_buffer(placeholder_23, [1, 1, 192, 96], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_27 = T.match_buffer(placeholder_24, [1, 1, 1, 96], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_5 = T.match_buffer(T_cast_4, [1, 28, 28, 96], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_1 = T.allocate([150528], "int16", "global")
        for i0_i1_fused_1 in T.serial(0, 28):
            for i2_1, i3_1 in T.grid(28, 192):
                T.store(PaddedInput_1, (((i0_i1_fused_1*5376) + (i2_1*192)) + i3_1), T.load("int16", placeholder_25.data, (((i0_i1_fused_1*5376) + (i2_1*192)) + i3_1)), True)
        for ax0_ax1_fused_ax2_fused_1 in T.serial(0, 784):
            Conv2dOutput_1 = T.allocate([1], "int32", "global")
            for ax3_1 in T.serial(0, 96):
                T.store(Conv2dOutput_1, 0, 0, True)
                for rc_1 in T.serial(0, 192):
                    T.store(Conv2dOutput_1, 0, (T.load("int32", Conv2dOutput_1, 0) + (T.cast(T.load("int16", PaddedInput_1, ((ax0_ax1_fused_ax2_fused_1*192) + rc_1)), "int32")*T.cast(T.load("int16", placeholder_26.data, ((rc_1*96) + ax3_1)), "int32"))), True)
                T.store(T_cast_5.data, ((ax0_ax1_fused_ax2_fused_1*96) + ax3_1), T.cast(T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_1, 0) + T.load("int32", placeholder_27.data, ax3_1)), 1201322342, 31, -6, dtype="int32"), 255), 0), "uint8"), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_28, [1, 112, 112, 64], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_7 = T.match_buffer(T_cast_6, [1, 56, 56, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_2 = T.allocate([200704], "uint8", "global")
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    T.store(tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init), T.uint8(0), True)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    T.store(tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2), T.max(T.load("uint8", tensor_2, (((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)), T.if_then_else(((((ax0_ax1_fused_4*2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)), T.load("uint8", placeholder_29.data, (((((ax0_ax1_fused_4*14336) + (T.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (T.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)), T.uint8(0), dtype="uint8")), True)
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T.store(T_cast_7.data, (((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3), T.cast(T.load("uint8", tensor_2, (((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_2(placeholder_30: T.handle, placeholder_31: T.handle, placeholder_32: T.handle, T_cast_8: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_2", "tir.noalias": True})
        placeholder_33 = T.match_buffer(placeholder_30, [1, 28, 28, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_34 = T.match_buffer(placeholder_31, [1, 1, 192, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_35 = T.match_buffer(placeholder_32, [1, 1, 1, 64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_9 = T.match_buffer(T_cast_8, [1, 28, 28, 64], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_2 = T.allocate([150528], "int16", "global")
        for i0_i1_fused_2 in T.serial(0, 28):
            for i2_2, i3_2 in T.grid(28, 192):
                T.store(PaddedInput_2, (((i0_i1_fused_2*5376) + (i2_2*192)) + i3_2), T.load("int16", placeholder_33.data, (((i0_i1_fused_2*5376) + (i2_2*192)) + i3_2)), True)
        for ax0_ax1_fused_ax2_fused_2 in T.serial(0, 784):
            Conv2dOutput_2 = T.allocate([64], "int32", "global")
            for ff_1 in T.serial(0, 64):
                T.store(Conv2dOutput_2, ff_1, 0, True)
                for rc_2 in T.serial(0, 192):
                    T.store(Conv2dOutput_2, ff_1, (T.load("int32", Conv2dOutput_2, ff_1) + (T.cast(T.load("int16", PaddedInput_2, ((ax0_ax1_fused_ax2_fused_2*192) + rc_2)), "int32")*T.cast(T.load("int16", placeholder_34.data, ((rc_2*64) + ff_1)), "int32"))), True)
            for ax3_inner_4 in T.serial(0, 64):
                T.store(T_cast_9.data, ((ax0_ax1_fused_ax2_fused_2*64) + ax3_inner_4), T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_2, ax3_inner_4) + T.load("int32", placeholder_35.data, ax3_inner_4)), 1663316467, 31, -7, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast_1(placeholder_36: T.handle, T_cast_10: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast_1", "tir.noalias": True})
        placeholder_37 = T.match_buffer(placeholder_36, [1, 28, 28, 192], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_cast_11 = T.match_buffer(T_cast_10, [1, 28, 28, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        tensor_3 = T.allocate([150528], "uint8", "global")
        for ax0_ax1_fused_6 in T.serial(0, 28):
            for ax2_6 in T.serial(0, 28):
                for ax3_outer_init_1, ax3_inner_init_1 in T.grid(3, 64):
                    T.store(tensor_3, ((((ax0_ax1_fused_6*5376) + (ax2_6*192)) + (ax3_outer_init_1*64)) + ax3_inner_init_1), T.uint8(0), True)
                for rv0_rv1_fused_2, ax3_outer_2, ax3_inner_5 in T.grid(9, 3, 64):
                    T.store(tensor_3, ((((ax0_ax1_fused_6*5376) + (ax2_6*192)) + (ax3_outer_2*64)) + ax3_inner_5), T.max(T.load("uint8", tensor_3, ((((ax0_ax1_fused_6*5376) + (ax2_6*192)) + (ax3_outer_2*64)) + ax3_inner_5)), T.if_then_else(((((1 <= (T.floordiv(rv0_rv1_fused_2, 3) + ax0_ax1_fused_6)) and ((T.floordiv(rv0_rv1_fused_2, 3) + ax0_ax1_fused_6) < 29)) and (1 <= (ax2_6 + T.floormod(rv0_rv1_fused_2, 3)))) and ((ax2_6 + T.floormod(rv0_rv1_fused_2, 3)) < 29)), T.load("uint8", placeholder_37.data, (((((((T.floordiv(rv0_rv1_fused_2, 3)*5376) + (ax0_ax1_fused_6*5376)) + (ax2_6*192)) + (T.floormod(rv0_rv1_fused_2, 3)*192)) + (ax3_outer_2*64)) + ax3_inner_5) - 5568)), T.uint8(0), dtype="uint8")), True)
        for ax0_ax1_fused_7 in T.serial(0, 28):
            for ax2_7, ax3_4 in T.grid(28, 192):
                T.store(T_cast_11.data, (((ax0_ax1_fused_7*5376) + (ax2_7*192)) + ax3_4), T.cast(T.load("uint8", tensor_3, (((ax0_ax1_fused_7*5376) + (ax2_7*192)) + ax3_4)), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__2(placeholder_38: T.handle, placeholder_39: T.handle, placeholder_40: T.handle, T_cast_12: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__2", "tir.noalias": True})
        placeholder_41 = T.match_buffer(placeholder_38, [1, 28, 28, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_42 = T.match_buffer(placeholder_39, [1, 1, 192, 32], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_43 = T.match_buffer(placeholder_40, [1, 1, 1, 32], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_13 = T.match_buffer(T_cast_12, [1, 28, 28, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_3 = T.allocate([150528], "int16", "global")
        for i0_i1_fused_3 in T.serial(0, 28):
            for i2_3, i3_3 in T.grid(28, 192):
                T.store(PaddedInput_3, (((i0_i1_fused_3*5376) + (i2_3*192)) + i3_3), T.load("int16", placeholder_41.data, (((i0_i1_fused_3*5376) + (i2_3*192)) + i3_3)), True)
        for ax0_ax1_fused_ax2_fused_3 in T.serial(0, 784):
            Conv2dOutput_3 = T.allocate([1], "int32", "global")
            for ax3_5 in T.serial(0, 32):
                T.store(Conv2dOutput_3, 0, 0, True)
                for rc_3 in T.serial(0, 192):
                    T.store(Conv2dOutput_3, 0, (T.load("int32", Conv2dOutput_3, 0) + (T.cast(T.load("int16", PaddedInput_3, ((ax0_ax1_fused_ax2_fused_3*192) + rc_3)), "int32")*T.cast(T.load("int16", placeholder_42.data, ((rc_3*32) + ax3_5)), "int32"))), True)
                T.store(T_cast_13.data, ((ax0_ax1_fused_ax2_fused_3*32) + ax3_5), T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_3, 0) + T.load("int32", placeholder_43.data, ax3_5)), 1811141736, 31, -6, dtype="int32"), 255), 0), "uint8"), "int32"), 1136333842, 31, 0, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2(placeholder_44: T.handle, placeholder_45: T.handle, placeholder_46: T.handle, T_cast_14: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2", "tir.noalias": True})
        placeholder_47 = T.match_buffer(placeholder_44, [1, 28, 28, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_48 = T.match_buffer(placeholder_45, [1, 1, 192, 16], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_49 = T.match_buffer(placeholder_46, [1, 1, 1, 16], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_15 = T.match_buffer(T_cast_14, [1, 28, 28, 16], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_4 = T.allocate([150528], "int16", "global")
        for i0_i1_fused_4 in T.serial(0, 28):
            for i2_4, i3_4 in T.grid(28, 192):
                T.store(PaddedInput_4, (((i0_i1_fused_4*5376) + (i2_4*192)) + i3_4), T.load("int16", placeholder_47.data, (((i0_i1_fused_4*5376) + (i2_4*192)) + i3_4)), True)
        for ax0_ax1_fused_ax2_fused_4 in T.serial(0, 784):
            Conv2dOutput_4 = T.allocate([1], "int32", "global")
            for ax3_6 in T.serial(0, 16):
                T.store(Conv2dOutput_4, 0, 0, True)
                for rc_4 in T.serial(0, 192):
                    T.store(Conv2dOutput_4, 0, (T.load("int32", Conv2dOutput_4, 0) + (T.cast(T.load("int16", PaddedInput_4, ((ax0_ax1_fused_ax2_fused_4*192) + rc_4)), "int32")*T.cast(T.load("int16", placeholder_48.data, ((rc_4*16) + ax3_6)), "int32"))), True)
                T.store(T_cast_15.data, ((ax0_ax1_fused_ax2_fused_4*16) + ax3_6), T.cast(T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_4, 0) + T.load("int32", placeholder_49.data, ax3_6)), 1764006585, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__1(placeholder_50: T.handle, placeholder_51: T.handle, placeholder_52: T.handle, T_cast_16: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__1", "tir.noalias": True})
        placeholder_53 = T.match_buffer(placeholder_50, [1, 28, 28, 16], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_54 = T.match_buffer(placeholder_51, [3, 3, 16, 32], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_55 = T.match_buffer(placeholder_52, [1, 1, 1, 32], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_17 = T.match_buffer(T_cast_16, [1, 28, 28, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_5 = T.allocate([14400], "int16", "global")
        for i0_i1_fused_5 in T.serial(0, 30):
            for i2_5, i3_5 in T.grid(30, 16):
                T.store(PaddedInput_5, (((i0_i1_fused_5*480) + (i2_5*16)) + i3_5), T.if_then_else(((((1 <= i0_i1_fused_5) and (i0_i1_fused_5 < 29)) and (1 <= i2_5)) and (i2_5 < 29)), T.load("int16", placeholder_53.data, ((((i0_i1_fused_5*448) + (i2_5*16)) + i3_5) - 464)), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_5 in T.serial(0, 784):
            Conv2dOutput_5 = T.allocate([1], "int32", "global")
            for ax3_7 in T.serial(0, 32):
                T.store(Conv2dOutput_5, 0, 0, True)
                for ry, rx, rc_5 in T.grid(3, 3, 16):
                    T.store(Conv2dOutput_5, 0, (T.load("int32", Conv2dOutput_5, 0) + (T.cast(T.load("int16", PaddedInput_5, (((((T.floordiv(ax0_ax1_fused_ax2_fused_5, 28)*480) + (ry*480)) + (rx*16)) + (T.floormod(ax0_ax1_fused_ax2_fused_5, 28)*16)) + rc_5)), "int32")*T.cast(T.load("int16", placeholder_54.data, ((((ry*1536) + (rx*512)) + (rc_5*32)) + ax3_7)), "int32"))), True)
                T.store(T_cast_17.data, ((ax0_ax1_fused_ax2_fused_5*32) + ax3_7), T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_5, 0) + T.load("int32", placeholder_55.data, ax3_7)), 1131968888, 31, -6, dtype="int32"), 255), 0), "uint8"), "int32"), 1900719667, 31, 0, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320_(placeholder_56: T.handle, placeholder_57: T.handle, placeholder_58: T.handle, T_cast_18: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320_", "tir.noalias": True})
        placeholder_59 = T.match_buffer(placeholder_56, [1, 28, 28, 96], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_60 = T.match_buffer(placeholder_57, [3, 3, 96, 128], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_61 = T.match_buffer(placeholder_58, [1, 1, 1, 128], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_19 = T.match_buffer(T_cast_18, [1, 28, 28, 128], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_6 = T.allocate([86400], "int16", "global")
        for i0_i1_fused_6 in T.serial(0, 30):
            for i2_6, i3_6 in T.grid(30, 96):
                T.store(PaddedInput_6, (((i0_i1_fused_6*2880) + (i2_6*96)) + i3_6), T.if_then_else(((((1 <= i0_i1_fused_6) and (i0_i1_fused_6 < 29)) and (1 <= i2_6)) and (i2_6 < 29)), T.load("int16", placeholder_59.data, ((((i0_i1_fused_6*2688) + (i2_6*96)) + i3_6) - 2784)), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_6 in T.serial(0, 784):
            Conv2dOutput_6 = T.allocate([64], "int32", "global")
            for ax3_outer_3 in T.serial(0, 2):
                for ff_2 in T.serial(0, 64):
                    T.store(Conv2dOutput_6, ff_2, 0, True)
                    for ry_1, rx_1, rc_6 in T.grid(3, 3, 96):
                        T.store(Conv2dOutput_6, ff_2, (T.load("int32", Conv2dOutput_6, ff_2) + (T.cast(T.load("int16", PaddedInput_6, (((((T.floordiv(ax0_ax1_fused_ax2_fused_6, 28)*2880) + (ry_1*2880)) + (rx_1*96)) + (T.floormod(ax0_ax1_fused_ax2_fused_6, 28)*96)) + rc_6)), "int32")*T.cast(T.load("int16", placeholder_60.data, (((((ry_1*36864) + (rx_1*12288)) + (rc_6*128)) + (ax3_outer_3*64)) + ff_2)), "int32"))), True)
                for ax3_inner_6 in T.serial(0, 64):
                    T.store(T_cast_19.data, (((ax0_ax1_fused_ax2_fused_6*128) + (ax3_outer_3*64)) + ax3_inner_6), T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_6, ax3_inner_6) + T.load("int32", placeholder_61.data, ((ax3_outer_3*64) + ax3_inner_6))), 1374050734, 31, -7, dtype="int32"), 255), 0), "uint8"), "int32"), 1544713713, 31, 0, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", "T.noalias": True})
        placeholder_65 = T.match_buffer(placeholder_62, [1, 224, 224, 3], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_66 = T.match_buffer(placeholder_63, [7, 7, 3, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_67 = T.match_buffer(placeholder_64, [1, 1, 1, 64], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_21 = T.match_buffer(T_cast_20, [1, 112, 112, 64], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_7 = T.allocate([157323], "int16", "global")
        for i0_i1_fused_7 in T.serial(0, 229):
            for i2_7, i3_7 in T.grid(229, 3):
                T.store(PaddedInput_7, (((i0_i1_fused_7*687) + (i2_7*3)) + i3_7), T.if_then_else(((((2 <= i0_i1_fused_7) and (i0_i1_fused_7 < 226)) and (2 <= i2_7)) and (i2_7 < 226)), T.load("int16", placeholder_65.data, ((((i0_i1_fused_7*672) + (i2_7*3)) + i3_7) - 1350)), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_7 in T.serial(0, 12544):
            Conv2dOutput_7 = T.allocate([64], "int32", "global")
            for ff_3 in T.serial(0, 64):
                T.store(Conv2dOutput_7, ff_3, 0, True)
                for ry_2, rx_2, rc_7 in T.grid(7, 7, 3):
                    T.store(Conv2dOutput_7, ff_3, (T.load("int32", Conv2dOutput_7, ff_3) + (T.cast(T.load("int16", PaddedInput_7, (((((T.floordiv(ax0_ax1_fused_ax2_fused_7, 112)*1374) + (ry_2*687)) + (T.floormod(ax0_ax1_fused_ax2_fused_7, 112)*6)) + (rx_2*3)) + rc_7)), "int32")*T.cast(T.load("int16", placeholder_66.data, ((((ry_2*1344) + (rx_2*192)) + (rc_7*64)) + ff_3)), "int32"))), True)
            for ax3_inner_7 in T.serial(0, 64):
                T.store(T_cast_21.data, ((ax0_ax1_fused_ax2_fused_7*64) + ax3_inner_7), T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_7, ax3_inner_7) + T.load("int32", placeholder_67.data, ax3_inner_7)), 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1(placeholder_68: T.handle, placeholder_69: T.handle, placeholder_70: T.handle, T_cast_22: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", "tir.noalias": True})
        placeholder_71 = T.match_buffer(placeholder_68, [1, 56, 56, 64], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_72 = T.match_buffer(placeholder_69, [3, 3, 64, 192], dtype="int16", elem_offset=0, align=128, offset_factor=1)
        placeholder_73 = T.match_buffer(placeholder_70, [1, 1, 1, 192], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        T_cast_23 = T.match_buffer(T_cast_22, [1, 56, 56, 192], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        PaddedInput_8 = T.allocate([215296], "int16", "global")
        for i0_i1_fused_8 in T.serial(0, 58):
            for i2_8, i3_8 in T.grid(58, 64):
                T.store(PaddedInput_8, (((i0_i1_fused_8*3712) + (i2_8*64)) + i3_8), T.if_then_else(((((1 <= i0_i1_fused_8) and (i0_i1_fused_8 < 57)) and (1 <= i2_8)) and (i2_8 < 57)), T.load("int16", placeholder_71.data, ((((i0_i1_fused_8*3584) + (i2_8*64)) + i3_8) - 3648)), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_8 in T.serial(0, 3136):
            Conv2dOutput_8 = T.allocate([64], "int32", "global")
            for ax3_outer_4 in T.serial(0, 3):
                for ff_4 in T.serial(0, 64):
                    T.store(Conv2dOutput_8, ff_4, 0, True)
                    for ry_3, rx_3, rc_8 in T.grid(3, 3, 64):
                        T.store(Conv2dOutput_8, ff_4, (T.load("int32", Conv2dOutput_8, ff_4) + (T.cast(T.load("int16", PaddedInput_8, (((((T.floordiv(ax0_ax1_fused_ax2_fused_8, 56)*3712) + (ry_3*3712)) + (rx_3*64)) + (T.floormod(ax0_ax1_fused_ax2_fused_8, 56)*64)) + rc_8)), "int32")*T.cast(T.load("int16", placeholder_72.data, (((((ry_3*36864) + (rx_3*12288)) + (rc_8*192)) + (ax3_outer_4*64)) + ff_4)), "int32"))), True)
                for ax3_inner_8 in T.serial(0, 64):
                    T.store(T_cast_23.data, (((ax0_ax1_fused_ax2_fused_8*192) + (ax3_outer_4*64)) + ax3_inner_8), T.cast(T.max(T.min(T.q_multiply_shift((T.load("int32", Conv2dOutput_8, ax3_inner_8) + T.load("int32", placeholder_73.data, ((ax3_outer_4*64) + ax3_inner_8))), 1139793473, 31, -6, dtype="int32"), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_run_model(input: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_run_model", "runner_function": True})
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        sid_32 = T.allocate([301056], "int8", "global")
        sid_20 = T.allocate([150528], "int8", "global")
        sid_6 = T.allocate([401408], "int8", "global")
        sid_9 = T.allocate([301056], "int8", "global")
        sid_7 = T.allocate([401408], "int8", "global")
        sid_8 = T.allocate([802816], "int8", "global")
        sid_2 = T.allocate([50176], "int8", "global")
        sid_3 = T.allocate([301056], "int8", "global")
        sid_19 = T.allocate([100352], "int8", "global")
        sid_4 = T.allocate([150528], "int8", "global")
        sid_5 = T.allocate([602112], "int8", "global")
        sid_25 = T.allocate([25088], "int8", "global")
        sid_26 = T.allocate([25088], "int8", "global")
        sid_31 = T.allocate([25088], "int8", "global")
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input, T.lookup_param("p0", dtype="handle"), sid_9, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", sid_9, T.lookup_param("p1", dtype="handle"), T.lookup_param("p2", dtype="handle"), sid_8, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_max_pool2d_cast", sid_8, sid_7, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", sid_7, T.lookup_param("p3", dtype="handle"), T.lookup_param("p4", dtype="handle"), sid_6, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_1", sid_6, T.lookup_param("p5", dtype="handle"), T.lookup_param("p6", dtype="handle"), sid_5, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_max_pool2d", sid_5, sid_4, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_cast", sid_4, sid_3, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_2", sid_3, T.lookup_param("p7", dtype="handle"), T.lookup_param("p8", dtype="handle"), sid_2, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", sid_3, T.lookup_param("p9", dtype="handle"), T.lookup_param("p10", dtype="handle"), sid_20, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320_", sid_20, T.lookup_param("p11", dtype="handle"), T.lookup_param("p12", dtype="handle"), sid_19, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_2", sid_3, T.lookup_param("p13", dtype="handle"), T.lookup_param("p14", dtype="handle"), sid_26, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__1", sid_26, T.lookup_param("p15", dtype="handle"), T.lookup_param("p16", dtype="handle"), sid_25, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_max_pool2d_cast_1", sid_4, sid_32, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_fixed_point_multiply_cli_4464294615199028320__2", sid_32, T.lookup_param("p17", dtype="handle"), T.lookup_param("p18", dtype="handle"), sid_31, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_concatenate", sid_2, sid_19, sid_25, sid_31, output, dtype="int32"))
    __tvm_meta__ = None
# fmt: on


def test_inception_structure():
    global_ws_pool = usmp_utils.PoolInfo(
        pool_name="global_workspace",
        target_access={Target("c"): usmp_utils.PoolInfo.READ_WRITE_ACCESS},
    )
    tir_mod = InceptionStructure
    tir_mod = _assign_poolinfos_to_allocates_in_irmodule(tir_mod, [global_ws_pool])
    main_func = tir_mod["tvmgen_default_run_model"]
    buffer_info_map = tvm.tir.usmp.analysis.extract_buffer_info(main_func, tir_mod)
    buffer_info_map = _replace_stmt_with_buf_var_names(buffer_info_map)

    # check conflicts
    _verify_conflicts("sid_5", ["Conv2dOutput_8", "sid_4"], buffer_info_map)
    _verify_conflicts(
        "Conv2dOutput_2", ["PaddedInput_2", "sid_4", "sid_3", "sid_2"], buffer_info_map
    )
    _verify_conflicts("sid_9", ["PaddedInput_7"], buffer_info_map)
    _verify_conflicts("PaddedInput_7", ["sid_9", "Conv2dOutput_7"], buffer_info_map)
    _verify_conflicts(
        "sid_26", ["sid_19", "Conv2dOutput_4", "sid_2", "sid_4", "PaddedInput_5"], buffer_info_map
    )
    _verify_conflicts("Conv2dOutput", ["PaddedInput", "sid_6"], buffer_info_map)
    _verify_conflicts(
        "PaddedInput_4", ["sid_19", "sid_2", "sid_4", "sid_3", "Conv2dOutput_4"], buffer_info_map
    )
    _verify_conflicts("sid_8", ["Conv2dOutput_7", "tensor_2"], buffer_info_map)
    _verify_conflicts("tensor_3", ["sid_25", "sid_19", "sid_2", "sid_4", "sid_32"], buffer_info_map)
    _verify_conflicts(
        "sid_3",
        [
            "sid_4",
            "PaddedInput_2",
            "Conv2dOutput_2",
            "sid_2",
            "PaddedInput_1",
            "Conv2dOutput_1",
            "sid_20",
            "PaddedInput_6",
            "Conv2dOutput_6",
            "sid_19",
            "PaddedInput_4",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "sid_32", ["tensor_3", "sid_25", "sid_19", "sid_2", "PaddedInput_3"], buffer_info_map
    )
    _verify_conflicts("PaddedInput_8", ["sid_6", "Conv2dOutput_8"], buffer_info_map)
    _verify_conflicts(
        "Conv2dOutput_6", ["PaddedInput_6", "sid_2", "sid_4", "sid_3", "sid_19"], buffer_info_map
    )
    _verify_conflicts(
        "sid_4",
        [
            "sid_5",
            "sid_3",
            "PaddedInput_2",
            "Conv2dOutput_2",
            "sid_2",
            "PaddedInput_1",
            "Conv2dOutput_1",
            "sid_20",
            "PaddedInput_6",
            "Conv2dOutput_6",
            "sid_19",
            "PaddedInput_4",
            "Conv2dOutput_4",
            "sid_26",
            "PaddedInput_5",
            "Conv2dOutput_5",
            "sid_25",
            "tensor_3",
        ],
        buffer_info_map,
    )
    _verify_conflicts("PaddedInput_2", ["sid_3", "sid_4", "Conv2dOutput_2"], buffer_info_map)
    _verify_conflicts(
        "Conv2dOutput_4", ["sid_19", "sid_2", "sid_4", "PaddedInput_4", "sid_26"], buffer_info_map
    )
    _verify_conflicts(
        "PaddedInput_1", ["sid_2", "sid_4", "sid_3", "Conv2dOutput_1"], buffer_info_map
    )
    _verify_conflicts("sid_6", ["Conv2dOutput", "PaddedInput_8"], buffer_info_map)
    _verify_conflicts("Conv2dOutput_8", ["PaddedInput_8", "sid_5"], buffer_info_map)
    _verify_conflicts(
        "sid_25",
        [
            "Conv2dOutput_5",
            "sid_19",
            "sid_2",
            "sid_4",
            "tensor_3",
            "sid_32",
            "PaddedInput_3",
            "Conv2dOutput_3",
            "sid_31",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "PaddedInput_6", ["sid_20", "sid_2", "sid_4", "sid_3", "Conv2dOutput_6"], buffer_info_map
    )
    _verify_conflicts(
        "sid_7",
        [
            "tensor_2",
            "PaddedInput",
        ],
        buffer_info_map,
    )
    _verify_conflicts("sid_31", ["Conv2dOutput_3", "sid_25", "sid_19", "sid_2"], buffer_info_map)
    _verify_conflicts("tensor_2", ["sid_8", "sid_7"], buffer_info_map)
    _verify_conflicts(
        "sid_2",
        [
            "Conv2dOutput_2",
            "sid_4",
            "sid_3",
            "PaddedInput_1",
            "Conv2dOutput_1",
            "sid_20",
            "PaddedInput_6",
            "Conv2dOutput_6",
            "sid_19",
            "PaddedInput_4",
            "Conv2dOutput_4",
            "sid_26",
            "PaddedInput_5",
            "Conv2dOutput_5",
            "sid_25",
            "tensor_3",
            "sid_32",
            "PaddedInput_3",
            "Conv2dOutput_3",
            "sid_31",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_3", ["sid_25", "PaddedInput_3", "sid_19", "sid_2", "sid_31"], buffer_info_map
    )
    _verify_conflicts("PaddedInput", ["sid_7", "Conv2dOutput"], buffer_info_map)
    _verify_conflicts(
        "Conv2dOutput_1", ["PaddedInput_1", "sid_2", "sid_4", "sid_3", "sid_20"], buffer_info_map
    )
    _verify_conflicts(
        "PaddedInput_5", ["sid_26", "sid_19", "sid_2", "sid_4", "Conv2dOutput_5"], buffer_info_map
    )
    _verify_conflicts(
        "PaddedInput_3", ["sid_32", "sid_25", "sid_19", "sid_2", "Conv2dOutput_3"], buffer_info_map
    )
    _verify_conflicts(
        "sid_19",
        [
            "Conv2dOutput_6",
            "sid_2",
            "sid_4",
            "sid_3",
            "PaddedInput_4",
            "Conv2dOutput_4",
            "sid_26",
            "PaddedInput_5",
            "Conv2dOutput_5",
            "sid_25",
            "tensor_3",
            "sid_32",
            "PaddedInput_3",
            "Conv2dOutput_3",
            "sid_31",
        ],
        buffer_info_map,
    )
    _verify_conflicts(
        "Conv2dOutput_5", ["PaddedInput_5", "sid_19", "sid_2", "sid_4", "sid_25"], buffer_info_map
    )
    _verify_conflicts("Conv2dOutput_7", ["PaddedInput_7", "sid_8"], buffer_info_map)
    _verify_conflicts(
        "sid_20", ["sid_2", "Conv2dOutput_1", "sid_4", "sid_3", "PaddedInput_6"], buffer_info_map
    )

    # check sizes
    assert buffer_info_map["sid_20"].size_bytes == 150528
    assert buffer_info_map["tensor_2"].size_bytes == 200704
    assert buffer_info_map["sid_5"].size_bytes == 602112
    assert buffer_info_map["sid_9"].size_bytes == 301056
    assert buffer_info_map["Conv2dOutput_3"].size_bytes == 4
    assert buffer_info_map["sid_26"].size_bytes == 25088
    assert buffer_info_map["Conv2dOutput_2"].size_bytes == 256
    assert buffer_info_map["PaddedInput_5"].size_bytes == 28800
    assert buffer_info_map["sid_8"].size_bytes == 802816
    assert buffer_info_map["Conv2dOutput_5"].size_bytes == 4
    assert buffer_info_map["sid_3"].size_bytes == 301056
    assert buffer_info_map["Conv2dOutput"].size_bytes == 256
    assert buffer_info_map["PaddedInput_3"].size_bytes == 301056
    assert buffer_info_map["sid_32"].size_bytes == 301056
    assert buffer_info_map["PaddedInput_8"].size_bytes == 430592
    assert buffer_info_map["sid_4"].size_bytes == 150528
    assert buffer_info_map["PaddedInput_7"].size_bytes == 314646
    assert buffer_info_map["sid_6"].size_bytes == 401408
    assert buffer_info_map["Conv2dOutput_8"].size_bytes == 256
    assert buffer_info_map["sid_25"].size_bytes == 25088
    assert buffer_info_map["PaddedInput"].size_bytes == 401408
    assert buffer_info_map["sid_7"].size_bytes == 401408
    assert buffer_info_map["Conv2dOutput_1"].size_bytes == 4
    assert buffer_info_map["Conv2dOutput_4"].size_bytes == 4
    assert buffer_info_map["PaddedInput_2"].size_bytes == 301056
    assert buffer_info_map["sid_31"].size_bytes == 25088
    assert buffer_info_map["PaddedInput_1"].size_bytes == 301056
    assert buffer_info_map["Conv2dOutput_6"].size_bytes == 256
    assert buffer_info_map["PaddedInput_4"].size_bytes == 301056
    assert buffer_info_map["sid_2"].size_bytes == 50176
    assert buffer_info_map["tensor_3"].size_bytes == 150528
    assert buffer_info_map["Conv2dOutput_7"].size_bytes == 256
    assert buffer_info_map["sid_19"].size_bytes == 100352
    assert buffer_info_map["PaddedInput_6"].size_bytes == 172800


if __name__ == "__main__":
    pytest.main([__file__])
