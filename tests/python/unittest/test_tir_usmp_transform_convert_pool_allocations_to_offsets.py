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
import sys

import tvm
from tvm.script import tir as T
from tvm.tir import stmt_functor
from tvm.tir.usmp import utils as usmp_utils
from tvm.target import Target


def _get_primfuncs_from_module(module):
    primfuncs = list()
    for gv, primfunc in module.functions.items():
        primfuncs.append(primfunc)
    return primfuncs


def assign_poolinfos_to_allocates_in_primfunc(primfunc, pool_infos):
    """Helper to assign poolinfos to allocate nodes in a tir.PrimFunc"""

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


def assign_poolinfos_to_allocates_in_irmodule(mod, pool_infos):
    """Helper to assign poolinfos to allocate nodes in a IRModule"""
    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, tvm.tir.PrimFunc):
            ret[global_var] = assign_poolinfos_to_allocates_in_primfunc(basefunc, pool_infos)
    return ret


def _assign_targets_to_primfuncs_irmodule(mod, target):
    """Helper to assign target for PrimFunc in a IRModule"""
    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, tvm.tir.PrimFunc):
            ret[global_var] = basefunc.with_attr("target", target)
    return ret


# fmt: off
@tvm.script.ir_module
class LinearStructure:
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
    def __tvm_main__(input: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "__tvm_main__", "runner_function": True})
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        sid_9 = T.allocate([301056], "int8", "global")
        sid_8 = T.allocate([802816], "int8", "global")
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input, T.lookup_param("p0", dtype="handle"), sid_9, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", sid_9, T.lookup_param("p1", dtype="handle"), T.lookup_param("p2", dtype="handle"), sid_8, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_max_pool2d_cast", sid_8, output, dtype="int32"))
# fmt: on


# fmt: off
@tvm.script.ir_module
class LinearStructurePlanned:
    @T.prim_func
    def __tvm_main__(input: T.handle, fast_memory_0_var: T.handle, slow_memory_1_var: T.handle,  output: T.handle) -> None:
        fast_memory_0_buffer_var = T.match_buffer(fast_memory_0_var, [200704], dtype="uint8", strides=[1], elem_offset=1, align=16)
        slow_memory_1_buffer_var = T.match_buffer(slow_memory_1_var, [1418528], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        sid_9_let: T.handle = T.address_of(T.load("uint8", slow_memory_1_buffer_var.data, 1117472), dtype="handle")
        sid_8_let: T.handle = T.address_of(T.load("uint8", slow_memory_1_buffer_var.data, 0), dtype="handle")
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input, T.lookup_param("p0", dtype="handle"), sid_9_let, fast_memory_0_buffer_var.data, slow_memory_1_buffer_var.data, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", sid_9_let, T.lookup_param("p1", dtype="handle"), T.lookup_param("p2", dtype="handle"), sid_8_let, fast_memory_0_buffer_var.data, slow_memory_1_buffer_var.data, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_max_pool2d_cast", sid_8_let, output, fast_memory_0_buffer_var.data, slow_memory_1_buffer_var.data, dtype="int32"))

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle, fast_memory_6_var: T.handle, slow_memory_7_var: T.handle) -> None:
        placeholder_29 = T.match_buffer(placeholder_28, [1, 112, 112, 64], dtype="uint8")
        T_cast_7 = T.match_buffer(T_cast_6, [1, 56, 56, 64], dtype="int16")
        fast_memory_6_buffer_var = T.match_buffer(fast_memory_6_var, [200704], dtype="uint8", strides=[1], elem_offset=1, align=16)
        slow_memory_7_buffer_var = T.match_buffer(slow_memory_7_var, [1418528], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        tensor_2_let: T.handle = T.address_of(T.load("uint8", fast_memory_6_buffer_var.data, 0), dtype="handle")
        for ax0_ax1_fused_4, ax2_4 in T.grid(56, 56):
            for ax3_init in T.serial(0, 64):
                T.store(tensor_2_let, ax0_ax1_fused_4 * 3584 + ax2_4 * 64 + ax3_init, T.uint8(0), True)
            for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                T.store(tensor_2_let, ax0_ax1_fused_4 * 3584 + ax2_4 * 64 + ax3_2, T.max(T.load("uint8", tensor_2_let, ax0_ax1_fused_4 * 3584 + ax2_4 * 64 + ax3_2), T.if_then_else(ax0_ax1_fused_4 * 2 + rv0_rv1_fused_1 // 3 < 112 and ax2_4 * 2 + rv0_rv1_fused_1 % 3 < 112, T.load("uint8", placeholder_29.data, ax0_ax1_fused_4 * 14336 + rv0_rv1_fused_1 // 3 * 7168 + ax2_4 * 128 + rv0_rv1_fused_1 % 3 * 64 + ax3_2), T.uint8(0), dtype="uint8")), True)
        for ax0_ax1_fused_5, ax2_5, ax3_3 in T.grid(56, 56, 64):
            T.store(T_cast_7.data, ax0_ax1_fused_5 * 3584 + ax2_5 * 64 + ax3_3, T.cast(T.load("uint8", tensor_2_let, ax0_ax1_fused_5 * 3584 + ax2_5 * 64 + ax3_3), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle, fast_memory_2_var: T.handle, slow_memory_3_var: T.handle) -> None:
        placeholder_4 = T.match_buffer(placeholder_2, [1, 224, 224, 3], dtype="uint8")
        placeholder_5 = T.match_buffer(placeholder_3, [], dtype="int16")
        T_subtract_1 = T.match_buffer(T_subtract, [1, 224, 224, 3], dtype="int16")
        fast_memory_2_buffer_var = T.match_buffer(fast_memory_2_var, [200704], dtype="uint8", strides=[1], elem_offset=1, align=16)
        slow_memory_3_buffer_var = T.match_buffer(slow_memory_3_var, [1418528], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        for ax0_ax1_fused_1, ax2_1, ax3_inner_1 in T.grid(224, 224, 3):
            T.store(T_subtract_1.data, ax0_ax1_fused_1 * 672 + ax2_1 * 3 + ax3_inner_1, T.cast(T.load("uint8", placeholder_4.data, ax0_ax1_fused_1 * 672 + ax2_1 * 3 + ax3_inner_1), "int16") - T.load("int16", placeholder_5.data, 0), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle, fast_memory_4_var: T.handle, slow_memory_5_var: T.handle) -> None:
        placeholder_65 = T.match_buffer(placeholder_62, [1, 224, 224, 3], dtype="int16")
        placeholder_66 = T.match_buffer(placeholder_63, [7, 7, 3, 64], dtype="int16")
        placeholder_67 = T.match_buffer(placeholder_64, [1, 1, 1, 64], dtype="int32")
        T_cast_21 = T.match_buffer(T_cast_20, [1, 112, 112, 64], dtype="uint8")
        fast_memory_4_buffer_var = T.match_buffer(fast_memory_4_var, [200704], dtype="uint8", strides=[1], elem_offset=1, align=16)
        slow_memory_5_buffer_var = T.match_buffer(slow_memory_5_var, [1418528], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        PaddedInput_7_let: T.handle = T.address_of(T.load("uint8", slow_memory_5_buffer_var.data, 802816), dtype="handle")
        for i0_i1_fused_7, i2_7, i3_7 in T.grid(229, 229, 3):
            T.store(PaddedInput_7_let, i0_i1_fused_7 * 687 + i2_7 * 3 + i3_7, T.if_then_else(2 <= i0_i1_fused_7 and i0_i1_fused_7 < 226 and 2 <= i2_7 and i2_7 < 226, T.load("int16", placeholder_65.data, i0_i1_fused_7 * 672 + i2_7 * 3 + i3_7 - 1350), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_7 in T.serial(0, 12544):
            Conv2dOutput_7_let: T.handle = T.address_of(T.load("uint8", fast_memory_4_buffer_var.data, 0), dtype="handle")
            for ff_3 in T.serial(0, 64):
                T.store(Conv2dOutput_7_let, ff_3, 0, True)
                for ry_2, rx_2, rc_7 in T.grid(7, 7, 3):
                    T.store(Conv2dOutput_7_let, ff_3, T.load("int32", Conv2dOutput_7_let, ff_3) + T.cast(T.load("int16", PaddedInput_7_let, ax0_ax1_fused_ax2_fused_7 // 112 * 1374 + ry_2 * 687 + ax0_ax1_fused_ax2_fused_7 % 112 * 6 + rx_2 * 3 + rc_7), "int32") * T.cast(T.load("int16", placeholder_66.data, ry_2 * 1344 + rx_2 * 192 + rc_7 * 64 + ff_3), "int32"), True)
            for ax3_inner_7 in T.serial(0, 64):
                T.store(T_cast_21.data, ax0_ax1_fused_ax2_fused_7 * 64 + ax3_inner_7, T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput_7_let, ax3_inner_7) + T.load("int32", placeholder_67.data, ax3_inner_7), 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8"), True)
# fmt: on


def test_mobilenet_subgraph():
    target = Target("c")
    fast_memory_pool = usmp_utils.PoolInfo(
        pool_name="fast_memory",
        target_access={target: usmp_utils.PoolInfo.READ_WRITE_ACCESS},
        size_hint_bytes=200704,
    )
    slow_memory_pool = usmp_utils.PoolInfo(
        pool_name="slow_memory", target_access={target: usmp_utils.PoolInfo.READ_WRITE_ACCESS}
    )
    tir_mod = LinearStructure
    tir_mod = _assign_targets_to_primfuncs_irmodule(tir_mod, target)
    tir_mod = assign_poolinfos_to_allocates_in_irmodule(
        tir_mod, [fast_memory_pool, slow_memory_pool]
    )
    main_func = tir_mod["__tvm_main__"]
    buffer_analysis = tvm.tir.usmp.analysis.extract_buffer_info(main_func, tir_mod)
    buffer_info_map = buffer_analysis.buffer_info_stmts

    fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
    buffer_info_arr = fcreate_array_bi(buffer_info_map)
    fusmp_algo_greedy_by_size = tvm.get_global_func("tir.usmp.algo.greedy_by_size")
    buffer_pool_allocations = fusmp_algo_greedy_by_size(
        buffer_info_arr, buffer_analysis.memory_pressure
    )
    fassign_stmt_pool_allocations = tvm.get_global_func("tir.usmp.AssignStmtPoolAllocations")
    pool_allocations = fassign_stmt_pool_allocations(buffer_info_map, buffer_pool_allocations)
    tir_mod_with_offsets = tvm.tir.usmp.transform.convert_pool_allocations_to_offsets(
        pool_allocations, emit_tvmscript_printable=True
    )(tir_mod)

    tir_mod_with_offsets_ref = LinearStructurePlanned
    tir_mod_with_offsets_ref = tvm.script.from_source(
        tir_mod_with_offsets_ref.script(show_meta=False)
    )
    # The TIR produced fails on roundtrip TVMScript testing.
    # Therefore, indicates the TVMScript produced here and/or the parser
    # is lacking functionality. Thus for these tests, uses a string
    # version of the TVMScript for each function as a check instead.
    for gv, func in tir_mod_with_offsets_ref.functions.items():
        assert str(tir_mod_with_offsets_ref[gv.name_hint].script()) == str(
            tir_mod_with_offsets[gv.name_hint].script()
        )


# fmt: off
@tvm.script.ir_module
class ResnetStructure:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast(placeholder: T.handle, placeholder_1: T.handle, T_cast: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [1, 75, 75, 64], dtype="uint8")
        placeholder_3 = T.match_buffer(placeholder_1, [64], dtype="int32")
        T_cast_1 = T.match_buffer(T_cast, [1, 75, 75, 64], dtype="int16")
        # body
        for ax0_ax1_fused, ax2, ax3_outer, ax3_inner in T.grid(75, 75, 4, 16):
            T.store(T_cast_1.data, ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner, T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.load("uint8", placeholder_2.data, ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner), "int32") - 94, 1843157232, 31, 1, dtype="int32") + T.load("int32", placeholder_3.data, ax3_outer * 16 + ax3_inner), 255), 0), "uint8"), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1(placeholder_10: T.handle, placeholder_11: T.handle, placeholder_12: T.handle, T_cast_4: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", "tir.noalias": True})
        placeholder_13 = T.match_buffer(placeholder_10, [1, 75, 75, 64], dtype="int16")
        placeholder_14 = T.match_buffer(placeholder_11, [3, 3, 64, 64], dtype="int16")
        placeholder_15 = T.match_buffer(placeholder_12, [1, 1, 1, 64], dtype="int32")
        T_cast_5 = T.match_buffer(T_cast_4, [1, 75, 75, 64], dtype="int16")
        # body
        PaddedInput_1 = T.allocate([379456], "int16", "global")
        for i0_i1_fused_1, i2_1, i3_1 in T.grid(77, 77, 64):
            T.store(PaddedInput_1, i0_i1_fused_1 * 4928 + i2_1 * 64 + i3_1, T.if_then_else(1 <= i0_i1_fused_1 and i0_i1_fused_1 < 76 and 1 <= i2_1 and i2_1 < 76, T.load("int16", placeholder_13.data, i0_i1_fused_1 * 4800 + i2_1 * 64 + i3_1 - 4864), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_1 in T.serial(0, 5625):
            Conv2dOutput_1 = T.allocate([64], "int32", "global")
            for ff_1 in T.serial(0, 64):
                T.store(Conv2dOutput_1, ff_1, 0, True)
                for ry, rx, rc_1 in T.grid(3, 3, 64):
                    T.store(Conv2dOutput_1, ff_1, T.load("int32", Conv2dOutput_1, ff_1) + T.cast(T.load("int16", PaddedInput_1, T.floordiv(ax0_ax1_fused_ax2_fused_1, 75) * 4928 + ry * 4928 + rx * 64 + T.floormod(ax0_ax1_fused_ax2_fused_1, 75) * 64 + rc_1), "int32") * T.cast(T.load("int16", placeholder_14.data, ry * 12288 + rx * 4096 + rc_1 * 64 + ff_1), "int32"), True)
            for ax3_inner_2 in T.serial(0, 64):
                T.store(T_cast_5.data, ax0_ax1_fused_ax2_fused_1 * 64 + ax3_inner_2, T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput_1, ax3_inner_2) + T.load("int32", placeholder_15.data, ax3_inner_2), 1608879842, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_(placeholder_16: T.handle, placeholder_17: T.handle, placeholder_18: T.handle, T_add: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_", "tir.noalias": True})
        placeholder_19 = T.match_buffer(placeholder_16, [1, 75, 75, 64], dtype="int16")
        placeholder_20 = T.match_buffer(placeholder_17, [1, 1, 64, 256], dtype="int16")
        placeholder_21 = T.match_buffer(placeholder_18, [1, 1, 1, 256], dtype="int32")
        T_add_1 = T.match_buffer(T_add, [1, 75, 75, 256], dtype="int32")
        # body
        PaddedInput_2 = T.allocate([360000], "int16", "global")
        for i0_i1_fused_2, i2_2, i3_2 in T.grid(75, 75, 64):
            T.store(PaddedInput_2, i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2, T.load("int16", placeholder_19.data, i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2), True)
        for ax0_ax1_fused_ax2_fused_2 in T.serial(0, 5625):
            Conv2dOutput_2 = T.allocate([64], "int32", "global")
            for ax3_outer_1 in T.serial(0, 4):
                for ff_2 in T.serial(0, 64):
                    T.store(Conv2dOutput_2, ff_2, 0, True)
                    for rc_2 in T.serial(0, 64):
                        T.store(Conv2dOutput_2, ff_2, T.load("int32", Conv2dOutput_2, ff_2) + T.cast(T.load("int16", PaddedInput_2, ax0_ax1_fused_ax2_fused_2 * 64 + rc_2), "int32") * T.cast(T.load("int16", placeholder_20.data, rc_2 * 256 + ax3_outer_1 * 64 + ff_2), "int32"), True)
                for ax3_inner_3 in T.serial(0, 64):
                    T.store(T_add_1.data, ax0_ax1_fused_ax2_fused_2 * 256 + ax3_outer_1 * 64 + ax3_inner_3, T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput_2, ax3_inner_3) + T.load("int32", placeholder_21.data, ax3_outer_1 * 64 + ax3_inner_3), 1711626602, 31, -8, dtype="int32") + 132, 255), 0), "uint8"), "int32") - 132, 2094289803, 31, -2, dtype="int32") + 136, True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_(placeholder_22: T.handle, placeholder_23: T.handle, placeholder_24: T.handle, placeholder_25: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_22, [1, 75, 75, 64], dtype="int16")
        placeholder_27 = T.match_buffer(placeholder_23, [1, 1, 64, 256], dtype="int16")
        placeholder_26 = T.match_buffer(placeholder_24, [1, 1, 1, 256], dtype="int32")
        placeholder_28 = T.match_buffer(placeholder_25, [1, 75, 75, 256], dtype="int32")
        T_cast_7 = T.match_buffer(T_cast_6, [1, 75, 75, 256], dtype="uint8")
        # body
        PaddedInput_3 = T.allocate([360000], "int16", "global")
        for i0_i1_fused_3, i2_3, i3_3 in T.grid(75, 75, 64):
            T.store(PaddedInput_3, i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3, T.load("int16", placeholder_29.data, i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3), True)
        for ax0_ax1_fused_ax2_fused_3 in T.serial(0, 5625):
            Conv2dOutput_3 = T.allocate([64], "int32", "global")
            for ax3_outer_2 in T.serial(0, 4):
                for ff_3 in T.serial(0, 64):
                    T.store(Conv2dOutput_3, ff_3, 0, True)
                    for rc_3 in T.serial(0, 64):
                        T.store(Conv2dOutput_3, ff_3, T.load("int32", Conv2dOutput_3, ff_3) + T.cast(T.load("int16", PaddedInput_3, ax0_ax1_fused_ax2_fused_3 * 64 + rc_3), "int32") * T.cast(T.load("int16", placeholder_27.data, rc_3 * 256 + ax3_outer_2 * 64 + ff_3), "int32"), True)
                for ax3_inner_4 in T.serial(0, 64):
                    T.store(T_cast_7.data, ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4, T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput_3, ax3_inner_4) + T.load("int32", placeholder_26.data, ax3_outer_2 * 64 + ax3_inner_4), 1343014664, 31, -8, dtype="int32") + 136, 255), 0), "uint8"), "int32") - 136, 1073903788, 31, 1, dtype="int32") + T.load("int32", placeholder_28.data, ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4), 255), 0), "uint8"), True)

    @T.prim_func
    def __tvm_main__(input: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "__tvm_main__", "runner_function": True})
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        sid_2 = T.allocate([720000], "int8", "global")
        sid_6 = T.allocate([5760000], "int8", "global")
        sid_7 = T.allocate([720000], "int8", "global")
        sid_8 = T.allocate([720000], "int8", "global")
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast", input, T.lookup_param("p0", dtype="handle"), sid_2, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", sid_2, T.lookup_param("p3", dtype="handle"), T.lookup_param("p4", dtype="handle"), sid_8, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", sid_8, T.lookup_param("p5", dtype="handle"), T.lookup_param("p6", dtype="handle"), sid_7, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_", sid_7, T.lookup_param("p7", dtype="handle"), T.lookup_param("p8", dtype="handle"), sid_6, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_", sid_2, T.lookup_param("p1", dtype="handle"), T.lookup_param("p2", dtype="handle"), sid_6, output, dtype="int32"))

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast(placeholder_4: T.handle, placeholder_5: T.handle, placeholder_6: T.handle, T_cast_2: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", "tir.noalias": True})
        placeholder_7 = T.match_buffer(placeholder_4, [1, 75, 75, 64], dtype="int16")
        placeholder_8 = T.match_buffer(placeholder_5, [1, 1, 64, 64], dtype="int16")
        placeholder_9 = T.match_buffer(placeholder_6, [1, 1, 1, 64], dtype="int32")
        T_cast_3 = T.match_buffer(T_cast_2, [1, 75, 75, 64], dtype="int16")
        # body
        PaddedInput = T.allocate([360000], "int16", "global")
        for i0_i1_fused, i2, i3 in T.grid(75, 75, 64):
            T.store(PaddedInput, i0_i1_fused * 4800 + i2 * 64 + i3, T.load("int16", placeholder_7.data, i0_i1_fused * 4800 + i2 * 64 + i3), True)
        for ax0_ax1_fused_ax2_fused in T.serial(0, 5625):
            Conv2dOutput = T.allocate([64], "int32", "global")
            for ff in T.serial(0, 64):
                T.store(Conv2dOutput, ff, 0, True)
                for rc in T.serial(0, 64):
                    T.store(Conv2dOutput, ff, T.load("int32", Conv2dOutput, ff) + T.cast(T.load("int16", PaddedInput, ax0_ax1_fused_ax2_fused * 64 + rc), "int32") * T.cast(T.load("int16", placeholder_8.data, rc * 64 + ff), "int32"), True)
            for ax3_inner_1 in T.serial(0, 64):
                T.store(T_cast_3.data, ax0_ax1_fused_ax2_fused * 64 + ax3_inner_1, T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput, ax3_inner_1) + T.load("int32", placeholder_9.data, ax3_inner_1), 1843106743, 31, -6, dtype="int32"), 255), 0), "uint8"), "int16"), True)
# fmt: on


# fmt: off
@tvm.script.ir_module
class ResnetStructurePlanned:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast(placeholder: T.handle, placeholder_1: T.handle, T_cast: T.handle, global_workspace_1_var: T.handle) -> None:
        placeholder_2 = T.match_buffer(placeholder, [1, 75, 75, 64], dtype="uint8")
        placeholder_3 = T.match_buffer(placeholder_1, [64], dtype="int32")
        T_cast_1 = T.match_buffer(T_cast, [1, 75, 75, 64], dtype="int16")
        global_workspace_1_buffer_var = T.match_buffer(global_workspace_1_var, [7920256], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        for ax0_ax1_fused, ax2, ax3_outer, ax3_inner in T.grid(75, 75, 4, 16):
            T.store(T_cast_1.data, ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner, T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.load("uint8", placeholder_2.data, ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner), "int32") - 94, 1843157232, 31, 1, dtype="int32") + T.load("int32", placeholder_3.data, ax3_outer * 16 + ax3_inner), 255), 0), "uint8"), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_(placeholder_22: T.handle, placeholder_23: T.handle, placeholder_24: T.handle, placeholder_25: T.handle, T_cast_6: T.handle, global_workspace_5_var: T.handle) -> None:
        placeholder_29 = T.match_buffer(placeholder_22, [1, 75, 75, 64], dtype="int16")
        placeholder_27 = T.match_buffer(placeholder_23, [1, 1, 64, 256], dtype="int16")
        placeholder_26 = T.match_buffer(placeholder_24, [1, 1, 1, 256], dtype="int32")
        placeholder_28 = T.match_buffer(placeholder_25, [1, 75, 75, 256], dtype="int32")
        T_cast_7 = T.match_buffer(T_cast_6, [1, 75, 75, 256], dtype="uint8")
        global_workspace_5_buffer_var = T.match_buffer(global_workspace_5_var, [7920256], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        PaddedInput_3_let: T.handle = T.address_of(T.load("uint8", global_workspace_5_buffer_var.data, 6480000), dtype="handle")
        for i0_i1_fused_3, i2_3, i3_3 in T.grid(75, 75, 64):
            T.store(PaddedInput_3_let, i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3, T.load("int16", placeholder_29.data, i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3), True)
        for ax0_ax1_fused_ax2_fused_3 in T.serial(0, 5625):
            Conv2dOutput_3_let: T.handle = T.address_of(T.load("uint8", global_workspace_5_buffer_var.data, 7200000), dtype="handle")
            for ax3_outer_2 in T.serial(0, 4):
                for ff_3 in T.serial(0, 64):
                    T.store(Conv2dOutput_3_let, ff_3, 0, True)
                    for rc_3 in T.serial(0, 64):
                        T.store(Conv2dOutput_3_let, ff_3, T.load("int32", Conv2dOutput_3_let, ff_3) + T.cast(T.load("int16", PaddedInput_3_let, ax0_ax1_fused_ax2_fused_3 * 64 + rc_3), "int32") * T.cast(T.load("int16", placeholder_27.data, rc_3 * 256 + ax3_outer_2 * 64 + ff_3), "int32"), True)
                for ax3_inner_4 in T.serial(0, 64):
                    T.store(T_cast_7.data, ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4, T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput_3_let, ax3_inner_4) + T.load("int32", placeholder_26.data, ax3_outer_2 * 64 + ax3_inner_4), 1343014664, 31, -8, dtype="int32") + 136, 255), 0), "uint8"), "int32") - 136, 1073903788, 31, 1, dtype="int32") + T.load("int32", placeholder_28.data, ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4), 255), 0), "uint8"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_(placeholder_16: T.handle, placeholder_17: T.handle, placeholder_18: T.handle, T_add: T.handle, global_workspace_4_var: T.handle) -> None:
        placeholder_19 = T.match_buffer(placeholder_16, [1, 75, 75, 64], dtype="int16")
        placeholder_20 = T.match_buffer(placeholder_17, [1, 1, 64, 256], dtype="int16")
        placeholder_21 = T.match_buffer(placeholder_18, [1, 1, 1, 256], dtype="int32")
        T_add_1 = T.match_buffer(T_add, [1, 75, 75, 256], dtype="int32")
        global_workspace_4_buffer_var = T.match_buffer(global_workspace_4_var, [7920256], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        PaddedInput_2_let: T.handle = T.address_of(T.load("uint8", global_workspace_4_buffer_var.data, 7200000), dtype="handle")
        for i0_i1_fused_2, i2_2, i3_2 in T.grid(75, 75, 64):
            T.store(PaddedInput_2_let, i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2, T.load("int16", placeholder_19.data, i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2), True)
        for ax0_ax1_fused_ax2_fused_2 in T.serial(0, 5625):
            Conv2dOutput_2_let: T.handle = T.address_of(T.load("uint8", global_workspace_4_buffer_var.data, 7920000), dtype="handle")
            for ax3_outer_1 in T.serial(0, 4):
                for ff_2 in T.serial(0, 64):
                    T.store(Conv2dOutput_2_let, ff_2, 0, True)
                    for rc_2 in T.serial(0, 64):
                        T.store(Conv2dOutput_2_let, ff_2, T.load("int32", Conv2dOutput_2_let, ff_2) + T.cast(T.load("int16", PaddedInput_2_let, ax0_ax1_fused_ax2_fused_2 * 64 + rc_2), "int32") * T.cast(T.load("int16", placeholder_20.data, rc_2 * 256 + ax3_outer_1 * 64 + ff_2), "int32"), True)
                for ax3_inner_3 in T.serial(0, 64):
                    T.store(T_add_1.data, ax0_ax1_fused_ax2_fused_2 * 256 + ax3_outer_1 * 64 + ax3_inner_3, T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput_2_let, ax3_inner_3) + T.load("int32", placeholder_21.data, ax3_outer_1 * 64 + ax3_inner_3), 1711626602, 31, -8, dtype="int32") + 132, 255), 0), "uint8"), "int32") - 132, 2094289803, 31, -2, dtype="int32") + 136, True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast(placeholder_4: T.handle, placeholder_5: T.handle, placeholder_6: T.handle, T_cast_2: T.handle, global_workspace_2_var: T.handle) -> None:
        placeholder_7 = T.match_buffer(placeholder_4, [1, 75, 75, 64], dtype="int16")
        placeholder_8 = T.match_buffer(placeholder_5, [1, 1, 64, 64], dtype="int16")
        placeholder_9 = T.match_buffer(placeholder_6, [1, 1, 1, 64], dtype="int32")
        T_cast_3 = T.match_buffer(T_cast_2, [1, 75, 75, 64], dtype="int16")
        global_workspace_2_buffer_var = T.match_buffer(global_workspace_2_var, [7920256], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        PaddedInput_let: T.handle = T.address_of(T.load("uint8", global_workspace_2_buffer_var.data, 7200000), dtype="handle")
        for i0_i1_fused, i2, i3 in T.grid(75, 75, 64):
            T.store(PaddedInput_let, i0_i1_fused * 4800 + i2 * 64 + i3, T.load("int16", placeholder_7.data, i0_i1_fused * 4800 + i2 * 64 + i3), True)
        for ax0_ax1_fused_ax2_fused in T.serial(0, 5625):
            Conv2dOutput_let: T.handle = T.address_of(T.load("uint8", global_workspace_2_buffer_var.data, 7920000), dtype="handle")
            for ff in T.serial(0, 64):
                T.store(Conv2dOutput_let, ff, 0, True)
                for rc in T.serial(0, 64):
                    T.store(Conv2dOutput_let, ff, T.load("int32", Conv2dOutput_let, ff) + T.cast(T.load("int16", PaddedInput_let, ax0_ax1_fused_ax2_fused * 64 + rc), "int32") * T.cast(T.load("int16", placeholder_8.data, rc * 64 + ff), "int32"), True)
            for ax3_inner_1 in T.serial(0, 64):
                T.store(T_cast_3.data, ax0_ax1_fused_ax2_fused * 64 + ax3_inner_1, T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput_let, ax3_inner_1) + T.load("int32", placeholder_9.data, ax3_inner_1), 1843106743, 31, -6, dtype="int32"), 255), 0), "uint8"), "int16"), True)

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1(placeholder_10: T.handle, placeholder_11: T.handle, placeholder_12: T.handle, T_cast_4: T.handle, global_workspace_3_var: T.handle) -> None:
        placeholder_13 = T.match_buffer(placeholder_10, [1, 75, 75, 64], dtype="int16")
        placeholder_14 = T.match_buffer(placeholder_11, [3, 3, 64, 64], dtype="int16")
        placeholder_15 = T.match_buffer(placeholder_12, [1, 1, 1, 64], dtype="int32")
        T_cast_5 = T.match_buffer(T_cast_4, [1, 75, 75, 64], dtype="int16")
        global_workspace_3_buffer_var = T.match_buffer(global_workspace_3_var, [7920256], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        PaddedInput_1_let: T.handle = T.address_of(T.load("uint8", global_workspace_3_buffer_var.data, 0), dtype="handle")
        for i0_i1_fused_1, i2_1, i3_1 in T.grid(77, 77, 64):
            T.store(PaddedInput_1_let, i0_i1_fused_1 * 4928 + i2_1 * 64 + i3_1, T.if_then_else(1 <= i0_i1_fused_1 and i0_i1_fused_1 < 76 and 1 <= i2_1 and i2_1 < 76, T.load("int16", placeholder_13.data, i0_i1_fused_1 * 4800 + i2_1 * 64 + i3_1 - 4864), T.int16(0), dtype="int16"), True)
        for ax0_ax1_fused_ax2_fused_1 in T.serial(0, 5625):
            Conv2dOutput_1_let: T.handle = T.address_of(T.load("uint8", global_workspace_3_buffer_var.data, 7200000), dtype="handle")
            for ff_1 in T.serial(0, 64):
                T.store(Conv2dOutput_1_let, ff_1, 0, True)
                for ry, rx, rc_1 in T.grid(3, 3, 64):
                    T.store(Conv2dOutput_1_let, ff_1, T.load("int32", Conv2dOutput_1_let, ff_1) + T.cast(T.load("int16", PaddedInput_1_let, ax0_ax1_fused_ax2_fused_1 // 75 * 4928 + ry * 4928 + rx * 64 + ax0_ax1_fused_ax2_fused_1 % 75 * 64 + rc_1), "int32") * T.cast(T.load("int16", placeholder_14.data, ry * 12288 + rx * 4096 + rc_1 * 64 + ff_1), "int32"), True)
            for ax3_inner_2 in T.serial(0, 64):
                T.store(T_cast_5.data, ax0_ax1_fused_ax2_fused_1 * 64 + ax3_inner_2, T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.load("int32", Conv2dOutput_1_let, ax3_inner_2) + T.load("int32", placeholder_15.data, ax3_inner_2), 1608879842, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16"), True)

    @T.prim_func
    def __tvm_main__(input: T.handle, global_workspace_0_var: T.handle, output: T.handle) -> None:
        global_workspace_0_buffer_var = T.match_buffer(global_workspace_0_var, [7920256], dtype="uint8", strides=[1], elem_offset=1, align=16)
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        sid_2_let: T.handle = T.address_of(T.load("uint8", global_workspace_0_buffer_var.data, 5760000), dtype="handle")
        sid_6_let: T.handle = T.address_of(T.load("uint8", global_workspace_0_buffer_var.data, 0), dtype="handle")
        sid_7_let: T.handle = T.address_of(T.load("uint8", global_workspace_0_buffer_var.data, 6480000), dtype="handle")
        sid_8_let: T.handle = T.address_of(T.load("uint8", global_workspace_0_buffer_var.data, 6480000), dtype="handle")
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast", input, T.lookup_param("p0", dtype="handle"), sid_2_let, global_workspace_0_buffer_var.data, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast", sid_2_let, T.lookup_param("p3", dtype="handle"), T.lookup_param("p4", dtype="handle"), sid_8_let, global_workspace_0_buffer_var.data, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", sid_8_let, T.lookup_param("p5", dtype="handle"), T.lookup_param("p6", dtype="handle"), sid_7_let, global_workspace_0_buffer_var.data, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_", sid_7_let, T.lookup_param("p7", dtype="handle"), T.lookup_param("p8", dtype="handle"), sid_6_let, global_workspace_0_buffer_var.data, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_", sid_2_let, T.lookup_param("p1", dtype="handle"), T.lookup_param("p2", dtype="handle"), sid_6_let, output, global_workspace_0_buffer_var.data, dtype="int32"))
    __tvm_meta__ = None
# fmt: on


def test_resnet_subgraph():
    target = Target("c")
    global_workspace_pool = usmp_utils.PoolInfo(
        pool_name="global_workspace",
        target_access={target: usmp_utils.PoolInfo.READ_WRITE_ACCESS},
    )
    tir_mod = ResnetStructure
    tir_mod = _assign_targets_to_primfuncs_irmodule(tir_mod, target)
    tir_mod = assign_poolinfos_to_allocates_in_irmodule(tir_mod, [global_workspace_pool])
    main_func = tir_mod["__tvm_main__"]
    buffer_analysis = tvm.tir.usmp.analysis.extract_buffer_info(main_func, tir_mod)
    buffer_info_map = buffer_analysis.buffer_info_stmts

    fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
    buffer_info_arr = fcreate_array_bi(buffer_info_map)
    fusmp_algo_greedy_by_size = tvm.get_global_func("tir.usmp.algo.greedy_by_size")
    buffer_pool_allocations = fusmp_algo_greedy_by_size(
        buffer_info_arr, buffer_analysis.memory_pressure
    )
    fassign_stmt_pool_allocations = tvm.get_global_func("tir.usmp.AssignStmtPoolAllocations")
    pool_allocations = fassign_stmt_pool_allocations(buffer_info_map, buffer_pool_allocations)
    tir_mod_with_offsets = tvm.tir.usmp.transform.convert_pool_allocations_to_offsets(
        pool_allocations, emit_tvmscript_printable=True
    )(tir_mod)

    tir_mod_with_offsets_ref = ResnetStructurePlanned

    # The TIR produced fails on roundtrip TVMScript testing.
    # Therefore, indicates the TVMScript produced here and/or the parser
    # is lacking functionality. Thus for these tests, uses a string
    # version of the TVMScript for each function as a check instead.
    for gv, func in tir_mod_with_offsets_ref.functions.items():
        assert str(tir_mod_with_offsets_ref[gv.name_hint].script()) == str(
            tir_mod_with_offsets[gv.name_hint].script()
        )


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
