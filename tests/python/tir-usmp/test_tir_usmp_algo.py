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
from tvm.script import tir as T
from tvm.tir import stmt_functor
from tvm.tir.usmp import utils as usmp_utils
from tvm.target import Target
from tvm import WorkspacePoolInfo, PoolInfoProperties


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


def _assign_targets_to_primfuncs_irmodule(mod, target):
    """helper to assign target for PrimFunc in a IRModule"""
    ret = tvm.IRModule()
    for global_var, basefunc in mod.functions.items():
        if isinstance(basefunc, tvm.tir.PrimFunc):
            ret[global_var] = basefunc.with_attr("target", target)
    return ret


def _check_max_workspace_size(buffer_pool_allocations, pool_info, size):
    max_workspace_size = 0
    for buffer_info, pool_allocation in buffer_pool_allocations.items():
        if pool_allocation.pool_info == pool_info:
            size_candidate = pool_allocation.byte_offset + buffer_info.size_bytes
            if size_candidate > max_workspace_size:
                max_workspace_size = size_candidate
    assert max_workspace_size == size


def test_no_pool_error():
    target = Target("c")
    tiny_workspace_pool = WorkspacePoolInfo(
        "tiny_workspace",
        [target],
        PoolInfoProperties(size_hint_bytes=10),
    )
    bi_a = usmp_utils.BufferInfo(
        name_hint="bi_a", size_bytes=10, pool_candidates=[tiny_workspace_pool]
    )
    bi_b = usmp_utils.BufferInfo(
        name_hint="bi_b", size_bytes=10, pool_candidates=[tiny_workspace_pool]
    )
    bi_c = usmp_utils.BufferInfo(
        name_hint="bi_c", size_bytes=10, pool_candidates=[tiny_workspace_pool]
    )
    bi_a.set_conflicts([bi_b])
    bi_b.set_conflicts([bi_c])
    bi_c.set_conflicts([bi_a])
    buffer_info_arr = [bi_a, bi_b, bi_c]
    fusmp_algo = tvm.get_global_func(f"tir.usmp.algo.greedy_by_size")
    with pytest.raises(
        tvm.TVMError, match="TVM USMP Error: the space available in the provided pools exceeded"
    ):
        buffer_pool_allocations = fusmp_algo(buffer_info_arr, 0)


@pytest.mark.parametrize("algorithm", ["greedy_by_size", "greedy_by_conflicts", "hill_climb"])
def test_name_based_ordering(algorithm):
    """This checks when the size and conlicts are same a stable result is generated"""

    def _test():
        target = Target("c")
        global_workspace_pool = WorkspacePoolInfo(
            "global_workspace",
            [target],
        )
        bi_a = usmp_utils.BufferInfo(
            name_hint="bi_a", size_bytes=10, pool_candidates=[global_workspace_pool]
        )
        bi_b = usmp_utils.BufferInfo(
            name_hint="bi_b", size_bytes=10, pool_candidates=[global_workspace_pool]
        )
        bi_c = usmp_utils.BufferInfo(
            name_hint="bi_c", size_bytes=10, pool_candidates=[global_workspace_pool]
        )
        bi_a.set_conflicts([bi_b, bi_c])
        bi_b.set_conflicts([bi_c, bi_a])
        bi_c.set_conflicts([bi_a, bi_b])

        buffer_info_arr = [bi_a, bi_b, bi_c]
        fusmp_algo = tvm.get_global_func(f"tir.usmp.algo.{algorithm}")
        buffer_pool_allocations = fusmp_algo(buffer_info_arr, 0)
        assert buffer_pool_allocations[bi_a].byte_offset == 20
        assert buffer_pool_allocations[bi_b].byte_offset == 10
        assert buffer_pool_allocations[bi_c].byte_offset == 0

    # This is tested for several times to check stability
    for x in range(0, 10):
        _test()


@pytest.mark.parametrize(
    ["algorithm", "workspace_size"],
    [("greedy_by_size", 140), ("greedy_by_conflicts", 140), ("hill_climb", 140)],
)
def test_linear(algorithm, workspace_size):
    """
    The test case here represent BufferInfo objects
    that could get generated for a linear sequence
    such as :
    (Op A)
    |
    bi_a
    |
    (Op B)
    |
    bi_b
    |
    .
    .
    .
    (Op F)
    |
    bi_f
    """
    target = Target("c")
    global_workspace_pool = WorkspacePoolInfo(
        "global_workspace",
        [target],
    )
    bi_a = usmp_utils.BufferInfo(
        name_hint="bi_a", size_bytes=10, pool_candidates=[global_workspace_pool]
    )
    bi_b = usmp_utils.BufferInfo(
        name_hint="bi_b", size_bytes=20, pool_candidates=[global_workspace_pool]
    )
    bi_c = usmp_utils.BufferInfo(
        name_hint="bi_c", size_bytes=100, pool_candidates=[global_workspace_pool]
    )
    bi_d = usmp_utils.BufferInfo(
        name_hint="bi_d", size_bytes=40, pool_candidates=[global_workspace_pool]
    )
    bi_e = usmp_utils.BufferInfo(
        name_hint="bi_e", size_bytes=50, pool_candidates=[global_workspace_pool]
    )
    bi_f = usmp_utils.BufferInfo(
        name_hint="bi_f", size_bytes=50, pool_candidates=[global_workspace_pool]
    )

    # Creating conflicts for a linear graph
    bi_a.set_conflicts([bi_b])
    bi_b.set_conflicts([bi_a, bi_c])
    bi_c.set_conflicts([bi_b, bi_d])
    bi_d.set_conflicts([bi_c, bi_e])
    bi_e.set_conflicts([bi_d, bi_f])
    bi_f.set_conflicts([bi_e])

    buffer_info_arr = [bi_a, bi_b, bi_c, bi_d, bi_e, bi_f]
    fusmp_algo = tvm.get_global_func(f"tir.usmp.algo.{algorithm}")
    buffer_pool_allocations = fusmp_algo(buffer_info_arr, 0)
    _check_max_workspace_size(buffer_pool_allocations, global_workspace_pool, workspace_size)


@pytest.mark.parametrize(
    ["algorithm", "workspace_size"],
    [("greedy_by_size", 190), ("greedy_by_conflicts", 320), ("hill_climb", 190)],
)
def test_fanout(algorithm, workspace_size):
    """
    The test case here represent BufferInfo objects
    that could get generated for a fanout topology
    such as :
    (Op A)
    |
    bi_a ---------
    |            |
    (Op B)     (Op C)
    |            |
    bi_b        bi_c
    |            |
    (Op D)     (Op E)
    |            |
    bi_d        bi_e
    |            |
    (Op F) ------
    |
    bi_f
    |
    (Op G)
    |
    bi_g
    """
    target = Target("c")
    global_workspace_pool = WorkspacePoolInfo(
        "global_workspace",
        targets=[target],
    )
    bi_a = usmp_utils.BufferInfo(
        name_hint="bi_a", size_bytes=10, pool_candidates=[global_workspace_pool]
    )
    bi_b = usmp_utils.BufferInfo(
        name_hint="bi_b", size_bytes=20, pool_candidates=[global_workspace_pool]
    )
    bi_c = usmp_utils.BufferInfo(
        name_hint="bi_c", size_bytes=100, pool_candidates=[global_workspace_pool]
    )
    bi_d = usmp_utils.BufferInfo(
        name_hint="bi_d", size_bytes=40, pool_candidates=[global_workspace_pool]
    )
    bi_e = usmp_utils.BufferInfo(
        name_hint="bi_e", size_bytes=50, pool_candidates=[global_workspace_pool]
    )
    bi_f = usmp_utils.BufferInfo(
        name_hint="bi_f", size_bytes=60, pool_candidates=[global_workspace_pool]
    )
    bi_g = usmp_utils.BufferInfo(
        name_hint="bi_g", size_bytes=70, pool_candidates=[global_workspace_pool]
    )

    # Creating conflicts for a linear graph
    bi_a.set_conflicts([bi_b, bi_c])
    bi_b.set_conflicts([bi_a, bi_c, bi_e])
    bi_c.set_conflicts([bi_e, bi_a, bi_b, bi_d])
    bi_d.set_conflicts([bi_b, bi_f, bi_c, bi_e])
    bi_e.set_conflicts([bi_c, bi_f, bi_b, bi_d])
    bi_f.set_conflicts([bi_d, bi_e, bi_f])
    bi_g.set_conflicts([bi_f])

    buffer_info_arr = [bi_a, bi_b, bi_c, bi_d, bi_e, bi_f, bi_g]
    fusmp_algo = tvm.get_global_func(f"tir.usmp.algo.{algorithm}")
    buffer_pool_allocations = fusmp_algo(buffer_info_arr, 0)
    _check_max_workspace_size(buffer_pool_allocations, global_workspace_pool, workspace_size)


# fmt: off
@tvm.script.ir_module
class MobilenetStructure:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [150528], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast(placeholder_62: T.handle, placeholder_63: T.handle, placeholder_64: T.handle, T_cast_20: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast", "tir.noalias": True})
        placeholder_65 = T.match_buffer(placeholder_62, [150528], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        placeholder_66 = T.match_buffer(placeholder_63, [9408], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        placeholder_67 = T.match_buffer(placeholder_64, [64], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        T_cast_21 = T.match_buffer(T_cast_20, [802816], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        # body
        PaddedInput_7 = T.decl_buffer([157323], "int16")
        for i0_i1_fused_7 in T.serial(0, 229):
            for i2_7, i3_7 in T.grid(229, 3):
                PaddedInput_7[(((i0_i1_fused_7*687) + (i2_7*3)) + i3_7)] = T.if_then_else(((((2 <= i0_i1_fused_7) and (i0_i1_fused_7 < 226)) and (2 <= i2_7)) and (i2_7 < 226)), placeholder_65[((((i0_i1_fused_7*672) + (i2_7*3)) + i3_7) - 1350)], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_7 in T.serial(0, 12544):
            Conv2dOutput_7 = T.decl_buffer([64], "int32")
            for ff_3 in T.serial(0, 64):
                Conv2dOutput_7[ff_3] = 0
                for ry_2, rx_2, rc_7 in T.grid(7, 7, 3):
                    Conv2dOutput_7[ff_3] = (Conv2dOutput_7[ff_3] + (T.cast(PaddedInput_7[(((((T.floordiv(ax0_ax1_fused_ax2_fused_7, 112)*1374) + (ry_2*687)) + (T.floormod(ax0_ax1_fused_ax2_fused_7, 112)*6)) + (rx_2*3)) + rc_7)], "int32")*T.cast(placeholder_66[((((ry_2*1344) + (rx_2*192)) + (rc_7*64)) + ff_3)], "int32")))
            for ax3_inner_7 in T.serial(0, 64):
                T_cast_21[((ax0_ax1_fused_ax2_fused_7*64) + ax3_inner_7)] = T.cast(T.max(T.min(T.q_multiply_shift((Conv2dOutput_7[ax3_inner_7] + placeholder_67[ax3_inner_7]), 1939887962, 31, -9, dtype="int32"), 255), 0), "uint8")

    @T.prim_func
    def tvmgen_default_fused_nn_max_pool2d_cast(placeholder_28: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_28, [802816], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        T_cast_7 = T.match_buffer(T_cast_6, [200704], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        tensor_2 = T.decl_buffer([200704], "uint8")
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_init)] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)] = T.max(tensor_2[(((ax0_ax1_fused_4*3584) + (ax2_4*64)) + ax3_2)], T.if_then_else(((((ax0_ax1_fused_4*2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112) and (((ax2_4*2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)), placeholder_29[(((((ax0_ax1_fused_4*14336) + (T.floordiv(rv0_rv1_fused_1, 3)*7168)) + (ax2_4*128)) + (T.floormod(rv0_rv1_fused_1, 3)*64)) + ax3_2)], T.uint8(0), dtype="uint8"))
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T_cast_7[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)] = T.cast(tensor_2[(((ax0_ax1_fused_5*3584) + (ax2_5*64)) + ax3_3)], "int16")

    @T.prim_func
    def run_model(input: T.handle, output: T.handle) -> None:
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


@pytest.mark.parametrize(
    ["algorithm", "fast_memory_size", "slow_memory_size"],
    [
        ("greedy_by_size", 200704, 1418528),
        ("greedy_by_conflicts", 200704, 1418528),
        ("hill_climb", 200704, 1117462),
    ],
)
def test_mobilenet_subgraph(algorithm, fast_memory_size, slow_memory_size):
    target = Target("c")
    fast_memory_pool = WorkspacePoolInfo(
        "fast_memory",
        [target],
        PoolInfoProperties(size_hint_bytes=200704),
    )
    slow_memory_pool = WorkspacePoolInfo(
        "slow_memory",
        [target],
    )
    tir_mod = MobilenetStructure
    tir_mod = _assign_targets_to_primfuncs_irmodule(tir_mod, target)
    tir_mod = _assign_poolinfos_to_allocates_in_irmodule(
        tir_mod, [fast_memory_pool, slow_memory_pool]
    )
    main_func = tir_mod["run_model"]
    buffer_info_analysis = tvm.tir.usmp.analysis.extract_buffer_info(main_func, tir_mod)
    assert buffer_info_analysis.memory_pressure == 1117718

    fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
    buffer_info_arr = fcreate_array_bi(buffer_info_analysis.buffer_info_stmts)
    fusmp_algo = tvm.get_global_func(f"tir.usmp.algo.{algorithm}")
    buffer_pool_allocations = fusmp_algo(buffer_info_arr, buffer_info_analysis.memory_pressure)

    buffer_info_map_names = dict()
    for buf_info in buffer_info_arr:
        buffer_info_map_names[buf_info.name_hint] = buf_info

    # check conflicts
    _verify_conflicts("PaddedInput_7", ["sid_9", "sid_8", "Conv2dOutput_7"], buffer_info_map_names)
    _verify_conflicts("tensor_2", ["sid_8"], buffer_info_map_names)
    _verify_conflicts("sid_9", ["PaddedInput_7"], buffer_info_map_names)
    _verify_conflicts(
        "sid_8", ["PaddedInput_7", "Conv2dOutput_7", "tensor_2"], buffer_info_map_names
    )
    _verify_conflicts("Conv2dOutput_7", ["sid_8", "PaddedInput_7"], buffer_info_map_names)

    _check_max_workspace_size(buffer_pool_allocations, slow_memory_pool, slow_memory_size)
    _check_max_workspace_size(buffer_pool_allocations, fast_memory_pool, fast_memory_size)


# fmt: off
@tvm.script.ir_module
class ResnetStructure:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast(placeholder: T.handle, placeholder_1: T.handle, T_cast: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [360000], dtype="uint8")
        placeholder_3 = T.match_buffer(placeholder_1, [64], dtype="int32")
        T_cast_1 = T.match_buffer(T_cast, [360000], dtype="int16")
        # body
        for ax0_ax1_fused, ax2, ax3_outer, ax3_inner in T.grid(75, 75, 4, 16):
            T_cast_1[ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(T.cast(placeholder_2[ax0_ax1_fused * 4800 + ax2 * 64 + ax3_outer * 16 + ax3_inner], "int32") - 94, 1843157232, 31, 1, dtype="int32") + placeholder_3[ax3_outer * 16 + ax3_inner], 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1(placeholder_10: T.handle, placeholder_11: T.handle, placeholder_12: T.handle, T_cast_4: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_1", "tir.noalias": True})
        placeholder_13 = T.match_buffer(placeholder_10, [360000], dtype="int16")
        placeholder_14 = T.match_buffer(placeholder_11, [36864], dtype="int16")
        placeholder_15 = T.match_buffer(placeholder_12, [64], dtype="int32")
        T_cast_5 = T.match_buffer(T_cast_4, [360000], dtype="int16")
        # body
        PaddedInput_1 = T.decl_buffer([379456], "int16")
        for i0_i1_fused_1, i2_1, i3_1 in T.grid(77, 77, 64):
            PaddedInput_1[i0_i1_fused_1 * 4928 + i2_1 * 64 + i3_1] = T.if_then_else(1 <= i0_i1_fused_1 and i0_i1_fused_1 < 76 and 1 <= i2_1 and i2_1 < 76, placeholder_13[i0_i1_fused_1 * 4800 + i2_1 * 64 + i3_1 - 4864], T.int16(0), dtype="int16")
        for ax0_ax1_fused_ax2_fused_1 in T.serial(0, 5625):
            Conv2dOutput_1 = T.decl_buffer([64], "int32")
            for ff_1 in T.serial(0, 64):
                Conv2dOutput_1[ff_1] = 0
                for ry, rx, rc_1 in T.grid(3, 3, 64):
                    Conv2dOutput_1[ff_1] = Conv2dOutput_1[ff_1] + T.cast(PaddedInput_1[T.floordiv(ax0_ax1_fused_ax2_fused_1, 75) * 4928 + ry * 4928 + rx * 64 + T.floormod(ax0_ax1_fused_ax2_fused_1, 75) * 64 + rc_1], "int32") * T.cast(placeholder_14[ry * 12288 + rx * 4096 + rc_1 * 64 + ff_1], "int32")
            for ax3_inner_2 in T.serial(0, 64):
                T_cast_5[ax0_ax1_fused_ax2_fused_1 * 64 + ax3_inner_2] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_1[ax3_inner_2] + placeholder_15[ax3_inner_2], 1608879842, 31, -7, dtype="int32"), 255), 0), "uint8"), "int16")

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_(placeholder_16: T.handle, placeholder_17: T.handle, placeholder_18: T.handle, T_add: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_15934180698220515269_", "tir.noalias": True})
        placeholder_19 = T.match_buffer(placeholder_16, [360000], dtype="int16")
        placeholder_20 = T.match_buffer(placeholder_17, [16384], dtype="int16")
        placeholder_21 = T.match_buffer(placeholder_18, [256], dtype="int32")
        T_add_1 = T.match_buffer(T_add, [1440000], dtype="int32")
        # body
        PaddedInput_2 = T.decl_buffer([360000], "int16")
        for i0_i1_fused_2, i2_2, i3_2 in T.grid(75, 75, 64):
            PaddedInput_2[i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2] = placeholder_19[i0_i1_fused_2 * 4800 + i2_2 * 64 + i3_2]
        for ax0_ax1_fused_ax2_fused_2 in T.serial(0, 5625):
            Conv2dOutput_2 = T.decl_buffer([64], "int32")
            for ax3_outer_1 in T.serial(0, 4):
                for ff_2 in T.serial(0, 64):
                    Conv2dOutput_2[ff_2] = 0
                    for rc_2 in T.serial(0, 64):
                        Conv2dOutput_2[ff_2] = Conv2dOutput_2[ff_2] + T.cast(PaddedInput_2[ax0_ax1_fused_ax2_fused_2 * 64 + rc_2], "int32") * T.cast(placeholder_20[rc_2 * 256 + ax3_outer_1 * 64 + ff_2], "int32")
                for ax3_inner_3 in T.serial(0, 64):
                    T_add_1[ax0_ax1_fused_ax2_fused_2 * 256 + ax3_outer_1 * 64 + ax3_inner_3] = T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_2[ax3_inner_3] + placeholder_21[ax3_outer_1 * 64 + ax3_inner_3], 1711626602, 31, -8, dtype="int32") + 132, 255), 0), "uint8"), "int32") - 132, 2094289803, 31, -2, dtype="int32") + 136

    @T.prim_func
    def tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_(placeholder_22: T.handle, placeholder_23: T.handle, placeholder_24: T.handle, placeholder_25: T.handle, T_cast_6: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_nn_conv2d_add_fixed_point_multiply_add_clip_cast_cast_subtract_fixed_point_4200876283395191415_", "tir.noalias": True})
        placeholder_29 = T.match_buffer(placeholder_22, [360000], dtype="int16")
        placeholder_27 = T.match_buffer(placeholder_23, [16384], dtype="int16")
        placeholder_26 = T.match_buffer(placeholder_24, [256], dtype="int32")
        placeholder_28 = T.match_buffer(placeholder_25, [1440000], dtype="int32")
        T_cast_7 = T.match_buffer(T_cast_6, [1440000], dtype="uint8")
        # body
        PaddedInput_3 = T.decl_buffer([360000], "int16")
        for i0_i1_fused_3, i2_3, i3_3 in T.grid(75, 75, 64):
            PaddedInput_3[i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3] = placeholder_29[i0_i1_fused_3 * 4800 + i2_3 * 64 + i3_3]
        for ax0_ax1_fused_ax2_fused_3 in T.serial(0, 5625):
            Conv2dOutput_3 = T.decl_buffer([64], "int32")
            for ax3_outer_2 in T.serial(0, 4):
                for ff_3 in T.serial(0, 64):
                    Conv2dOutput_3[ff_3] = 0
                    for rc_3 in T.serial(0, 64):
                        Conv2dOutput_3[ff_3] = Conv2dOutput_3[ff_3] + T.cast(PaddedInput_3[ax0_ax1_fused_ax2_fused_3 * 64 + rc_3], "int32") * T.cast(placeholder_27[rc_3 * 256 + ax3_outer_2 * 64 + ff_3], "int32")
                for ax3_inner_4 in T.serial(0, 64):
                    T_cast_7[ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4] = T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput_3[ax3_inner_4] + placeholder_26[ax3_outer_2 * 64 + ax3_inner_4], 1343014664, 31, -8, dtype="int32") + 136, 255), 0), "uint8"), "int32") - 136, 1073903788, 31, 1, dtype="int32") + placeholder_28[ax0_ax1_fused_ax2_fused_3 * 256 + ax3_outer_2 * 64 + ax3_inner_4], 255), 0), "uint8")

    @T.prim_func
    def tvmgen_default_run_model(input: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_run_model", "runner_function": True})
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
        placeholder_7 = T.match_buffer(placeholder_4, [360000], dtype="int16")
        placeholder_8 = T.match_buffer(placeholder_5, [4096], dtype="int16")
        placeholder_9 = T.match_buffer(placeholder_6, [64], dtype="int32")
        T_cast_3 = T.match_buffer(T_cast_2, [360000], dtype="int16")
        # body
        PaddedInput = T.decl_buffer([360000], "int16")
        for i0_i1_fused, i2, i3 in T.grid(75, 75, 64):
            PaddedInput[i0_i1_fused * 4800 + i2 * 64 + i3] = placeholder_7[i0_i1_fused * 4800 + i2 * 64 + i3]
        for ax0_ax1_fused_ax2_fused in T.serial(0, 5625):
            Conv2dOutput = T.decl_buffer([64], "int32")
            for ff in T.serial(0, 64):
                Conv2dOutput[ff] = 0
                for rc in T.serial(0, 64):
                    Conv2dOutput[ff] = Conv2dOutput[ff] + T.cast(PaddedInput[ax0_ax1_fused_ax2_fused * 64 + rc], "int32") * T.cast(placeholder_8[rc * 64 + ff], "int32")
            for ax3_inner_1 in T.serial(0, 64):
                T_cast_3[ax0_ax1_fused_ax2_fused * 64 + ax3_inner_1] = T.cast(T.cast(T.max(T.min(T.q_multiply_shift(Conv2dOutput[ax3_inner_1] + placeholder_9[ax3_inner_1], 1843106743, 31, -6, dtype="int32"), 255), 0), "uint8"), "int16")
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize(
    ["algorithm", "workspace_size"],
    [("greedy_by_size", 7920256), ("greedy_by_conflicts", 7200256), ("hill_climb", 7200256)],
)
def test_resnet_subgraph(algorithm, workspace_size):
    target = Target("c")
    global_workspace_pool = WorkspacePoolInfo(
        "global_workspace",
        [target],
    )
    tir_mod = ResnetStructure
    tir_mod = _assign_targets_to_primfuncs_irmodule(tir_mod, target)
    tir_mod = _assign_poolinfos_to_allocates_in_irmodule(tir_mod, [global_workspace_pool])
    main_func = tir_mod["tvmgen_default_run_model"]
    buffer_info_analysis = tvm.tir.usmp.analysis.extract_buffer_info(main_func, tir_mod)
    assert buffer_info_analysis.memory_pressure == 7200256

    fcreate_array_bi = tvm.get_global_func("tir.usmp.CreateArrayBufferInfo")
    buffer_info_arr = fcreate_array_bi(buffer_info_analysis.buffer_info_stmts)
    fusmp_algo = tvm.get_global_func(f"tir.usmp.algo.{algorithm}")
    buffer_pool_allocations = fusmp_algo(buffer_info_arr, buffer_info_analysis.memory_pressure)

    buffer_info_map_names = dict()
    for buf_info in buffer_info_arr:
        buffer_info_map_names[buf_info.name_hint] = buf_info

    # check conflicts
    _verify_conflicts(
        "sid_7",
        [
            "PaddedInput_1",
            "sid_2",
            "Conv2dOutput_1",
            "PaddedInput_2",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "Conv2dOutput_3",
        [
            "PaddedInput_3",
            "sid_6",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "sid_6",
        [
            "Conv2dOutput_2",
            "PaddedInput_2",
            "sid_2",
            "PaddedInput_3",
            "Conv2dOutput_3",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "Conv2dOutput",
        [
            "sid_8",
            "sid_2",
            "PaddedInput",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "PaddedInput_3",
        [
            "sid_6",
            "sid_2",
            "Conv2dOutput_3",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "Conv2dOutput_2",
        [
            "PaddedInput_2",
            "sid_2",
            "sid_6",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "PaddedInput_1",
        [
            "sid_8",
            "sid_2",
            "sid_7",
            "Conv2dOutput_1",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "Conv2dOutput_1",
        [
            "sid_7",
            "PaddedInput_1",
            "sid_2",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "PaddedInput",
        [
            "sid_2",
            "sid_8",
            "Conv2dOutput",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "sid_8",
        [
            "PaddedInput",
            "sid_2",
            "Conv2dOutput",
            "PaddedInput_1",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "sid_2",
        [
            "PaddedInput",
            "sid_8",
            "Conv2dOutput",
            "PaddedInput_1",
            "sid_7",
            "Conv2dOutput_1",
            "PaddedInput_2",
            "Conv2dOutput_2",
            "sid_6",
            "PaddedInput_3",
        ],
        buffer_info_map_names,
    )
    _verify_conflicts(
        "PaddedInput_2",
        [
            "sid_7",
            "sid_2",
            "Conv2dOutput_2",
            "sid_6",
        ],
        buffer_info_map_names,
    )

    _check_max_workspace_size(buffer_pool_allocations, global_workspace_pool, workspace_size)


def test_custom_algo():
    target = Target("c")
    global_workspace_pool = WorkspacePoolInfo(
        "global_workspace",
        [target],
    )
    tir_mod = ResnetStructure
    tir_mod = _assign_targets_to_primfuncs_irmodule(tir_mod, target)
    tir_mod = _assign_poolinfos_to_allocates_in_irmodule(tir_mod, [global_workspace_pool])
    tir_mod = tir_mod.with_attr("executor", tvm.relay.backend.Executor("aot"))
    tir_mod = tir_mod.with_attr("runtime", tvm.relay.backend.Runtime("crt"))
    tir_mod["__tvm_main__"] = tir_mod[
        "tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_clip_cast_cast"
    ]

    algo_called = False

    @tvm.register_func("tir.usmp.algo.trivial")
    def _trivial_algo(buf_infos, mem_pressure):
        nonlocal algo_called
        algo_called = True
        out_layout = {}
        offset = 0
        for buf_info in buf_infos:
            pool_info = buf_info.pool_candidates[0]
            out_layout[buf_info] = usmp_utils.PoolAllocation(pool_info, offset)
            offset += buf_info.size_bytes
        return out_layout

    usmp_pass = tvm.get_global_func("tir.transform.UnifiedStaticMemoryPlanner")
    usmp_pass()(tir_mod)
    assert not algo_called

    with tvm.transform.PassContext(config={"tir.usmp.custom_algorithm": "trivial"}):
        usmp_pass()(tir_mod)

    assert algo_called

    with pytest.raises(
        tvm.TVMError, match="The selected custom USMP algorithm : invalid is not defined"
    ):
        with tvm.transform.PassContext(config={"tir.usmp.custom_algorithm": "invalid"}):
            usmp_pass()(tir_mod)
