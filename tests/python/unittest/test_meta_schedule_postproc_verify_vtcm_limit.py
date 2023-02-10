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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T


def _create_context(mod, target) -> ms.TuneContext:
    return ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[ms.postproc.VerifyVTCMLimit()],
            mutator_probs={},
        ),
        task_name="test",
    )


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable,misplaced-comparison-constant
# fmt: off


@tvm.script.ir_module
class Conv2dNCHWcVTCM:
    @T.prim_func
    def main(p0: T.Buffer((T.int64(1), T.int64(2), T.int64(56), T.int64(56), T.int64(32)), "uint8"), p1: T.Buffer((T.int64(2), T.int64(2), T.int64(3), T.int64(3), T.int64(8), T.int64(32), T.int64(4)), "uint8"), conv2d_NCHWc_int8: T.Buffer((T.int64(1), T.int64(2), T.int64(54), T.int64(54), T.int64(32)), "int32")):
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        p0_global_vtcm = T.alloc_buffer([T.int64(1), T.int64(2), T.int64(56), T.int64(56), T.int64(32)], dtype="uint8", scope="global.vtcm")
        p1_global_vtcm = T.alloc_buffer([T.int64(2), T.int64(2), T.int64(3), T.int64(3), T.int64(8), T.int64(32), T.int64(4)], dtype="uint8", scope="global.vtcm")
        for n_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step":16, "pragma_unroll_explicit":1}):
            for oc_chunk_0, oh_0, ow_0, oc_block_0_0 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(1)):
                for oc_chunk_1_init, oh_1_init, ow_1_init, oc_chunk_2_init, oh_2_init, ow_2_init in T.grid(T.int64(1), T.int64(27), T.int64(3), T.int64(1), T.int64(1), T.int64(9)):
                    with T.block("conv2d_NCHWc_int8_o_init"):
                        v_n = T.axis.spatial(T.int64(1), T.int64(0))
                        v_oc_chunk = T.axis.spatial(T.int64(2), oc_chunk_1_init + oc_chunk_2_init + oc_chunk_0)
                        v_oh = T.axis.spatial(T.int64(54), oh_2_init + oh_0 * T.int64(27) + oh_1_init)
                        v_ow = T.axis.spatial(T.int64(54), ow_0 * T.int64(27) + ow_1_init * T.int64(9) + ow_2_init)
                        v_oc_block_o = T.axis.spatial(T.int64(1), T.int64(0))
                        T.reads()
                        T.writes(conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, T.int64(0) : T.int64(32)])
                        for oc_block_1 in T.vectorized(T.int64(32)):
                            with T.block("conv2d_NCHWc_int8_init"):
                                v_oc_block_i_init = T.axis.spatial(T.int64(32), oc_block_1)
                                T.reads()
                                T.writes(conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i_init])
                                conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i_init] = 0
                for kh_0_kw_0_ic_outer_0_ic_f_inner_0_ic_s_inner_0_0_fused in T.serial(T.int64(2), annotations={"software_pipeline_async_stages":[0], "software_pipeline_order":[0, 1, 2], "software_pipeline_stage":[0, 0, 1]}):
                    for ax0_ax1_ax2_ax3_ax4_fused in T.serial(T.int64(26912)):
                        with T.block("p0_global.vtcm"):
                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v1 = T.axis.spatial(T.int64(2), ax0_ax1_ax2_ax3_ax4_fused // T.int64(13456))
                            v2 = T.axis.spatial(T.int64(56), oh_0 * T.int64(27) + ax0_ax1_ax2_ax3_ax4_fused % T.int64(13456) // T.int64(464))
                            v3 = T.axis.spatial(T.int64(56), ow_0 * T.int64(27) + ax0_ax1_ax2_ax3_ax4_fused % T.int64(464) // T.int64(16))
                            v4 = T.axis.spatial(T.int64(32), kh_0_kw_0_ic_outer_0_ic_f_inner_0_ic_s_inner_0_0_fused * T.int64(16) + ax0_ax1_ax2_ax3_ax4_fused % T.int64(16))
                            T.reads(p0[v0, v1, v2, v3, v4])
                            T.writes(p0_global_vtcm[v0, v1, v2, v3, v4])
                            p0_global_vtcm[v0, v1, v2, v3, v4] = p0[v0, v1, v2, v3, v4]
                    for ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused in T.serial(T.int64(9216)):
                        with T.block("p1_global.vtcm"):
                            v0 = T.axis.spatial(T.int64(2), oc_chunk_0)
                            v1 = T.axis.spatial(T.int64(2), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused // T.int64(4608))
                            v2 = T.axis.spatial(T.int64(3), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % T.int64(4608) // T.int64(1536))
                            v3 = T.axis.spatial(T.int64(3), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % T.int64(1536) // T.int64(512))
                            v4 = T.axis.spatial(T.int64(8), kh_0_kw_0_ic_outer_0_ic_f_inner_0_ic_s_inner_0_0_fused * T.int64(4) + ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % T.int64(512) // T.int64(128))
                            v5 = T.axis.spatial(T.int64(32), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % T.int64(128) // T.int64(4))
                            v6 = T.axis.spatial(T.int64(4), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % T.int64(4))
                            T.reads(p1[v0, v1, v2, v3, v4, v5, v6])
                            T.writes(p1_global_vtcm[v0, v1, v2, v3, v4, v5, v6])
                            p1_global_vtcm[v0, v1, v2, v3, v4, v5, v6] = p1[v0, v1, v2, v3, v4, v5, v6]
                    for n_1, oc_chunk_1, oh_1, ow_1, oc_block_0_1, kh_1, kw_1, ic_outer_1, ic_f_inner_1, ic_s_inner_0_1, n_2, oc_chunk_2, oh_2, ow_2, oc_block_0_2 in T.grid(T.int64(1), T.int64(1), T.int64(27), T.int64(3), T.int64(1), T.int64(3), T.int64(3), T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(9), T.int64(1)):
                        with T.block("conv2d_NCHWc_int8_o_update"):
                            v_n = T.axis.spatial(T.int64(1), T.int64(0))
                            v_oc_chunk = T.axis.spatial(T.int64(2), oc_chunk_1 + oc_chunk_2 + oc_chunk_0)
                            v_oh = T.axis.spatial(T.int64(54), oh_2 + oh_0 * T.int64(27) + oh_1)
                            v_ow = T.axis.spatial(T.int64(54), ow_0 * T.int64(27) + ow_1 * T.int64(9) + ow_2)
                            v_oc_block_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v_kh, v_kw, v_ic_outer = T.axis.remap("RRR", [kh_1, kw_1, ic_outer_1])
                            v_ic_f_inner = T.axis.reduce(T.int64(8), kh_0_kw_0_ic_outer_0_ic_f_inner_0_ic_s_inner_0_0_fused * T.int64(4) + ic_f_inner_1)
                            v_ic_s_inner_o = T.axis.reduce(T.int64(1), T.int64(0))
                            T.reads(conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, T.int64(0) : T.int64(32)], p0_global_vtcm[v_n, v_ic_outer, v_oh + v_kh, v_ow + v_kw, v_ic_f_inner * T.int64(4) : v_ic_f_inner * T.int64(4) + T.int64(4)], p1_global_vtcm[v_oc_chunk, v_ic_outer, v_kh, v_kw, v_ic_f_inner, T.int64(0) : T.int64(32), T.int64(0) : T.int64(4)])
                            T.writes(conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, T.int64(0) : T.int64(32)])
                            for oc_block_1, ic_s_inner_1 in T.grid(T.int64(32), T.int64(4)):
                                with T.block("conv2d_NCHWc_int8"):
                                    v_oc_block_i, v_ic_s_inner_i = T.axis.remap("SR", [oc_block_1, ic_s_inner_1])
                                    T.reads(conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i], p0_global_vtcm[v_n, v_ic_outer, v_oh + v_kh, v_ow + v_kw, v_ic_f_inner * T.int64(4) + v_ic_s_inner_i], p1_global_vtcm[v_oc_chunk, v_ic_outer, v_kh, v_kw, v_ic_f_inner, v_oc_block_i, v_ic_s_inner_i])
                                    T.writes(conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i])
                                    T.block_attr({"meta_schedule.tiling_structure":"SRSRS"})
                                    conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i] = conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i] + T.Cast("int32", p0_global_vtcm[v_n, v_ic_outer, v_oh + v_kh, v_ow + v_kw, v_ic_f_inner * T.int64(4) + v_ic_s_inner_i]) * T.Cast("int32", p1_global_vtcm[v_oc_chunk, v_ic_outer, v_kh, v_kw, v_ic_f_inner, v_oc_block_i, v_ic_s_inner_i])

#fmt on


def test_conv2d_vtcm():
    def get_target(vtcm_cap):
        target = tvm.target.hexagon("v68", vtcm_capacity=vtcm_cap)
        return tvm.target.Target(target, host=target)

    sch = tir.Schedule(Conv2dNCHWcVTCM, debug_mask="all")

    ctx = _create_context(Conv2dNCHWcVTCM, target=get_target(70000))
    assert not ctx.space_generator.postprocs[0].apply(sch)

    ctx = _create_context(Conv2dNCHWcVTCM, target=get_target(75000))
    assert ctx.space_generator.postprocs[0].apply(sch)


if __name__ == "__main__":
    tvm.testing.main()
