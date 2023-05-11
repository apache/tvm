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

""" Test different strategies for loading data into vtcm before running HVX workloads. """

import numpy as np
import pytest
import tvm
from tvm.script import tir as T

VRMPY_SIZE_B = 128
VRMPY_SIZE_INT32 = 32

# pylint: disable=invalid-name
@T.prim_func
def conv2d_async_non_contig(
    p0: T.Buffer((T.int64(1), T.int64(1), T.int64(56), T.int64(56), T.int64(4)), "uint8"),
    fused_constant_1: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(1), T.int64(32), T.int64(4)),
        "uint8",
    ),
    conv2d_NCHWc_int8: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(54), T.int64(54), T.int64(32)), "int32"
    ),
):
    """Non contiguous memory access is used in this conv2d taken from MS."""
    # pylint: disable=no-self-argument
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    p0_global_vtcm = T.alloc_buffer(
        [T.int64(1), T.int64(1), T.int64(56), T.int64(56), T.int64(4)],
        dtype="uint8",
        scope="global.vtcm",
    )
    fused_constant_global_vtcm = T.alloc_buffer(
        [T.int64(1), T.int64(1), T.int64(3), T.int64(3), T.int64(1), T.int64(32), T.int64(4)],
        dtype="uint8",
        scope="global.vtcm",
    )
    for oh_0 in T.serial(T.int64(3)):
        for ow_0 in T.serial(
            T.int64(3),
            annotations={
                "software_pipeline_async_stages": [0],
                "software_pipeline_order": [0, 1, 2],
                "software_pipeline_stage": [0, 0, 1],
            },
        ):
            for ax0_ax1_ax2_ax3_ax4_fused in T.serial(T.int64(1600)):
                with T.block("p0_global.vtcm"):
                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v2 = T.axis.spatial(
                        T.int64(56), oh_0 * T.int64(18) + ax0_ax1_ax2_ax3_ax4_fused // T.int64(80)
                    )
                    v3 = T.axis.spatial(
                        T.int64(56),
                        ow_0 * T.int64(18) + ax0_ax1_ax2_ax3_ax4_fused % T.int64(80) // T.int64(4),
                    )
                    v4 = T.axis.spatial(T.int64(4), ax0_ax1_ax2_ax3_ax4_fused % T.int64(4))
                    T.reads(p0[v0, v1, v2, v3, v4])
                    T.writes(p0_global_vtcm[v0, v1, v2, v3, v4])
                    p0_global_vtcm[v0, v1, v2, v3, v4] = p0[v0, v1, v2, v3, v4]
            for ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused in T.serial(T.int64(1152)):
                with T.block("fused_constant_global.vtcm"):
                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v2 = T.axis.spatial(
                        T.int64(3), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused // T.int64(384)
                    )
                    v3 = T.axis.spatial(
                        T.int64(3), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % T.int64(384) // T.int64(128)
                    )
                    v4 = T.axis.spatial(T.int64(1), T.int64(0))
                    v5 = T.axis.spatial(
                        T.int64(32), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % T.int64(128) // T.int64(4)
                    )
                    v6 = T.axis.spatial(T.int64(4), ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % T.int64(4))
                    T.reads(fused_constant_1[v0, v1, v2, v3, v4, v5, v6])
                    T.writes(fused_constant_global_vtcm[v0, v1, v2, v3, v4, v5, v6])
                    fused_constant_global_vtcm[v0, v1, v2, v3, v4, v5, v6] = fused_constant_1[
                        v0, v1, v2, v3, v4, v5, v6
                    ]
            for oh_1, ow_1 in T.grid(T.int64(3), T.int64(6)):
                for oh_2_init, ow_2_init in T.grid(T.int64(6), T.int64(3)):
                    with T.block("conv2d_NCHWc_int8_o_init"):
                        v_n = T.axis.spatial(T.int64(1), T.int64(0))
                        v_oc_chunk = T.axis.spatial(T.int64(1), T.int64(0))
                        v_oh = T.axis.spatial(
                            T.int64(54), oh_0 * T.int64(18) + oh_1 * T.int64(6) + oh_2_init
                        )
                        v_ow = T.axis.spatial(
                            T.int64(54), ow_0 * T.int64(18) + ow_1 * T.int64(3) + ow_2_init
                        )
                        T.reads()
                        T.writes(
                            conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, T.int64(0) : T.int64(32)]
                        )
                        for oc_block_1 in T.vectorized(T.int64(32)):
                            with T.block("conv2d_NCHWc_int8_init"):
                                v_oc_block_i_init = T.axis.spatial(T.int64(32), oc_block_1)
                                T.reads()
                                T.writes(
                                    conv2d_NCHWc_int8[
                                        v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i_init
                                    ]
                                )
                                conv2d_NCHWc_int8[
                                    v_n, v_oc_chunk, v_oh, v_ow, v_oc_block_i_init
                                ] = 0
                for kh_1, kw_1, oh_2, ow_2 in T.grid(
                    T.int64(3), T.int64(3), T.int64(6), T.int64(3)
                ):
                    with T.block("conv2d_NCHWc_int8_o_update"):
                        v_n = T.axis.spatial(T.int64(1), T.int64(0))
                        v_oc_chunk = T.axis.spatial(T.int64(1), T.int64(0))
                        v_oh = T.axis.spatial(
                            T.int64(54), oh_0 * T.int64(18) + oh_1 * T.int64(6) + oh_2
                        )
                        v_ow = T.axis.spatial(
                            T.int64(54), ow_0 * T.int64(18) + ow_1 * T.int64(3) + ow_2
                        )
                        v_kh, v_kw = T.axis.remap("RR", [kh_1, kw_1])
                        v_ic_outer = T.axis.reduce(T.int64(1), T.int64(0))
                        v_ic_f_inner = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(
                            conv2d_NCHWc_int8[
                                v_n, v_oc_chunk, v_oh, v_ow, T.int64(0) : T.int64(32)
                            ],
                            p0_global_vtcm[
                                v_n,
                                v_ic_outer,
                                v_oh + v_kh,
                                v_ow + v_kw,
                                v_ic_f_inner * T.int64(4) : v_ic_f_inner * T.int64(4) + T.int64(4),
                            ],
                            fused_constant_global_vtcm[
                                v_oc_chunk,
                                v_ic_outer,
                                v_kh,
                                v_kw,
                                v_ic_f_inner,
                                T.int64(0) : T.int64(32),
                                T.int64(0) : T.int64(4),
                            ],
                        )
                        T.writes(
                            conv2d_NCHWc_int8[v_n, v_oc_chunk, v_oh, v_ow, T.int64(0) : T.int64(32)]
                        )
                        A = T.match_buffer(
                            p0_global_vtcm[
                                v_n,
                                v_ic_outer,
                                v_oh + v_kh,
                                v_ow + v_kw,
                                v_ic_f_inner * T.int64(4) : v_ic_f_inner * T.int64(4) + T.int64(4),
                            ],
                            [T.int64(4)],
                            dtype="uint8",
                            scope="global.vtcm",
                            offset_factor=1,
                        )
                        B = T.match_buffer(
                            fused_constant_global_vtcm[
                                v_oc_chunk,
                                v_ic_outer,
                                v_kh,
                                v_kw,
                                v_ic_f_inner,
                                T.int64(0) : T.int64(32),
                                T.int64(0) : T.int64(4),
                            ],
                            [T.int64(32), T.int64(4)],
                            dtype="uint8",
                            scope="global.vtcm",
                            offset_factor=1,
                        )
                        C = T.match_buffer(
                            conv2d_NCHWc_int8[
                                v_n, v_oc_chunk, v_oh, v_ow, T.int64(0) : T.int64(32)
                            ],
                            [T.int64(32)],
                            dtype="int32",
                            offset_factor=1,
                        )
                        A_u8x4: T.uint8x4 = A[T.int64(0) : T.int64(4)]
                        A_i32: T.int32 = T.reinterpret(A_u8x4, dtype="int32")
                        B_i8x128 = B[T.int64(0), T.int64(0) : T.int64(128)]
                        B_i32x32: T.int32x32 = T.reinterpret(B_i8x128, dtype="int32x32")
                        C[0:32] = T.call_llvm_pure_intrin(
                            T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.acc.128B"),
                            T.uint32(3),
                            C[0:32],
                            B_i32x32,
                            A_i32,
                            dtype="int32x32",
                        )


def conv_approximation(size_a, size_w):
    """Conv approximation."""
    a_shape = (size_a, VRMPY_SIZE_B)
    w_shape = (size_w, VRMPY_SIZE_B)
    out_shape = (size_a, VRMPY_SIZE_INT32)

    @T.prim_func
    def operator(a_input: T.handle, b_input: T.handle, c_output: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        a_buffer = T.match_buffer(a_input, a_shape, dtype="uint8")
        w_buffer = T.match_buffer(b_input, w_shape, dtype="uint8")
        c_buffer = T.match_buffer(c_output, out_shape, dtype="int32")
        for n, index_0 in T.grid(size_a, size_w):
            with T.block("c_buffer"):
                vn_index, vi_index = T.axis.remap("SR", [n, index_0])
                T.reads(
                    a_buffer[vn_index, 0:VRMPY_SIZE_B],
                    w_buffer[vi_index, 0:VRMPY_SIZE_B],
                    c_buffer[vn_index, 0:VRMPY_SIZE_INT32],
                )
                T.writes(c_buffer[vn_index, 0:VRMPY_SIZE_INT32])
                with T.init():
                    for x in T.serial(VRMPY_SIZE_INT32):
                        c_buffer[vn_index, x] = 0
                c_buffer[vn_index, T.ramp(0, 1, 32)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.acc.128B"),
                    T.uint32(3),
                    c_buffer[vn_index, T.ramp(0, 1, 32)],
                    T.reinterpret(a_buffer[vn_index, T.ramp(0, 1, 128)], dtype="int32x32"),
                    T.reinterpret(w_buffer[vi_index, T.ramp(0, 1, 128)], dtype="int32x32"),
                    dtype="int32x32",
                )
        # Currently async DMA lowering does not add any wait to the end of schedules so
        # for timing purposes we are manually adding a wait to ensure that all copies
        # are complete when the schedule exits.
        T.evaluate(
            T.tvm_call_packed(
                "device_api.hexagon.dma_wait",
                0,  # QueueId
                0,  # Wait for 0 in flight
                dtype="int32",
            )
        )

    return tvm.tir.Schedule(operator)


def evaluate(
    hexagon_session,
    sch,
    a_data,
    b_data,
    c_data,
    expected_output=None,
    use_async_copy=0,
):
    """Evaluate function."""
    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    with tvm.transform.PassContext(
        config={
            "tir.use_async_copy": use_async_copy,
            "tir.experimental_dma_bypass_cache": 1,
        }
    ):
        func_tir = tvm.build(
            sch.mod["main"], target=tvm.target.Target(target_hexagon, host=target_hexagon)
        )
    module = hexagon_session.load_module(func_tir)

    a_hexagon = tvm.runtime.ndarray.array(a_data, device=hexagon_session.device)
    b_hexagon = tvm.runtime.ndarray.array(b_data, device=hexagon_session.device)
    c_hexagon = tvm.runtime.ndarray.array(c_data, device=hexagon_session.device)

    if tvm.testing.utils.IS_IN_CI:
        # Run with reduced number and repeat for CI
        timer = module.time_evaluator("__tvm_main__", hexagon_session.device, number=1, repeat=1)
    else:
        timer = module.time_evaluator("__tvm_main__", hexagon_session.device, number=10, repeat=10)

    time = timer(a_hexagon, b_hexagon, c_hexagon)
    if expected_output is not None:
        tvm.testing.assert_allclose(c_hexagon.asnumpy(), expected_output)
    return round(time.mean * 1000, 4)


def get_fake_conv_vtcm_schedule(size_a, size_w, blocks=2):
    """Generate fake conv schedule with VTCM."""
    sch = conv_approximation(size_a, size_w)

    compute_block = sch.get_block("c_buffer")
    sch.cache_read(compute_block, 1, "global.vtcm")

    n = sch.get_loops(compute_block)[0]
    n_outer, _ = sch.split(n, [blocks, None])

    cache_read_block_a = sch.cache_read(compute_block, 0, "global.vtcm")
    sch.compute_at(cache_read_block_a, n_outer)
    sch.fuse(*sch.get_loops(cache_read_block_a)[1:])

    cache_write_block_c = sch.cache_write(compute_block, 0, "global.vtcm")
    sch.reverse_compute_at(cache_write_block_c, n_outer)
    sch.fuse(*sch.get_loops(cache_write_block_c)[1:])

    return sch


def get_multi_input_fake_conv_vtcm_schedule(size_a, size_w, blocks=2):
    """Generate multi input fake Conv using VTCM."""
    sch = conv_approximation(size_a, size_w)

    compute_block = sch.get_block("c_buffer")

    n = sch.get_loops(compute_block)[0]
    n_outer, _ = sch.split(n, [blocks, None])

    cache_read_block_a = sch.cache_read(compute_block, 0, "global.vtcm")
    sch.compute_at(cache_read_block_a, n_outer)
    sch.fuse(*sch.get_loops(cache_read_block_a)[1:])

    cache_read_block_b = sch.cache_read(compute_block, 1, "global.vtcm")
    sch.compute_at(cache_read_block_b, n_outer)
    sch.fuse(*sch.get_loops(cache_read_block_b)[1:])

    cache_write_block_c = sch.cache_write(compute_block, 0, "global.vtcm")
    sch.reverse_compute_at(cache_write_block_c, n_outer)
    sch.fuse(*sch.get_loops(cache_write_block_c)[1:])

    return sch


def print_results(test_key, runtimes):
    print(test_key)
    for runtime in runtimes.items():
        print("-{} took {} ms".format(runtime[0], runtime[1]))
    print()


class TestAsyncDMAPipeline:
    """Async DMA pipeline test class."""

    # Removed most of these to speedup CI.
    size_a = tvm.testing.parameter(
        1024,
        64 * 64,
        # 128 * 64, # Only works on 8Gen1 HDK's
    )

    size_w = tvm.testing.parameter(
        1 * 1,
        3 * 3,
        9 * 9,
    )

    @tvm.testing.fixture
    def input_a(self, size_a):
        return np.random.randint(0, 8, (size_a, VRMPY_SIZE_B), dtype="uint8")

    @tvm.testing.fixture
    def input_w(self, size_w):
        return np.random.randint(0, 8, (size_w, VRMPY_SIZE_B), dtype="uint8")

    @tvm.testing.fixture
    def expected_output(self, size_a, size_w, input_a, input_w):
        """Generate expected output."""
        if tvm.testing.utils.IS_IN_CI and (size_a > 1024 or size_w > 1):
            pytest.skip("Skipping test since it takes too long in CI.")
        expected_result = np.zeros((size_a, VRMPY_SIZE_INT32), dtype="int32")
        for n in range(size_a):
            for x in range(size_w):
                for index_0 in range(VRMPY_SIZE_INT32):
                    for r_index in range(4):
                        expected_result[n, index_0] += np.uint32(
                            input_a[n, index_0 * 4 + r_index]
                        ) * np.uint32(input_w[x, index_0 * 4 + r_index])
        return expected_result

    @tvm.testing.requires_hexagon
    def test_loading_vtcm_for_vrmpy(
        self,
        hexagon_session,
        size_a,
        size_w,
        input_a,
        input_w,
        expected_output,
    ):
        """VTCM for VRMPY test."""

        if tvm.testing.utils.IS_IN_CI and (size_a > 1024 or size_w > 1):
            pytest.skip("Skipping test since it takes too long in CI.")

        sch = conv_approximation(size_a, size_w)
        base_runtime = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        base_vtcm_runtime = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("c_buffer"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [0])
        async_input_runtime = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("c_buffer"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [0, 2])
        async_input_output = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("c_buffer"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 3, 6])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [0, 6])
        async_larger_buffers = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
        )

        sch = get_multi_input_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("c_buffer"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2, 3])
        sch.annotate(n, "software_pipeline_async_stages", [0, 2])
        async_multi_input_output = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("c_buffer"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [2])
        async_output_runtime = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
        )

        # Total transfer size is equal to the size of
        # a_buffer + w_buffer + c_buffer which is equal to 2 * size_a * 128 + size_w * 128
        transfer_mb = round((2 * size_a * VRMPY_SIZE_B + size_w * VRMPY_SIZE_B) / 1e6, 2)

        # Total number of operations can be calculated given
        # the total number of vrmpy calls (size_a * size_w) * operations
        # per vrmpy accumulate (128 multiplies + 3 adds for reduction
        # per lane + 1 add for accumulate per lane)
        complexity = round(size_a * size_w * (VRMPY_SIZE_B * 4) / 1e9, 3)
        print_results(
            (
                f"Test with a_buffer.size: {size_a * VRMPY_SIZE_B}, w_buffer.size:"
                f" {size_w * VRMPY_SIZE_B}, computational complexity of {complexity} GOPs"
                f", and total memory transfer of {transfer_mb} MB..."
            ),
            {
                "without_vtcm": base_runtime,
                "base_vtcm": base_vtcm_runtime,
                "async_dma_input": async_input_runtime,
                "async_dma_output": async_output_runtime,
                "async_dma_input_output": async_input_output,
                "async_dma_multi_input_output": async_multi_input_output,
                "async_input_output_runtime_larger_buffers": async_larger_buffers,
            },
        )


# from tvm.script import tir as T
@tvm.script.ir_module
class ModulePipelined:
    """Pipelined module class."""

    # pylint: disable=no-self-argument
    @T.prim_func
    def main(
        p0_buffer: T.Buffer((1, 1, 230, 230, 4), "uint8"),
        p1_buffer: T.Buffer((2, 1, 7, 7, 1, 32, 4), "int8"),
        t_cast: T.Buffer((1, 2, 112, 112, 32), "int32"),
    ) -> None:
        # pylint: disable=missing-function-docstring
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        conv2d_nchwc_int8 = T.alloc_buffer([1, 2, 112, 112, 32], dtype="int32", scope="global.vtcm")
        p0_global_vtcm = T.alloc_buffer([1, 1, 230, 230, 4], dtype="uint8", scope="global.vtcm")
        p1_global_vtcm = T.alloc_buffer([2, 1, 7, 7, 1, 32, 4], dtype="int8", scope="global.vtcm")
        for ax0, ax1, ax2, ax3, ax4, ax5, ax6 in T.grid(2, 1, 7, 7, 1, 32, 4):
            with T.block("p1_global.vtcm"):
                v0_ind, v1_ind, v2_ind, v3_ind, v4_ind, v5_ind, v6_ind = T.axis.remap(
                    "SSSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
                )
                T.reads(p1_buffer[v0_ind, v1_ind, v2_ind, v3_ind, v4_ind, v5_ind, v6_ind])
                T.writes(p1_global_vtcm[v0_ind, v1_ind, v2_ind, v3_ind, v4_ind, v5_ind, v6_ind])
                p1_global_vtcm[v0_ind, v1_ind, v2_ind, v3_ind, v4_ind, v5_ind, v6_ind] = p1_buffer[
                    v0_ind, v1_ind, v2_ind, v3_ind, v4_ind, v5_ind, v6_ind
                ]
        for p_outer in T.serial(4):
            for index_0 in T.serial(55876):
                with T.block("p0_global.vtcm"):
                    v0_ind = T.axis.spatial(1, 0)
                    v1_ind = T.axis.spatial(1, 0)
                    v2_ind = T.axis.spatial(230, p_outer * 56 + index_0 // 916)
                    v3_ind = T.axis.spatial(230, index_0 % 916 // 4)
                    v4_ind = T.axis.spatial(4, index_0 % 4)
                    T.reads(p0_buffer[v0_ind, v1_ind, v2_ind, v3_ind, v4_ind])
                    T.writes(p0_global_vtcm[v0_ind, v1_ind, v2_ind, v3_ind, v4_ind])
                    p0_global_vtcm[v0_ind, v1_ind, v2_ind, v3_ind, v4_ind] = p0_buffer[
                        v0_ind, v1_ind, v2_ind, v3_ind, v4_ind
                    ]
            for index_0 in T.parallel(28):
                for index_1, index_2, index_3 in T.grid(2, 14, 8):
                    with T.block("conv2d_NCHWc_int8_o_init"):
                        n = T.axis.spatial(1, 0)
                        oc_chunk = T.axis.spatial(2, index_1)
                        o_height = T.axis.spatial(
                            112, (p_outer * 28 + index_0) // 14 * 14 + index_2
                        )
                        o_width = T.axis.spatial(112, (p_outer * 28 + index_0) % 14 * 8 + index_3)
                        oc_block_o = T.axis.spatial(1, 0)  # pylint: disable=unused-variable
                        T.reads()
                        T.writes(conv2d_nchwc_int8[n, oc_chunk, o_height, o_width, 0:32])
                        for i4_1 in T.vectorized(32):
                            with T.block("conv2d_NCHWc_int8_init"):
                                oc_block_i_init = T.axis.spatial(32, i4_1)
                                T.reads()
                                T.writes(
                                    conv2d_nchwc_int8[
                                        n, oc_chunk, o_height, o_width, oc_block_i_init
                                    ]
                                )
                                conv2d_nchwc_int8[
                                    n, oc_chunk, o_height, o_width, oc_block_i_init
                                ] = 0
                for i1_1, i5_1, i6_1, i2_2, i3_2 in T.grid(2, 7, 7, 14, 8):
                    with T.block("conv2d_NCHWc_int8_o_update"):
                        n = T.axis.spatial(1, 0)
                        oc_chunk = T.axis.spatial(2, i1_1)
                        o_height = T.axis.spatial(112, (p_outer * 28 + index_0) // 14 * 14 + i2_2)
                        o_width = T.axis.spatial(112, (p_outer * 28 + index_0) % 14 * 8 + i3_2)
                        oc_block_o = T.axis.spatial(1, 0)  # pylint: disable=unused-variable
                        k_height = T.axis.reduce(7, i5_1)
                        k_width = T.axis.reduce(7, i6_1)
                        ic_outer = T.axis.reduce(1, 0)
                        ic_f_inner = T.axis.reduce(1, 0)
                        ic_s_inner_o = T.axis.reduce(1, 0)  # pylint: disable=unused-variable
                        T.reads(
                            conv2d_nchwc_int8[n, oc_chunk, o_height, o_width, 0:32],
                            p0_global_vtcm[
                                n,
                                ic_outer,
                                o_height * 2 + k_height,
                                o_width * 2 + k_width,
                                ic_f_inner * 4 : ic_f_inner * 4 + 4,
                            ],
                            p1_global_vtcm[
                                oc_chunk, ic_outer, k_height, k_width, ic_f_inner, 0:32, 0:4
                            ],
                        )
                        T.writes(conv2d_nchwc_int8[n, oc_chunk, o_height, o_width, 0:32])
                        a_buffer = T.match_buffer(
                            p0_global_vtcm[
                                n,
                                ic_outer,
                                o_height * 2 + k_height,
                                o_width * 2 + k_width,
                                ic_f_inner * 4 : ic_f_inner * 4 + 4,
                            ],
                            [4],
                            dtype="uint8",
                            offset_factor=1,
                            scope="global.vtcm",
                        )
                        b_buffer = T.match_buffer(
                            p1_global_vtcm[
                                oc_chunk, ic_outer, k_height, k_width, ic_f_inner, 0:32, 0:4
                            ],
                            [32, 4],
                            dtype="int8",
                            offset_factor=1,
                            scope="global.vtcm",
                        )
                        c_buffer = T.match_buffer(
                            conv2d_nchwc_int8[n, oc_chunk, o_height, o_width, 0:32],
                            [32],
                            dtype="int32",
                            offset_factor=1,
                            scope="global.vtcm",
                        )
                        a_u8x4: T.uint8x4 = a_buffer[0:4]
                        a_i32: T.int32 = T.reinterpret(a_u8x4, dtype="int32")
                        b_i8x128 = b_buffer[0, 0:128]
                        b_i32x32: T.int32x32 = T.reinterpret(b_i8x128, dtype="int32x32")
                        c_buffer[0:32] = T.call_llvm_pure_intrin(
                            T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.acc.128B"),
                            T.uint32(3),
                            c_buffer[0:32],
                            T.broadcast(a_i32, 32),
                            b_i32x32,
                            dtype="int32x32",
                        )
            for index_0 in T.serial(200704):
                with T.block("conv2d_nchwc_int8.vtcm"):
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.spatial(2, index_0 % 7168 // 3584)
                    ax2_1 = T.axis.spatial(
                        112, (p_outer * 28 + index_0 // 7168) // 14 * 14 + index_0 % 3584 // 256
                    )
                    ax3_1 = T.axis.spatial(
                        112, (p_outer * 28 + index_0 // 7168) % 14 * 8 + index_0 % 256 // 32
                    )
                    ax4 = T.axis.spatial(32, index_0 % 32)
                    T.reads(conv2d_nchwc_int8[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                    T.writes(t_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                    t_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4] = conv2d_nchwc_int8[
                        ax0_1, ax1_1, ax2_1, ax3_1, ax4
                    ]


# from tvm.script import tir as T
@tvm.script.ir_module
class ModuleBase:
    """Base module test class."""

    # pylint: disable=no-self-argument
    @T.prim_func
    def main(
        p0_buffer: T.Buffer((1, 1, 230, 230, 4), "uint8"),
        p1_buffer: T.Buffer((2, 1, 7, 7, 1, 32, 4), "int8"),
        t_cast: T.Buffer((1, 2, 112, 112, 32), "int32"),
    ) -> None:
        # pylint: disable=missing-function-docstring
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # buffer definition
        # body
        # with T.block("root")
        conv2d_nchwc_int8 = T.alloc_buffer([1, 2, 112, 112, 32], dtype="int32")
        for i0_0_i1_0_i2_0_i3_0_fused in T.parallel(
            112, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}
        ):
            for i4_0_0 in T.serial(1):  # pylint: disable=unused-variable
                for i1_1_init, i2_1_init, i3_1_init, i1_2_init, i2_2_init, i3_2_init in T.grid(
                    2, 1, 1, 1, 14, 8
                ):
                    with T.block("conv2d_NCHWc_int8_o_init"):
                        n = T.axis.spatial(1, 0)
                        oc_chunk = T.axis.spatial(2, i1_1_init + i1_2_init)
                        o_height = T.axis.spatial(
                            112, i0_0_i1_0_i2_0_i3_0_fused // 14 * 14 + i2_1_init * 14 + i2_2_init
                        )
                        o_width = T.axis.spatial(
                            112, i0_0_i1_0_i2_0_i3_0_fused % 14 * 8 + i3_1_init * 8 + i3_2_init
                        )
                        oc_block_o = T.axis.spatial(1, 0)  # pylint: disable=unused-variable
                        T.reads()
                        T.writes(conv2d_nchwc_int8[n, oc_chunk, o_height, o_width, 0:32])
                        for i4_1 in T.vectorized(32):
                            with T.block("conv2d_NCHWc_int8_init"):
                                oc_block_i_init = T.axis.spatial(32, i4_1)
                                T.reads()
                                T.writes(
                                    conv2d_nchwc_int8[
                                        n, oc_chunk, o_height, o_width, oc_block_i_init
                                    ]
                                )
                                conv2d_nchwc_int8[
                                    n, oc_chunk, o_height, o_width, oc_block_i_init
                                ] = 0
                for i5_0, i6_0, i7_0, i8_0, i9_0_0 in T.grid(  # pylint: disable=unused-variable
                    1, 1, 1, 1, 1
                ):  # pylint: disable=unused-variable
                    for (
                        i0_1,  # pylint: disable=unused-variable
                        i1_1,
                        i2_1,
                        i3_1,
                        i4_0_1,  # pylint: disable=unused-variable
                        i5_1,
                        i6_1,
                        i7_1,  # pylint: disable=unused-variable
                        i8_1,  # pylint: disable=unused-variable
                        i9_0_1,  # pylint: disable=unused-variable
                        i0_2,  # pylint: disable=unused-variable
                        i1_2,
                        i2_2,
                        i3_2,
                        i4_0_2,  # pylint: disable=unused-variable
                    ) in T.grid(1, 2, 1, 1, 1, 7, 7, 1, 1, 1, 1, 1, 14, 8, 1):
                        with T.block("conv2d_NCHWc_int8_o_update"):
                            n = T.axis.spatial(1, 0)
                            oc_chunk = T.axis.spatial(2, i1_1 + i1_2)
                            o_height = T.axis.spatial(
                                112, i0_0_i1_0_i2_0_i3_0_fused // 14 * 14 + i2_1 * 14 + i2_2
                            )
                            o_width = T.axis.spatial(
                                112, i0_0_i1_0_i2_0_i3_0_fused % 14 * 8 + i3_1 * 8 + i3_2
                            )
                            oc_block_o = T.axis.spatial(1, 0)  # pylint: disable=unused-variable
                            k_height = T.axis.reduce(7, i5_0 * 7 + i5_1)
                            k_width = T.axis.reduce(7, i6_0 * 7 + i6_1)
                            ic_outer = T.axis.reduce(1, 0)
                            ic_f_inner = T.axis.reduce(1, 0)
                            ic_s_inner_o = T.axis.reduce(1, 0)  # pylint: disable=unused-variable
                            T.reads(
                                conv2d_nchwc_int8[n, oc_chunk, o_height, o_width, 0:32],
                                p0_buffer[
                                    n,
                                    ic_outer,
                                    o_height * 2 + k_height,
                                    o_width * 2 + k_width,
                                    ic_f_inner * 4 : ic_f_inner * 4 + 4,
                                ],
                                p1_buffer[
                                    oc_chunk, ic_outer, k_height, k_width, ic_f_inner, 0:32, 0:4
                                ],
                            )
                            T.writes(conv2d_nchwc_int8[n, oc_chunk, o_height, o_width, 0:32])
                            a_buffer = T.match_buffer(
                                p0_buffer[
                                    n,
                                    ic_outer,
                                    o_height * 2 + k_height,
                                    o_width * 2 + k_width,
                                    ic_f_inner * 4 : ic_f_inner * 4 + 4,
                                ],
                                [4],
                                dtype="uint8",
                                offset_factor=1,
                            )
                            b_buffer = T.match_buffer(
                                p1_buffer[
                                    oc_chunk, ic_outer, k_height, k_width, ic_f_inner, 0:32, 0:4
                                ],
                                [32, 4],
                                dtype="int8",
                                offset_factor=1,
                            )
                            c_buffer = T.match_buffer(
                                conv2d_nchwc_int8[n, oc_chunk, o_height, o_width, 0:32],
                                [32],
                                dtype="int32",
                                offset_factor=1,
                            )
                            a_u8x4: T.uint8x4 = a_buffer[0:4]
                            a_i32: T.int32 = T.reinterpret(a_u8x4, dtype="int32")
                            b_i8x128 = b_buffer[0, 0:128]
                            b_i32x32: T.int32x32 = T.reinterpret(b_i8x128, dtype="int32x32")
                            c_buffer[0:32] = T.call_llvm_pure_intrin(
                                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.acc.128B"),
                                T.uint32(3),
                                c_buffer[0:32],
                                T.broadcast(a_i32, 32),
                                b_i32x32,
                                dtype="int32x32",
                            )
                    for ax0, ax1, ax2, ax3 in T.grid(1, 2, 14, 8):
                        for ax4_fused in T.vectorized(32):
                            with T.block("T_cast_2"):
                                ax0_1, ax1_1 = T.axis.remap("SS", [ax0, ax1])
                                ax2_1 = T.axis.spatial(
                                    112, i0_0_i1_0_i2_0_i3_0_fused // 14 * 14 + ax2
                                )
                                ax3_1 = T.axis.spatial(
                                    112, i0_0_i1_0_i2_0_i3_0_fused % 14 * 8 + ax3
                                )
                                ax4 = T.axis.spatial(32, ax4_fused)
                                T.reads(conv2d_nchwc_int8[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                                T.writes(t_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                                t_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4] = conv2d_nchwc_int8[
                                    ax0_1, ax1_1, ax2_1, ax3_1, ax4
                                ]


@tvm.testing.requires_hexagon
def test_meta(hexagon_session):
    """Test meta."""
    if tvm.testing.utils.IS_IN_CI:
        pytest.skip("Skipping test since it takes too long in CI.")

    a_data = np.random.randint(1, 8, (1, 1, 230, 230, 4), dtype="uint8")
    w_data = np.random.randint(1, 8, (2, 1, 7, 7, 1, 32, 4), dtype="int8")
    c_data = np.zeros((1, 2, 112, 112, 32), dtype="int32")

    sch = tvm.tir.Schedule(ModuleBase)
    base_runtime = evaluate(hexagon_session, sch, a_data, w_data, c_data)

    sch = tvm.tir.Schedule(ModulePipelined)
    compute_block = sch.get_block("conv2d_NCHWc_int8_o_update")
    outer = sch.get_loops(compute_block)[0]

    unscheduled_vtcm_runtime = evaluate(
        hexagon_session, sch, a_data, w_data, c_data, use_async_copy=1
    )

    sch = tvm.tir.Schedule(ModulePipelined)
    compute_block = sch.get_block("conv2d_NCHWc_int8_o_update")
    outer = sch.get_loops(compute_block)[0]

    sch.annotate(outer, "software_pipeline_stage", [0, 1, 2])
    sch.annotate(outer, "software_pipeline_order", [0, 1, 2])
    sch.annotate(outer, "software_pipeline_async_stages", [0, 2])

    pipeline_runtime = evaluate(hexagon_session, sch, a_data, w_data, c_data, use_async_copy=1)

    transfer_mb = round((a_data.size + w_data.size + c_data.size) / 1e6, 2)
    print_results(
        (
            f"Test with a_buffer.size: {a_data.size}, w_buffer.size: {w_data.size}"
            f", and total memory transfer of {transfer_mb} MB..."
        ),
        {
            "without_vtcm": base_runtime,
            "unscheduled_vtcm_runtime": unscheduled_vtcm_runtime,
            "pipeline_runtime": pipeline_runtime,
        },
    )


if __name__ == "__main__":
    tvm.testing.main()
