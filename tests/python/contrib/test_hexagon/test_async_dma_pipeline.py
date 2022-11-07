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
import tvm
import pytest

from tvm.script import tir as T
from numpy.random import default_rng

VRMPY_SIZE_B = 128
VRMPY_SIZE_INT32 = 32


def conv_approximation(size_a, size_w):
    a_shape = (size_a, VRMPY_SIZE_B)
    w_shape = (size_w, VRMPY_SIZE_B)
    out_shape = (size_a, VRMPY_SIZE_INT32)

    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, a_shape, dtype="uint8")
        W = T.match_buffer(b, w_shape, dtype="uint8")
        C = T.match_buffer(c, out_shape, dtype="int32")
        for n, i in T.grid(size_a, size_w):
            with T.block("C"):
                vn, vi = T.axis.remap("SR", [n, i])
                T.reads(A[vn, 0:VRMPY_SIZE_B], W[vi, 0:VRMPY_SIZE_B], C[vn, 0:VRMPY_SIZE_INT32])
                T.writes(C[vn, 0:VRMPY_SIZE_INT32])
                with T.init():
                    for x in T.serial(VRMPY_SIZE_INT32):
                        C[vn, x] = 0
                C[vn, T.ramp(0, 1, 32)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.acc.128B"),
                    T.uint32(3),
                    C[vn, T.ramp(0, 1, 32)],
                    T.reinterpret(A[vn, T.ramp(0, 1, 128)], dtype="int32x32"),
                    T.reinterpret(W[vi, T.ramp(0, 1, 128)], dtype="int32x32"),
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
    a,
    b,
    c,
    expected_output=None,
    use_async_copy=0,
    merge_async_commit_queue_scope=False,
):
    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    with tvm.transform.PassContext(
        config={
            "tir.use_async_copy": use_async_copy,
            "tir.merge_async_commit_queue_scope": merge_async_commit_queue_scope,
        }
    ):
        func_tir = tvm.build(
            sch.mod["main"], target=tvm.target.Target(target_hexagon, host=target_hexagon)
        )
    module = hexagon_session.load_module(func_tir)

    a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device)
    b_hexagon = tvm.runtime.ndarray.array(b, device=hexagon_session.device)
    c_hexagon = tvm.runtime.ndarray.array(c, device=hexagon_session.device)

    if tvm.testing.utils.IS_IN_CI:
        # Run with reduced number and repeat for CI
        timer = module.time_evaluator("__tvm_main__", hexagon_session.device, number=1, repeat=1)
    else:
        timer = module.time_evaluator("__tvm_main__", hexagon_session.device, number=10, repeat=10)

    time = timer(a_hexagon, b_hexagon, c_hexagon)
    if expected_output is not None:
        tvm.testing.assert_allclose(c_hexagon.asnumpy(), expected_output)
    return round(time.mean * 1000, 4)


@tvm.testing.fixture
def input_a(size_a):
    return default_rng().integers(0, 8, (size_a, VRMPY_SIZE_B), dtype="uint8")


@tvm.testing.fixture
def input_w(size_w):
    return default_rng().integers(0, 8, (size_w, VRMPY_SIZE_B), dtype="uint8")


@tvm.testing.fixture
def expected_output(size_a, size_w, input_a, input_w):
    if tvm.testing.utils.IS_IN_CI and (size_a > 1024 or size_w > 1):
        pytest.skip("Skipping test since it takes too long in CI.")
    expected_output = np.zeros((size_a, VRMPY_SIZE_INT32), dtype="int32")
    for n in range(size_a):
        for x in range(size_w):
            for i in range(VRMPY_SIZE_INT32):
                for r in range(4):
                    expected_output[n, i] += np.uint32(input_a[n, i * 4 + r]) * np.uint32(
                        input_w[x, i * 4 + r]
                    )
    return expected_output


def get_single_dma_schedule(size_a, size_w):
    a_shape = (size_a, VRMPY_SIZE_B)
    w_shape = (size_w, VRMPY_SIZE_B)
    out_shape = (size_a, VRMPY_SIZE_INT32)

    a_bytes = size_a * VRMPY_SIZE_B
    w_bytes = size_w * VRMPY_SIZE_B

    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, a_shape, dtype="uint8", mem_scope="global")
        W = T.match_buffer(b, w_shape, dtype="uint8", mem_scope="global")
        C = T.match_buffer(c, out_shape, dtype="int32", mem_scope="global")
        A_global_vtcm = T.alloc_buffer(a_shape, dtype="uint8", mem_scope="global.vtcm")
        W_global_vtcm = T.alloc_buffer(w_shape, dtype="uint8", mem_scope="global.vtcm")
        C_global_vtcm = T.alloc_buffer(out_shape, dtype="int32", mem_scope="global.vtcm")
        T.evaluate(
            T.tvm_call_packed(
                "device_api.hexagon.mem_copy_DLTensor",
                T.tvm_stack_make_array(
                    A_global_vtcm.data,
                    T.tvm_stack_make_shape(size_a, VRMPY_SIZE_B, dtype="handle"),
                    0,
                    2,
                    A_global_vtcm.dtype,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    A.data,
                    T.tvm_stack_make_shape(size_a, VRMPY_SIZE_B, dtype="handle"),
                    0,
                    2,
                    A.dtype,
                    0,
                    dtype="handle",
                ),
                T.cast(a_bytes, dtype="int"),
                dtype="int32",
            )
        )
        T.evaluate(
            T.tvm_call_packed(
                "device_api.hexagon.mem_copy_DLTensor",
                T.tvm_stack_make_array(
                    W_global_vtcm.data,
                    T.tvm_stack_make_shape(size_w, VRMPY_SIZE_B, dtype="handle"),
                    0,
                    2,
                    W_global_vtcm.dtype,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    W.data,
                    T.tvm_stack_make_shape(size_w, VRMPY_SIZE_B, dtype="handle"),
                    0,
                    2,
                    W.dtype,
                    0,
                    dtype="handle",
                ),
                T.cast(w_bytes, dtype="int"),
                dtype="int32",
            )
        )
        for n, i in T.grid(size_a, size_w):
            with T.block("C"):
                vn, vi = T.axis.remap("SR", [n, i])
                T.reads(
                    A_global_vtcm[vn, 0:VRMPY_SIZE_B],
                    W_global_vtcm[vi, 0:VRMPY_SIZE_B],
                    C_global_vtcm[vn, 0:VRMPY_SIZE_INT32],
                )
                T.writes(C_global_vtcm[vn, 0:VRMPY_SIZE_INT32])
                with T.init():
                    for x in T.serial(VRMPY_SIZE_INT32):
                        C_global_vtcm[vn, x] = 0
                C_global_vtcm[vn, T.ramp(0, 1, 32)] += T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.128B"),
                    T.uint32(2),
                    T.reinterpret(A_global_vtcm[vn, T.ramp(0, 1, 128)], dtype="int32x32"),
                    T.reinterpret(W_global_vtcm[vi, T.ramp(0, 1, 128)], dtype="int32x32"),
                    dtype="int32x32",
                )
        T.evaluate(
            T.tvm_call_packed(
                "device_api.hexagon.mem_copy_DLTensor",
                T.tvm_stack_make_array(
                    C.data,
                    T.tvm_stack_make_shape(size_a, VRMPY_SIZE_B, dtype="handle"),
                    0,
                    2,
                    C.dtype,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    C_global_vtcm.data,
                    T.tvm_stack_make_shape(size_a, VRMPY_SIZE_B, dtype="handle"),
                    0,
                    2,
                    C_global_vtcm.dtype,
                    0,
                    dtype="handle",
                ),
                T.cast(a_bytes, dtype="int"),
                dtype="int32",
            )
        )

    sch = tvm.tir.Schedule(operator)

    return sch


def get_fake_conv_vtcm_schedule(size_a, size_w, blocks=2):
    sch = conv_approximation(size_a, size_w)

    compute_block = sch.get_block("C")
    sch.cache_read(compute_block, 1, "global.vtcm")

    n = sch.get_loops(compute_block)[0]
    no, _ = sch.split(n, [blocks, None])

    cache_read_block_a = sch.cache_read(compute_block, 0, "global.vtcm")
    sch.compute_at(cache_read_block_a, no)
    sch.fuse(*sch.get_loops(cache_read_block_a)[1:])

    cache_write_block_c = sch.cache_write(compute_block, 0, "global.vtcm")
    sch.reverse_compute_at(cache_write_block_c, no)
    sch.fuse(*sch.get_loops(cache_write_block_c)[1:])

    return sch


def get_multi_input_fake_conv_vtcm_schedule(size_a, size_w, blocks=2):
    sch = conv_approximation(size_a, size_w)

    compute_block = sch.get_block("C")

    n = sch.get_loops(compute_block)[0]
    no, _ = sch.split(n, [blocks, None])

    cache_read_block_a = sch.cache_read(compute_block, 0, "global.vtcm")
    sch.compute_at(cache_read_block_a, no)
    sch.fuse(*sch.get_loops(cache_read_block_a)[1:])

    cache_read_block_b = sch.cache_read(compute_block, 1, "global.vtcm")
    sch.compute_at(cache_read_block_b, no)
    sch.fuse(*sch.get_loops(cache_read_block_b)[1:])

    cache_write_block_c = sch.cache_write(compute_block, 0, "global.vtcm")
    sch.reverse_compute_at(cache_write_block_c, no)
    sch.fuse(*sch.get_loops(cache_write_block_c)[1:])

    return sch


def print_results(test_key, runtimes):
    print(test_key)
    for runtime in runtimes.items():
        print("-{} took {} ms".format(runtime[0], runtime[1]))
    print()


class TestAsyncDMAPipeline:
    # Removed most of these to speedup CI.
    size_a = tvm.testing.parameter(
        1024,
        64 * 64,
        128 * 64,
    )

    size_w = tvm.testing.parameter(
        1 * 1,
        3 * 3,
        9 * 9,
    )

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
        n = sch.get_loops(sch.get_block("C"))[0]
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
        n = sch.get_loops(sch.get_block("C"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [0, 2])
        async_input_output_runtime = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("C"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 3, 6])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [0, 6])
        async_input_output_runtime_larger_buffers = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
        )

        sch = get_multi_input_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("C"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2, 3])
        sch.annotate(n, "software_pipeline_async_stages", [0, 2])
        async_multi_input_output_runtime = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
            use_async_copy=1,
            merge_async_commit_queue_scope=False,
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("C"))[0]
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

        sch = get_single_dma_schedule(size_a, size_w)
        single_dma_runtime = evaluate(
            hexagon_session,
            sch,
            input_a,
            input_w,
            np.zeros(expected_output.shape, "int32"),
            expected_output,
        )

        # Total transfer size is equal to the size of A + W + C which is equal to 2 * size_a * 128 + size_w * 128
        transfer_mb = round((2 * size_a * VRMPY_SIZE_B + size_w * VRMPY_SIZE_B) / 1e6, 2)

        # Total number of operations can be calculated given the total number of vrmpy calls (size_a * size_w) * operations per vrmpy accumulate (128 multiplies + 3 adds for reduction per lane + 1 add for accumulate per lane)
        complexity = round(size_a * size_w * (VRMPY_SIZE_B * 4) / 1e9, 3)
        print_results(
            f"Test with A.size: {size_a * VRMPY_SIZE_B}, W.size: {size_w * VRMPY_SIZE_B}, computational complexity of {complexity} GOPs, and total memory transfer of {transfer_mb} MB...",
            {
                "without_vtcm": base_runtime,
                "synchronous_dma": single_dma_runtime,
                "base_vtcm": base_vtcm_runtime,
                "async_dma_input": async_input_runtime,
                "async_dma_output": async_output_runtime,
                "async_dma_input_output": async_input_output_runtime,
                "async_dma_multi_input_output": async_multi_input_output_runtime,
                "async_input_output_runtime_larger_buffers": async_input_output_runtime_larger_buffers,
            },
        )


# from tvm.script import tir as T
@tvm.script.ir_module
class ModulePipelined:
    @T.prim_func
    def main(
        p0: T.Buffer[(1, 1, 230, 230, 4), "uint8"],
        p1: T.Buffer[(2, 1, 7, 7, 1, 32, 4), "int8"],
        T_cast: T.Buffer[(1, 2, 112, 112, 32), "int32"],
    ) -> None:
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        conv2d_NCHWc_int8 = T.alloc_buffer([1, 2, 112, 112, 32], dtype="int32", scope="global.vtcm")
        p0_global_vtcm = T.alloc_buffer([1, 1, 230, 230, 4], dtype="uint8", scope="global.vtcm")
        p1_global_vtcm = T.alloc_buffer([2, 1, 7, 7, 1, 32, 4], dtype="int8", scope="global.vtcm")
        for ax0, ax1, ax2, ax3, ax4, ax5, ax6 in T.grid(2, 1, 7, 7, 1, 32, 4):
            with T.block("p1_global.vtcm"):
                v0, v1, v2, v3, v4, v5, v6 = T.axis.remap(
                    "SSSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
                )
                T.reads(p1[v0, v1, v2, v3, v4, v5, v6])
                T.writes(p1_global_vtcm[v0, v1, v2, v3, v4, v5, v6])
                p1_global_vtcm[v0, v1, v2, v3, v4, v5, v6] = p1[v0, v1, v2, v3, v4, v5, v6]
        for po in T.serial(4):
            for i in T.serial(55876):
                with T.block("p0_global.vtcm"):
                    v0 = T.axis.spatial(1, 0)
                    v1 = T.axis.spatial(1, 0)
                    v2 = T.axis.spatial(230, po * 56 + i // 916)
                    v3 = T.axis.spatial(230, i % 916 // 4)
                    v4 = T.axis.spatial(4, i % 4)
                    T.reads(p0[v0, v1, v2, v3, v4])
                    T.writes(p0_global_vtcm[v0, v1, v2, v3, v4])
                    p0_global_vtcm[v0, v1, v2, v3, v4] = p0[v0, v1, v2, v3, v4]
            for i in T.parallel(28):
                for ii, iii, iiii in T.grid(2, 14, 8):
                    with T.block("conv2d_NCHWc_int8_o_init"):
                        n = T.axis.spatial(1, 0)
                        oc_chunk = T.axis.spatial(2, ii)
                        oh = T.axis.spatial(112, (po * 28 + i) // 14 * 14 + iii)
                        ow = T.axis.spatial(112, (po * 28 + i) % 14 * 8 + iiii)
                        oc_block_o = T.axis.spatial(1, 0)
                        T.reads()
                        T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:32])
                        for i4_1 in T.vectorized(32):
                            with T.block("conv2d_NCHWc_int8_init"):
                                oc_block_i_init = T.axis.spatial(32, i4_1)
                                T.reads()
                                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init])
                                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                for i1_1, i5_1, i6_1, i2_2, i3_2 in T.grid(2, 7, 7, 14, 8):
                    with T.block("conv2d_NCHWc_int8_o_update"):
                        n = T.axis.spatial(1, 0)
                        oc_chunk = T.axis.spatial(2, i1_1)
                        oh = T.axis.spatial(112, (po * 28 + i) // 14 * 14 + i2_2)
                        ow = T.axis.spatial(112, (po * 28 + i) % 14 * 8 + i3_2)
                        oc_block_o = T.axis.spatial(1, 0)
                        kh = T.axis.reduce(7, i5_1)
                        kw = T.axis.reduce(7, i6_1)
                        ic_outer = T.axis.reduce(1, 0)
                        ic_f_inner = T.axis.reduce(1, 0)
                        ic_s_inner_o = T.axis.reduce(1, 0)
                        T.reads(
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:32],
                            p0_global_vtcm[
                                n,
                                ic_outer,
                                oh * 2 + kh,
                                ow * 2 + kw,
                                ic_f_inner * 4 : ic_f_inner * 4 + 4,
                            ],
                            p1_global_vtcm[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:32, 0:4],
                        )
                        T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:32])
                        A = T.match_buffer(
                            p0_global_vtcm[
                                n,
                                ic_outer,
                                oh * 2 + kh,
                                ow * 2 + kw,
                                ic_f_inner * 4 : ic_f_inner * 4 + 4,
                            ],
                            [4],
                            dtype="uint8",
                            offset_factor=1,
                            scope="global.vtcm",
                        )
                        B = T.match_buffer(
                            p1_global_vtcm[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:32, 0:4],
                            [32, 4],
                            dtype="int8",
                            offset_factor=1,
                            scope="global.vtcm",
                        )
                        C = T.match_buffer(
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:32],
                            [32],
                            dtype="int32",
                            offset_factor=1,
                            scope="global.vtcm",
                        )
                        A_u8x4: T.uint8x4 = A[0:4]
                        A_i32: T.int32 = T.reinterpret(A_u8x4, dtype="int32")
                        B_i8x128 = B[0, 0:128]
                        B_i32x32: T.int32x32 = T.reinterpret(B_i8x128, dtype="int32x32")
                        C[0:32] = T.call_llvm_pure_intrin(
                            4217,
                            T.uint32(3),
                            C[0:32],
                            T.broadcast(A_i32, 32),
                            B_i32x32,
                            dtype="int32x32",
                        )
            for i in T.serial(200704):
                with T.block("conv2d_NCHWc_int8.vtcm"):
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.spatial(2, i % 7168 // 3584)
                    ax2_1 = T.axis.spatial(112, (po * 28 + i // 7168) // 14 * 14 + i % 3584 // 256)
                    ax3_1 = T.axis.spatial(112, (po * 28 + i // 7168) % 14 * 8 + i % 256 // 32)
                    ax4 = T.axis.spatial(32, i % 32)
                    T.reads(conv2d_NCHWc_int8[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                    T.writes(T_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                    T_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4] = conv2d_NCHWc_int8[
                        ax0_1, ax1_1, ax2_1, ax3_1, ax4
                    ]


# from tvm.script import tir as T
@tvm.script.ir_module
class ModuleBase:
    @T.prim_func
    def main(
        p0: T.Buffer[(1, 1, 230, 230, 4), "uint8"],
        p1: T.Buffer[(2, 1, 7, 7, 1, 32, 4), "int8"],
        T_cast: T.Buffer[(1, 2, 112, 112, 32), "int32"],
    ) -> None:
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # buffer definition
        # body
        # with T.block("root")
        conv2d_NCHWc_int8 = T.alloc_buffer([1, 2, 112, 112, 32], dtype="int32")
        for i0_0_i1_0_i2_0_i3_0_fused in T.parallel(
            112, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}
        ):
            for i4_0_0 in T.serial(1):
                for i1_1_init, i2_1_init, i3_1_init, i1_2_init, i2_2_init, i3_2_init in T.grid(
                    2, 1, 1, 1, 14, 8
                ):
                    with T.block("conv2d_NCHWc_int8_o_init"):
                        n = T.axis.spatial(1, 0)
                        oc_chunk = T.axis.spatial(2, i1_1_init + i1_2_init)
                        oh = T.axis.spatial(
                            112, i0_0_i1_0_i2_0_i3_0_fused // 14 * 14 + i2_1_init * 14 + i2_2_init
                        )
                        ow = T.axis.spatial(
                            112, i0_0_i1_0_i2_0_i3_0_fused % 14 * 8 + i3_1_init * 8 + i3_2_init
                        )
                        oc_block_o = T.axis.spatial(1, 0)
                        T.reads()
                        T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:32])
                        for i4_1 in T.vectorized(32):
                            with T.block("conv2d_NCHWc_int8_init"):
                                oc_block_i_init = T.axis.spatial(32, i4_1)
                                T.reads()
                                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init])
                                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                for i5_0, i6_0, i7_0, i8_0, i9_0_0 in T.grid(1, 1, 1, 1, 1):
                    for (
                        i0_1,
                        i1_1,
                        i2_1,
                        i3_1,
                        i4_0_1,
                        i5_1,
                        i6_1,
                        i7_1,
                        i8_1,
                        i9_0_1,
                        i0_2,
                        i1_2,
                        i2_2,
                        i3_2,
                        i4_0_2,
                    ) in T.grid(1, 2, 1, 1, 1, 7, 7, 1, 1, 1, 1, 1, 14, 8, 1):
                        with T.block("conv2d_NCHWc_int8_o_update"):
                            n = T.axis.spatial(1, 0)
                            oc_chunk = T.axis.spatial(2, i1_1 + i1_2)
                            oh = T.axis.spatial(
                                112, i0_0_i1_0_i2_0_i3_0_fused // 14 * 14 + i2_1 * 14 + i2_2
                            )
                            ow = T.axis.spatial(
                                112, i0_0_i1_0_i2_0_i3_0_fused % 14 * 8 + i3_1 * 8 + i3_2
                            )
                            oc_block_o = T.axis.spatial(1, 0)
                            kh = T.axis.reduce(7, i5_0 * 7 + i5_1)
                            kw = T.axis.reduce(7, i6_0 * 7 + i6_1)
                            ic_outer = T.axis.reduce(1, 0)
                            ic_f_inner = T.axis.reduce(1, 0)
                            ic_s_inner_o = T.axis.reduce(1, 0)
                            T.reads(
                                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:32],
                                p0[
                                    n,
                                    ic_outer,
                                    oh * 2 + kh,
                                    ow * 2 + kw,
                                    ic_f_inner * 4 : ic_f_inner * 4 + 4,
                                ],
                                p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:32, 0:4],
                            )
                            T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:32])
                            A = T.match_buffer(
                                p0[
                                    n,
                                    ic_outer,
                                    oh * 2 + kh,
                                    ow * 2 + kw,
                                    ic_f_inner * 4 : ic_f_inner * 4 + 4,
                                ],
                                [4],
                                dtype="uint8",
                                offset_factor=1,
                            )
                            B = T.match_buffer(
                                p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:32, 0:4],
                                [32, 4],
                                dtype="int8",
                                offset_factor=1,
                            )
                            C = T.match_buffer(
                                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:32],
                                [32],
                                dtype="int32",
                                offset_factor=1,
                            )
                            A_u8x4: T.uint8x4 = A[0:4]
                            A_i32: T.int32 = T.reinterpret(A_u8x4, dtype="int32")
                            B_i8x128 = B[0, 0:128]
                            B_i32x32: T.int32x32 = T.reinterpret(B_i8x128, dtype="int32x32")
                            C[0:32] = T.call_llvm_pure_intrin(
                                4217,
                                T.uint32(3),
                                C[0:32],
                                T.broadcast(A_i32, 32),
                                B_i32x32,
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
                                T.reads(conv2d_NCHWc_int8[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                                T.writes(T_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                                T_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4] = conv2d_NCHWc_int8[
                                    ax0_1, ax1_1, ax2_1, ax3_1, ax4
                                ]


@tvm.testing.requires_hexagon
def test_meta(hexagon_session):
    if tvm.testing.utils.IS_IN_CI:
        pytest.skip("Skipping test since it takes too long in CI.")

    a = default_rng().integers(1, 8, (1, 1, 230, 230, 4), dtype="uint8")
    w = default_rng().integers(1, 8, (2, 1, 7, 7, 1, 32, 4), dtype="int8")
    c = np.zeros((1, 2, 112, 112, 32), dtype="int32")

    sch = tvm.tir.Schedule(ModuleBase)
    base_runtime = evaluate(hexagon_session, sch, a, w, c)

    sch = tvm.tir.Schedule(ModulePipelined)
    compute_block = sch.get_block("conv2d_NCHWc_int8_o_update")
    o = sch.get_loops(compute_block)[0]

    unscheduled_vtcm_runtime = evaluate(hexagon_session, sch, a, w, c, use_async_copy=1)

    sch = tvm.tir.Schedule(ModulePipelined)
    compute_block = sch.get_block("conv2d_NCHWc_int8_o_update")
    o = sch.get_loops(compute_block)[0]

    sch.annotate(o, "software_pipeline_stage", [0, 1, 2])
    sch.annotate(o, "software_pipeline_order", [0, 1, 2])
    sch.annotate(o, "software_pipeline_async_stages", [0, 2])

    pipeline_runtime = evaluate(hexagon_session, sch, a, w, c, use_async_copy=1)

    transfer_mb = round((a.size + w.size + c.size) / 1e6, 2)
    print_results(
        f"Test with A.size: {a.size}, W.size: {w.size}, and total memory transfer of {transfer_mb} MB...",
        {
            "without_vtcm": base_runtime,
            "unscheduled_vtcm_runtime": unscheduled_vtcm_runtime,
            "pipeline_runtime": pipeline_runtime,
        },
    )
