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

from tvm.tir.function import TensorIntrin

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


def evaluate(hexagon_session, sch, a, b, size_a, expected_output, use_async_copy=0):
    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    with tvm.transform.PassContext(config={"tir.use_async_copy": use_async_copy}):
        func_tir = tvm.build(
            sch.mod["main"], target=tvm.target.Target(target_hexagon, host=target_hexagon)
        )
    module = hexagon_session.load_module(func_tir)

    a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device)
    b_hexagon = tvm.runtime.ndarray.array(b, device=hexagon_session.device)
    c_hexagon = tvm.runtime.ndarray.array(
        np.zeros((size_a, VRMPY_SIZE_INT32), dtype="int32"), device=hexagon_session.device
    )

    if tvm.testing.utils.IS_IN_CI:
        # Run with reduced number and repeat for CI
        timer = module.time_evaluator("__tvm_main__", hexagon_session.device, number=1, repeat=1)
    else:
        timer = module.time_evaluator("__tvm_main__", hexagon_session.device, number=10, repeat=10)

    time = timer(a_hexagon, b_hexagon, c_hexagon)
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

    cache_read_block_c = sch.cache_write(compute_block, 0, "global.vtcm")
    sch.reverse_compute_at(cache_read_block_c, no)
    sch.fuse(*sch.get_loops(cache_read_block_c)[1:])

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
        128 * 128,
    )

    size_w = tvm.testing.parameter(
        1 * 1,
        3 * 3,
        7 * 7,
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
        base_runtime = evaluate(hexagon_session, sch, input_a, input_w, size_a, expected_output)

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        base_vtcm_runtime = evaluate(
            hexagon_session, sch, input_a, input_w, size_a, expected_output, use_async_copy=1
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("C"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [0])
        async_input_runtime = evaluate(
            hexagon_session, sch, input_a, input_w, size_a, expected_output, use_async_copy=1
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("C"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [0, 2])
        async_input_output_runtime = evaluate(
            hexagon_session, sch, input_a, input_w, size_a, expected_output, use_async_copy=1
        )

        sch = get_fake_conv_vtcm_schedule(size_a, size_w)
        n = sch.get_loops(sch.get_block("C"))[0]
        sch.annotate(n, "software_pipeline_stage", [0, 1, 2])
        sch.annotate(n, "software_pipeline_order", [0, 1, 2])
        sch.annotate(n, "software_pipeline_async_stages", [2])
        async_output_runtime = evaluate(
            hexagon_session, sch, input_a, input_w, size_a, expected_output, use_async_copy=1
        )

        sch = get_single_dma_schedule(size_a, size_w)
        single_dma_runtime = evaluate(
            hexagon_session, sch, input_a, input_w, size_a, expected_output
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
            },
        )
