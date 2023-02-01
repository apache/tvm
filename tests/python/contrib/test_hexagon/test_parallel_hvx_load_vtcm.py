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
from tvm.script import tir as T

from .infrastructure import get_hexagon_target

TEST_OUTPUT_TEMPLATE = (
    "Test with {} MB of data to load... \n"
    "    -No VTCM: {} Gops \n    -Basic VTCM: {} Gops \n"
    "    -Vectorized: {} Gops\n    -Vectorized and"
    " Parallelized: {} Gops\n    -Preallocated and Vectorized: {} Gops\n"
    "    -Preallocated, Vectorized, and Parallelized: {} Gops\n"
    "    -Single DMA: {} Gops\n    -Preloaded: {} Gops\n"
)


def apply_parallel_unroll_vectorize(sch, blocks, outer_split, unroll_split, vector_split):
    """Apply parallel unroll vectorized."""
    for block in blocks:
        vb_index, vi_index = sch.get_loops(block)
        v = sch.fuse(vb_index, vi_index)
        vbo, vbi, vio, vii = sch.split(  # pylint: disable=unused-variable
            v, factors=[outer_split, None, unroll_split, vector_split]
        )  # pylint: disable=unused-variable
        sch.vectorize(vii)
        sch.unroll(vio)
        sch.parallel(vbo)
    return sch


def apply_unroll_vectorize(sch, blocks, unroll_split, vector_split):
    for block in blocks:
        vb_index, vi_index = sch.get_loops(block)
        v = sch.fuse(vb_index, vi_index)
        _, vio, vii = sch.split(v, factors=[None, unroll_split, vector_split])
        sch.vectorize(vii)
        sch.unroll(vio)
    return sch


def apply_vrmpy_parallelization(sch):
    block = sch.get_block("c_buffer")
    b = sch.get_loops(block)
    b_outer, _ = sch.split(b[0], factors=[4, None])
    sch.parallel(b_outer)
    return sch


def apply_vtcm_cache_read_write(sch):
    block = sch.get_block("c_buffer")
    sch.cache_read(block, 0, "global.vtcm")
    sch.cache_read(block, 1, "global.vtcm")
    sch.cache_write(block, 0, "global.vtcm")
    return sch


def vrmpy(operations):
    """Generate VRMPY operator"""

    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        a_buffer = T.match_buffer(a, [operations, 128], dtype="uint8", align=128)
        b_buffer = T.match_buffer(b, [operations, 128], dtype="uint8", align=128)
        c_buffer = T.match_buffer(c, [operations, 32], dtype="int32", align=128)
        for n in T.grid(operations):
            with T.block("c_buffer"):
                vn_ind = T.axis.remap("S", [n])
                c_buffer[vn_ind, T.ramp(0, 1, 32)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.128B"),
                    T.uint32(2),
                    T.reinterpret(a_buffer[vn_ind, T.ramp(0, 1, 128)], dtype="int32x32"),
                    T.reinterpret(b_buffer[vn_ind, T.ramp(0, 1, 128)], dtype="int32x32"),
                    dtype="int32x32",
                )

    return operator


def preloaded_vrmpy(operations):
    """Generate preloaded VRMPY operator."""

    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        a_buffer = T.match_buffer(
            a,
            [T.cast(operations, "int32") * 128],
            dtype="uint8",
            align=128,
            scope="global.vtcm",
        )
        b_buffer = T.match_buffer(
            b,
            [T.cast(operations, "int32") * 128],
            dtype="uint8",
            align=128,
            scope="global.vtcm",
        )
        c_buffer = T.match_buffer(
            c, [T.cast(operations, "int32") * 32], dtype="int32", align=128, scope="global.vtcm"
        )
        for n in T.grid(operations):
            with T.block("c_buffer"):
                vn_ind = T.axis.remap("S", [n])
                c_buffer[T.ramp(T.cast(vn_ind, "int32") * 32, 1, 32)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.128B"),
                    T.uint32(2),
                    T.reinterpret(
                        a_buffer[T.ramp(T.cast(vn_ind, "int32") * 128, 1, 128)], dtype="int32x32"
                    ),
                    T.reinterpret(
                        b_buffer[T.ramp(T.cast(vn_ind, "int32") * 128, 1, 128)], dtype="int32x32"
                    ),
                    dtype="int32x32",
                )

    return operator


def preallocated_vrmpy(operations):
    """Generate preallocated VRMPY operator."""
    size = operations * 128
    out_size = operations * 32

    @T.prim_func
    def operator(
        a: T.handle, b: T.handle, c: T.handle, a_v: T.handle, b_v: T.handle, c_v: T.handle
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        a_buffer = T.match_buffer(a, [operations, 128], dtype="uint8", align=128, scope="global")
        b_buffer = T.match_buffer(b, [operations, 128], dtype="uint8", align=128, scope="global")
        c_buffer = T.match_buffer(c, [operations, 32], dtype="int32", align=128, scope="global")
        a_global_vtcm = T.match_buffer(a_v, [size], dtype="uint8", align=128, scope="global.vtcm")
        b_global_vtcm = T.match_buffer(b_v, [size], dtype="uint8", align=128, scope="global.vtcm")
        c_global_vtcm = T.match_buffer(
            c_v, [out_size], dtype="int32", align=128, scope="global.vtcm"
        )
        for n, i in T.grid(operations, 128):
            with T.block("a_buffer_global.vtcm"):
                vn_ind, vi_index = T.axis.remap("SS", [n, i])
                a_global_vtcm[vn_ind * 128 + vi_index] = a_buffer[vn_ind, vi_index]
        for n, i in T.grid(operations, 128):
            with T.block("b_buffer_global.vtcm"):
                vn_ind, vi_index = T.axis.remap("SS", [n, i])
                b_global_vtcm[vn_ind * 128 + vi_index] = b_buffer[vn_ind, vi_index]
        for n in T.grid(operations):
            with T.block("c_buffer"):
                vn_ind = T.axis.remap("S", [n])
                c_global_vtcm[T.ramp(T.cast(vn_ind, "int32") * 32, 1, 32)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.128B"),
                    T.uint32(2),
                    T.reinterpret(
                        a_global_vtcm[T.ramp(T.cast(vn_ind, "int32") * 128, 1, 128)],
                        dtype="int32x32",
                    ),
                    T.reinterpret(
                        b_global_vtcm[T.ramp(T.cast(vn_ind, "int32") * 128, 1, 128)],
                        dtype="int32x32",
                    ),
                    dtype="int32x32",
                )
        for n, i in T.grid(operations, 32):
            with T.block("c_buffer_global.vtcm"):
                vn_ind, vi_index = T.axis.remap("SS", [n, i])
                c_buffer[vn_ind, vi_index] = c_global_vtcm[vn_ind * 32 + vi_index]

    return operator


def preallocated_single_dma_vrmpy(operations):
    """Generate preallocated single DMA VRMPY operator."""
    size = operations * 128
    out_size = operations * 32

    @T.prim_func
    def operator(
        a: T.handle,
        b: T.handle,
        c: T.handle,
        a_v: T.handle,
        b_v: T.handle,
        c_v: T.handle,
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        a_buffer = T.match_buffer(a, [operations, 128], dtype="uint8", align=128, scope="global")
        b_buffer = T.match_buffer(b, [operations, 128], dtype="uint8", align=128, scope="global")
        c_buffer = T.match_buffer(c, [operations, 32], dtype="int32", align=128, scope="global")
        a_global_vtcm = T.match_buffer(a_v, [size], dtype="uint8", align=128, scope="global.vtcm")
        b_global_vtcm = T.match_buffer(b_v, [size], dtype="uint8", align=128, scope="global.vtcm")
        c_global_vtcm = T.match_buffer(
            c_v, [out_size], dtype="int32", align=128, scope="global.vtcm"
        )
        T.evaluate(
            T.tvm_call_packed(
                "device_api.hexagon.dma_copy_dltensor",
                T.tvm_stack_make_array(
                    a_global_vtcm.data,
                    T.tvm_stack_make_shape(size, dtype="handle"),
                    0,
                    1,
                    a_global_vtcm.dtype,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    a_buffer.data,
                    T.tvm_stack_make_shape(size, dtype="handle"),
                    0,
                    1,
                    a_buffer.dtype,
                    0,
                    dtype="handle",
                ),
                T.cast(size, dtype="int"),
                True,  # bypass cache
                dtype="int32",
            )
        )
        T.evaluate(
            T.tvm_call_packed(
                "device_api.hexagon.dma_copy_dltensor",
                T.tvm_stack_make_array(
                    b_global_vtcm.data,
                    T.tvm_stack_make_shape(size, dtype="handle"),
                    0,
                    1,
                    b_global_vtcm.dtype,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    b_buffer.data,
                    T.tvm_stack_make_shape(size, dtype="handle"),
                    0,
                    1,
                    b_buffer.dtype,
                    0,
                    dtype="handle",
                ),
                T.cast(size, dtype="int"),
                True,  # bypass cache
                dtype="int32",
            )
        )
        for n in T.grid(operations):
            with T.block("c_buffer"):
                vn_ind = T.axis.remap("S", [n])
                c_global_vtcm[T.ramp(T.cast(vn_ind, "int32") * 32, 1, 32)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.128B"),
                    T.uint32(2),
                    T.reinterpret(
                        a_global_vtcm[T.ramp(T.cast(vn_ind, "int32") * 128, 1, 128)],
                        dtype="int32x32",
                    ),
                    T.reinterpret(
                        b_global_vtcm[T.ramp(T.cast(vn_ind, "int32") * 128, 1, 128)],
                        dtype="int32x32",
                    ),
                    dtype="int32x32",
                )
        T.evaluate(
            T.tvm_call_packed(
                "device_api.hexagon.dma_copy_dltensor",
                T.tvm_stack_make_array(
                    c_buffer.data,
                    T.tvm_stack_make_shape(size, dtype="handle"),
                    0,
                    1,
                    c_buffer.dtype,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    c_global_vtcm.data,
                    T.tvm_stack_make_shape(size, dtype="handle"),
                    0,
                    1,
                    c_global_vtcm.dtype,
                    0,
                    dtype="handle",
                ),
                T.cast(size, dtype="int"),
                True,  # bypass cache
                dtype="int32",
            )
        )

    return operator


def evaluate_result(operations, tag, time, result, expected_output):
    transfer_mb = round(3 * operations * 128 / 1e6, 2)
    gops = round(operations * 128 * 3 / time.mean / 1e9, 3)
    mean_ms = round(time.mean * 1000, 6)

    print(f"\ntest_{transfer_mb}MB_{tag} took {mean_ms} ms @ GOPS: {gops}")
    tvm.testing.assert_allclose(result, expected_output)


def setup_and_run(hexagon_session, sch, a, b, c, operations, mem_scope="global"):
    """Setup and run operator."""
    func_tir = tvm.build(sch.mod["main"], target=get_hexagon_target("v69"))
    module = hexagon_session.load_module(func_tir)

    a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device, mem_scope=mem_scope)
    b_hexagon = tvm.runtime.ndarray.array(b, device=hexagon_session.device, mem_scope=mem_scope)
    c_hexagon = tvm.runtime.ndarray.array(c, device=hexagon_session.device, mem_scope=mem_scope)

    # These are reduced for CI but number=100 and repeat=10 does a good job of removing noise.
    number = 1
    repeat = 1

    timer = module.time_evaluator(
        "__tvm_main__", hexagon_session.device, number=number, repeat=repeat
    )
    time = timer(a_hexagon, b_hexagon, c_hexagon)
    gops = round(operations * 128 * 3 / time.mean / 1e9, 4)
    return gops, c_hexagon.asnumpy()


def setup_and_run_preallocated(hexagon_session, sch, a, b, c, operations):
    """Setup and run for preallocated."""
    func_tir = tvm.build(sch.mod["main"], target=get_hexagon_target("v69"))
    module = hexagon_session.load_module(func_tir)

    a_vtcm = np.zeros((a.size), dtype="uint8")
    b_vtcm = np.zeros((b.size), dtype="uint8")
    c_vtcm = np.zeros((c.size), dtype="int32")

    a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device, mem_scope="global")
    b_hexagon = tvm.runtime.ndarray.array(b, device=hexagon_session.device, mem_scope="global")
    c_hexagon = tvm.runtime.ndarray.array(c, device=hexagon_session.device, mem_scope="global")
    a_vtcm_hexagon = tvm.runtime.ndarray.array(
        a_vtcm, device=hexagon_session.device, mem_scope="global.vtcm"
    )
    b_vtcm_hexagon = tvm.runtime.ndarray.array(
        b_vtcm, device=hexagon_session.device, mem_scope="global.vtcm"
    )
    c_vtcm_hexagon = tvm.runtime.ndarray.array(
        c_vtcm, device=hexagon_session.device, mem_scope="global.vtcm"
    )

    # These are reduced for CI but number=100 and repeat=10 does a good job of removing noise.
    number = 1
    repeat = 1

    timer = module.time_evaluator(
        "__tvm_main__", hexagon_session.device, number=number, repeat=repeat
    )
    time = timer(a_hexagon, b_hexagon, c_hexagon, a_vtcm_hexagon, b_vtcm_hexagon, c_vtcm_hexagon)
    gops = round(operations * 128 * 3 / time.mean / 1e9, 4)
    return gops, c_hexagon.asnumpy()


class TestMatMulVec:
    """MatMul test class."""

    # Removed most of these to speedup CI.
    operations = tvm.testing.parameter(
        1024,
        # 2048,
        # 4096,
        # 5 * 2048,  # 3.93MB of total transfer
        # 16384, #Only works on 8Gen1 HDK's
        # 5 * 4096,  # 7.86MB of total transfer. Only works on 8Gen1 HDK's
    )

    # Experimentally best configurations for the memcopy
    outer_split = tvm.testing.parameter(4)
    unroll_split = tvm.testing.parameter(8)
    vector_split = tvm.testing.parameter(64)
    c_vector_split = tvm.testing.parameter(16)
    c_vector_split_unallocated = tvm.testing.parameter(8)

    @tvm.testing.fixture
    def input_a(self, operations):
        return np.random.randint(0, 16, (operations, 128), dtype="uint8")

    @tvm.testing.fixture
    def input_b(self, operations):
        return np.random.randint(0, 16, (operations, 128), dtype="uint8")

    @tvm.testing.fixture
    def input_c(self, operations):
        return np.zeros((operations, 32), dtype="int32")

    @tvm.testing.fixture
    def expected_output(self, operations, input_a, input_b, input_c):
        expected_output = np.zeros(input_c.shape, dtype="int32")
        for n in range(operations):
            for i in range(32):
                for r_ind in range(4):  # pylint: disable=unused-variable
                    expected_output[n, i] = expected_output[n, i] + np.uint32(
                        input_a[n, i * 4 + r_ind]
                    ) * np.uint32(input_b[n, i * 4 + r_ind])
        return expected_output

    @tvm.testing.requires_hexagon
    def test_loading_vtcm_for_vrmpy(
        self,
        hexagon_session,
        operations,
        input_a,
        input_b,
        input_c,
        expected_output,
        outer_split,
        unroll_split,
        vector_split,
        c_vector_split,
        c_vector_split_unallocated,
    ):
        """Load VTCM for VRMPY operator test."""
        # Run parallel vrmpy without loading to VTCM.
        sch = tvm.tir.Schedule(vrmpy(operations))
        sch = apply_vrmpy_parallelization(sch)
        base_runtime, result = setup_and_run(
            hexagon_session, sch, input_a, input_b, input_c, operations
        )
        tvm.testing.assert_allclose(result, expected_output)

        # Run parallel vrmpy with basic memory loads to VTCM.
        sch = tvm.tir.Schedule(vrmpy(operations))
        sch = apply_vtcm_cache_read_write(sch)
        sch = apply_vrmpy_parallelization(sch)
        basic_load_runtime, result = setup_and_run(
            hexagon_session, sch, input_a, input_b, input_c, operations
        )
        tvm.testing.assert_allclose(result, expected_output)

        # Run parallel vrmpy with vectorized memory loads to VTCM.
        sch = tvm.tir.Schedule(vrmpy(operations))
        sch = apply_vtcm_cache_read_write(sch)
        sch = apply_vrmpy_parallelization(sch)
        sch = apply_unroll_vectorize(
            sch,
            [sch.get_block("a_buffer_global.vtcm"), sch.get_block("b_buffer_global.vtcm")],
            unroll_split,
            vector_split,
        )
        sch = apply_unroll_vectorize(
            sch, [sch.get_block("c_buffer_global.vtcm")], unroll_split, c_vector_split_unallocated
        )
        vectorized_runtime, result = setup_and_run(
            hexagon_session, sch, input_a, input_b, input_c, operations
        )
        tvm.testing.assert_allclose(result, expected_output)

        # Run parallel vrmpy with vectorized and parallelized memory loads to VTCM.
        sch = tvm.tir.Schedule(vrmpy(operations))
        sch = apply_vtcm_cache_read_write(sch)
        sch = apply_vrmpy_parallelization(sch)
        sch = apply_parallel_unroll_vectorize(
            sch,
            [sch.get_block("a_buffer_global.vtcm"), sch.get_block("b_buffer_global.vtcm")],
            outer_split,
            unroll_split,
            vector_split,
        )
        sch = apply_parallel_unroll_vectorize(
            sch,
            [sch.get_block("c_buffer_global.vtcm")],
            outer_split,
            unroll_split,
            c_vector_split_unallocated,
        )
        vectorized_parallelized_runtime, result = setup_and_run(
            hexagon_session, sch, input_a, input_b, input_c, operations
        )
        tvm.testing.assert_allclose(result, expected_output)

        # Run parallel vrmpy with preallocated and vectorized memory loads to VTCM.
        sch = tvm.tir.Schedule(preallocated_vrmpy(operations))
        sch = apply_vrmpy_parallelization(sch)
        sch = apply_unroll_vectorize(
            sch,
            [sch.get_block("a_buffer_global.vtcm"), sch.get_block("b_buffer_global.vtcm")],
            unroll_split,
            vector_split,
        )
        sch = apply_unroll_vectorize(
            sch, [sch.get_block("c_buffer_global.vtcm")], unroll_split, c_vector_split
        )
        preallocated_vectorized_runtime, result = setup_and_run_preallocated(
            hexagon_session, sch, input_a, input_b, input_c, operations
        )
        result = result.reshape((operations, 32))
        tvm.testing.assert_allclose(result, expected_output)

        # Run parallel vrmpy with preallocated, vectorized, and parallelized memory loads to VTCM.
        sch = tvm.tir.Schedule(preallocated_vrmpy(operations))
        sch = apply_vrmpy_parallelization(sch)
        sch = apply_parallel_unroll_vectorize(
            sch,
            [sch.get_block("a_buffer_global.vtcm"), sch.get_block("b_buffer_global.vtcm")],
            outer_split,
            unroll_split,
            vector_split,
        )
        sch = apply_parallel_unroll_vectorize(
            sch, [sch.get_block("c_buffer_global.vtcm")], outer_split, unroll_split, c_vector_split
        )
        prealloc_vector_parallelized, result = setup_and_run_preallocated(
            hexagon_session, sch, input_a, input_b, input_c, operations
        )
        result = result.reshape((operations, 32))
        tvm.testing.assert_allclose(result, expected_output)

        # Run parallel vrmpy with preallocated single dma memory load to VTCM.
        sch = tvm.tir.Schedule(preallocated_single_dma_vrmpy(operations))
        sch = apply_vrmpy_parallelization(sch)
        single_dma_runtime, result = setup_and_run_preallocated(
            hexagon_session, sch, input_a, input_b, input_c, operations
        )
        result = result.reshape((operations, 32))
        tvm.testing.assert_allclose(result, expected_output)

        # Run parallel vrmpy with data preloaded in VTCM.
        sch = tvm.tir.Schedule(preloaded_vrmpy(operations))
        sch = apply_vrmpy_parallelization(sch)
        input_a = input_a.reshape(operations * 128)
        input_b = input_b.reshape(operations * 128)
        input_c = input_c.reshape(operations * 32)
        preloaded_runtime, result = setup_and_run(
            hexagon_session, sch, input_a, input_b, input_c, operations, "global.vtcm"
        )
        result = result.reshape((operations, 32))
        tvm.testing.assert_allclose(result, expected_output)

        transfer_mb = round(3 * operations * 128 / 1e6, 2)
        print(
            TEST_OUTPUT_TEMPLATE.format(
                transfer_mb,
                base_runtime,
                basic_load_runtime,
                vectorized_runtime,
                vectorized_parallelized_runtime,
                preallocated_vectorized_runtime,
                prealloc_vector_parallelized,
                single_dma_runtime,
                preloaded_runtime,
            )
        )


if __name__ == "__main__":
    tvm.testing.main()
