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
# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for GPU operators."""
import math
from typing import Literal, Optional

from tvm import DataType, tir
from tvm.target import Target
from tvm.tir.stmt import ForKind

from ..base.roller.rasterization import NoRasterization
from ..base import analysis
from .base import GPUScheduleRule
from .matmul_analysis import (
    auto_inline_consumer_chain,
    auto_inline_consumers,
    auto_inline_producers,
    get_index_map,
    get_reduction_blocks,
    normalize_to_matmul,
)


class MatmulTensorizationWMMA(GPUScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        micro_size_x = 16
        micro_size_y = 16
        micro_size_k = 16

        warp_size = 32
        vector_size = 4

        i_factors, j_factors, k_factors = (
            [None, 1, 4, 2],
            [1, None, 4, 2],
            [None, 4],
        )

        num_ty = i_factors[2] * j_factors[2]
        x_pad_factor = i_factors[2] * i_factors[3]
        y_pad_factor = j_factors[2] * j_factors[3]
        k_pad_factor = k_factors[1]

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                micro_size_x * x_pad_factor,
                micro_size_y * y_pad_factor,
                micro_size_k * k_pad_factor,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k = sch.get_loops(block)

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_x])
        j, j_inner = sch.split(j, factors=[None, micro_size_y])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = block
        block_outer = sch.blockize(i_inner)

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, k_factors)
        sch.annotate(k0, "software_pipeline_order", [0, 3, 1, 4, 5, 2, 6])
        sch.annotate(k0, "software_pipeline_stage", [0, 0, 0, 0, 0, 1, 1])
        sch.annotate(k1, "software_pipeline_order", [0, 1, 2])
        sch.annotate(k1, "software_pipeline_stage", [0, 0, 1])

        sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3)

        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        sch.bind(batch, "blockIdx.z")
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])

            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)

            sch.storage_align(block_read, 0, axis=-2, factor=16, offset=8)
            sch.annotate(block_read, "tir.manifest_shared_memory_local_stage", 1)
            sch.annotate(block_read, "double_buffer_scope", 0)
            return block_read

        a_g2s = fetch_to_shared(block_outer, 0, 2)
        b_g2s = fetch_to_shared(block_outer, 1, 2)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared.dyn")
        sch.storage_align(accumulator_shared_to_global, 0, -2, 16, 4)

        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(store, thread_idy)
        sch.reverse_compute_at(accumulator_shared_to_global, thread_idy)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype="float16",
            out_dtype="float32",
            trans_b=True,
        )

        try:
            i, j = sch.get_loops(A_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_a"])

            i, j = sch.get_loops(B_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            return None

        # Try to tensorize the init, store and compute block with f16 or f32 intrinsics
        tensorize_success: bool = False

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        try:
            tensorize_init_store_compute()
            tensorize_success = True
        except:  # pylint: disable=bare-except
            intrin_group = get_wmma_intrin_group(
                load_scope="shared.dyn",
                store_scope="shared.dyn",
                in_dtype="float16",
                out_dtype="float16",
                trans_b=True,
            )

        if not tensorize_success:
            try:
                tensorize_init_store_compute()
                tensorize_success = True
            except:  # pylint: disable=bare-except
                return None
        auto_inline_consumer_chain(sch, accumulator_shared_to_global)

        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
        _, f1, f2 = sch.split(fused, factors=[None, warp_size, vector_size])
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)

        return sch if tensorize_success else None


class MatmulInt8Tensorization(GPUScheduleRule):
    """
    The schedule rule for int8 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        micro_size_x = 16
        micro_size_y = 16
        micro_size_k = 16

        warp_size = 32
        vector_size = 4

        i_factors, j_factors, k_factors = (
            [None, 1, 4, 2],
            [1, None, 4, 2],
            [None, 1],
        )

        num_ty = i_factors[2] * j_factors[2]
        x_pad_factor = i_factors[2] * i_factors[3]
        y_pad_factor = j_factors[2] * j_factors[3]
        k_pad_factor = k_factors[1]

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                micro_size_x * x_pad_factor,
                micro_size_y * y_pad_factor,
                micro_size_k * k_pad_factor,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k = sch.get_loops(block)

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_x])
        j, j_inner = sch.split(j, factors=[None, micro_size_y])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = block
        block_outer = sch.blockize(i_inner)

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, k_factors)
        sch.annotate(k0, "software_pipeline_order", [0, 3, 1, 4, 5, 2, 6])
        sch.annotate(k0, "software_pipeline_stage", [0, 0, 0, 0, 0, 1, 1])
        sch.annotate(k1, "software_pipeline_order", [0, 1, 2])
        sch.annotate(k1, "software_pipeline_stage", [0, 0, 1])

        sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3)

        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        sch.bind(batch, "blockIdx.z")
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])

            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)

            sch.storage_align(block_read, 0, axis=-2, factor=32, offset=16)
            sch.annotate(block_read, "tir.manifest_shared_memory_local_stage", 1)
            sch.annotate(block_read, "double_buffer_scope", 0)
            return block_read

        a_g2s = fetch_to_shared(block_outer, 0, 2)
        b_g2s = fetch_to_shared(block_outer, 1, 2)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared.dyn")
        sch.storage_align(accumulator_shared_to_global, 0, -2, 16, 4)

        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(store, thread_idy)
        sch.reverse_compute_at(accumulator_shared_to_global, thread_idy)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype="int8",
            out_dtype="int32",
            trans_b=True,
        )

        try:
            i, j = sch.get_loops(A_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_a"])

            i, j = sch.get_loops(B_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            return None

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        try:
            tensorize_init_store_compute()
        except:  # pylint: disable=bare-except
            return None

        auto_inline_consumer_chain(sch, accumulator_shared_to_global)

        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
        _, f1, f2 = sch.split(fused, factors=[None, warp_size, vector_size])
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)

        return sch


class MatmulTensorizationLegacy(GPUScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        micro_size_x = 16
        micro_size_y = 16
        micro_size_k = 16

        warp_size = 32
        vector_size = 4

        i_factors, j_factors, k_factors = (
            [None, 1, 4, 2],
            [1, None, 4, 2],
            [None, 4],
        )

        num_ty = i_factors[2] * j_factors[2]
        x_pad_factor = i_factors[2] * i_factors[3]
        y_pad_factor = j_factors[2] * j_factors[3]
        k_pad_factor = k_factors[1]

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                micro_size_x * x_pad_factor,
                micro_size_y * y_pad_factor,
                micro_size_k * k_pad_factor,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k = sch.get_loops(block)

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_x])
        j, j_inner = sch.split(j, factors=[None, micro_size_y])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = block
        block_outer = sch.blockize(i_inner)

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, k_factors)
        sch.annotate(k0, "software_pipeline_order", [0, 3, 1, 4, 5, 2, 6])
        sch.annotate(k0, "software_pipeline_stage", [0, 0, 0, 0, 0, 1, 1])
        sch.annotate(k1, "software_pipeline_order", [0, 1, 2])
        sch.annotate(k1, "software_pipeline_stage", [0, 0, 1])

        sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3)

        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        sch.bind(batch, "blockIdx.z")
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])

            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)

            sch.storage_align(block_read, 0, axis=-2, factor=16, offset=8)
            sch.annotate(block_read, "tir.manifest_shared_memory_local_stage", 1)
            sch.annotate(block_read, "double_buffer_scope", 0)
            return block_read

        a_g2s = fetch_to_shared(block_outer, 0, 2)
        b_g2s = fetch_to_shared(block_outer, 1, 2)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared.dyn")
        sch.storage_align(accumulator_shared_to_global, 0, -2, 16, 4)

        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(store, thread_idy)
        sch.reverse_compute_at(accumulator_shared_to_global, thread_idy)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype="float16",
            out_dtype="float32",
            trans_b=True,
        )

        try:
            i, j = sch.get_loops(A_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_a"])

            i, j = sch.get_loops(B_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            return None

        # Try to tensorize the init, store and compute block with f16 or f32 intrinsics
        tensorize_success: bool = False

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        try:
            tensorize_init_store_compute()
            tensorize_success = True
        except:  # pylint: disable=bare-except
            intrin_group = get_wmma_intrin_group(
                load_scope="shared.dyn",
                store_scope="shared.dyn",
                in_dtype="float16",
                out_dtype="float16",
                trans_b=True,
            )

        if not tensorize_success:
            try:
                tensorize_init_store_compute()
                tensorize_success = True
            except:  # pylint: disable=bare-except
                return None
        auto_inline_consumer_chain(sch, accumulator_shared_to_global)

        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
        _, f1, f2 = sch.split(fused, factors=[None, warp_size, vector_size])
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)

        return sch if tensorize_success else None

    def apply_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        intrin_info = config.intrin_info
        warp_row_tiles = config.warp[0]
        warp_col_tiles = config.warp[1]
        block_row_warps = config.block[0] // warp_row_tiles
        block_col_warps = config.block[1] // warp_col_tiles
        stage = config.pipeline_stage
        use_async = config.use_async
        chunk = config.rstep[0]

        micro_size_x = 16
        micro_size_y = 16
        micro_size_k = 16

        warp_size = 32

        i_factors, j_factors, k_factors = (
            [None, 1, block_row_warps, warp_row_tiles // micro_size_x],
            [1, None, block_col_warps, warp_col_tiles // micro_size_y],
            [None, chunk // micro_size_k],
        )

        num_ty = i_factors[2] * j_factors[2]
        x_pad_factor = i_factors[2] * i_factors[3]
        y_pad_factor = j_factors[2] * j_factors[3]
        k_pad_factor = k_factors[1]

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]/B[S, K, J]
        if not (func.attrs is not None and "dlight.tensorcore_prenormlized" in func.attrs.keys()):
            sch = normalize_to_matmul(sch, main_block, ["a", "a", "a"])

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                micro_size_x * x_pad_factor,
                micro_size_y * y_pad_factor,
                micro_size_k * k_pad_factor,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k = sch.get_loops(block)

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_x])
        j, j_inner = sch.split(j, factors=[None, micro_size_y])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = block
        block_outer = sch.blockize(i_inner)

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, k_factors)

        sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3)

        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        # plan rasteration
        if (
            not isinstance(config.rasterization_plan, NoRasterization)
            and sch.get(batch).extent.value == 1
        ):
            device_func, invoke_func = config.rasterization_plan.get_code()
            factor = config.rasterization_plan.panel_width_

            # TODO(lei): this is a trick for rasterization implementation
            # wait for https://github.com/apache/tvm/pull/16113 to be merged
            # require a solution for general block rasterization
            factor = 8  # should be divisible by block_idy
            if sch.get(block_idy).extent.value % factor == 0:
                block_k, block_idy = sch.split(block_idy, factors=[None, factor])
                sch.bind(block_k, "blockIdx.z")
        else:
            sch.bind(batch, "blockIdx.z")

        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim, vec_len, dtype="float16"):
            block_read = sch.cache_read(block, idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vec_len])

            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)
            offset: int = 0
            if dtype == "float16":
                offset = 8
            elif dtype == "int8":
                offset = 16
            # todo(lei): the pad value should be varied according to the data type
            sch.storage_align(block_read, 0, axis=-2, factor=16, offset=offset)
            return block_read

        a_g2s = fetch_to_shared(
            block_outer,
            0,
            2,
            vec_len=list(config.vectorize.values())[0],
            dtype=intrin_info.in_dtype,
        )
        b_g2s = fetch_to_shared(
            block_outer,
            1,
            2,
            vec_len=list(config.vectorize.values())[1],
            dtype=intrin_info.in_dtype,
        )

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared.dyn")
        sch.storage_align(accumulator_shared_to_global, 0, -2, 16, 4)

        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(store, thread_idy)
        sch.reverse_compute_at(accumulator_shared_to_global, thread_idy)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype=intrin_info.in_dtype,
            out_dtype=intrin_info.out_dtype,
            trans_b=intrin_info.trans_b,
        )

        try:
            i, j = sch.get_loops(A_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_a"])

            i, j = sch.get_loops(B_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            return None

        # Try to tensorize the init, store and compute block with f16 or f32 intrinsics
        tensorize_success: bool = False

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        try:
            tensorize_init_store_compute()
            tensorize_success = True
        except:  # pylint: disable=bare-except
            return None

        auto_inline_consumer_chain(sch, accumulator_shared_to_global)

        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
        _, f1, f2 = sch.split(
            fused, factors=[None, warp_size, max(list(config.vectorize.values()))]
        )
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)

        if stage > 1:
            sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
            sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
        if use_async:
            sch.annotate(k0, "software_pipeline_async_stages", [0])

        return sch if tensorize_success else None

