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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from functools import reduce

from tvm import tir
from tvm.target import Target
from tvm.tir import PrimExpr
from tvm.tir import IndexMap
from .base import GPUScheduleRule
from ..base import ScheduleRule, analysis
from ..base.roller.rasterization import NoRasterization
from ..base.analysis import (
    IterKind,
    IterTrait,
    detect_iter_traits,
    get_reduction_blocks,
    get_index_map,
    get_in_out_dtypes,
    normalize_with_tensorcore,
)



def _collect_producers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def auto_inline_producers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumer_chain(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    auto_inline_consumers(sch, block)
    remaining_consumers = sch.get_consumers(block)

    if len(remaining_consumers) != 0:
        # Some blocks have failed to be inlined to the producer cache-write stage.
        # This could be due to another producer block that has not been scheduled.
        for c in remaining_consumers:
            for p in sch.get_producers(c):
                if sch.get(p) != sch.get(block):
                    auto_inline_producers(sch, p)
                    sch.compute_inline(p)

        # Try inlining into the cache-write stage again, this time it should succeed.
        auto_inline_consumers(sch, block)

    msg = "There are some consumers of the cache-write stage that are not properly inlined."
    assert len(sch.get_consumers(block)) == 0, msg


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


class MatmulAdvancedTensorization(ScheduleRule):
    """
    The advanced schedule rule for float16 tensor core matmul computation.
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

        def make_iter_fusion_index_map(
            traits: List[IterTrait],
            kind_order: List[IterKind],
            keep_last_dims: int = 0,
        ) -> tir.IndexMap:
            fused_iters: Dict[IterKind, PrimExpr] = {}
            keep_iters: List[tir.Var] = []
            input_iters: List[tir.Var] = []
            for i, trait in enumerate(traits):
                v_i = tir.Var(f"i{i}", trait.extent.dtype)
                input_iters.append(v_i)
                if trait.kind == IterKind.kIter_T:
                    continue
                if trait.kind not in kind_order:
                    raise ValueError(f"Unknown iter kind {trait.kind}")
                if i + keep_last_dims < len(traits):
                    if trait.kind in fused_iters:
                        fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
                    else:
                        fused_iters[trait.kind] = v_i
                else:
                    keep_iters.append(v_i)
            final_indices: List[tir.PrimExpr] = [
                fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
            ]
            final_indices.extend(keep_iters)

            return tir.IndexMap(input_iters, final_indices, None)

        def get_index_map(block: tir.Block) -> Optional[Tuple[tir.IndexMap, ...]]:
            """Get index maps for the block

            Parameters
            ----------
            block : tir.Block
                The block to be analyzed

            Returns
            -------
            index_maps : Optional[Tuple[tir.IndexMap]]
                The index maps for the block, or None if the block is not a gemm-liked kernel
            """
            traits = detect_iter_traits(block)
            if traits is None:
                return None
            A_traits, B_traits, C_traits, block_traits = traits

            A_index_map = make_iter_fusion_index_map(
                A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K], keep_last_dims=2
            )
            B_index_map = make_iter_fusion_index_map(
                B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K], keep_last_dims=2
            )
            C_index_map = make_iter_fusion_index_map(
                C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J], keep_last_dims=3
            )
            matmul_index_map = make_iter_fusion_index_map(
                block_traits,
                [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K],
                keep_last_dims=3,
            )

            return (
                matmul_index_map,
                A_index_map,
                B_index_map,
                C_index_map,
            )

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
                x_pad_factor,
                y_pad_factor,
                k_pad_factor,
                micro_size_x,
                micro_size_y,
                micro_size_k,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k, i_inner, j_inner, k_inner = sch.get_loops(block)

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


class MatmulAdvancedTensorizationMMA(ScheduleRule):
    """
    The advanced schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            LDMATRIX_f16_A_INTRIN,
            LDMATRIX_f16_B_INTRIN,
            LDMATRIX_f16_B_TRANS_INTRIN,
            MMA_f16f16f16_INTRIN,
            MMA_f16f16f16_TRANS_B_INTRIN,
            MMA_fill_16x16_f16_INTRIN,
            MMA_store_16x16_f16_shared_INTRIN,
            shared_16x16_to_mma_32x8_layout,
            ldmatrix_32x8_to_shared_16x16_layout,
            ldmatrix_trans_32x8_to_shared_16x16_layout,
        )

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        def make_iter_fusion_index_map(
            traits: List[IterTrait],
            kind_order: List[IterKind],
            keep_last_dims: int = 0,
        ) -> tir.IndexMap:
            fused_iters: Dict[IterKind, PrimExpr] = {}
            keep_iters: List[tir.Var] = []
            input_iters: List[tir.Var] = []
            for i, trait in enumerate(traits):
                v_i = tir.Var(f"i{i}", trait.extent.dtype)
                input_iters.append(v_i)
                if trait.kind == IterKind.kIter_T:
                    continue
                if trait.kind not in kind_order:
                    raise ValueError(f"Unknown iter kind {trait.kind}")
                if i + keep_last_dims < len(traits):
                    if trait.kind in fused_iters:
                        fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
                    else:
                        fused_iters[trait.kind] = v_i
                else:
                    keep_iters.append(v_i)
            final_indices: List[tir.PrimExpr] = [
                fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
            ]
            final_indices.extend(keep_iters)

            return tir.IndexMap(input_iters, final_indices, None)

        def get_index_map(block: tir.Block) -> Optional[Tuple[tir.IndexMap, ...]]:
            """Get index maps for the block

            Parameters
            ----------
            block : tir.Block
                The block to be analyzed

            Returns
            -------
            index_maps : Optional[Tuple[tir.IndexMap]]
                The index maps for the block, or None if the block is not a gemm-liked kernel
            """
            traits = detect_iter_traits(block)
            if traits is None:
                return None
            A_traits, B_traits, C_traits, block_traits = traits

            A_index_map = make_iter_fusion_index_map(
                A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K], keep_last_dims=2
            )
            B_index_map = make_iter_fusion_index_map(
                B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K], keep_last_dims=2
            )
            C_index_map = make_iter_fusion_index_map(
                C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J], keep_last_dims=3
            )
            matmul_index_map = make_iter_fusion_index_map(
                block_traits,
                [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K],
                keep_last_dims=3,
            )

            return (
                matmul_index_map,
                A_index_map,
                B_index_map,
                C_index_map,
            )

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        if func.attrs is not None and "transform_kind" in func.attrs.keys():
            if func.attrs["transform_kind"] >= 2:

                def ldmatrix_permutation_16x16_32x8_16x16(kernel_i, kernel_j):
                    thread_id = kernel_i * 2 + kernel_j // 8
                    local_id = kernel_j % 8
                    return ldmatrix_32x8_to_shared_16x16_layout(thread_id, local_id)

                def ldmatrix_trans_permutation_16x16_32x8_16x16(kernel_i, kernel_j):
                    thread_id = kernel_i * 2 + kernel_j // 8
                    local_id = kernel_j % 8
                    return ldmatrix_trans_32x8_to_shared_16x16_layout(thread_id, local_id)

                index_map = IndexMap.from_func(ldmatrix_permutation_16x16_32x8_16x16)
                inversed_index_map = index_map.inverse([16, 16])

                def A_permutation_inverse(i, j, kernel_i, kernel_j):
                    return (i, j, *inversed_index_map.map_indices([kernel_i, kernel_j]))

                sch.transform_layout(
                    sch.get_block("A_reindex_reindex"), ("read", 0), A_permutation_inverse
                )
                index_map = IndexMap.from_func(ldmatrix_trans_permutation_16x16_32x8_16x16)
                inversed_index_map = index_map.inverse([16, 16])

                def B_permutation_inverse(i, j, kernel_i, kernel_j):
                    return (i, j, *inversed_index_map.map_indices([kernel_i, kernel_j]))

                sch.transform_layout(
                    sch.get_block("B_reindex_reindex"), ("read", 0), B_permutation_inverse
                )

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
            [None, 2],
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
                x_pad_factor,
                y_pad_factor,
                k_pad_factor,
                micro_size_x,
                micro_size_y,
                micro_size_k,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k, i_inner, j_inner, k_inner = sch.get_loops(block)

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
        sch.bind(batch, "blockIdx.z")
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared")
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

        a_g2s = fetch_to_shared(block_outer, 0, 4)
        b_g2s = fetch_to_shared(block_outer, 1, 4)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "warp")
        B_mat = sch.cache_read(block_outer, 1, "warp")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared")
        sch.storage_align(accumulator_shared_to_global, 0, -2, 16, 4)

        store = sch.cache_write(block_outer, 0, "warp")
        sch.reverse_compute_at(store, thread_idy)
        sch.reverse_compute_at(accumulator_shared_to_global, thread_idy)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]

        def index_map(*args):
            other_args = args[:-2]
            inner_i, inner_j = args[-2:]
            return (
                *other_args,
                *shared_16x16_to_mma_32x8_layout(inner_i, inner_j),
            )

        sch.transform_layout(A_mat, ("write", 0), index_map)
        sch.transform_layout(B_mat, ("write", 0), index_map)
        sch.transform_layout(store, ("read", 0), index_map)
        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        intrin_group = {
            "load_a": LDMATRIX_f16_A_INTRIN,
            "load_b": LDMATRIX_f16_B_TRANS_INTRIN,
            "init": MMA_fill_16x16_f16_INTRIN,
            "compute": MMA_f16f16f16_TRANS_B_INTRIN,
            "store": MMA_store_16x16_f16_shared_INTRIN,
        }

        try:
            i, j, ii, jj = sch.get_loops(A_mat)[-4:]
            sch.tensorize(ii, intrin_group["load_a"])

            i, j, ii, jj = sch.get_loops(B_mat)[-4:]
            sch.tensorize(ii, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            return None

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        tensorize_init_store_compute()

        auto_inline_consumer_chain(sch, accumulator_shared_to_global)

        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-4:])
        _, f1, f2 = sch.split(fused, factors=[None, warp_size, vector_size])
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)

        return sch


class MatmulTensorization(GPUScheduleRule):
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
            sch = normalize_with_tensorcore(sch, main_block)

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
        if not isinstance(config.rasterization_plan, NoRasterization) and sch.get(batch).extent.value == 1:
            device_func, invoke_func = config.rasterization_plan.get_code()
            factor = config.rasterization_plan.panel_width_

            # TODO(lei): this is a trick for rasterization implementation
            # require a solution for general block rasterization
            factor = 8 # should be divisible by block_idy
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
            offset:int = 0
            if dtype == "float16":
                offset = 8
            elif dtype == "int8":
                offset = 16
            # todo(lei): the pad value should be varied according to the data type
            sch.storage_align(block_read, 0, axis=-2, factor=16, offset=offset)
            return block_read

        a_g2s = fetch_to_shared(block_outer, 0, 2, vec_len=list(config.vectorize.values())[0], dtype=intrin_info.in_dtype)
        b_g2s = fetch_to_shared(block_outer, 1, 2, vec_len=list(config.vectorize.values())[1], dtype=intrin_info.in_dtype)

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


class Matmul(GPUScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vthread_x: int = 1
        vthread_y: int = 1
        micro_size_x: int = 4
        micro_size_y: int = 4
        micro_size_k: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name == "cuda" or target.kind.name == "rocm":
            return Matmul.Config(
                block_size_x=8,
                block_size_y=16,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=4,
                micro_size_y=4,
                micro_size_k=16,
                vector_size=2,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        elif target.kind.name == "opencl" and "android" in str(target.host):
            return Matmul.Config(
                block_size_x=8,
                block_size_y=8,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=8,
                micro_size_y=2,
                micro_size_k=16,
                vector_size=8,
                unroll=64,
                use_shared=False,
                storage_align=False,
                inner_x=True,
            )
        else:
            return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        sch = normalize_with_tensorcore(sch, main_block)
     
        # Step 1. Check Tensor Core support
        # Tensorization config:
        # If any value of I, J, K is fixed and less than this threshold,
        # tensorization rule will not be applied.
        minimal_tensorize_threshold = 64
        block_stmt = sch.get(main_block)
        if target.kind.name == "cuda" and check_sm_version(target.arch) >= 70:
            apply_tensorization: bool = True
            # the batch dimension is not taken into consideration.
            for item_var in block_stmt.iter_vars[1:]:
                extent = item_var.dom.extent
                if isinstance(extent, tir.expr.IntImm):
                    if extent.value <= minimal_tensorize_threshold:
                        apply_tensorization = False
            if apply_tensorization:
                # Analyze read/write buffers and choose correct tensorizer: int8 or fp16.
                in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
                if in_dtype == "int8" and out_dtype == "int32":
                    tensorize_sch = MatmulInt8Tensorization().apply(func, target, _)
                else:
                    tensorize_sch = MatmulTensorization().apply(func, target, _)
                if tensorize_sch is not None:
                    return tensorize_sch

        # Step 2. Get schedule config.
        config = self.get_configs(target)

        # Step 3. Schedule matmul
        y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y
        x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, config.micro_size_k],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, config.micro_size_k],
            )
            batch, x, y, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_size_y, config.micro_size_y]
        )
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_size_x, config.micro_size_x]
        )
        ko, ki = sch.split(k, factors=[None, config.micro_size_k])
        sch.reorder(by, bx, vy, vx, ty, tx, ko, ki, yi, xi)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        inner_loop = config.micro_size_x if config.inner_x else config.micro_size_y
        if inner_loop % config.vector_size == 0:
            _, v = sch.split(xi, [None, config.vector_size])
            sch.vectorize(v)

        if config.unroll > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=config.unroll)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        if config.micro_size_x % config.vector_size == 0:
            _, v = sch.split(sch.get_loops(l2g)[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.use_shared:

            def _cooperative_fetch(index, vec_len):
                block = sch.cache_read(main_block, index, "shared")
                num_loops = len(sch.get_loops(block))
                sch.compute_at(block, ko, preserve_unit_loops=True)
                loops = sch.get_loops(block)[-num_loops:]
                ty, tx, _, vec = sch.split(
                    sch.fuse(*loops),
                    factors=[config.block_size_y, config.block_size_x, None, vec_len],
                )
                sch.vectorize(vec)
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                if config.storage_align:
                    sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
                return block

            a_g2s = _cooperative_fetch(0, vec_len=config.vector_size)
            b_g2s = _cooperative_fetch(1, vec_len=config.vector_size)

            auto_inline_producers(sch, a_g2s)
            auto_inline_producers(sch, b_g2s)
        else:
            auto_inline_producers(sch, main_block)

        auto_inline_consumer_chain(sch, l2g)

        sch.decompose_reduction(main_block, ko)
        return sch

    def apply_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> tir.Schedule:
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        # in some case conv template will use this rule, but the tile config is not
        # analyzed by matmul expr.
        assert len(config.block) == 2, "Matmul Only support 2D block"
        
        if config.use_tc:
            tensorize_sch = MatmulTensorization().apply_config(func, config)
            if tensorize_sch is not None:
                return tensorize_sch
        
        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)

        # cuda core prefer b is [k, j] layout without swizzling.
        index_maps = get_index_map(block_stmt, ['n', 'n', 'n'])
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Get schedule config.
        block_row_warps = config.block[0] // (config.thread[0] * config.step[0])
        block_col_warps = config.block[1] // (config.thread[1] * config.step[1])
        thread_row_tiles = config.thread[1] // (config.step[0] * 2)
        thread_col_tiles = config.thread[1] // (config.step[1] * 2)
        vthread_row_tiles = config.step[0] * 2 # expand vtrhead to avoid load band conflict
        vthread_col_tiles = config.step[1] * 2 # expand vtrhead to avoid load band conflict
        chunk = config.rstep[0]

        # Step 3. Schedule matmul
        BM = block_row_warps * vthread_row_tiles * thread_row_tiles
        BN = block_col_warps * vthread_col_tiles * thread_col_tiles
        BK = chunk

        sch.pad_einsum(
            main_block,
            [1, BM, BN, BK],
        )
        batch, y, x, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(y, [None, vthread_row_tiles, block_row_warps, thread_row_tiles])
        bx, vx, tx, xi = sch.split(x, [None, vthread_col_tiles, block_col_warps, thread_col_tiles])
        ko, ki = sch.split(k, factors=[None, BK])
        sch.reorder(by, bx, vy, vx, ty, tx, ko, ki, yi, xi)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        def prod(iterable):
            return reduce(lambda x, y: x * y, iterable, 1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)

        def _cooperative_fetch(index, vec_len):
            block = sch.cache_read(main_block, index, "shared")
            num_loops = len(sch.get_loops(block))
            block_local = sch.cache_read(main_block, index, "local")
            sch.compute_at(block_local, ki, preserve_unit_loops=True)
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            _, ty, tx, vec = sch.split(
                sch.fuse(*loops),
                factors=[None, block_row_warps, block_col_warps, vec_len],
            )

            auto_inline_producers(sch, block)

            def is_trivial_load(block):
                # avoid vectorize under global[v2, v1]] shared[v1, v2] case
                reads = sch.get(block).reads
                writes = sch.get(block).writes
                if len(reads) != 1 or len(writes) != 1:
                    return False
                return all(
                    read.region[-1] == write.region[-1] for read, write in zip(reads, writes)
                )

            if is_trivial_load(block):
                sch.vectorize(vec)

            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")

            _, vec = sch.split(
                sch.fuse(*sch.get_loops(block_local)[-2:]),
                [None, vec_len // prod(config.step)],
            )
            sch.vectorize(vec)

            return block

        for i, input_region in enumerate(sch.get(main_block).reads):
            _buffer_name = input_region.buffer.name.replace("_reindex", "").replace("_pad", "")
            if _buffer_name not in config.cached_tensors:
                print(
                    f"Warning: {_buffer_name} is not in cached_tensors {config.cached_tensors}, skip."
                )
                continue

            # otherwise cooperative fetch in shared memory.
            if _buffer_name in config.vectorize:
                vectorize = config.vectorize[_buffer_name]
            else:
                vectorize = 1

            _cooperative_fetch(i, vec_len=vectorize)

        auto_inline_consumer_chain(sch, l2g)

        _, vec = sch.split(
            sch.fuse(*sch.get_loops(l2g)[-2:]), [None, vectorize // prod(config.step)]
        )
        sch.vectorize(vec)

        sch.decompose_reduction(main_block, ko)
        return sch
