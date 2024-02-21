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
from typing import Literal, Optional, List

from tvm import tir
from tvm.target import Target

from ..base.roller.rasterization import NoRasterization
from ..base import analysis
from .base import GPUScheduleRule
from .matmul_mma_dequantize import MatmulTensorizationMMAWithDequantizeInfo
from .matmul_analysis import (
    auto_inline_consumer_chain,
    is_transpose_block,
    is_identity_block,
    _collect_producers,
    inline_transpose_block,
    auto_inline_producers,
    get_index_map,
    get_reduction_blocks,
    get_dequantize_block,
    normalize_to_matmul,
    get_propagate_map,
)


def get_index_map_3d(index_map, l=16, r=16):
    def index_map_3d(b, i, j):
        return (
            b,
            i // l,
            j // r,
            *index_map(i % l, j % r),
        )

    return index_map_3d


def get_index_map_5d(index_map):
    """
    for layout transformed gemm, the index map should be 5d
    """

    def index_map_5d(b, i, j, ii, jj):
        return (
            b,
            i,
            j,
            *index_map(ii, jj),
        )

    return index_map_5d


def get_warp_index_map(index_map, l=16, r=16, is_5d=False):
    if is_5d:
        return get_index_map_5d(index_map)
    return get_index_map_3d(index_map, l, r)


class MatmulTensorizationMMA(GPUScheduleRule):
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
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        # We first inline all transpose blocks for later analysis of transposed A and B
        blocks = inline_transpose_block(sch, blocks)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        dequantize_block = get_dequantize_block(sch, blocks)

        main_block = reduction_blocks[0]
        main_block_stmt = sch.get(main_block)

        # Supported data types:
        # fp16, fp16, fp16: fp16 precision
        # fp16, fp16, fp32: fp16 mixed precision
        dtype_a = main_block_stmt.reads[0].buffer.dtype
        dtype_b = main_block_stmt.reads[1].buffer.dtype
        dtype_c = main_block_stmt.writes[0].buffer.dtype
        if dtype_a != dtype_b:
            return None

        # Get index maps
        index_maps = get_index_map(main_block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # Tensorization by hardware intrinsics
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_mma_intrin_group,
            shared_16x16_to_mma_32x8_layout,
        )

        # tile size
        block_m, block_n, block_k = 128, 128, 32

        # tensor core intrinsic size
        micro_size_m, micro_size_n, micro_size_k = 16, 16, 16

        # thread size
        # thread_x == warp_size
        thread_z, thread_y, thread_x = 2, 2, 32

        vector_size = 8
        unroll_depth = 4

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        is_transpose_a = is_transpose_block(sch.get(block))
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        is_transpose_b = is_identity_block(sch.get(block))
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        batch, i, j, k = sch.get_loops(main_block)

        swizzle_factor_for_l2_m = [1, None]
        swizzle_factor_for_l2_n = [1, None]

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                swizzle_factor_for_l2_m[0] * block_m,
                swizzle_factor_for_l2_n[0] * block_n,
                block_k,
            ],
        )

        # Step 3. Reorder loops for tiling

        # Step 3.1 inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_m])
        j, j_inner = sch.split(j, factors=[None, micro_size_n])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = main_block
        block_outer = sch.blockize(i_inner)

        # Step 3.2 outer loops for tiling
        # split factors for i, j, and k
        micro_block_cnt_in_warp_m = block_m // thread_z // micro_size_m
        micro_block_cnt_in_warp_n = block_n // thread_y // micro_size_n
        micro_block_cnt_in_warp_k = block_k // micro_size_k

        i_factors = swizzle_factor_for_l2_m + [thread_z, micro_block_cnt_in_warp_m]
        j_factors = swizzle_factor_for_l2_n + [thread_y, micro_block_cnt_in_warp_n]
        k_factors = [None, micro_block_cnt_in_warp_k]

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, factors=k_factors)

        sch.reorder(i0, j0, i1, j1, i2, j2, k0, k1, i3, j3)

        block_axis = sch.fuse(batch, i0, j0, i1, j1)
        sch.bind(block_axis, "blockIdx.x")

        sch.bind(i2, "threadIdx.z")
        sch.bind(j2, "threadIdx.y")

        # Step 4. Read/write to shared mem and register
        def fetch_input(block_outer, read_buffer_idx, tensor_name: Literal["A", "B"], is_transpose):
            # 1) Read to shared memory
            block_read_smem = sch.cache_read(block_outer, read_buffer_idx, "shared.dyn")
            sch.compute_at(block_read_smem, k0)
            auto_inline_producers(
                sch, block_read_smem, [dequantize_block] if dequantize_block else []
            )

            # For transposed read, we directly load transposed tensor from global
            # Then use ldmatrix.trans to handle transpose later
            if (tensor_name == "A" and is_transpose) or (tensor_name == "B" and not is_transpose):
                # specifical handle transpose read (for NN matmul or TT matmul)
                v0, v1 = sch.get_loops(block_read_smem)[-2:]
                sch.reorder(v1, v0)
                sch.transform_layout(block_read_smem, ("write", 0), lambda b, i, j: (b, j, i))

            # bind loops
            fused = sch.fuse(*sch.get_loops(block_read_smem)[-2:])
            f0, f1, f2, f3, f4 = sch.split(fused, [None, thread_z, thread_y, thread_x, vector_size])
            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)

            # swizzling
            sch.annotate(block_read_smem, ann_key="permuted_layout", ann_val=1)

            # 2) Read to register
            block_read_reg = sch.cache_read(block_outer, read_buffer_idx, "warp")
            sch.compute_at(block_read_reg, k1)

            # bind_loops
            micro_size_spatial = micro_size_m if tensor_name == "A" else micro_size_n
            micro_size_1, micro_size_2 = (
                (micro_size_spatial, micro_size_k)
                if not is_transpose
                else (micro_size_k, micro_size_spatial)
            )
            v00, v01 = sch.split(sch.get_loops(block_read_reg)[-2], [None, micro_size_1])
            v10, v11 = sch.split(sch.get_loops(block_read_reg)[-1], [None, micro_size_2])
            sch.reorder(v00, v10, v01, v11)

            # reorder read axis to match the layout of ldmatrix
            sch.transform_layout(
                block_read_reg,
                ("write", 0),
                lambda v0, v1, v2: (
                    v0,
                    v1 // micro_size_1,
                    v2 // micro_size_2,
                    *shared_16x16_to_mma_32x8_layout(v1 % micro_size_1, v2 % micro_size_2),
                ),
            )

            # swizzling
            mma_read_block = sch.blockize(sch.get_loops(block_read_reg)[-2])
            sch.annotate(mma_read_block, ann_key="permuted_layout", ann_val=1)

            return block_read_smem, block_read_reg

        block_read_a, block_read_reg_a = fetch_input(block_outer, 0, "A", is_transpose_a)
        block_read_b, block_read_reg_b = fetch_input(block_outer, 1, "B", is_transpose_b)

        # Write to register, and then smem
        def store_output(block_outer, write_buffer_idx):
            # 1) Write to shared memory
            block_write_smem = sch.cache_write(block_outer, write_buffer_idx, "shared.dyn")
            sch.reverse_compute_at(block_write_smem, block_axis)
            auto_inline_consumer_chain(sch, block_write_smem)

            # bind loops
            fused = sch.fuse(*sch.get_loops(block_write_smem)[-2:])
            f0, f1, f2 = sch.split(fused, [None, thread_x, vector_size])
            sch.bind(f1, "threadIdx.x")
            sch.vectorize(f2)

            # 2) Write to register
            block_write_reg = sch.cache_write(block_outer, write_buffer_idx, "warp")

            # bind loops
            v0, v1, v2 = sch.get_loops(block_write_reg)[-3:]
            v11, v12, v13 = sch.split(v1, factors=[thread_z, None, micro_size_m])
            v21, v22, v23 = sch.split(v2, factors=[thread_y, None, micro_size_n])
            sch.reorder(v11, v21, v12, v22, v13, v23)
            sch.bind(v11, "threadIdx.z")
            sch.bind(v21, "threadIdx.y")

            # reorder write axis to match the layout of ldmatrix
            sch.transform_layout(
                block_write_reg,
                ("read", 0),
                lambda v0, v1, v2: (
                    v0,
                    v1 // micro_size_m,
                    v2 // micro_size_n,
                    *shared_16x16_to_mma_32x8_layout(v1 % micro_size_m, v2 % micro_size_n),
                ),
            )

            return block_write_smem, block_write_reg

        block_write_smem, block_write_reg = store_output(block_outer, 0)

        # Step 5. Schedule tensor core computation
        block_init = sch.decompose_reduction(block_outer, k0)
        block_init_inner = sch.get_child_blocks(block_init)[0]

        intrin_group = get_mma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype=str(dtype_a),
            out_dtype=str(dtype_c),
            trans_a=is_transpose_a,
            trans_b=is_transpose_b,
            not_use_mma_store_intrinic=False,
        )

        sch.tensorize(sch.get_loops(block_init_inner)[-2], intrin_group["init"])
        sch.tensorize(sch.get_loops(block_read_reg_a)[-2], intrin_group["load_a"])
        sch.tensorize(sch.get_loops(block_read_reg_b)[-2], intrin_group["load_b"])
        sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])
        sch.tensorize(sch.get_loops(block_write_reg)[-2], intrin_group["store"])

        # Step 6. Async pipeline
        sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
        sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
        sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

        # Step 7. Handle dequantize block
        # Now we just add a dummy kernel to compute dequantize
        if dequantize_block is not None:
            auto_inline_producers(sch, dequantize_block)
            loops = sch.get_loops(dequantize_block)
            loop = sch.fuse(*loops)
            v0, v1, v2, v3 = sch.split(loop, [None, 128, 2, 4])
            sch.bind(v0, "blockIdx.x")
            sch.bind(v1, "threadIdx.x")
            sch.unroll(v2)
            sch.vectorize(v3)
        return sch

    def apply_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> Optional[tir.Schedule]:
        if "dequantize_info" in func.attrs:
            dequantize_rule = MatmulTensorizationMMAWithDequantizeInfo()
            return dequantize_rule.apply_config(func, config)

        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_mma_intrin_group,
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
        output_blocks = [sch.get(block) for block in sch.get_output_blocks(root_block)]
        def check_require_cache(func:tir.PrimFunc):
            conditions:List[bool] = []
            # check if has dynamic symbolic
            def check_has_dynamic(func:tir.PrimFunc):
                for param in func.params:
                    if param not in func.buffer_map:
                        continue
                    arg = func.buffer_map[param]
                    for i in arg.shape:
                        if isinstance(i, tir.Var):
                            return True
                return False
            conditions.append(check_has_dynamic(func))
            # check if has post process
            conditions.append(sch.get(main_block) not in output_blocks)
            return any(conditions)
        cache_write_required = check_require_cache(func)

        shared_scope = "shared"

        intrin_info = config.intrin_info
        intrin_group = get_mma_intrin_group(
            load_scope=shared_scope,
            store_scope=shared_scope if cache_write_required else "global",
            in_dtype=intrin_info.in_dtype,
            out_dtype=intrin_info.out_dtype,
            trans_a=intrin_info.trans_a,
            trans_b=intrin_info.trans_b,
            smooth_a=intrin_info.smooth_a,
            smooth_b=intrin_info.smooth_b,
            not_use_mma_store_intrinic=False,
        )

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        warp_row_tiles = config.warp[0]
        warp_col_tiles = config.warp[1]
        block_row_warps = config.block[0] // warp_row_tiles
        block_col_warps = config.block[1] // warp_col_tiles
        stage = config.pipeline_stage
        use_async = config.use_async
        chunk = config.rstep[0]
        
        micro_size_x, micro_size_y, micro_size_k = intrin_group["micro_kernel"]

        # get the axis for layout transform
        def get_axis(l, r, trans):
            return (r, l) if trans else (l, r)

        a_lr = get_axis(micro_size_x, micro_size_k, intrin_info.trans_a)
        b_lr = get_axis(micro_size_k, micro_size_y, intrin_info.trans_b)

        def can_enable_swizzle(dtype: str, smooth: bool):
            # inject_permuted_layout only support float16 currently
            if dtype == "float16":
                # if we use smooth layout, we don't need to do swizzling
                return not smooth
            return False

        can_swizzle_a = can_enable_swizzle(intrin_info.in_dtype, intrin_info.smooth_a)
        can_swizzle_b = can_enable_swizzle(intrin_info.in_dtype, intrin_info.smooth_b)

        warp_size = 32

        i_factors, j_factors, k_factors = (
            [None, 1, block_row_warps, warp_row_tiles // micro_size_x],
            [1, None, block_col_warps, warp_col_tiles // micro_size_y],
            [None, chunk // micro_size_k],
        )

        num_ty = i_factors[2]
        num_tz = j_factors[2]
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

        sch.reorder(i0, j0, i1, j1, i2, j2, k0, k1, i3, j3)

        block_idy = sch.fuse(i0, j0)
        block_idx = sch.fuse(i1, j1)
        thread_idy = i2
        thread_idz = j2

        # plan rasteration
        if (
            not isinstance(config.rasterization_plan, NoRasterization)
            and sch.get(batch).extent.value == 1
        ):
            device_func, invoke_func = config.rasterization_plan.get_code()
            factor = config.rasterization_plan.panel_width_

            # TODO(lei): this is a trick for rasterization implementation
            # is not optimal. (5% performance loss)
            # require a solution for general block rasterization
            factor = 4  # should be divisible by block_idx
            if sch.get(block_idx).extent.value % factor == 0:
                block_k, block_idx = sch.split(block_idx, factors=[None, factor])
                sch.bind(block_k, "blockIdx.z")
        else:
            sch.bind(batch, "blockIdx.z")

        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")
        sch.bind(thread_idz, "threadIdx.z")

        # rewrite smooth layout of shared memory
        def smooth_smem_layout_rewrite(block, scope, l=16, r=16, enable=True):
            if not enable:
                return
            sch.transform_layout(
                block,
                scope,
                lambda b, i, j: (
                    b,
                    i // l,
                    j // r,
                    i % l,
                    j % r,
                ),
            )

        smooth_smem_layout_rewrite(block_outer, ("read", 0), *a_lr, enable=intrin_info.smooth_a)
        smooth_smem_layout_rewrite(block_outer, ("read", 1), *b_lr, enable=intrin_info.smooth_b)
        smooth_smem_layout_rewrite(block_outer, ("write", 0), enable=True)

        def fetch_to_shared(block, idx, vec_len, can_swizzle=False, is_smooth=False, trans=False):
            block_read = sch.cache_read(block, idx, shared_scope)
            sch.compute_at(block_read, k0, preserve_unit_loops=True)
            ndim = len(sch.get(block_read).iter_vars)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            f_0, f_1, f_2, f_3, f_4 = sch.split(
                fused, factors=[num_ty, num_tz, None, warp_size, vec_len]
            )

            sch.bind(f_3, "threadIdx.x")
            sch.bind(f_1, "threadIdx.z")
            sch.bind(f_0, "threadIdx.y")
            sch.vectorize(f_4)
            sch.unroll(f_2)
            # Apply Swizzling
            sch.annotate(block_read, ann_key="permuted_layout", ann_val=can_swizzle)
            # if not, apply padding to alleviate bank conflict
            if not (can_swizzle or is_smooth):
                pad_offset = 8 if intrin_info.in_dtype == "float16" else 16
                sch.storage_align(block_read, 0, axis=-2, factor=16, offset=pad_offset)
            sch.annotate(f_2, "pragma_unroll_explicit", False)
            return block_read

        a_g2s = fetch_to_shared(
            block_outer,
            0,
            vec_len=list(config.vectorize.values())[0],
            can_swizzle=can_swizzle_a,
            is_smooth=intrin_info.smooth_a,
            trans=intrin_info.trans_a,
        )
        b_g2s = fetch_to_shared(
            block_outer,
            1,
            vec_len=list(config.vectorize.values())[1],
            can_swizzle=can_swizzle_b,
            is_smooth=intrin_info.smooth_b,
            trans=intrin_info.trans_b,
        )

        # rewrite global smooth layout
        def smooth_gmem_layout_rewrite(sch, block, enable=True, trans=False):
            if not enable:
                return
            # step1: find the first producer block
            # Notes: we assume the layout propagate happens in the first producer block
            # otherwise, the layout transform will have no effect as it will transform both
            # read and write buffer
            producers = _collect_producers(sch, block)

            propagate_block: tir.Block = producers[-1]

            # step2: transform the layout with inverse permutation
            _, inverse_indexmap = get_propagate_map(trans=trans, dtype=intrin_info.in_dtype)

            def inverse_permutation(i, j, ii, jj):
                return (i, j, *inverse_indexmap.map_indices([ii, jj]))

            sch.transform_layout(propagate_block, ("read", 0), inverse_permutation)

        smooth_gmem_layout_rewrite(sch, a_g2s, intrin_info.smooth_a, intrin_info.trans_a)
        smooth_gmem_layout_rewrite(sch, b_g2s, intrin_info.smooth_b, intrin_info.trans_b)
        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "warp")
        B_mat = sch.cache_read(block_outer, 1, "warp")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        if cache_write_required:
            accumulator_shared_to_global = sch.cache_write(block_outer, 0, shared_scope)

        store = sch.cache_write(block_outer, 0, "warp")
        sch.reverse_compute_at(store, j2)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]
        i0, i1 = sch.split(i, factors=[None, micro_size_x])
        j0, j1 = sch.split(j, factors=[None, micro_size_y])
        sch.reorder(i0, j0, i1, j1)

        if cache_write_required:
            auto_inline_consumer_chain(sch, accumulator_shared_to_global)
            sch.reverse_compute_at(
                accumulator_shared_to_global, sch.get_loops(store)[-3], preserve_unit_loops=True
            )

            fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-5:])
            f0, f1, f2 = sch.split(
                fused, factors=[None, warp_size, max(list(config.vectorize.values()))]
            )
            sch.bind(f1, "threadIdx.x")
            sch.vectorize(f2)
            sch.unroll(f0)
            sch.annotate(f0, "pragma_unroll_explicit", False)
        else:
            auto_inline_consumer_chain(sch, store)

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        index_map_a, index_map_b, index_map_c = intrin_group["index_map"]

        sch.transform_layout(
            A_mat, ("write", 0), get_warp_index_map(index_map_a, *a_lr, intrin_info.smooth_a)
        )
        sch.transform_layout(
            B_mat, ("write", 0), get_warp_index_map(index_map_b, *b_lr, intrin_info.smooth_b)
        )
        sch.transform_layout(
            store,
            ("read", 0),
            get_warp_index_map(index_map_c, is_5d=True),
        )

        i, j = sch.get_loops(A_mat)[-2:]
        i0, i1 = sch.split(i, factors=[None, a_lr[0]])
        j0, j1 = sch.split(j, factors=[None, a_lr[1]])
        sch.reorder(i0, j0, i1, j1)
        ba = sch.blockize(i1)
        sch.annotate(ba, ann_key="permuted_layout", ann_val=can_swizzle_a)
        sch.tensorize(ba, intrin_group["load_a"])

        i, j = sch.get_loops(B_mat)[-2:]
        i0, i1 = sch.split(i, factors=[None, b_lr[0]])
        j0, j1 = sch.split(j, factors=[None, b_lr[1]])
        sch.reorder(i0, j0, i1, j1)
        bb = sch.blockize(i1)
        sch.annotate(bb, ann_key="permuted_layout", ann_val=can_swizzle_b)
        sch.tensorize(bb, intrin_group["load_b"])

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        tensorize_init_store_compute()

        if stage > 1:
            sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
            sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
        if use_async:
            sch.annotate(k0, "software_pipeline_async_stages", [0])

        return sch
