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
from typing import Literal, Optional

from tvm import tir
from tvm.target import Target

from ..base import ScheduleRule, analysis
from .matmul_analysis import (
    auto_inline_consumer_chain,
    auto_inline_producers,
    get_dequantize_block,
    get_index_map,
    get_reduction_blocks,
    inline_transpose_block,
    is_identity_block,
    is_transpose_block,
)


class MatmulTensorizationMMA(ScheduleRule):
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
            shared_16x16_to_ldmatrix_32x8_layout,
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
        # swizzle_factor_for_l2_m = [4, None]
        # swizzle_factor_for_l2_n = [4, None]

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

        sch.reorder(i0, j0, i1, j1, k0, i2, j2, k1, i3, j3)

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
            sch.annotate(block_read_smem, ann_key="permuted_layout", ann_val=f"g2s_{tensor_name}")

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
                    *shared_16x16_to_ldmatrix_32x8_layout(v1 % micro_size_1, v2 % micro_size_2),
                ),
            )

            # swizzling
            mma_read_block = sch.blockize(sch.get_loops(block_read_reg)[-2])
            sch.annotate(mma_read_block, ann_key="permuted_layout", ann_val=f"s2l_{tensor_name}")

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
            f0, f1, f2, f3, f4 = sch.split(fused, [None, thread_z, thread_y, thread_x, vector_size])
            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)

            # swizzling
            sch.annotate(block_write_smem, ann_key="permuted_layout", ann_val=f"s2g_C")

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
                    *shared_16x16_to_ldmatrix_32x8_layout(v1 % micro_size_m, v2 % micro_size_n),
                ),
            )

            # swizzling
            mma_read_block = sch.blockize(sch.get_loops(block_write_reg)[-2])
            sch.annotate(mma_read_block, ann_key="permuted_layout", ann_val=f"l2s_C")

            return block_write_smem, block_write_reg

        block_write_smem, block_write_reg = store_output(block_outer, 0)

        # Step 5. Schedule tensor core computation
        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # unroll k
        # Profiling result shows unrolling k0 is not helpful on A100
        # sch.unroll(k0)
        # k00, k01 = sch.split(k0, factors=[None, 8])
        # sch.unroll(k01)

        intrin_group = get_mma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype=str(dtype_a),
            out_dtype=str(dtype_c),
            trans_a=is_transpose_a,
            trans_b=is_transpose_b,
        )

        sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
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
