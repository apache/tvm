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

from ..base.roller.rasterization import NoRasterization
from ..base import analysis
from .base import GPUScheduleRule
from .matmul_analysis import (
    auto_inline_consumer_chain,
    auto_inline_producers,
    get_reduction_blocks,
    get_dequantize_block,
    normalize_to_matmul,
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


def get_index_map(index_map, l=16, r=16, is_5d=False):
    if is_5d:
        return get_index_map_5d(index_map)
    return get_index_map_3d(index_map, l, r)


class MatmulTensorizationMMAWithDequantizeInfo(GPUScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def sch_dequantize_in_register_with_config(
        self,
        func: tir.PrimFunc,
        config,
    ):
        """
        Simple dequantize schedule without shared memory prefetch.
            quantized weight
                |
                V
            dequantized in register
                |
                V
            save into shared memory
                |
                V
            compute
        """

        return None

    def sch_shared_memory_prefetch_with_config(
        self,
        func: tir.PrimFunc,
        config,
    ):
        """
        For A100 Like devices, the shared memory prefetch(async) is required
        to achieve optimal performance.
            quantized weight
                |
                V
            shared memory prefetch (with async copy)
                |
                V
            dequantized into shared memory
                |
                V
            compute
        """
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_mma_intrin_group,
        )
        from .intrin.lop3 import get_lop3_intrin_group

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        # always enable shared memory rewrite
        cache_write_required = True

        # Check Dequantize Info
        # TODO(leiwang): this is a hack to get the configuaration, can be improved by writing a pass to analysis the dequantize block.
        dequantize_info = func.attrs["dequantize_info"]

        def check_dequantize_info(dequantize_info):
            conditions = []
            # currently only support weight only dequantization
            conditions.append(len(dequantize_info) == 1)
            # TODO(@lei) check if the dequantize value name is weight
            return all(conditions)

        assert check_dequantize_info(dequantize_info)

        (B_decode_info,) = list(dequantize_info.values())

        def check_b_decode_info(B_decode_info):
            conditions = []
            # check source format in ["int", "fp", "af"]
            conditions.append("source_format" in B_decode_info)
            conditions.append(B_decode_info["source_format"]["format"] in ["int", "fp", "af"])
            # check source bits in [1, 2, 4, 8]
            conditions.append(B_decode_info["source_format"]["bits"] in [1, 2, 4, 8])
            # check target format in ["float16", "int8"]
            conditions.append("target_format" in B_decode_info)
            conditions.append(B_decode_info["target_format"] in ["float16", "int8"])
            return all(conditions)

        assert check_b_decode_info(B_decode_info)

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        intrin_info = config.intrin_info
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
            # is not optimal.
            # require a solution for general block rasterization
            factor = 8  # should be divisible by block_idy
            if sch.get(block_idx).extent.value % factor == 0:
                block_k, block_idx = sch.split(block_idx, factors=[None, factor])
                sch.bind(block_k, "blockIdx.z")
        else:
            sch.bind(batch, "blockIdx.z")

        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")
        sch.bind(thread_idz, "threadIdx.z")

        def smooth_layout_recover(block, scope, l=16, r=16, enable=True):
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

        smooth_layout_recover(block_outer, ("read", 0), *a_lr, enable=intrin_info.smooth_a)
        smooth_layout_recover(
            block_outer,
            ("read", 1),
            *b_lr,
            enable=intrin_info.smooth_b,
        )
        smooth_layout_recover(block_outer, ("write", 0), enable=True)

        def fetch_to_shared(block, idx, vec_len, can_swizzle=False, is_smooth=False):
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
        )

        auto_inline_producers(sch, a_g2s)

        def decode_fetch_to_shared(block, idx):
            # step1. create memory hierarchy
            # global -> local -> shared
            block_shared = sch.cache_read(block, idx, shared_scope)
            sch.compute_at(block_shared, k0, preserve_unit_loops=True)

            # TODO(lei): the factor shoule be analyzed more deeper.
            _, B_shared_vi, _ = sch.split(sch.get_loops(block_shared)[-1], factors=[None, 1, 8])
            block_shared_local = sch.cache_read(block_shared, 0, "local")
            # global -> dequantzed_local -> shared
            # step2. inline to local block
            auto_inline_producers(sch, block_shared_local)

            # get target dequantize buffer's idx
            def get_idx():
                # for LUT dequantize, the expr is LUT(w), the idx is 1
                # maybe we can use a more general and structual based way
                # to analysis the idx
                if B_decode_info["source_format"]["format"] == "af":
                    return 1
                return 0

            b_idx = get_idx()
            # global -> prefetch_local -> dequantzed_local -> shared
            block_shared_local_local = sch.cache_read(block_shared_local, b_idx, "local")
            # global -> prefetch_shared -> vector load -> dequantzed_local -> shared
            block_shared_local_local_shared = sch.cache_read(
                block_shared_local_local, 0, shared_scope
            )
            sch.compute_at(block_shared_local, B_shared_vi, preserve_unit_loops=True)
            sch.compute_at(block_shared_local_local, B_shared_vi, preserve_unit_loops=True)

            dequantize_block = block_shared_local
            # fast type conversion
            if "fast_decoding" in B_decode_info and B_decode_info["fast_decoding"]:
                intrin_group = get_lop3_intrin_group(
                    in_dtype="int8", out_dtype="float16", storage_nbit=4, with_scale=False
                )
                sch.tensorize(sch.get_loops(dequantize_block)[-1], intrin_group["compute"])
                sch.annotate(
                    thread_idz, ann_key="pragma_import_c", ann_val=intrin_group["c_source"]
                )

            sch.annotate(block_shared, ann_key="permuted_layout", ann_val=can_swizzle_b)
            union_len = (2 + 4) if intrin_info.smooth_b else (2 + 2)
            B_shared_fused = sch.fuse(*sch.get_loops(block_shared)[-union_len:-2])
            _, B_shared_ty, B_shared_tz, B_shared_tx = sch.split(
                B_shared_fused, factors=[None, num_ty, num_tz, warp_size]
            )
            if not (can_swizzle_b or intrin_info.smooth_b):
                pad_offset = 8 if intrin_info.in_dtype == "float16" else 16
                sch.storage_align(block_shared, 0, axis=-2, factor=16, offset=pad_offset)
            sch.bind(B_shared_tx, "threadIdx.x")
            sch.bind(B_shared_ty, "threadIdx.y")
            sch.bind(B_shared_tz, "threadIdx.z")
            sch.vectorize(sch.get_loops(block_shared)[-1])
            sch.vectorize(sch.get_loops(block_shared_local_local)[-1])

            sch.compute_at(block_shared_local_local_shared, k0, preserve_unit_loops=True)
            ndim = len(sch.get(block_shared_local_local_shared).iter_vars)
            fused = sch.fuse(*sch.get_loops(block_shared_local_local_shared)[-ndim:])

            f_0, f_1, f_2, f_3, f_4 = sch.split(
                fused, factors=[None, num_tz, num_ty, warp_size, 16]  # int8x16 = 128bits
            )

            sch.bind(f_3, "threadIdx.x")
            sch.bind(f_2, "threadIdx.y")
            sch.bind(f_1, "threadIdx.z")
            sch.vectorize(f_4)
            sch.unroll(f_0)
            sch.annotate(f_0, "pragma_unroll_explicit", False)

            # cache small tensors, e.g. LUT
            if b_idx:
                block_shared_lut = sch.cache_read(dequantize_block, 0, shared_scope)
                sch.reverse_compute_at(block_shared_lut, j2)
                _, B_shared_tx = sch.split(
                    sch.get_loops(block_shared_lut)[-1], factors=[None, warp_size]
                )
                sch.bind(B_shared_tx, "threadIdx.x")
            return block_shared_local

        dequantize_block = decode_fetch_to_shared(block_outer, 1)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "warp")
        B_mat = sch.cache_read(block_outer, 1, "warp")
        sch.compute_at(A_mat, k1, preserve_unit_loops=True)
        sch.compute_at(B_mat, k1, preserve_unit_loops=True)

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
            A_mat, ("write", 0), get_index_map(index_map_a, *a_lr, intrin_info.smooth_a)
        )
        sch.transform_layout(
            B_mat, ("write", 0), get_index_map(index_map_b, *b_lr, intrin_info.smooth_b)
        )
        sch.transform_layout(
            store,
            ("read", 0),
            get_index_map(index_map_c, is_5d=True),
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
            sch.annotate(
                k0, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1, stage - 1]
            )
            sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2, 3])
        if use_async:
            sch.annotate(k0, "software_pipeline_async_stages", [0])
        return sch

    def apply_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> Optional[tir.Schedule]:
        def check_sm_version(arch: str) -> int:
            sm_version = arch.replace("sm_", "")
            return int(sm_version) if sm_version.isdigit() else -1

        if check_sm_version(config.arch.target.arch) < 80:
            """MMA Template only support sm_80 and above"""
            return None

        if (
            config.arch.target.kind.name == "cuda"
            and check_sm_version(config.arch.target.arch) == 80
        ):
            return self.sch_shared_memory_prefetch_with_config(func, config)
        else:
            return self.sch_with_config(func, config)
