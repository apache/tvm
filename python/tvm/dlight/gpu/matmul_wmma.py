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
from typing import Optional

from tvm import DataType, tir
from tvm.target import Target
from tvm.tir.stmt import ForKind

from ..base import ScheduleRule, analysis
from .matmul_analysis import (
    auto_inline_consumer_chain,
    auto_inline_consumers,
    auto_inline_producers,
    get_index_map,
    get_reduction_blocks,
)


class MatmulTensorizationWMMA(ScheduleRule):
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

        block_m = 128
        block_n = 128
        block_k = 32

        # tensor core intrinsic size
        micro_size_m = 16
        micro_size_n = 16
        micro_size_k = 16

        thread_z = 2
        thread_y = 2
        warp_size = 32
        thread_cnt = thread_y * thread_z * warp_size

        vector_size = 8
        unroll_depth = 256

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        batch, i, j, k = sch.get_loops(main_block)
        main_block_stmt = sch.get(main_block)
        buffer_regions = list(main_block_stmt.reads) + list(main_block_stmt.writes)

        # Supported data types:
        # fp16, fp16, fp16: fp16 precision
        # fp16, fp16, fp32: fp16 mixed precision
        dtype_a, dtype_b, dtype_c = [DataType(region.buffer.dtype) for region in buffer_regions]
        input_b, input_m, input_n, input_k = [sch.get(loop).extent for loop in [batch, i, j, k]]
        l2_size = target.l2_cache_size_bytes
        dtype_a_bytes, dtype_b_bytes, dtype_c_bytes = [
            math.ceil(d.bits / 8) for d in [dtype_a, dtype_b, dtype_c]
        ]

        def get_z_order_factor(l2_size, input_k, dtype_bytes, input_spatial, block_size):
            if l2_size != 0 and isinstance(input_k, (int, tir.IntImm)):
                z_order_factor = l2_size / 3 / int(input_k) / dtype_bytes / block_size
                if isinstance(input_spatial, (int, tir.IntImm)):
                    block_cnt = math.ceil(int(input_spatial) / block_size)
                    z_order_factor = math.ceil(block_cnt / math.ceil(block_cnt / z_order_factor))
                else:
                    z_order_factor = math.floor(z_order_factor)
                return [None, z_order_factor]
            else:
                return [4, None]

        z_order_factor_m = get_z_order_factor(l2_size, input_k, dtype_a_bytes, input_m, block_m)
        z_order_factor_n = get_z_order_factor(l2_size, input_k, dtype_b_bytes, input_n, block_n)

        z_order_factor_m = [1, None]
        z_order_factor_n = [1, None]

        print(f"z_order_factor_m={z_order_factor_m}, z_order_factor_n={z_order_factor_n}")

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                (z_order_factor_m[0] or z_order_factor_m[1]) * block_m,
                (z_order_factor_n[0] or z_order_factor_n[1]) * block_n,
                block_k,
            ],
        )

        # Step 3. Schedule matmul to use tensor core

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_m])
        j, j_inner = sch.split(j, factors=[None, micro_size_n])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = main_block
        block_outer = sch.blockize(i_inner)

        # split factors for i, j, and k
        in_wrap_block_cnt_m = block_m // thread_z // micro_size_m
        in_wrap_block_cnt_n = block_n // thread_y // micro_size_n
        in_wrap_block_cnt_k = block_k // micro_size_k

        i_factors = z_order_factor_m + [thread_z, in_wrap_block_cnt_m]
        j_factors = z_order_factor_n + [thread_y, in_wrap_block_cnt_n]
        k_factors = [None, in_wrap_block_cnt_k]

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, factors=k_factors)

        sch.reorder(i0, j0, i1, j1, k0, i2, j2, k1, i3, j3)
        block_axis = sch.fuse(batch, i0, j0, i1, j1)

        sch.bind(block_axis, "blockIdx.x")
        sch.bind(i2, "threadIdx.z")
        sch.bind(j2, "threadIdx.y")

        def fetch_input(block_outer, read_buffer_idx, read_loop_ndim, block_sizes, wmma_name):
            block_read = sch.cache_read(block_outer, read_buffer_idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-read_loop_ndim:])

            f0, f1, f2, f3, f4 = sch.split(
                fused, [None, thread_z, thread_y, warp_size, vector_size]
            )

            block_m, block_k, micro_size_m, micro_size_k = block_sizes

            sch.transform_layout(
                block_read,
                ("write", 0),
                lambda v0, v1, v2: (
                    v1 // block_m,
                    v2 // block_k,
                    v1 % block_m // micro_size_m,
                    v2 % block_k // micro_size_k,
                    v1 % micro_size_m,
                    v2 % micro_size_k,
                ),
            )

            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)
            # sch.storage_align(block_read, 0, axis=-2, factor=16, offset=8)

            auto_inline_producers(sch, block_read)

            wmma_read = sch.cache_read(block_outer, read_buffer_idx, wmma_name)
            sch.compute_at(wmma_read, k1)
            return wmma_read

        wmma_read_a = fetch_input(
            block_outer, 0, 2, [block_m, block_k, micro_size_m, micro_size_k], "wmma.matrix_a"
        )
        wmma_read_b = fetch_input(
            block_outer, 1, 2, [block_n, block_k, micro_size_n, micro_size_k], "wmma.matrix_b"
        )

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        def store_output(block_outer, write_buffer_idx, write_loop_ndim, block_sizes, wmma_name):
            block_write = sch.cache_write(block_outer, write_buffer_idx, "shared.dyn")
            sch.reverse_compute_at(block_write, block_axis)

            fused = sch.fuse(*sch.get_loops(block_write)[-write_loop_ndim:])

            f0, f1, f2, f3, f4 = sch.split(
                fused, [None, thread_z, thread_y, warp_size, vector_size]
            )

            block_m, block_n, micro_size_m, micro_size_n = block_sizes
            sch.transform_layout(
                block_write,
                ("read", 0),
                lambda v0, v1, v2: (
                    v1 // block_m,
                    v2 // block_n,
                    v1 % block_m // micro_size_m,
                    v2 % block_n // micro_size_n,
                    v1 % micro_size_m,
                    v2 % micro_size_n,
                ),
            )

            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)
            auto_inline_consumers(sch, block_write)

            store = sch.cache_write(block_outer, write_buffer_idx, wmma_name)
            v0, v1, v2, v3, v4, v5 = sch.get_loops(store)[-6:]
            v21, v22 = sch.split(v2, factors=[thread_z, None])
            v31, v32 = sch.split(v3, factors=[thread_z, None])
            sch.reorder(v21, v31, v22, v32)
            sch.bind(v21, "threadIdx.z")
            sch.bind(v31, "threadIdx.y")
            return store

        store = store_output(
            block_outer, 0, 2, [block_m, block_n, micro_size_m, micro_size_n], "wmma.accumulator"
        )

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # unroll k
        sch.unroll(k0)

        # Tensorization by hardware intrinsics
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype=str(dtype_a),
            out_dtype=str(dtype_c),
            trans_b=True,
        )

        sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
        sch.tensorize(sch.get_loops(wmma_read_a)[-2], intrin_group["load_a"])
        sch.tensorize(sch.get_loops(wmma_read_b)[-2], intrin_group["load_b"])
        sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])
        sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])

        return sch


class MatmulInt8Tensorization(ScheduleRule):
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
