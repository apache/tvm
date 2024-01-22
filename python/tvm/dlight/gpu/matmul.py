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
from typing import Optional

from tvm import tir
from tvm.target import Target
from tvm.tir.stmt import ForKind

from ..base import analysis
from .base import GPUScheduleRule
from . import utils
from .matmul_analysis import (
    auto_inline_consumer_chain,
    auto_inline_producers,
    get_in_out_dtypes,
    get_index_map,
    get_reduction_blocks,
)
from .matmul_mma import MatmulTensorizationMMA
from .matmul_wmma import MatmulInt8Tensorization, MatmulTensorizationWMMA, MatmulTensorizationLegacy
from functools import reduce

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
        sch = normalize_to_matmul(sch, main_block)
        if sch is None:
            return None

        # Step 1. Check Tensor Core support
        # Tensorization config:
        # If any value of I, J, K is fixed and less than this threshold,
        # tensorization rule will not be applied.
        minimal_tensorize_threshold = 64
        block_stmt = sch.get(main_block)
        if target.kind.name == "cuda" and utils.get_sm_version(target) >= 70:
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
                elif utils.get_sm_version(target) >= 80:
                    # For A100(sm_80) or more advanced gpu, use MMA tensorization.
                    tensorize_sch = MatmulTensorizationMMA().apply(func, target, _)
                else:
                    # For other GPUs, use WMMA tensorization.
                    tensorize_sch = MatmulTensorizationWMMA().apply(func, target, _)
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

        # Step 4. Check if there are unbound blocks. Execute fallback scheduling to them.
        def is_scheduled(block: tir.schedule.BlockRV) -> bool:
            loops = sch.get_loops(block)
            loop_kinds = {sch.get(loop).kind for loop in loops}
            return loop_kinds != {ForKind.SERIAL}

        blocks = sch.get_child_blocks(root_block)
        max_threads_per_block = utils.max_threads_per_block(target)
        for block in blocks:
            if is_scheduled(block):
                continue
            # no axis of the block is bound to thread or block
            s_loops = sch.get_loops(block)
            bx, tx = sch.split(
                sch.fuse(*s_loops),
                factors=[
                    None,
                    256,
                ],
            )
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

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
            tensorize_sch = MatmulMMATensorization().apply_config(func, config)
            if tensorize_sch is not None:
                return tensorize_sch

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)

        # cuda core prefer b is [k, j] layout without swizzling.
        index_maps = get_index_map(block_stmt, ["n", "n", "n"])
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
        vthread_row_tiles = config.step[0] * 2  # expand vtrhead to avoid load band conflict
        vthread_col_tiles = config.step[1] * 2  # expand vtrhead to avoid load band conflict
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
