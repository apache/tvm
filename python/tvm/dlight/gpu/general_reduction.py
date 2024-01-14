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
# pylint: disable=invalid-name
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""
from typing import List, Union
from functools import reduce

from tvm import tir
from tvm.target import Target

from ..base import ScheduleRule, normalize_prim_func, try_inline_contiguous_spatial
from ..base.analysis import get_root_block, get_reduction_blocks


class GeneralReduction(ScheduleRule):
    """General Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc):
            return None

        if target.kind.name == "cuda":
            len_tx = 256
            unroll_depth = 256
        else:
            len_tx = 64
            unroll_depth = 64

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        dom_kind = block_infos[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        # Align the number of block iters of the last block.
        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter < len(dom_kind):
            index_map = tir.IndexMap.from_func(
                lambda *iters: (
                    [tir.const(0, iters[0].dtype)] * (len(dom_kind) - num_last_block_iter)
                    + list(iters)
                ),
                ndim=num_last_block_iter,
            )
            sch.transform_block_layout(block_infos[-1].block_rv, index_map)

        try:
            # TODO: fix num_leading_s = 0 case
            assert num_trailing_r > 0
            for block in block_infos[1:-1]:
                assert block.dom_kind() == dom_kind
            assert block_infos[-1].is_injective()
            assert len(block_infos[-1].dom_kind()) <= len(dom_kind)
        except AssertionError:
            return None

        loops = sch.get_loops(block_infos[-1].block_rv)
        bx = sch.fuse(*loops[:num_leading_s])
        r_loop, tx = sch.split(loops[-1], [None, len_tx])
        sch.reorder(tx, r_loop)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

        for block in reversed(block_infos[:-1]):
            block = block.block_rv
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, buffer_index=i, storage_scope="shared")
            sch.compute_at(block, bx, preserve_unit_loops=True)
            r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
            r_loop, tx = sch.split(r_loop, [None, len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(tx, "threadIdx.x")
            sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

        # TODO: It's just a workaround to avoid unroll spatial loops, because of the bug of
        # the pass lower-thread-allreduce. We should fix it in the future.
        # sch.annotate(bx, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        # sch.annotate(bx, ann_key="pragma_unroll_explicit", ann_val=1)
        return sch

    def sch_inner_reduction_with_config(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tir.PrimFunc,
        config,
    ):
        block_factors = config.block
        thread_factors = config.thread
        reduce_therad_factors = config.reduce_thread

        # For inter thread reduction case, one thread must only compute one element
        assert thread_factors == block_factors

        # inline all the other blocks
        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)

        schedule_block: tir.schedule.BlockRV = None
        reduction_blocks: List[tir.schedule.BlockRV] = []
        for block in block_infos:
            s_loops: List[tir.schedule.LoopRV] = []
            r_loops: List[tir.schedule.LoopRV] = []
            o_loops: List[tir.schedule.LoopRV] = []
            dom_kind = block.dom_kind()
            block_rv = block.block_rv

            if (
                any(
                    [
                        sch.get(loop_rv).thread_binding is not None
                        for loop_rv in sch.get_loops(block_rv)
                    ]
                )
                or len(sch.get_loops(block.block_rv)) == 0
            ):
                continue

            for loop, iter_type in zip(sch.get_loops(block_rv), dom_kind):
                {"S": s_loops, "R": r_loops, "O": o_loops}[iter_type].append(loop)

            if not s_loops:
                s_loops.append(sch.add_unit_loop(block_rv))
            if len(r_loops) > 0:
                # always use the last reduction block for scheduling
                schedule_block = block
                reduction_blocks.append(block_rv)

        # Align the number of block iters of the last block.
        dom_kind = schedule_block.dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        schedule_block = schedule_block.block_rv
        loops = sch.get_loops(schedule_block)
        s_loops = loops[:num_leading_s]
        r_loops = loops[-num_trailing_r:]

        block_axis = []
        thread_axis = []

        for s_loop, block_factor in zip(s_loops, block_factors):
            block_loop, thread_loop = sch.split(s_loop, factors=[None, block_factor])
            block_axis.append(block_loop)
            thread_axis.append(thread_loop)

        axis_order = block_axis + thread_axis

        sch.reorder(*axis_order)
        blck_fused = sch.fuse(*block_axis)
        thrd_fused = sch.fuse(*thread_axis)
        sch.bind(blck_fused, "blockIdx.x")
        sch.bind(thrd_fused, "threadIdx.y")

        reduce_outer_axis, reduce_inner_axis, reduce_inter_threads = [], [], []
        for i in config.raxis_order:
            loop = r_loops[i]
            ro, ri = sch.split(loop, factors=[None, config.rstep[i]])
            ri, thd = sch.split(ri, factors=[None, config.reduce_thread[i]])
            reduce_inter_threads.append(thd)
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)

        axis_order = reduce_inter_threads + reduce_outer_axis + reduce_inner_axis
        sch.reorder(*axis_order)
        fused_reduce_inter_threads = sch.fuse(*reduce_inter_threads)
        sch.bind(fused_reduce_inter_threads, "threadIdx.x")

        def prod(iterable):
            return reduce(lambda x, y: x * y, iterable, 1)

        reg_tile = sch.cache_write(schedule_block, 0, "local")
        # todo(lei): should add the shared_inputs/stride memory pad analysis at shared memory fusion stage.
        for i, input_region in enumerate(sch.get(schedule_block).reads):
            if input_region.buffer.name not in config.cached_tensors:
                continue

            # otherwise cooperative fetch in shared memory.
            cache_shared = sch.cache_read(schedule_block, i, "shared")
            sch.compute_at(cache_shared, reduce_outer_axis[-1])

            dim_offset = (
                len(reduce_inner_axis) + len(reduce_outer_axis) + 2
            )  # outer loops are: blck_fused, thrd_fused, vthread_axis, reduce_outer_axis
            if input_region.buffer.name in config.vectorize:
                vectorize = config.vectorize[input_region.buffer.name]
            else:
                vectorize = 1

            loops = sch.get_loops(cache_shared)
            if len(loops) == dim_offset:
                # handle fetching only one element
                loops.append(sch.add_unit_loop(schedule_block))
            assert len(loops) > dim_offset

            _, ty, tx, tv = sch.split(
                sch.fuse(*loops[dim_offset:]),
                factors=[
                    None,
                    int(prod(thread_factors)),
                    int(prod(reduce_therad_factors)),
                    vectorize,
                ],
            )
            sch.vectorize(tv)
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")

        sch.reverse_compute_at(reg_tile, thrd_fused)

        # resolve compute_at
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None
        return sch

    def sch_outer_reduction_with_config(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tir.PrimFunc,
        config,
    ):
        block_factors = config.block
        thread_factors = config.thread
        step_factors = config.step

        # inline all the other blocks
        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)

        schedule_block: tir.schedule.BlockRV = None
        for block in block_infos:
            s_loops: List[tir.schedule.LoopRV] = []
            r_loops: List[tir.schedule.LoopRV] = []
            o_loops: List[tir.schedule.LoopRV] = []
            dom_kind = block.dom_kind()
            block_rv = block.block_rv

            if (
                any(
                    [
                        sch.get(loop_rv).thread_binding is not None
                        for loop_rv in sch.get_loops(block_rv)
                    ]
                )
                or len(sch.get_loops(block.block_rv)) == 0
            ):
                continue

            for loop, iter_type in zip(sch.get_loops(block_rv), dom_kind):
                {"S": s_loops, "R": r_loops, "O": o_loops}[iter_type].append(loop)

            if not s_loops:
                s_loops.append(sch.add_unit_loop(block_rv))
            if len(r_loops) > 0:
                # always use the last reduction block for scheduling
                schedule_block = block

        # Align the number of block iters of the last block.
        dom_kind = schedule_block.dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter < len(dom_kind):
            index_map = tir.IndexMap.from_func(
                lambda *iters: (
                    [tir.const(0, iters[0].dtype)] * (len(dom_kind) - num_last_block_iter)
                    + list(iters)
                ),
                ndim=num_last_block_iter,
            )
            sch.transform_block_layout(schedule_block.block_rv, index_map)

        schedule_block = schedule_block.block_rv
        loops = sch.get_loops(schedule_block)
        s_loops = loops[:num_leading_s]
        r_loops = loops[-num_trailing_r:]

        reg_tile = sch.cache_write(schedule_block, 0, "local")

        block_axis = []
        vthread_axis = []
        thread_axis = []
        inner_axis = []
        for s_loop, block_factor, step_factor, thread_factor in zip(
            s_loops, block_factors, step_factors, thread_factors
        ):
            block_loop, inner_loop = sch.split(s_loop, factors=[None, block_factor])
            vthread_loop, inner_loop = sch.split(
                inner_loop, factors=[None, thread_factor * step_factor]
            )
            thread_loop, inner_loop = sch.split(inner_loop, factors=[None, step_factor])
            block_axis.append(block_loop)
            vthread_axis.append(vthread_loop)
            thread_axis.append(thread_loop)
            inner_axis.append(inner_loop)

        reduce_outer_axis, reduce_inner_axis = [], []
        for i in config.raxis_order:
            loop = r_loops[i]
            ro, ri = sch.split(loop, factors=[None, config.rstep[i]])
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)

        vthread_axis = list(reversed(vthread_axis))  # inner virtual thread first
        axis_order = (
            block_axis
            + vthread_axis
            + thread_axis
            + reduce_outer_axis
            + reduce_inner_axis
            + inner_axis
        )

        sch.reorder(*axis_order)
        blck_fused = sch.fuse(*block_axis)
        thrd_fused = sch.fuse(*thread_axis)
        sch.bind(blck_fused, "blockIdx.x")
        sch.bind(thrd_fused, "threadIdx.x")
        if len(vthread_axis) > 3:
            vthread_axis = vthread_axis[0:2] + [sch.fuse(*vthread_axis[2:])]
        for i, ax in enumerate(vthread_axis):
            sch.bind(ax, "vthread" + [".x", ".y", ".z"][i])

        # todo(lei): should add the shared_inputs/stride memory pad analysis at shared memory fusion stage.
        for i, input_region in enumerate(sch.get(schedule_block).reads):
            if input_region.buffer.name not in config.cached_tensors:
                continue

            # otherwise cooperative fetch in shared memory.
            cache_shared = sch.cache_read(schedule_block, i, "shared")
            sch.compute_at(cache_shared, reduce_outer_axis[-1])

            dim_offset = (
                len(vthread_axis) + len(reduce_outer_axis) + 2
            )  # outer loops are: blck_fused, thrd_fused, vthread_axis, reduce_outer_axis
            if input_region.buffer.name in config.vectorize:
                vectorize = config.vectorize[input_region.buffer.name]
            else:
                vectorize = 1

            loops = sch.get_loops(cache_shared)
            if len(loops) == dim_offset:
                # handle fetching only one element
                loops.append(sch.add_unit_loop(schedule_block))
            assert len(loops) > dim_offset

            def prod(iterable):
                return reduce(lambda x, y: x * y, iterable, 1)

            _, tx, tv = sch.split(
                sch.fuse(*loops[dim_offset:]), factors=[None, int(prod(thread_factors)), vectorize]
            )
            sch.vectorize(tv)
            sch.bind(tx, "threadIdx.x")

        sch.reverse_compute_at(reg_tile, thrd_fused)

        sch.decompose_reduction(schedule_block, reduce_outer_axis[0])

        # resolve compute_at
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        return sch

    def sch_mutiple_reductions_with_config(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tir.PrimFunc,
        config,
    ):
        block_factors = config.block
        thread_factors = config.thread
        reduce_therad_factors = config.reduce_thread

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        def prod(iterable):
            return reduce(lambda x, y: x * y, iterable, 1)

        len_tx = prod(thread_factors) * prod(reduce_therad_factors)
        block_factor = prod(block_factors)

        dom_kind = block_infos[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        # Align the number of block iters of the last block.
        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter < len(dom_kind):
            index_map = tir.IndexMap.from_func(
                lambda *iters: (
                    [tir.const(0, iters[0].dtype)] * (len(dom_kind) - num_last_block_iter)
                    + list(iters)
                ),
                ndim=num_last_block_iter,
            )
            sch.transform_block_layout(block_infos[-1].block_rv, index_map)

        try:
            # TODO: fix num_leading_s = 0 case
            assert num_trailing_r > 0
            for block in block_infos[1:-1]:
                assert block.dom_kind() == dom_kind
            assert block_infos[-1].is_injective()
            assert len(block_infos[-1].dom_kind()) <= len(dom_kind)
        except AssertionError:
            return None

        loops = sch.get_loops(block_infos[-1].block_rv)
        bx, _ = sch.split(sch.fuse(*loops[:num_leading_s]), factors=[None, block_factor])
        r_loop, tx = sch.split(loops[-1], [None, len_tx])
        sch.reorder(tx, r_loop)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        for block in reversed(block_infos[:-1]):
            block = block.block_rv
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, buffer_index=i, storage_scope="shared")
            sch.compute_at(block, bx, preserve_unit_loops=True)
            r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
            r_loop, tx = sch.split(r_loop, [None, len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(tx, "threadIdx.x")

        return sch

    def apply_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> tir.Schedule:
        # check the number of reduction blocks
        sch = tir.Schedule(func)
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = get_reduction_blocks(sch, blocks)
        if len(reduction_blocks) > 1:
            # schedule for multiple reduction blocks (e.g. softmax)
            return self.sch_mutiple_reductions_with_config(func, config)

        if any([t > 1 for t in config.reduce_thread]):
            # todo(lei) should implement block reduction schedule
            return self.sch_inner_reduction_with_config(func, config)
        else:
            return self.sch_outer_reduction_with_config(func, config)
