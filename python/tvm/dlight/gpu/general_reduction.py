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

from tvm import tir
from tvm.target import Target

from ..base import ScheduleRule, normalize_prim_func, try_inline_contiguous_spatial


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

    def apply_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> tir.Schedule:
        block_factors = config.block
        thread_factors = config.thread
        step_factors = config.step_factors
        reduce_thread = config.reduce_thread
        if any([t > 1 for t in reduce_thread]):
            # block reduction schedule
            return None

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        loops = sch.get_loops(block_infos[-1].block_rv)
        block_loops = []
        vthread_loops = []
        thread_loops = []
        inner_loops = []
        for s_loop, block_factor, step_factor, thread_factor in zip(s_loops, block_factors, step_factors, thread_factors):
            block_loop, inner_loop = sch.split(s_loop, factors=[None, block_factor])
            vthread_loop, inner_loop = sch.split(
            inner_loop, factors=[None, thread_factor * step_factor])
            thread_loop, inner_loop = sch.split(inner_loop, factors=[None, step_factor])
            block_loops.append(block_loop)
            vthread_loops.append(vthread_loop)
            thread_loops.append(thread_loop)
            inner_loops.append(inner_loop)

        reduce_outer_axis, reduce_inner_axis = [], []
        for i in config.raxis_order:
            loop = reduce_loops[i]
            ro, ri = sch.split(loop, factors=[None, config.rstep[i]])
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)

        vthd_axis = list(reversed(vthd_axis)) # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + reduce_outer_axis + reduce_inner_axis + tile_axis

        sch.reorder(*axis_order)
        blck_fused = sch.fuse(*blck_axis)
        thrd_fused = sch.fuse(*thrd_axis)
        sch.bind(blck_fused, "blockIdx.x")
        sch.bind(thrd_fused, "threadIdx.x")
        if len(vthd_axis) > 3:
            vthd_axis = vthd_axis[0:2] + [sch.fuse(*vthd_axis[2:])]
        for i, ax in enumerate(vthd_axis):
            sch.bind(ax, "vthread" + ['.x', '.y', '.z'][i])
        for ax in tile_axis:
            sch.unroll(ax)

        cached_stages = []
        for i, input_tensor in enumerate(self.reduce_op.input_tensors):
            SS = sch.cache_read(C, i, "shared")
            cached_stages.append(SS)
            if input_tensor in self.shared_inputs:
                sch.compute_at(SS, blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
                dim_offset = 1
            else:
                sch.compute_at(SS, reduce_outer_axis[-1])
                strides = Stride()
                dim_offset = len(vthd_axis) + len(reduce_outer_axis) + 2 # outer loops are: blck_fused, thrd_fused, vthd_axis, reduce_outer_axis
            if input_tensor.name in config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = config.vectorize[input_tensor.name]
            else:
                vectorize = 1
            self.cooperative_fetch(SS, dim_offset, strides, vectorize)

        sch.reverse_compute_at(CL, thrd_fused)
        if len(tile_axis) > 0:
            for ax in sch.get_loops(CL)[-len(tile_axis):]:
                sch.unroll(ax)
        
        sch.decompose_reduction(C, reduce_outer_axis[0])

        self.schedule_compute_inline()
        
        return sch