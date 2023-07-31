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
"""A rule for GEMV and DecodeGEMV."""
import re
from typing import List, Optional, Union

from tvm import DataType, arith, ir, tir
from tvm.target import Target

from ..base import (
    BlockInfo,
    ScheduleRule,
    collect_vars_used_in_access_region,
    detect_dominant_read,
    is_broadcast_epilogue,
    normalize_prim_func,
    try_inline_contiguous_spatial,
)


def _get_reduction_expr(block: tir.Block) -> Optional[tir.PrimExpr]:
    # Detect and return `Y` in `X[...] = X[...] + Y`
    buffer_store = block.body
    if not isinstance(buffer_store, tir.BufferStore):
        return None
    if not isinstance(buffer_store.value, tir.Add):
        return None
    if not ir.structural_equal(
        buffer_store.value.a,
        tir.BufferLoad(buffer_store.buffer, block.body.indices),
        map_free_vars=True,
    ):
        return None
    return buffer_store.value.b


def get_bytes(dtype: Union[DataType, str]) -> int:
    num = re.findall(r"\d+", dtype)
    if len(num) != 1:
        raise ValueError(f"Cannot get bytes from {dtype}")
    return int(num[0]) // 8


def is_gemv(sch: tir.Schedule, block_info: BlockInfo) -> Optional[List[tir.Buffer]]:
    """Check if the block is a GEMV.

    Parameters
    ----------

    sch : tir.Schedule
        The schedule

    block_info : BlockInfo
        The block info to be checked


    Returns
    -------
    ret : Optional[List[tir.Buffer]]
        The vector buffers used in the GEMV if it is a GEMV, otherwise None.
    """
    block = block_info.block_rv
    block_stmt = sch.get(block)
    conditions = []
    conditions.append(block_info.is_reduction())
    conditions.append(len(block_stmt.reads) >= 2)
    conditions.append(len(block_stmt.writes) == 1)
    conditions.append(_get_reduction_expr(block_stmt) is not None)
    conditions.append(len(collect_vars_used_in_access_region(block_stmt.writes[0].region)) > 0)
    if not all(conditions):
        return None

    iter_num = len(block_stmt.iter_vars)
    ret = [
        read.buffer
        for read in block_stmt.reads
        if len(collect_vars_used_in_access_region(read.region)) < iter_num
    ]
    return ret if 0 < len(ret) < len(block_stmt.reads) else None


def normalize(
    sch: tir.Schedule,
    block_info: BlockInfo,
) -> Optional[bool]:
    """Normalize the main block."""
    block_stmt: tir.Block = sch.get(block_info.block_rv)
    access = arith.normalize_to_iter_sum(
        detect_dominant_read(block_stmt),
        input_iters={i.var: i.dom for i in block_stmt.iter_vars},
    )

    buffers_use_vars = [collect_vars_used_in_access_region(buf.region) for buf in block_stmt.writes]
    buffers_use_vars.extend(
        [collect_vars_used_in_access_region(buf.region) for buf in block_stmt.reads]
    )
    if access.base != 0:
        return None
    iter_to_info = {i.var: i for i in block_info.iters}
    batch_loops, s_loops, r_loops, c_loops = [], [], [], []
    inner_axis = access.args[-1].source.source
    is_inner_reduction = iter_to_info[inner_axis].kind == "R"

    for split_expr in access.args:
        var = split_expr.source.source
        info = iter_to_info.get(var)
        loop = info.loop_rv
        is_reduction = info.kind == "R"
        if split_expr.lower_factor > 1:
            if c_loops:
                return None
            loop, c_loop = sch.split(loop, factors=[None, split_expr.lower_factor])
            c_loops.append(c_loop)
        if is_reduction:
            r_loops.append(loop)
        elif all([var in buf_vars for buf_vars in buffers_use_vars]):
            batch_loops.append(loop)
        else:
            s_loops.append(loop)

    assert s_loops
    assert r_loops
    if not c_loops:
        c_loops = [sch.add_unit_loop(block_info.block_rv)]
    if not batch_loops:
        batch_loops = [sch.add_unit_loop(block_info.block_rv)]
    sch.reorder(*batch_loops, *s_loops, *r_loops, *c_loops)
    sch.fuse(*batch_loops)
    sch.fuse(*s_loops)
    sch.fuse(*r_loops)
    return is_inner_reduction


class GEMV(ScheduleRule):
    """A rule for GEMV and DecodeGEMV."""

    def apply(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc):
            return None
        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if len(block_infos) == 1:
            epilogue = None
        elif len(block_infos) == 2:
            epilogue = block_infos[1]
            if not epilogue.is_injective():
                return None
        else:
            return None

        block_info = block_infos[0]
        block = block_info.block_rv
        vector_input_buffers = is_gemv(sch, block_info)
        if vector_input_buffers is None:
            return None

        # Step 1. Normalize the block, merge spatial and reduction iters
        is_inner_reduction = normalize(sch, block_info)

        # Step 2. Do the scheduling
        if is_inner_reduction:
            # print(func)
            self.sch_inner_reduction(sch, target, block, vector_input_buffers, epilogue)
            return sch
        else:
            # TODO: Need to handle GEMV with KN layout
            return None

    def sch_inner_reduction(  # pylint: disable=too-many-arguments
        self,
        sch: tir.Schedule,
        target: Target,
        block: tir.schedule.BlockRV,
        vector_input_buffers: List[tir.Buffer],
        epilogue_info: Optional[BlockInfo],
    ):
        """Schedule the inner reduction block."""
        # pylint: disable=invalid-name
        _, s, r, _ = sch.get_loops(block)
        # TODO: make it tunable
        vec_bytes = 16 if target.kind.name == "cuda" else 8
        unroll_number = 256 if target.kind.name == "cuda" else 64

        def get_extent(loop_rv: tir.schedule.LoopRV):
            loop: tir.For = sch.get(loop_rv)
            return loop.extent.value if isinstance(loop.extent, tir.IntImm) else 1

        # Specify the `len_tx` and `len_ty` according to the loop extent
        len_s, len_r = get_extent(s), get_extent(r)
        if len_r >= 4096 and len_r % 128 == 0:
            len_tx = 128
        elif 1024 < len_r <= 2048 and len_r % 64 == 0:
            len_tx = 64
        else:
            len_tx = 32

        if len_s >= 4096:
            len_ty = 8
        else:
            len_ty = min(len_s, 4)

        _, tx = sch.split(r, [None, len_tx], preserve_unit_iters=True)
        # Schedule the RF block
        rf = sch.rfactor(tx, 0)
        batch, bx, r, tx, _ = sch.get_loops(rf)
        sch.reorder(bx, tx, r)
        bx, ty = sch.split(bx, [None, len_ty], preserve_unit_iters=True)
        sch.bind(batch, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        unit = sch.add_unit_loop(r)
        sch.annotate(unit, "pragma_auto_unroll_max_step", unroll_number)
        sch.annotate(unit, "pragma_unroll_explicit", 1)

        if target.kind.name == "cuda":
            # Cache read the vector
            def cache_shared(index: int):
                block: tir.Block = sch.get(rf)
                type_bytes: int = get_bytes(block.reads[index].buffer.dtype)
                cache = sch.cache_read(rf, index, "shared")
                sch.compute_at(cache, unit, preserve_unit_loops=True)
                fused = sch.fuse(*sch.get_loops(cache)[5:])
                loop: tir.For = sch.get(fused)
                vec_length = vec_bytes // type_bytes
                if isinstance(loop.extent, tir.IntImm):
                    # avoid introducing predicates when vector length is too large
                    vec_length = min(loop.extent // len_ty // len_tx, vec_length)
                _, _ty, _tx, _vec = sch.split(fused, [None, len_ty, len_tx, vec_length])
                sch.bind(_ty, "threadIdx.y")
                sch.bind(_tx, "threadIdx.x")
                sch.vectorize(_vec)

            def cache_local(index: int):
                block: tir.Block = sch.get(rf)
                type_bytes: int = get_bytes(block.reads[index].buffer.dtype)
                vec_length = vec_bytes // type_bytes
                cache = sch.cache_read(rf, index, "local")
                sch.compute_at(cache, r, preserve_unit_loops=True)
                fused = sch.fuse(*sch.get_loops(cache)[6:])
                loop: tir.For = sch.get(fused)
                if isinstance(loop.extent, tir.IntImm) and loop.extent.value % vec_length == 0:
                    _, _vec = sch.split(fused, [None, vec_length])
                    sch.vectorize(_vec)
                elif isinstance(loop.extent, tir.IntImm) and loop.extent.value < vec_length:
                    sch.vectorize(fused)

            for buffer in vector_input_buffers:
                index = vector_input_buffers.index(buffer)
                cache_shared(index)
                cache_local(index)

            # TODO: cache scale buffer in Decode-GEMV to shared memory

        sch.set_scope(rf, 0, "local")
        sch.decompose_reduction(rf, r)
        # Schedule the write back block
        sch.reverse_compute_at(block, ty, preserve_unit_loops=True)
        _, _, _, tx, *s = sch.get_loops(block)
        s = sch.fuse(*s)
        sch.reorder(s, tx)
        sch.bind(tx, "threadIdx.x")
        # Schedule epilogue
        if epilogue_info is not None:
            epilogue = epilogue_info.block_rv
            if is_broadcast_epilogue(sch, block, epilogue):
                sch.reverse_compute_at(epilogue, bx)
                sch.set_scope(block, 0, "shared")
                _, _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                _, tx = sch.split(sch.fuse(*s), factors=[None, len_tx])
                sch.bind(tx, "threadIdx.x")
            else:
                # NOTE: Need to ensure tx_len == 32, so that can use `local` stage here
                sch.reverse_compute_at(epilogue, ty)
                sch.set_scope(block, 0, "local")
        # pylint: enable=invalid-name
