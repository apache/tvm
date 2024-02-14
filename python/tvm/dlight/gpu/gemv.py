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
from functools import reduce
from typing import List, Optional, Union

from tvm import DataType, arith, ir, tir
from tvm.target import Target

from ..base import (
    BlockInfo,
    collect_block_iter_vars_used_in_access_region,
    collect_vars_used_in_prim_expr,
    detect_dominant_read,
    is_broadcast_epilogue,
    normalize_prim_func,
    try_inline_contiguous_spatial,
)
from .base import GPUScheduleRule


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


def get_extent(sch: tir.Schedule, loop_rv: tir.schedule.LoopRV):
    loop: tir.For = sch.get(loop_rv)
    return loop.extent.value if isinstance(loop.extent, tir.IntImm) else loop.extent


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
    conditions.append(
        len(collect_block_iter_vars_used_in_access_region(block_stmt, block_stmt.writes[0].region))
        > 0
    )
    if not all(conditions):
        return None

    iter_num = len(block_stmt.iter_vars)
    ret = [
        read.buffer
        for read in block_stmt.reads
        if len(collect_block_iter_vars_used_in_access_region(block_stmt, read.region)) < iter_num
        and len(collect_block_iter_vars_used_in_access_region(block_stmt, read.region)) > 0
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
    buffers_use_vars = [
        collect_block_iter_vars_used_in_access_region(block_stmt, buf.region)
        for buf in block_stmt.writes
    ]
    buffers_use_vars.extend(
        [
            collect_block_iter_vars_used_in_access_region(block_stmt, buf.region)
            for buf in block_stmt.reads
        ]
    )
    if collect_vars_used_in_prim_expr(access.base) & set(
        iter_var.var for iter_var in block_stmt.iter_vars
    ):
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
            # we only support the reduction dim being grouped atm
            if not is_reduction:
                return None
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


class GEMV(GPUScheduleRule):
    """A rule for GEMV and DecodeGEMV."""

    def apply(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
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
        if len(block_info.iters) not in [2, 3]:
            # either [B, S, R] = [B, S, R] * [B, R]
            # or [S, R] = [S, R] * [R]
            return None
        block = block_info.block_rv
        vector_input_buffers = is_gemv(sch, block_info)
        if vector_input_buffers is None:
            return None

        # Step 1. Normalize the block, merge spatial and reduction iters
        is_inner_reduction = normalize(sch, block_info)

        # Step 2. Do the scheduling
        if is_inner_reduction is None:
            return None
        elif is_inner_reduction:
            self.sch_inner_reduction(sch, target, block, vector_input_buffers, epilogue)
            return sch
        else:
            return self.sch_outer_reduction(sch, target, block, vector_input_buffers, epilogue)

    def sch_inner_reduction(  # pylint: disable=too-many-arguments, invalid-name, unused-argument
        self,
        sch: tir.Schedule,
        target: Target,
        block: tir.schedule.BlockRV,
        vector_input_buffers: List[tir.Buffer],
        epilogue_info: Optional[BlockInfo],
    ):
        """Schedule the inner reduction block."""

        def get_max_factor(n, factors):
            factors = sorted(factors, reverse=True)
            for factor in factors:
                if n % factor == 0:
                    return factor
            return 1

        def apply(
            sch: tir.Schedule,
            gemv,
            TAG_S,
            TAG_R,
            TS,
            TR,
            TILE_S,
            TILE_R,
            VEC_LOAD,
            VEC_C,
            LOAD_V_SHARED,
            LOAD_V_VEC,
            UNROLL,
        ):
            # rfactor: reduce to tx * vec_c
            _, s, r, c = sch.get_loops(block=gemv)
            s = sch.fuse(_, s)
            r = sch.fuse(r, c)
            bx, ts, tile_s = sch.split(s, factors=[None, TS, TILE_S], preserve_unit_iters=True)
            r, tr, tile_r_vec_n, vec_c = sch.split(
                r, factors=[None, TR, TILE_R // VEC_C, VEC_C], preserve_unit_iters=True
            )
            sch.reorder(r, tile_r_vec_n, tr, vec_c)
            tr_vec_c = sch.fuse(tr, vec_c)
            rf = sch.rfactor(tr_vec_c, 0)

            # rfactor: reduce to tx
            bx, ts, tile_s, tr_vec_c = sch.get_loops(block=gemv)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            rf2 = sch.rfactor(tr, 0)

            # bind, vectorize compute
            bx, ts, tile_s, r, tile_r_vec_n, tr_vec_c = sch.get_loops(block=rf)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            sch.reorder(bx, ts, tr, r, tile_s, tile_r_vec_n, vec_c)
            sch.bind(bx, "blockIdx.x")
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            sch.vectorize(vec_c)

            shared_mem_usage = 0
            for buf in vector_input_buffers:
                buf_size = reduce(
                    lambda x, y: x * y, buf.shape, tir.IntImm(buf.shape[0].dtype, 1)
                ) * get_bytes(buf.dtype)
                shared_mem_usage += buf_size
            LOAD_V_SHARED = (
                LOAD_V_SHARED
                and isinstance(shared_mem_usage, tir.IntImm)
                and shared_mem_usage.value <= target.max_shared_memory_per_block
            )

            # vectorize load A
            # (TODO) this is now actually problematic since the number of loops is dependent on the
            # number of dimensions of A_q
            Aq_local = sch.cache_read(rf, read_buffer_index=1, storage_scope="local")
            sch.compute_at(Aq_local, r, preserve_unit_loops=True)
            s_local, r_local = sch.get_loops(block=Aq_local)[-2:]
            s_local, vec_load = sch.split(
                s_local, factors=[None, VEC_LOAD], preserve_unit_iters=True
            )
            sch.reorder(s_local, r_local, vec_load)  # either s_local or r_local should be 1
            sch.vectorize(vec_load)

            # load vector into shared memory, shape should be the whole vector
            if LOAD_V_SHARED:
                assert len(vector_input_buffers) == 1
                V_shared = sch.cache_read(rf, read_buffer_index=0, storage_scope="shared")
                sch.compute_at(V_shared, tr, preserve_unit_loops=True)
                l = sch.get_loops(block=V_shared)[-1]
                loop: tir.For = sch.get(l)
                if isinstance(loop.extent, tir.IntImm):
                    # avoid introducing predicates when vector length is too large
                    vec_length = max(
                        min(
                            get_max_factor(
                                (int)(loop.extent),
                                [TS * TR * 1, TS * TR * 2, TS * TR * 4, TS * TR * 8],
                            )
                            // TS
                            // TR,
                            LOAD_V_VEC,
                        ),
                        1,
                    )
                else:
                    vec_length = LOAD_V_VEC
                if TAG_R == "threadIdx.x":
                    _, ty, tx, vec = sch.split(
                        l, factors=[None, TS, TR, vec_length], preserve_unit_iters=True
                    )
                else:
                    _, ty, tx, vec = sch.split(
                        l, factors=[None, TR, TS, vec_length], preserve_unit_iters=True
                    )
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                sch.vectorize(vec)

            # reduce tile_s * tr * vec to tile_s * tr
            sch.reverse_compute_at(rf2, loop=bx, preserve_unit_loops=True)
            tr, vec_c, *ts_tile_s = sch.get_loops(block=rf2)[1:]
            ts_tile_s = sch.fuse(*ts_tile_s)
            ts, tile_s = sch.split(ts_tile_s, factors=[TS, None], preserve_unit_iters=True)
            tile_s, vec_s = sch.split(
                tile_s,
                factors=[None, get_max_factor(TILE_S, [1, 2, 4, 8])],
                preserve_unit_iters=True,
            )
            sch.reorder(ts, tr, tile_s, vec_s, vec_c)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            sch.vectorize(vec_s)

            # reduce tile_s * tr to tile_s
            sch.reverse_compute_at(gemv, loop=bx, preserve_unit_loops=True)
            tr, *ts_tile_s = sch.get_loops(block=gemv)[1:]
            ts_tile_s = sch.fuse(*ts_tile_s)
            ts, tile_s = sch.split(ts_tile_s, factors=[TS, None], preserve_unit_iters=True)
            sch.reorder(tile_s, ts, tr)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)

            sch.decompose_reduction(rf, loop=sch.get_loops(block=rf)[3])
            sch.decompose_reduction(rf2, loop=sch.get_loops(block=rf2)[-1])

            sch.set_scope(rf, buffer_index=0, storage_scope="local")
            sch.set_scope(rf2, buffer_index=0, storage_scope="local")

            unroll_factor = UNROLL

            sch.annotate(
                block_or_loop=sch.get_loops(rf)[3],
                ann_key="pragma_auto_unroll_max_step",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf)[3], ann_key="pragma_unroll_explicit", ann_val=1
            )

            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[3],
                ann_key="pragma_auto_unroll_max_step",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[3], ann_key="pragma_unroll_explicit", ann_val=1
            )

            if LOAD_V_SHARED:
                sch.annotate(
                    block_or_loop=sch.get_loops(V_shared)[-4],
                    ann_key="pragma_unroll_explicit",
                    ann_val=unroll_factor,
                )
                sch.annotate(
                    block_or_loop=sch.get_loops(V_shared)[-4], ann_key="pragma_vectorize", ann_val=1
                )

            # Schedule epilogue
            if epilogue_info is not None:
                epilogue = epilogue_info.block_rv
                if is_broadcast_epilogue(sch, block, epilogue):
                    sch.reverse_compute_at(epilogue, bx)
                    sch.set_scope(block, 0, "shared")
                    _, _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                    _, tx = sch.split(sch.fuse(*s), factors=[None, TX])
                    sch.bind(tx, "threadIdx.x")
                else:
                    sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
                    ts_tile_s = sch.fuse(*sch.get_loops(epilogue)[1:])
                    ts_tile_s = sch.get_loops(epilogue)[-1]
                    ts, tile_s = sch.split(ts_tile_s, factors=[TS, None], preserve_unit_iters=True)
                    sch.bind(ts, TAG_S)
                    sch.set_scope(block, 0, "local")
            # pylint: enable=invalid-name
            return sch

        # Specify the `len_tx` and `len_ty` according to the loop extent
        batch, s, r, c = sch.get_loops(block=block)
        len_batch, len_s, len_r, len_c = (
            get_extent(sch, batch),
            get_extent(sch, s),
            get_extent(sch, r),
            get_extent(sch, c),
        )
        len_S = len_batch * len_s
        len_R = len_r * len_c

        TAG_S, TAG_R = "threadIdx.y", "threadIdx.x"
        if target.kind.name == "cuda":
            VEC_C = 4
            LOAD_V_SHARED = True
            LOAD_V_VEC = 8
            UNROLL = 256
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 4, 64
                else:
                    TS, TR = 16, 32
        elif target.kind.name == "metal":
            # Note that the following tile size is tuned on M2 Ultra for 7B
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            VEC_C = 1
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 256
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 4, 16
                else:
                    TS, TR = 2, 64
        elif target.kind.name == "rocm":
            VEC_C = 4
            LOAD_V_SHARED = True
            LOAD_V_VEC = 8
            UNROLL = 256
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 1, 128
                else:
                    TS, TR = 8, 64
        elif target.kind.name == "opencl" and "android" in str(target.host):
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            VEC_C = 8
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 8
            TS, TR = 2, 32
        elif target.kind.name == "vulkan":
            VEC_C = 4
            LOAD_V_SHARED = True
            LOAD_V_VEC = 4
            UNROLL = 256
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 4, 32
                else:
                    TS, TR = 16, 32
        elif target.kind.name == "opencl" and "mali" in str(target.attrs):
            VEC_C = 8
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 64
            TS, TR = 1, 64
        else:
            VEC_C = 1
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 64
            TS, TR = 1, 64

        if not isinstance(len_S, int):
            TS, TR = 1, 64

        while TS * TR > target.max_num_threads:
            if TS > 1:
                TS //= 2
            else:
                TR //= 2

        TILE_S, TILE_R = (
            1,
            len_c
            if len_c > 1
            else max(get_max_factor(len_r, [TR * 1, TR * 2, TR * 4, TR * 8]) // TR, 1),
        )
        VEC_C = min(get_max_factor(TILE_R, [1, 2, 4, 8]), VEC_C)
        VEC_LOAD = 1

        return apply(
            sch,
            gemv=block,
            TAG_S=TAG_S,
            TAG_R=TAG_R,
            TS=TS,
            TR=TR,
            TILE_S=TILE_S,
            TILE_R=TILE_R,
            VEC_LOAD=VEC_LOAD,
            VEC_C=VEC_C,
            LOAD_V_SHARED=LOAD_V_SHARED,
            LOAD_V_VEC=LOAD_V_VEC,
            UNROLL=UNROLL,
        )

    def sch_outer_reduction(  # pylint: disable=too-many-arguments, invalid-name, unused-argument
        self,
        sch: tir.Schedule,
        target: Target,
        block: tir.schedule.BlockRV,
        vector_input_buffers: List[tir.Buffer],
        epilogue_info: Optional[BlockInfo],
    ):
        """Schedule the outer reduction block."""
        # NOTE: Only Android is supported so far
        if not (target.kind.name == "opencl" and "android" in str(target.host)):
            return None
        batch, s, r, c = sch.get_loops(block)
        len_s = get_extent(sch, s)

        # The config is designed for Adreno
        tx_len = 64
        vec_len = (4 if len_s > 4096 else 2) if isinstance(len_s, int) else 1
        inner_r = 4

        bx, tx, vec = sch.split(s, factors=[None, tx_len, vec_len])
        r0, r1 = sch.split(r, factors=[None, inner_r])
        sch.bind(batch, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.reorder(bx, tx, r0, r1, c, vec)

        sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=8)
        sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        cache_v = sch.cache_read(block, vector_input_buffers[0], "local")
        sch.compute_at(cache_v, r1, preserve_unit_loops=True)
        sch.vectorize(sch.get_loops(cache_v)[-1])

        sch.vectorize(vec)

        # Schedule epilogue
        if epilogue_info is not None:
            sch.reverse_compute_at(epilogue_info.block_rv, tx)

            sch.set_scope(block, 0, "local")

        sch.decompose_reduction(block, r0)

        return sch
