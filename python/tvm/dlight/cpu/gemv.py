from functools import reduce
from typing import List, Optional, Union

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

from tvm import arith, ir, tir
from .base import CPUScheduleRule

from tvm.target import Target

from .utils import auto_vectorize, get_bytes, get_extent


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


class GEMV(CPUScheduleRule):
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
        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None:
            return None
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
            return self.sch_inner_reduction(sch, target, block, vector_input_buffers, epilogue)
        else:
            # sch_outer reduction
            return None

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
            vector_width: int = 8,
            parallel_threads: int = 8,
            unroll_factor: int = 256
        ):
            batch, s, r, c = sch.get_loops(block)
            len_batch, len_s, len_r, len_c = (
                get_extent(sch, batch),
                get_extent(sch, s),
                get_extent(sch, r),
                get_extent(sch, c),
            )
            len_S = len_batch * len_s
            len_R = len_r * len_c

            if isinstance(len_S, int) and isinstance(len_R, int):
                if len_S > len_R:
                    tile_s, tile_r = 128, 64  # Larger tiling for s-axis when len_S is larger
                else:
                    tile_s, tile_r = 64, 128  # Larger tiling for r-axis when len_R is larger
            else:
                tile_s, tile_r = 64, 64  # Default tile sizes for unknown extents

            tile_c = min(vector_width, len_c)  # Ensure c-axis tiling aligns with SIMD vector width

            # Apply loop tiling (improves cache locality)
            s_outer, s_inner = sch.split(s, factors=[None, tile_s])
            r_outer, r_inner = sch.split(r, factors=[None, tile_r])
            c_outer, c_inner = sch.split(c, factors=[None, tile_c])

            # Apply vectorization (SIMD optimization)
            sch.vectorize(s_inner)  # Vectorize computation along c-axis for AVX/NEON

            # Enable parallel execution
            sch.parallel(s_outer)  # Parallelize along the s-axis (major computation loop)

            # Apply loop unrolling for better CPU performance
            sch.annotate(r_outer, "pragma_auto_unroll_max_step", unroll_factor)
            sch.annotate(r_outer, "pragma_unroll_explicit", 1)
            return sch

        return apply(
            sch,
            gemv=block,
        )
