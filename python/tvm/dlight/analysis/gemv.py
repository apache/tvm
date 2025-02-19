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
"""Analysis for GEMV."""
from typing import List, Optional

from tvm import arith, ir, tir

from .common_analysis import (
    BlockInfo,
    collect_block_iter_vars_used_in_access_region,
    collect_vars_used_in_prim_expr,
    detect_dominant_read,
)


def get_reduction_expr(block: tir.Block) -> Optional[tir.PrimExpr]:
    """Extracts the reduction expression from a TIR block.

    This function checks whether the given TIR block follows a reduction pattern
    of the form `X[...] = X[...] + Y` and returns `Y` as the reduction expression.

    Parameters:
    ----------
    block : tir.Block
        The TIR block to analyze.

    Returns:
    -------
    Optional[tir.PrimExpr]
        The reduction expression (`Y`) if detected, otherwise None.
    """

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
    conditions.append(get_reduction_expr(block_stmt) is not None)
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
