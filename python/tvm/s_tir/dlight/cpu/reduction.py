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
"""CPU reduction rule for operators including softmax, layer norm, RMS norm, etc."""

from tvm import s_tir, tirx
from tvm.target import Target
from tvm.target.codegen import llvm_get_vector_width

from ..analysis import normalize_prim_func
from ..base import get_extent
from .base import CPUScheduleRule


class Reduction(CPUScheduleRule):
    """CPU reduction rule for softmax, layer norm, RMS norm, and similar operators.

    Targets patterns with a mix of reduction (SR) and injective (SS) blocks,
    where all blocks share the same leading spatial axes.
    Example: softmax = maxelem(SR) -> exp(SS) -> expsum(SR) -> norm(SS).

    Schedule strategy:
      1. Parallelize leading spatial axes (batch dimension).
      2. Move all blocks under the spatial loop via compute_at.
      3. Vectorize injective blocks (exp, delta, norm) on their inner axis.
      4. Split reduction inner axis to VLEN-sized chunks and annotate for
         LLVM unrolling, preventing harmful full-unroll by the backend.

    Note: vectorized reduction via rfactor is not used here because TVM's
    rfactor primitive requires the reduction block to be the first child of
    its enclosing loop, which is incompatible with compute_at when multiple
    blocks share the same spatial loop. A follow-up using RVV reduction
    intrinsics (vfredmax/vfredusum) via tensorize can address this.
    """

    def apply(  # pylint: disable=too-many-locals,too-many-return-statements,too-many-branches
        self,
        func: tirx.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tirx.PrimFunc) or not self.is_target_available(target):
            return None

        sch = s_tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        if block_infos is None or len(block_infos) < 2:
            return None

        # Must have at least one reduction block and last block must be injective.
        has_reduction = any(not bi.is_injective() for bi in block_infos)
        if not has_reduction or not block_infos[-1].is_injective():
            return None

        # All blocks must have at least one leading spatial axis.
        for bi in block_infos:
            dk = bi.dom_kind()
            if not dk or dk[0] != "S":
                return None

        # Find the number of leading spatial axes (from the first reduction block).
        first_reduction = next(bi for bi in block_infos if not bi.is_injective())
        dom_kind = first_reduction.dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        if num_leading_s == 0:
            return None

        # Determine vector width from target.
        try:
            vlen_bits = llvm_get_vector_width(target)
        except Exception:  # pylint: disable=broad-except
            vlen_bits = 128
        dtype_bits = 32  # default float32
        vec_lanes = max(vlen_bits // dtype_bits, 4)

        # --- Phase 1: Parallelize spatial on the last block ---
        last_block = block_infos[-1]
        loops = sch.get_loops(last_block.block_rv)
        if num_leading_s > 1:
            spatial = sch.fuse(*loops[:num_leading_s])
        else:
            spatial = loops[0]
        sch.parallel(spatial)

        # --- Phase 2: Vectorize the last (injective) block ---
        inner_loops = sch.get_loops(last_block.block_rv)
        if len(inner_loops) > 1:
            inner = inner_loops[-1]
            extent = get_extent(sch, inner)
            if isinstance(extent, int) and extent > vec_lanes:
                _, vec_loop = sch.split(inner, factors=[None, vec_lanes])
                sch.vectorize(vec_loop)
            elif isinstance(extent, int):
                sch.vectorize(inner)

        # --- Phase 3: compute_at all preceding blocks under spatial ---
        for block_info in reversed(block_infos[:-1]):
            sch.compute_at(block_info.block_rv, spatial, preserve_unit_loops=True)

        # --- Phase 4: Vectorize injective, split+unroll reduction blocks ---
        for block_info in block_infos[:-1]:
            block = block_info.block_rv
            block_loops = sch.get_loops(block)
            if len(block_loops) <= 1:
                continue
            inner = block_loops[-1]
            extent = get_extent(sch, inner)

            if block_info.is_injective():
                # Injective blocks (e.g. exp, delta): vectorize directly.
                if isinstance(extent, int) and extent > vec_lanes:
                    _, vec_loop = sch.split(inner, factors=[None, vec_lanes])
                    sch.vectorize(vec_loop)
                elif isinstance(extent, int) and extent >= 2:
                    sch.vectorize(inner)
            else:
                # Reduction blocks (e.g. max, sum): split inner to vec_lanes
                # and annotate for unrolling. This prevents LLVM from doing
                # harmful full-unroll of the 185-element loop and gives it
                # a vec_lanes-sized inner loop to auto-vectorize.
                if isinstance(extent, int) and extent > vec_lanes:
                    _, inner_loop = sch.split(inner, factors=[None, vec_lanes])
                    sch.annotate(
                        inner_loop,
                        ann_key="pragma_auto_unroll_max_step",
                        ann_val=vec_lanes,
                    )
                    sch.annotate(
                        inner_loop,
                        ann_key="pragma_unroll_explicit",
                        ann_val=1,
                    )

        return sch
