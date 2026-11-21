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
"""dlight schedule for RDNA (gfx11/gfx12) WMMA: fp16 x fp16 -> fp32 GEMMs on wave32.

Applies to a matmul already normalized by `Matmul` into C[S, I, J] += A[S, I, K] * B[S, J, K]
(B in [N, K] layout). The block tile is (16 * TI * WY) x (16 * TJ * WX): WY x WX waves per
workgroup, each owning TI x TJ 16x16 WMMA tiles; K advances in steps of 16 * KS through LDS.
"""

from tvm import s_tir

from ..analysis import get_root_block
from .matmul import (
    auto_inline_consumer_chain,
    auto_inline_producers,
    get_index_map,
    get_reduction_blocks,
)


def is_rdna_target(target) -> bool:
    if target.kind.name != "rocm":
        return False
    mcpu = str(target.attrs.get("mcpu", ""))
    return (
        mcpu.startswith(("gfx11", "gfx12")) and int(target.attrs.get("thread_warp_size", 64)) == 32
    )


def _all_blocks_in_kernels(sch: s_tir.Schedule) -> bool:
    """True if every top-level block runs under a blockIdx binding (nothing left on the host)."""
    for block in sch.get_child_blocks(get_root_block(sch)):
        tags = [
            str(sch.get(lp).annotations.get("thread_binding", "")) for lp in sch.get_loops(block)
        ]
        if not any(tag.startswith("blockIdx") for tag in tags):
            return False
    return True


def _inline_pad_copy(sch: s_tir.Schedule, reindex_c_name: str) -> None:
    """With a dynamic M, pad_einsum leaves a copy out of the padded C buffer in front of the C
    reindex block; fold it into that block so no copy runs outside a kernel."""
    for blk in sch.get_child_blocks(get_root_block(sch)):
        if sch.get(blk).name_hint == reindex_c_name + "_pad":
            sch.compute_inline(blk)


class RocmWmmaMatmul:
    """Tensorized schedule; returns None if the block does not fit (dtype / shape)."""

    def __init__(
        self,
        TI: int = 4,
        TJ: int = 4,
        WY: int = 2,
        WX: int = 2,
        KS: int = 2,
        stage_c_in_shared: bool = False,
    ):
        self.TI, self.TJ, self.WY, self.WX, self.KS = TI, TJ, WY, WX, KS
        # Staging C in LDS avoids an fp32 round trip through global memory but needs
        # 16*16*TI*TJ*WY*WX*4 bytes of LDS (64 KB limit), i.e. smaller tiles.
        self.stage_c_in_shared = stage_c_in_shared

    def apply(self, func, target):
        """Normalize the matmul (as MetalMatmul does), then tensorize; None if not applicable."""
        if not is_rdna_target(target):
            return None
        sch = s_tir.Schedule(func)
        blocks = sch.get_child_blocks(get_root_block(sch))
        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None
        main_block = reduction_blocks[0]
        stmt = sch.get(main_block)
        # Small static M (decode GEMVs) would mostly pad a 16*TI*WY-row tile: leave them to Matmul.
        spatial = [iv.dom.extent for iv in stmt.iter_vars if iv.iter_type == 0]
        if spatial and all(hasattr(e, "value") for e in spatial[:-1]):
            m = 1
            for e in spatial[:-1]:
                m *= int(e.value)
            if m < 64:
                return None
        in_dtype = str(stmt.reads[0].source.ty.dtype)
        out_dtype = str(stmt.writes[0].source.ty.dtype)
        reindex_a = sch.reindex(main_block, ("read", 0))
        reindex_b = sch.reindex(main_block, ("read", 1))
        reindex_c = sch.reindex(main_block, ("write", 0))
        reindex_c_name = sch.get(reindex_c).name_hint
        index_maps = get_index_map(sch.get(main_block))
        if index_maps is None:
            return None
        matmul_map, a_map, b_map, c_map = index_maps
        sch.transform_layout(reindex_a, ("write", 0), a_map)
        sch.transform_layout(reindex_b, ("write", 0), b_map)
        sch.transform_layout(reindex_c, ("read", 0), c_map)
        sch.transform_block_layout(main_block, matmul_map)
        sch.compute_inline(reindex_a)
        sch.compute_inline(reindex_b)
        out = self.schedule(
            sch,
            main_block,
            in_dtype,
            out_dtype,
            reindex_c_name=reindex_c_name if self.stage_c_in_shared else None,
        )
        if out is None:
            return None
        if not self.stage_c_in_shared:  # C reindex as a separate thread-bound copy epilogue
            rc = sch.get_sblock(reindex_c_name)
            _inline_pad_copy(sch, reindex_c_name)
            auto_inline_consumer_chain(sch, rc)  # cast / bias / elementwise epilogues
            fused = sch.fuse(*sch.get_loops(rc))
            bx, tx = sch.split(fused, [None, 256])
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")
        if not _all_blocks_in_kernels(sch):
            return None  # e.g. a prologue that could not be computed in the LDS fill
        return sch

    def schedule(
        self,
        sch: s_tir.Schedule,
        main_block,
        in_dtype: str,
        out_dtype: str,
        reindex_c_name: str | None = None,
    ):
        # pylint: disable=import-outside-toplevel
        from tvm.s_tir.tensor_intrin.rocm_wmma import (
            ROCM_WMMA_F16F16F32_INTRIN,
            ROCM_WMMA_FILL_16x16_F32_INTRIN,
            ROCM_WMMA_LOAD_16x16_A_SHARED_F16_INTRIN,
            ROCM_WMMA_LOAD_16x16_B_TRANS_SHARED_F16_INTRIN,
            ROCM_WMMA_STORE_16x16_F32_INTRIN,
            ROCM_WMMA_STORE_16x16_F32_SHARED_INTRIN,
            shared_16x16_to_local_32x8_layout_C,
            shared_16x16_to_local_32x16_layout_A,
        )

        if in_dtype != "float16" or out_dtype != "float32":
            return None
        TI, TJ, WY, WX, KS = self.TI, self.TJ, self.WY, self.WX, self.KS
        warp = 32
        BM, BN, BK = 16 * TI * WY, 16 * TJ * WX, 16 * KS

        sch.pad_einsum(main_block, [1, BM, BN, BK])
        batch, i, j, k = sch.get_loops(main_block)
        i, i_tc = sch.split(i, [None, 16])
        j, j_tc = sch.split(j, [None, 16])
        k, k_tc = sch.split(k, [None, 16])
        sch.reorder(i, j, k, i_tc, j_tc, k_tc)
        block_inner = main_block
        block_outer = sch.blockize(i_tc)

        i0, i1, i2 = sch.split(i, [None, WY, TI])
        j0, j1, j2 = sch.split(j, [None, WX, TJ])
        k0, k1 = sch.split(k, [None, KS])
        sch.reorder(batch, i0, j0, i1, j1, k0, k1, i2, j2)
        sch.bind(batch, "blockIdx.z")
        block_loop = sch.fuse(i0, j0)
        sch.bind(block_loop, "blockIdx.x")
        wave = sch.fuse(i1, j1)
        sch.bind(wave, "threadIdx.y")

        def fetch_to_shared(idx):
            rd = sch.cache_read(block_outer, idx, "shared")
            sch.compute_at(rd, k0)
            fused = sch.fuse(*sch.get_loops(rd)[-2:])
            _, ty, tx, vec = sch.split(fused, [None, WY * WX, warp, 8])
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            sch.vectorize(vec)
            return rd

        a_g2s = fetch_to_shared(0)
        b_g2s = fetch_to_shared(1)
        # Padding (dynamic M) and prologues such as weight dequantization are computed while
        # filling LDS, once per block tile.
        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        a_loc = sch.cache_read(block_outer, 0, "local")
        b_loc = sch.cache_read(block_outer, 1, "local")
        sch.compute_at(a_loc, k1)
        sch.compute_at(b_loc, k1)
        c_loc = sch.cache_write(block_outer, 0, "local")
        sch.reverse_compute_at(c_loc, wave)
        stage_c_in_shared = reindex_c_name is not None
        if stage_c_in_shared:
            # C reindex copy per wave out of LDS: no fp32 round trip through global memory.
            _inline_pad_copy(sch, reindex_c_name)
            rc = sch.get_sblock(reindex_c_name)
            auto_inline_consumer_chain(sch, rc)
            sch.reverse_compute_at(rc, wave)
            sch.set_scope(c_loc, 0, "shared")
            rc_loops = sch.get_loops(rc)
            wave_stmt = sch.get(wave)
            pos = next(i for i, lp in enumerate(rc_loops) if sch.get(lp).same_as(wave_stmt))
            fused = sch.fuse(*rc_loops[pos + 1 :])
            _, tx = sch.split(fused, [None, 32])
            sch.bind(tx, "threadIdx.x")
        ii, jj = sch.get_loops(c_loc)[-2:]
        io, ii = sch.split(ii, [None, 16])
        jo, ji = sch.split(jj, [None, 16])
        sch.reorder(io, jo, ii, ji)

        init = sch.decompose_reduction(block_outer, k0)

        def frag(blk):
            a, b = sch.get_loops(blk)[-2:]
            a0, a1 = sch.split(a, [None, 16])
            b0, b1 = sch.split(b, [None, 16])
            sch.reorder(a0, b0, a1, b1)
            return a1

        la, lb = frag(a_loc), frag(b_loc)

        def tile(mapping):
            return lambda *idx: (
                *idx[:-2],
                idx[-2] // 16,
                idx[-1] // 16,
                *mapping(idx[-2] % 16, idx[-1] % 16),
            )

        sch.transform_layout(a_loc, ("write", 0), tile(shared_16x16_to_local_32x16_layout_A))
        sch.transform_layout(
            b_loc, ("write", 0), tile(shared_16x16_to_local_32x16_layout_A)
        )  # B is [N, K]
        sch.transform_layout(c_loc, ("read", 0), tile(shared_16x16_to_local_32x8_layout_C))
        sch.tensorize(la, ROCM_WMMA_LOAD_16x16_A_SHARED_F16_INTRIN)
        sch.tensorize(lb, ROCM_WMMA_LOAD_16x16_B_TRANS_SHARED_F16_INTRIN)
        sch.tensorize(sch.get_loops(block_inner)[-3], ROCM_WMMA_F16F16F32_INTRIN)
        inner_init = sch.get_child_blocks(init)
        init_loops = sch.get_loops(inner_init[0] if inner_init else init)
        sch.tensorize(init_loops[-2], ROCM_WMMA_FILL_16x16_F32_INTRIN)
        store = (
            ROCM_WMMA_STORE_16x16_F32_SHARED_INTRIN
            if stage_c_in_shared
            else ROCM_WMMA_STORE_16x16_F32_INTRIN
        )
        sch.tensorize(sch.get_loops(c_loc)[-2], store)
        return sch
