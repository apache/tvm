# SPDX-License-Identifier: MIT
"""DPU register encoding — packs field values into 64-bit register commands.

Block IDs, OP codes, the npuop() encoder, and the gen_* functions that
write register command arrays for CNA+DPU and bias/residual RDMA blocks.
"""

import math

import numpy as np

from .hardware import (
    # PC registers
    PC_OPERATION_ENABLE, PC_BASE_ADDRESS, PC_REGISTER_AMOUNTS,
    # CNA registers
    CNA_CONV_CON1, CNA_CONV_CON2, CNA_CONV_CON3,
    CNA_DATA_SIZE0, CNA_DATA_SIZE1, CNA_DATA_SIZE2, CNA_DATA_SIZE3,
    CNA_WEIGHT_SIZE0, CNA_WEIGHT_SIZE1, CNA_WEIGHT_SIZE2,
    CNA_CBUF_CON0, CNA_CBUF_CON1,
    CNA_CVT_CON0, CNA_CVT_CON1, CNA_CVT_CON2, CNA_CVT_CON3, CNA_CVT_CON4, CNA_CVT_CON5,
    CNA_FC_CON0, CNA_FC_CON1, CNA_FC_CON2,
    CNA_PAD_CON0, CNA_PAD_CON1,
    CNA_FEATURE_DATA_ADDR,
    CNA_DMA_CON0, CNA_DMA_CON1, CNA_DMA_CON2,
    CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE1,
    CNA_DCOMP_CTRL, CNA_DCOMP_REGNUM, CNA_DCOMP_ADDR0,
    CNA_DCOMP_AMOUNT, CNA_DCOMP_AMOUNT15,
    # CORE registers
    CORE_MISC_CFG, CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_1,
    CORE_CLIP_TRUNCATE, CORE_3030,
    # DPU registers
    DPU_S_POINTER, DPU_FEATURE_MODE_CFG, DPU_DATA_FORMAT, DPU_OFFSET_PEND,
    DPU_DST_BASE_ADD, DPU_DST_SURF_STRIDE,
    DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_NOTCH, DPU_DATA_CUBE_CHANNEL,
    DPU_BS_CFG, DPU_BS_ALU_CFG, DPU_BS_MUL_CFG, DPU_BS_RELUX_CMP,
    DPU_BS_OW_CFG, DPU_BS_OW_OP,
    DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_1,
    DPU_BN_CFG, DPU_BN_ALU_CFG, DPU_BN_MUL_CFG, DPU_BN_RELUX_CMP,
    DPU_EW_CFG, DPU_EW_CVT_OFFSET, DPU_EW_CVT_SCALE, DPU_EW_RELUX_CMP,
    DPU_OUT_CVT_OFFSET, DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SHIFT,
    DPU_EW_OP_VALUE_0, DPU_EW_OP_VALUE_7,
    DPU_SURFACE_ADD, DPU_40C4,
    DPU_LUT_ACCESS_CFG, DPU_LUT_ACCESS_DATA, DPU_LUT_CFG, DPU_LUT_INFO,
    DPU_LUT_LE_START, DPU_LUT_LE_END, DPU_LUT_LO_START, DPU_LUT_LO_END,
    DPU_LUT_LE_SLOPE_SCALE, DPU_LUT_LE_SLOPE_SHIFT,
    DPU_LUT_LO_SLOPE_SCALE, DPU_LUT_LO_SLOPE_SHIFT,
    # DPU_RDMA registers
    DPU_RDMA_S_POINTER, DPU_RDMA_500C, DPU_RDMA_5010, DPU_RDMA_5014,
    DPU_RDMA_SRC_A_ADDR, DPU_RDMA_501C, DPU_RDMA_5020,
    DPU_RDMA_5028, DPU_RDMA_502C, DPU_RDMA_5034,
    DPU_RDMA_EW_SRC_ADDR, DPU_RDMA_EW_SURF_STRIDE,
    DPU_RDMA_LINE_STRIDE, DPU_RDMA_SURF_STRIDE,
    DPU_RDMA_504C, DPU_RDMA_5064, DPU_RDMA_BURST_CFG, DPU_RDMA_506C,
    # Task masks
    TASK_EW_ENABLE_MASK,
    TASK_ENABLE_MASK_MODE6,
    REGCMD_COUNT_EW,
    REGCMD_COUNT_LUT_UPLOAD,
    REGCMD_COUNT_LUT_COMBINED,
    REGCFG_AMOUNT_LUT_UPLOAD,
    REGCFG_AMOUNT_EW,
)
from .alignment import align_up, pad_m
from ._cna_regcfg import CnaDesc, CoreDesc, DpuDesc


# =============================================================================
# Block IDs and Op Codes
# =============================================================================

BLOCK_PC = 0x0100
BLOCK_CNA = 0x0200
BLOCK_CORE = 0x0800
BLOCK_DPU = 0x1000
BLOCK_DPU_RDMA = 0x2000
BLOCK_PPU = 0x4000
BLOCK_PPU_RDMA = 0x8000

OP_01 = 0x01
OP_40 = 0x41       # bit 6 + bit 0
OP_ENABLE = 0x81   # bit 7 + bit 0
OP_NONE = 0x0000

OP_REG_PC = BLOCK_PC | OP_01
OP_REG_CNA = BLOCK_CNA | OP_01
OP_REG_CORE = BLOCK_CORE | OP_01
OP_REG_DPU = BLOCK_DPU | OP_01
OP_REG_DPU_RDMA = BLOCK_DPU_RDMA | OP_01
OP_REG_PPU = BLOCK_PPU | OP_01
OP_REG_PPU_RDMA = BLOCK_PPU_RDMA | OP_01

# PC_OPERATION_ENABLE flags
PC_ENABLE = 0x01
PC_ENABLE_CNA = 0x04
PC_ENABLE_DPU = 0x08
PC_ENABLE_DPU_RDMA = 0x10


# =============================================================================
# Encoding Helpers
# =============================================================================

def npuop(op: int, value: int, reg: int) -> int:
    """Encode a 64-bit NPU register command.

    Format: [op:16][value:32][reg:16]
    """
    return (
        ((op & 0xFFFF) << 48)
        | ((value & 0xFFFFFFFF) << 16)
        | (reg & 0xFFFF)
    )


def _write_pc_tail(ops: np.ndarray, idx: int, enable_mask: int) -> None:
    """Write the 4-command PC control tail starting at ops[idx]."""
    ops[idx]     = npuop(OP_NONE, 0, 0)
    ops[idx + 1] = npuop(OP_REG_PC, 0, PC_REGISTER_AMOUNTS)
    ops[idx + 2] = npuop(OP_40, 0, 0)
    ops[idx + 3] = npuop(OP_ENABLE, enable_mask, PC_OPERATION_ENABLE)


# LUT register list used by gen_cna_dpu_regcmds.
_LUT_REGS = [
    DPU_LUT_ACCESS_CFG, DPU_LUT_ACCESS_DATA, DPU_LUT_CFG, DPU_LUT_INFO,
    DPU_LUT_LE_START, DPU_LUT_LE_END, DPU_LUT_LO_START, DPU_LUT_LO_END,
    DPU_LUT_LE_SLOPE_SCALE, DPU_LUT_LE_SLOPE_SHIFT,
    DPU_LUT_LO_SLOPE_SCALE, DPU_LUT_LO_SLOPE_SHIFT,
]


# =============================================================================
# gen_cna_dpu_regcmds
# =============================================================================

def gen_cna_dpu_regcmds(ops: np.ndarray, cna: CnaDesc, core: CoreDesc, dpu: DpuDesc):
    """Write 112 register commands for CNA + CORE + DPU into ops array."""
    # ---- CNA preamble (ops[0..3]) ----
    v = ((cna.weight_bank & 0xF) << 4) | (cna.data_bank & 0xF)
    ops[0] = npuop(OP_REG_CNA, v, CNA_CBUF_CON0)
    ops[1] = npuop(OP_REG_CNA, 0, CNA_DCOMP_REGNUM)
    ops[2] = npuop(OP_REG_CNA, 0, CNA_DCOMP_CTRL)
    v = ((cna.proc_precision & 0x7) << 7) | ((cna.in_precision & 0x7) << 4) | (cna.conv_mode & 0xF)
    ops[3] = npuop(OP_REG_CNA, v, CNA_CONV_CON1)

    # ---- DPU pointer (ops[4]) ----
    ops[4] = npuop(OP_REG_DPU, 0x0E, DPU_S_POINTER)

    # ---- CNA configuration (ops[5..52]) ----
    v = ((cna.proc_precision & 0x7) << 7) | ((cna.in_precision & 0x7) << 4) | (cna.conv_mode & 0xF)
    ops[5] = npuop(OP_REG_CNA, v, CNA_CONV_CON1)

    v = ((cna.kernel_groups & 0xFF) << 16) | ((cna.feature_grains & 0x3FF) << 4)
    ops[6] = npuop(OP_REG_CNA, v, CNA_CONV_CON2)

    v = ((cna.conv_y_stride & 0x7) << 3) | (cna.conv_x_stride & 0x7)
    ops[7] = npuop(OP_REG_CNA, v, CNA_CONV_CON3)

    v = ((cna.datain_width & 0x7FF) << 16) | (cna.datain_height & 0x7FF)
    ops[8] = npuop(OP_REG_CNA, v, CNA_DATA_SIZE0)

    upper = cna.datain_channel_upper if cna.datain_channel_upper is not None else (cna.datain_channel - 1)
    v = ((upper & 0xFFFF) << 16) | (cna.datain_channel & 0xFFFF)
    ops[9] = npuop(OP_REG_CNA, v, CNA_DATA_SIZE1)

    ops[10] = npuop(OP_REG_CNA, cna.dataout_width & 0x7FF, CNA_DATA_SIZE2)
    ops[11] = npuop(OP_REG_CNA, cna.dataout_atomics & 0x3FFFF, CNA_DATA_SIZE3)
    ops[12] = npuop(OP_REG_CNA, cna.weight_bytes & 0xFFFFFFFF, CNA_WEIGHT_SIZE0)
    ops[13] = npuop(OP_REG_CNA, cna.weight_bytes_per_kernel & 0x7FFFF, CNA_WEIGHT_SIZE1)

    v = ((cna.weight_width & 0x1F) << 24) | ((cna.weight_height & 0x1F) << 16) | (cna.weight_kernels & 0x3FFF)
    ops[14] = npuop(OP_REG_CNA, v, CNA_WEIGHT_SIZE2)

    v = ((cna.weight_bank & 0xF) << 4) | (cna.data_bank & 0xF)
    ops[15] = npuop(OP_REG_CNA, v, CNA_CBUF_CON0)

    ops[16] = npuop(OP_REG_CNA, cna.data_entries & 0x1FFF, CNA_CBUF_CON1)
    ops[17] = npuop(OP_REG_CNA, 0x0B, CNA_CVT_CON0)
    ops[18] = npuop(OP_REG_CNA, 0x00010000, CNA_CVT_CON1)
    ops[19] = npuop(OP_REG_CNA, 0x00010000, CNA_CVT_CON2)
    ops[20] = npuop(OP_REG_CNA, 0x00010000, CNA_CVT_CON3)
    ops[21] = npuop(OP_REG_CNA, 0x00010000, CNA_CVT_CON4)
    ops[22] = npuop(OP_REG_CNA, 0, CNA_FC_CON0)
    ops[23] = npuop(OP_REG_CNA, 0, CNA_FC_CON1)

    v = ((cna.pad_left & 0xF) << 4) | (cna.pad_top & 0xF)
    ops[24] = npuop(OP_REG_CNA, v, CNA_PAD_CON0)

    ops[25] = npuop(OP_REG_CNA, cna.feature_base_addr & 0xFFFFFFFF, CNA_FEATURE_DATA_ADDR)
    ops[26] = npuop(OP_REG_CNA, 0, CNA_FC_CON2)
    ops[27] = npuop(OP_REG_CNA, 0x000F000F, CNA_DMA_CON0)
    ops[28] = npuop(OP_REG_CNA, cna.line_stride & 0xFFFFFFF, CNA_DMA_CON1)
    ops[29] = npuop(OP_REG_CNA, cna.surf_stride & 0xFFFFFFF, CNA_DMA_CON2)

    v = ((cna.dma_width & 0x7FF) << 16) | (cna.dma_height & 0x7FF)
    ops[30] = npuop(OP_REG_CNA, v, CNA_FC_DATA_SIZE0)

    ops[31] = npuop(OP_REG_CNA, cna.dma_channel & 0xFFFF, CNA_FC_DATA_SIZE1)
    ops[32] = npuop(OP_REG_CNA, 0, CNA_DCOMP_CTRL)
    ops[33] = npuop(OP_REG_CNA, 0, CNA_DCOMP_REGNUM)
    ops[34] = npuop(OP_REG_CNA, cna.decompress_addr0 & 0xFFFFFFFF, CNA_DCOMP_ADDR0)
    for i, reg in enumerate(range(CNA_DCOMP_AMOUNT, CNA_DCOMP_AMOUNT15 + 4, 4)):
        ops[35 + i] = npuop(OP_REG_CNA, 0, reg)
    ops[51] = npuop(OP_REG_CNA, 0, CNA_CVT_CON5)

    v = ((cna.pad_right & 0xF) << 4) | (cna.pad_bottom & 0xF)
    ops[52] = npuop(OP_REG_CNA, v, CNA_PAD_CON1)

    # ---- CORE configuration (ops[53..57]) ----
    v = ((core.proc_precision & 0x7) << 8) | ((core.dw_flag & 0x1) << 1) | (core.qd_en & 0x1)
    if cna.conv_mode == 3:
        v |= (1 << 1)
    ops[53] = npuop(OP_REG_CORE, v, CORE_MISC_CFG)

    v = ((core.dataout_height & 0xFFFF) << 16) | (core.dataout_width & 0xFFFF)
    ops[54] = npuop(OP_REG_CORE, v, CORE_DATAOUT_SIZE_0)

    ops[55] = npuop(OP_REG_CORE, core.dataout_channel & 0xFFFF, CORE_DATAOUT_SIZE_1)
    ops[56] = npuop(OP_REG_CORE, 0, CORE_CLIP_TRUNCATE)
    ops[57] = npuop(OP_REG_CORE, 0, CORE_3030)

    # ---- DPU configuration (ops[58..107]) ----
    v = 0x1E4 | ((dpu.conv_mode & 0x3) << 3)
    ops[58] = npuop(OP_REG_DPU, v, DPU_FEATURE_MODE_CFG)

    v = ((dpu.out_precision & 0x7) << 29) | ((dpu.in_precision & 0x7) << 26) | (dpu.proc_precision & 0x7)
    ops[59] = npuop(OP_REG_DPU, v, DPU_DATA_FORMAT)

    ops[60] = npuop(OP_REG_DPU, 0, DPU_OFFSET_PEND)
    ops[61] = npuop(OP_REG_DPU, dpu.dst_base_addr & 0xFFFFFFFF, DPU_DST_BASE_ADD)
    ops[62] = npuop(OP_REG_DPU, (dpu.dst_surf_stride & 0xFFFFFFF) << 4, DPU_DST_SURF_STRIDE)
    ops[63] = npuop(OP_REG_DPU, dpu.width & 0x1FFF, DPU_DATA_CUBE_WIDTH)
    ops[64] = npuop(OP_REG_DPU, dpu.height & 0x1FFF, DPU_DATA_CUBE_HEIGHT)
    ops[65] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_NOTCH)

    v = ((dpu.channel & 0x1FFF) << 16) | (dpu.channel & 0x1FFF)
    ops[66] = npuop(OP_REG_DPU, v, DPU_DATA_CUBE_CHANNEL)

    # BS (Batch Scale)
    v = ((dpu.bs_relu_bypass & 0x1) << 6) | ((dpu.bs_mul_bypass & 0x1) << 4) | ((dpu.bs_alu_bypass & 0x1) << 1) | (dpu.bs_bypass & 0x1)
    if not dpu.bs_bypass:
        v |= (1 << 8) | (1 << 17)
    ops[67] = npuop(OP_REG_DPU, v, DPU_BS_CFG)
    ops[68] = npuop(OP_REG_DPU, 0, DPU_BS_ALU_CFG)
    ops[69] = npuop(OP_REG_DPU, 0, DPU_BS_MUL_CFG)
    ops[70] = npuop(OP_REG_DPU, 0, DPU_BS_RELUX_CMP)

    v = ((dpu.size_e_2 & 0x7) << 8) | ((dpu.size_e_1 & 0x7) << 5) | ((dpu.size_e_0 & 0x7) << 2) | (1 << 1)
    ops[71] = npuop(OP_REG_DPU, v, DPU_BS_OW_CFG)
    ops[72] = npuop(OP_REG_DPU, 0, DPU_BS_OW_OP)

    ops[73] = npuop(OP_REG_DPU, dpu.channel_wdma & 0x1FFF, DPU_WDMA_SIZE_0)
    v = ((dpu.height_wdma & 0x1FFF) << 16) | (dpu.width_wdma & 0x1FFF)
    ops[74] = npuop(OP_REG_DPU, v, DPU_WDMA_SIZE_1)

    # BN (Batch Norm)
    v = (
        ((dpu.bn_relux_compare_en & 0x1) << 7)
        | ((dpu.bn_relu_bypass & 0x1) << 6)
        | ((dpu.bn_mul_bypass & 0x1) << 4)
        | ((dpu.bn_alu_bypass & 0x1) << 1)
        | (dpu.bn_bypass & 0x1)
    )
    if dpu.bn_dw_mode:
        v |= 0x80
    ops[75] = npuop(OP_REG_DPU, v, DPU_BN_CFG)
    ops[76] = npuop(OP_REG_DPU, 0, DPU_BN_ALU_CFG)
    ops[77] = npuop(OP_REG_DPU, 0, DPU_BN_MUL_CFG)
    ops[78] = npuop(OP_REG_DPU, dpu.bn_relux_cmp & 0xFFFFFFFF, DPU_BN_RELUX_CMP)

    # EW (Element-Wise) — all bypass unless overridden
    ops[79] = npuop(OP_REG_DPU, dpu.ew_cfg_override or 0x383, DPU_EW_CFG)
    ops[80] = npuop(OP_REG_DPU, 0, DPU_EW_CVT_OFFSET)
    ops[81] = npuop(OP_REG_DPU, 1, DPU_EW_CVT_SCALE)
    ops[82] = npuop(OP_REG_DPU, 0, DPU_EW_RELUX_CMP)

    # Output conversion
    ops[83] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_OFFSET)
    v = ((dpu.fp32tofp16_en & 0x1) << 16) | 1
    ops[84] = npuop(OP_REG_DPU, v, DPU_OUT_CVT_SCALE)
    ops[85] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_SHIFT)

    # EW operand values (all zero)
    for i, reg in enumerate(range(DPU_EW_OP_VALUE_0, DPU_EW_OP_VALUE_7 + 4, 4)):
        ops[86 + i] = npuop(OP_REG_DPU, 0, reg)

    ops[94] = npuop(OP_REG_DPU, (dpu.surf_add & 0xFFFFFFF) << 4, DPU_SURFACE_ADD)
    ops[95] = npuop(OP_REG_DPU, 0, DPU_40C4)

    # LUT configuration — all zero
    for i, reg in enumerate(_LUT_REGS):
        ops[96 + i] = npuop(OP_REG_DPU, 0, reg)

    # ---- PC control tail (ops[108..111]) ----
    _write_pc_tail(ops, 108, PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE)


# =============================================================================
# EW (Elementwise) Operations
# =============================================================================

# EW operation configuration table (from codebook RE and RKNN-Lite probes)
_EW_OP_TABLE = {
    'add': {
        'feature_mode_cfg': 0x000001E5,
        'data_format': 0x48000002,
        'ew_cfg': 0x108202C0,
        'out_cvt_scale': 0x00010001,
        'rdma_5034': 0x40000008,
    },
    'mul': {
        'feature_mode_cfg': 0x000001E5,
        'data_format': 0x48000002,
        'ew_cfg': 0x108003C4,
        'out_cvt_scale': 0x00010001,
        'rdma_5034': 0x40000008,
    },
}


def gen_ew_op(
    ops: np.ndarray,
    op: str,
    cube_w: int,
    cube_h: int,
    cube_c: int,
    src_a_dma: int,
    src_b_dma: int,
    dst_dma: int,
    surf_stride: int,
    ew_surf_stride: int | None = None,
    relu: bool = False,
) -> None:
    """Generate 73 register commands for DPU+DPU_RDMA elementwise operation.

    Parameters
    ----------
    ops : numpy uint64 array of length REGCMD_COUNT_EW
    op : str
        'add' or 'mul'.
    cube_w, cube_h, cube_c : int
        0-based cube dimensions (value-1). E.g. width=C means cube_w=C-1.
    src_a_dma, src_b_dma, dst_dma : int
        DMA addresses for the two inputs and the output.
    surf_stride : int
        Bytes between channel surfaces.
    ew_surf_stride : int or None
        Override for DPU_RDMA_EW_SURF_STRIDE. Set to 0 for broadcast mode
        (broadcasts channel group 0 of source B across all output groups).
        None uses surf_stride (normal mode).
    """
    if ew_surf_stride is None:
        ew_surf_stride = surf_stride
    if op not in _EW_OP_TABLE:
        raise ValueError(f"Unknown op '{op}', expected 'add' or 'mul'")
    cfg = _EW_OP_TABLE[op]

    i = 0

    # ---- S_POINTERs ----
    ops[i] = npuop(OP_REG_DPU, 0x0E, DPU_S_POINTER); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x0E, DPU_RDMA_S_POINTER); i += 1

    # ---- DPU configuration ----
    ops[i] = npuop(OP_REG_DPU, cfg['feature_mode_cfg'], DPU_FEATURE_MODE_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, cfg['data_format'], DPU_DATA_FORMAT); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_OFFSET_PEND); i += 1
    ops[i] = npuop(OP_REG_DPU, dst_dma & 0xFFFFFFFF, DPU_DST_BASE_ADD); i += 1
    ops[i] = npuop(OP_REG_DPU, surf_stride & 0xFFFFFFFF, DPU_DST_SURF_STRIDE); i += 1

    # Cube dimensions (0-based)
    ops[i] = npuop(OP_REG_DPU, cube_w & 0x1FFF, DPU_DATA_CUBE_WIDTH); i += 1
    ops[i] = npuop(OP_REG_DPU, cube_h & 0x1FFF, DPU_DATA_CUBE_HEIGHT); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_NOTCH); i += 1
    cube_channel = ((cube_c & 0x1FFF) << 16) | (cube_c & 0x1FFF)
    ops[i] = npuop(OP_REG_DPU, cube_channel, DPU_DATA_CUBE_CHANNEL); i += 1

    # BS — bypassed
    ops[i] = npuop(OP_REG_DPU, 0x53, DPU_BS_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_ALU_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_MUL_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_RELUX_CMP); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x02, DPU_BS_OW_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_OW_OP); i += 1

    # WDMA size: size_0 = channels, size_1 = height<<16 | width
    ops[i] = npuop(OP_REG_DPU, cube_c & 0xFFFF, DPU_WDMA_SIZE_0); i += 1
    wdma_1 = ((cube_h & 0x1FFF) << 16) | (cube_w & 0x1FFF)
    ops[i] = npuop(OP_REG_DPU, wdma_1, DPU_WDMA_SIZE_1); i += 1

    # BN: optionally enable ReLU post-op for add+relu/relu stages.
    bn_cfg = 0x12 if relu else 0x53
    ops[i] = npuop(OP_REG_DPU, bn_cfg, DPU_BN_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BN_ALU_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BN_MUL_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BN_RELUX_CMP); i += 1

    # EW — ACTIVE
    ops[i] = npuop(OP_REG_DPU, cfg['ew_cfg'], DPU_EW_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_EW_CVT_OFFSET); i += 1
    ops[i] = npuop(OP_REG_DPU, 1, DPU_EW_CVT_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_EW_RELUX_CMP); i += 1

    # Output conversion
    ops[i] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_OFFSET); i += 1
    ops[i] = npuop(OP_REG_DPU, cfg['out_cvt_scale'], DPU_OUT_CVT_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_SHIFT); i += 1

    # EW operand values (all zero)
    for reg in range(DPU_EW_OP_VALUE_0, DPU_EW_OP_VALUE_7 + 4, 4):
        ops[i] = npuop(OP_REG_DPU, 0, reg); i += 1

    # Surface address
    ops[i] = npuop(OP_REG_DPU, surf_stride & 0xFFFFFFFF, DPU_SURFACE_ADD); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_40C4); i += 1

    # LUT (unused)
    for reg in _LUT_REGS:
        ops[i] = npuop(OP_REG_DPU, 0, reg); i += 1

    # ---- DPU_RDMA configuration (17 registers) ----
    ops[i] = npuop(OP_REG_DPU_RDMA, cube_w & 0x1FFF, DPU_RDMA_500C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, cube_h & 0x1FFF, DPU_RDMA_5010); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, cube_c & 0x1FFF, DPU_RDMA_5014); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, src_a_dma & 0xFFFFFFFF, DPU_RDMA_SRC_A_ADDR); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_501C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5020); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5028); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_502C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, cfg['rdma_5034'], DPU_RDMA_5034); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, src_b_dma & 0xFFFFFFFF, DPU_RDMA_EW_SRC_ADDR); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, ew_surf_stride & 0xFFFFFFFF, DPU_RDMA_EW_SURF_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x17849, DPU_RDMA_LINE_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_SURF_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_504C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5064); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x01010101, DPU_RDMA_BURST_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_506C); i += 1
    ops[i] = npuop(OP_NONE, 0, 0); i += 1  # Padding NOP (keep count even for PC chain)

    # ---- PC control tail (4 commands) ----
    _write_pc_tail(ops, i, TASK_EW_ENABLE_MASK)
    i += 4

    assert i == REGCMD_COUNT_EW, f"Expected {REGCMD_COUNT_EW} commands, wrote {i}"


# =============================================================================
# LUT (Lookup Table) Operations — exp(), sigmoid(), GELU
# =============================================================================

def gen_lut_upload(
    ops: np.ndarray,
    le_table: list[int],
    lo_table: list[int],
) -> int:
    """Generate register commands to upload LUT tables to DPU SRAM.

    Writes a 69-command DPU preamble (resets all DPU registers), then uploads
    513 LE table entries and 513 LO table entries via DPU_LUT_ACCESS_DATA,
    finishing with a 4-command PC tail.

    Total: REGCMD_COUNT_LUT_UPLOAD (1102) commands.

    Parameters
    ----------
    ops : numpy uint64 array of length >= REGCMD_COUNT_LUT_UPLOAD
    le_table : list of 513 uint16 values (Linear/Exponent table)
    lo_table : list of 513 uint16 values (Linear Offset table)

    Returns
    -------
    int : number of commands written
    """
    assert len(le_table) == 513, f"LE table must have 513 entries, got {len(le_table)}"
    assert len(lo_table) == 513, f"LO table must have 513 entries, got {len(lo_table)}"

    i = 0

    # ---- Preamble: 69 DPU + DPU_RDMA register reset ----
    # (Matches trace Task 0 structure exactly)

    # S_POINTERs
    ops[i] = npuop(OP_REG_DPU, 0x0E, DPU_S_POINTER); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x0E, DPU_RDMA_S_POINTER); i += 1

    # DPU configuration (all bypass/reset)
    ops[i] = npuop(OP_REG_DPU, 0x000001E5, DPU_FEATURE_MODE_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x48000002, DPU_DATA_FORMAT); i += 1  # FP16 output format
    ops[i] = npuop(OP_REG_DPU, 0, DPU_OFFSET_PEND); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_DST_BASE_ADD); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x10, DPU_DST_SURF_STRIDE); i += 1

    # Cube dimensions (minimal: 1x1 with 16 channels)
    ops[i] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_WIDTH); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_HEIGHT); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_NOTCH); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x000F000F, DPU_DATA_CUBE_CHANNEL); i += 1

    # BS — bypassed
    ops[i] = npuop(OP_REG_DPU, 0x53, DPU_BS_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_ALU_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_MUL_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_RELUX_CMP); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x02, DPU_BS_OW_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_OW_OP); i += 1

    # WDMA size
    ops[i] = npuop(OP_REG_DPU, 0x0F, DPU_WDMA_SIZE_0); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_WDMA_SIZE_1); i += 1

    # BN — bypassed
    ops[i] = npuop(OP_REG_DPU, 0x53, DPU_BN_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BN_ALU_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BN_MUL_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BN_RELUX_CMP); i += 1

    # EW — bypassed
    ops[i] = npuop(OP_REG_DPU, 0x383, DPU_EW_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_EW_CVT_OFFSET); i += 1
    ops[i] = npuop(OP_REG_DPU, 1, DPU_EW_CVT_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_EW_RELUX_CMP); i += 1

    # Output conversion
    ops[i] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_OFFSET); i += 1
    ops[i] = npuop(OP_REG_DPU, 1, DPU_OUT_CVT_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_SHIFT); i += 1

    # EW operand values (all zero)
    for reg in range(DPU_EW_OP_VALUE_0, DPU_EW_OP_VALUE_7 + 4, 4):
        ops[i] = npuop(OP_REG_DPU, 0, reg); i += 1

    # Surface address
    ops[i] = npuop(OP_REG_DPU, 0x10, DPU_SURFACE_ADD); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_40C4); i += 1

    # LUT registers — all reset to 0
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_ACCESS_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_ACCESS_DATA); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_INFO); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LE_START); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LE_END); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LO_START); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LO_END); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LE_SLOPE_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LE_SLOPE_SHIFT); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LO_SLOPE_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LO_SLOPE_SHIFT); i += 1

    # DPU_RDMA configuration
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_500C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5010); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x0F, DPU_RDMA_5014); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_SRC_A_ADDR); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_501C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5020); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5028); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_502C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x01, DPU_RDMA_5034); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_EW_SRC_ADDR); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_EW_SURF_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x7801, DPU_RDMA_LINE_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_SURF_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_504C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5064); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x01010101, DPU_RDMA_BURST_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_506C); i += 1

    assert i == 69, f"Preamble expected 69 commands, wrote {i}"

    # ---- LUT data upload: 1028 commands ----

    # Enable LE table write mode (bit 17=write enable, bit 16=0 for LE)
    ops[i] = npuop(OP_REG_DPU, 0x00020000, DPU_LUT_ACCESS_CFG); i += 1

    # Upload LE table (513 entries)
    for entry in le_table:
        ops[i] = npuop(OP_REG_DPU, entry & 0xFFFF, DPU_LUT_ACCESS_DATA); i += 1

    # Switch to LO table
    ops[i] = npuop(OP_REG_DPU, 0x00030000, DPU_LUT_ACCESS_CFG); i += 1

    # Upload LO table (513 entries)
    for entry in lo_table:
        ops[i] = npuop(OP_REG_DPU, entry & 0xFFFF, DPU_LUT_ACCESS_DATA); i += 1

    assert i == 69 + 1028, f"Data section expected 1028 commands, got {i - 69}"

    # ---- NOP padding (1 command) for even alignment ----
    # PC data amount uses scale=2 on RK3588, so total must be even.
    ops[i] = 0; i += 1

    # ---- PC control tail (4 commands) ----
    _write_pc_tail(ops, i, TASK_EW_ENABLE_MASK)
    i += 4

    assert i == REGCMD_COUNT_LUT_UPLOAD, f"Expected {REGCMD_COUNT_LUT_UPLOAD}, wrote {i}"
    return i


def gen_lut_combined(
    ops: np.ndarray,
    le_table: list[int],
    lo_table: list[int],
    cube_w: int,
    cube_h: int,
    cube_c: int,
    src_dma: int,
    dst_dma: int,
    surf_stride: int,
    bn_mul_cfg: int = 0x68000000,
    ew_cfg: int = 0x00000302,
    out_cvt_offset: int = 0x00000001,
    out_cvt_shift: int = 0x0000F000,
    lut_le_slope_scale: int = 0,
    lut_le_slope_shift: int = 0,
    lut_lo_slope_scale: int = 0,
    lut_lo_slope_shift: int = 0,
) -> int:
    """Generate combined LUT upload + eval in a single task.

    Structure: eval config (69) + LUT data (1028) + NOP (1) + PC tail (4) = 1102 total.

    The eval DPU/RDMA configuration comes FIRST (sets real addresses, active
    BN prescale, EW LUT mode, LUT range params), then the LUT tables are
    programmed into DPU SRAM via ACCESS_DATA writes, then the PC tail triggers
    OP_ENABLE to start data processing through the loaded LUT.

    Total: REGCMD_COUNT_LUT_COMBINED (1102) commands.
    """
    assert len(le_table) == 513, f"LE table must have 513 entries, got {len(le_table)}"
    assert len(lo_table) == 513, f"LO table must have 513 entries, got {len(lo_table)}"

    # [0-68]: DPU/RDMA eval config (69 commands, no PC tail)
    # Use gen_lut_eval to write the first 69 data commands
    temp_ops = np.zeros(_REGCMD_COUNT_LUT_EVAL, dtype=np.uint64)
    gen_lut_eval(
        temp_ops,
        cube_w=cube_w, cube_h=cube_h, cube_c=cube_c,
        src_dma=src_dma, dst_dma=dst_dma, surf_stride=surf_stride,
        bn_mul_cfg=bn_mul_cfg, ew_cfg=ew_cfg,
        out_cvt_offset=out_cvt_offset, out_cvt_shift=out_cvt_shift,
        lut_le_slope_scale=lut_le_slope_scale,
        lut_le_slope_shift=lut_le_slope_shift,
        lut_lo_slope_scale=lut_lo_slope_scale,
        lut_lo_slope_shift=lut_lo_slope_shift,
    )
    ops[:69] = temp_ops[:69]  # Copy only the 69 data commands, skip PC tail

    # [69-1096]: LUT data upload (1028 commands)
    i = 69
    # LE table (1 access_cfg + 513 data entries)
    ops[i] = npuop(OP_REG_DPU, 0x00020000, DPU_LUT_ACCESS_CFG); i += 1
    for entry in le_table:
        ops[i] = npuop(OP_REG_DPU, int(entry) & 0xFFFFFFFF, DPU_LUT_ACCESS_DATA); i += 1
    # LO table (1 access_cfg + 513 data entries)
    ops[i] = npuop(OP_REG_DPU, 0x00030000, DPU_LUT_ACCESS_CFG); i += 1
    for entry in lo_table:
        ops[i] = npuop(OP_REG_DPU, int(entry) & 0xFFFFFFFF, DPU_LUT_ACCESS_DATA); i += 1

    assert i == 1097, f"Expected 1097 data commands, wrote {i}"

    # [1097]: NOP padding for even alignment (PC data amount uses scale=2).
    ops[i] = 0; i += 1

    # [1098-1101]: PC control tail (4 commands)
    _write_pc_tail(ops, i, TASK_EW_ENABLE_MASK)
    i += 4

    assert i == REGCMD_COUNT_LUT_COMBINED, f"Expected {REGCMD_COUNT_LUT_COMBINED}, wrote {i}"
    return i


_REGCMD_COUNT_LUT_EVAL = 73   # 69 data + 4 PC tail


def gen_lut_eval(
    ops: np.ndarray,
    cube_w: int,
    cube_h: int,
    cube_c: int,
    src_dma: int,
    dst_dma: int,
    surf_stride: int,
    bn_mul_cfg: int = 0x68000000,
    ew_cfg: int = 0x00000302,
    out_cvt_offset: int = 0x00000001,
    out_cvt_shift: int = 0x0000F000,
    lut_le_slope_scale: int = 0,
    lut_le_slope_shift: int = 0,
    lut_lo_slope_scale: int = 0,
    lut_lo_slope_shift: int = 0,
) -> int:
    """Generate 73 register commands for DPU LUT evaluation (exp/sigmoid/GELU).

    Applies previously uploaded LUT to input data through the DPU pipeline:
      BN prescale (bn_mul_cfg) → LUT lookup → output conversion

    The LUT tables must be uploaded first (via gen_lut_upload or combined task).

    Parameters
    ----------
    ops : numpy uint64 array of length >= 73
    cube_w, cube_h, cube_c : int
        0-based cube dimensions (value-1).
    src_dma, dst_dma : int
        DMA addresses for input and output.
    surf_stride : int
        Bytes between channel surfaces.
    bn_mul_cfg : int
        BN multiply prescale value. Maps input range to LUT index range.
    ew_cfg : int
        EW configuration for LUT mode.
    out_cvt_offset, out_cvt_shift : int
        Output conversion parameters.
    lut_le_slope_scale, lut_le_slope_shift : int
        LE underflow extrapolation slope (0 = clamp to table edge).
    lut_lo_slope_scale, lut_lo_slope_shift : int
        LO overflow extrapolation slope (0 = clamp to table edge).

    Returns
    -------
    int : number of commands written (73)
    """
    i = 0

    # ---- S_POINTERs ----
    ops[i] = npuop(OP_REG_DPU, 0x0E, DPU_S_POINTER); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x0E, DPU_RDMA_S_POINTER); i += 1

    # ---- DPU configuration ----
    ops[i] = npuop(OP_REG_DPU, 0x000001E5, DPU_FEATURE_MODE_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x48000002, DPU_DATA_FORMAT); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_OFFSET_PEND); i += 1
    ops[i] = npuop(OP_REG_DPU, dst_dma & 0xFFFFFFFF, DPU_DST_BASE_ADD); i += 1
    ops[i] = npuop(OP_REG_DPU, surf_stride & 0xFFFFFFFF, DPU_DST_SURF_STRIDE); i += 1

    # Cube dimensions (0-based)
    ops[i] = npuop(OP_REG_DPU, cube_w & 0x1FFF, DPU_DATA_CUBE_WIDTH); i += 1
    ops[i] = npuop(OP_REG_DPU, cube_h & 0x1FFF, DPU_DATA_CUBE_HEIGHT); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_NOTCH); i += 1
    cube_channel = ((cube_c & 0x1FFF) << 16) | (cube_c & 0x1FFF)
    ops[i] = npuop(OP_REG_DPU, cube_channel, DPU_DATA_CUBE_CHANNEL); i += 1

    # BS — bypassed
    ops[i] = npuop(OP_REG_DPU, 0x53, DPU_BS_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_ALU_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_MUL_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_RELUX_CMP); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x02, DPU_BS_OW_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BS_OW_OP); i += 1

    # WDMA size
    ops[i] = npuop(OP_REG_DPU, cube_c & 0xFFFF, DPU_WDMA_SIZE_0); i += 1
    wdma_1 = ((cube_h & 0x1FFF) << 16) | (cube_w & 0x1FFF)
    ops[i] = npuop(OP_REG_DPU, wdma_1, DPU_WDMA_SIZE_1); i += 1

    # BN — ACTIVE for prescaling (multiply input by prescale constant)
    ops[i] = npuop(OP_REG_DPU, 0x00020040, DPU_BN_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x80000000, DPU_BN_ALU_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, bn_mul_cfg, DPU_BN_MUL_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_BN_RELUX_CMP); i += 1

    # EW — LUT mode active
    ops[i] = npuop(OP_REG_DPU, ew_cfg, DPU_EW_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_EW_CVT_OFFSET); i += 1
    ops[i] = npuop(OP_REG_DPU, 1, DPU_EW_CVT_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_EW_RELUX_CMP); i += 1

    # Output conversion
    ops[i] = npuop(OP_REG_DPU, out_cvt_offset, DPU_OUT_CVT_OFFSET); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x00010001, DPU_OUT_CVT_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, out_cvt_shift, DPU_OUT_CVT_SHIFT); i += 1

    # EW operand values (all zero)
    for reg in range(DPU_EW_OP_VALUE_0, DPU_EW_OP_VALUE_7 + 4, 4):
        ops[i] = npuop(OP_REG_DPU, 0, reg); i += 1

    # Surface address
    ops[i] = npuop(OP_REG_DPU, surf_stride & 0xFFFFFFFF, DPU_SURFACE_ADD); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_40C4); i += 1

    # LUT configuration — ACTIVE (defines table index range and slope)
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_ACCESS_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_ACCESS_DATA); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x00000068, DPU_LUT_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU, 0x00050500, DPU_LUT_INFO); i += 1
    ops[i] = npuop(OP_REG_DPU, 0xFFFFC000, DPU_LUT_LE_START); i += 1   # -16384
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LE_END); i += 1              # 0
    ops[i] = npuop(OP_REG_DPU, 0, DPU_LUT_LO_START); i += 1            # 0
    ops[i] = npuop(OP_REG_DPU, 0x00004000, DPU_LUT_LO_END); i += 1     # 16384
    ops[i] = npuop(OP_REG_DPU, lut_le_slope_scale, DPU_LUT_LE_SLOPE_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, lut_le_slope_shift, DPU_LUT_LE_SLOPE_SHIFT); i += 1
    ops[i] = npuop(OP_REG_DPU, lut_lo_slope_scale, DPU_LUT_LO_SLOPE_SCALE); i += 1
    ops[i] = npuop(OP_REG_DPU, lut_lo_slope_shift, DPU_LUT_LO_SLOPE_SHIFT); i += 1

    # ---- DPU_RDMA configuration (17 registers) ----
    ops[i] = npuop(OP_REG_DPU_RDMA, cube_w & 0x1FFF, DPU_RDMA_500C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, cube_h & 0x1FFF, DPU_RDMA_5010); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, cube_c & 0x1FFF, DPU_RDMA_5014); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, src_dma & 0xFFFFFFFF, DPU_RDMA_SRC_A_ADDR); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_501C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5020); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5028); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_502C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x01, DPU_RDMA_5034); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_EW_SRC_ADDR); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_EW_SURF_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x17849, DPU_RDMA_LINE_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_SURF_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_504C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5064); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x01010101, DPU_RDMA_BURST_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_506C); i += 1

    assert i == 69, f"LUT eval expected 69 data commands, wrote {i}"

    # ---- PC control tail (4 commands) ----
    _write_pc_tail(ops, i, TASK_EW_ENABLE_MASK)
    i += 4

    assert i == _REGCMD_COUNT_LUT_EVAL, f"Expected {_REGCMD_COUNT_LUT_EVAL} commands, wrote {i}"
    return i


# =============================================================================
# Conv2D Mode 6 (Direct Spatial Convolution)
# =============================================================================

def gen_conv_fp16_mode6(ops, M, K, N, input_dma, weights_dma, output_dma,
                        kH, kW, stride, H_out, W_out, H_in=32, W_in=32,
                        rdma_dma=None, relu=False):
    """Generate 130 register commands for FP16 conv using RKNN's mode 6.

    Mode 6 is the direct spatial convolution path used by RKNN.  Register
    formulas were decoded by comparing 5 conv traces with varying kH, kW,
    N, and stride (all with C=3, H_in=W_in=32).

    Parameters
    ----------
    M : int — H_out * W_out
    K : int — C_aligned * kH * kW (aligned to 32)
    N : int — output channels
    input_dma, weights_dma, output_dma : int — DMA addresses
    kH, kW : int — kernel size
    stride : int — stride (both H and W)
    H_out, W_out : int — output spatial dims
    H_in, W_in : int — input spatial dims
    rdma_dma : int | None — DPU_RDMA source address (defaults to output_dma)
    relu : bool — whether to apply ReLU
    """
    N_aligned = align_up(N, 16)
    M_padded = pad_m(M)

    # Decoded formulas (verified against 5 RKNN traces)
    conv_con2 = 0x200 | (math.ceil(kH / stride) << 4)
    conv_con3 = 9 * stride
    data_size0 = (H_in << 16) | (H_in - stride + 1 if stride > 1 else H_in)
    cbuf_con1 = W_in * (W_in - stride + 1)

    # CNA registers
    ops[0] = npuop(OP_REG_CNA, 0xB1, CNA_CBUF_CON0)                  # trace-frozen
    ops[1] = npuop(OP_REG_CNA, 0, CNA_DCOMP_REGNUM)
    ops[2] = npuop(OP_REG_CNA, 0, CNA_DCOMP_CTRL)
    ops[3] = npuop(OP_REG_CNA, 0x6000A120, CNA_CONV_CON1)            # lower bits: C=32, mode=0; upper 0x6000 from trace
    ops[4] = npuop(OP_REG_DPU, 0x0E, DPU_S_POINTER)
    ops[5] = npuop(OP_REG_DPU_RDMA, 0x0E, DPU_RDMA_S_POINTER)
    ops[6] = npuop(OP_REG_CNA, 0x6000A120, CNA_CONV_CON1)            # repeat
    ops[7] = npuop(OP_REG_CNA, conv_con2, CNA_CONV_CON2)
    ops[8] = npuop(OP_REG_CNA, conv_con3, CNA_CONV_CON3)
    ops[9] = npuop(OP_REG_CNA, data_size0, CNA_DATA_SIZE0)
    ops[10] = npuop(OP_REG_CNA, 0x00020008, CNA_DATA_SIZE1)           # constant for C_aligned=32
    ops[11] = npuop(OP_REG_CNA, H_out, CNA_DATA_SIZE2)
    ops[12] = npuop(OP_REG_CNA, M, CNA_DATA_SIZE3)
    ops[13] = npuop(OP_REG_CNA, N * K // 2, CNA_WEIGHT_SIZE0)
    ops[14] = npuop(OP_REG_CNA, K // 2, CNA_WEIGHT_SIZE1)
    ops[15] = npuop(OP_REG_CNA, (kH << 24) | (kW << 16) | N, CNA_WEIGHT_SIZE2)
    ops[16] = npuop(OP_REG_CNA, 0xB1, CNA_CBUF_CON0)                 # repeat
    ops[17] = npuop(OP_REG_CNA, cbuf_con1, CNA_CBUF_CON1)
    ops[18] = npuop(OP_REG_CNA, 0x0B, CNA_CVT_CON0)                  # data_sign=1, cvt_type=1, cvt_bypass=1
    ops[19] = npuop(OP_REG_CNA, 0x00010000, CNA_CVT_CON1)             # scale=1, offset=0
    ops[20] = npuop(OP_REG_CNA, 0x00010000, CNA_CVT_CON2)
    ops[21] = npuop(OP_REG_CNA, 0x00010000, CNA_CVT_CON3)
    ops[22] = npuop(OP_REG_CNA, 0x00010000, CNA_CVT_CON4)
    ops[23] = npuop(OP_REG_CNA, 0, CNA_FC_CON0)
    ops[24] = npuop(OP_REG_CNA, 0, CNA_FC_CON1)
    ops[25] = npuop(OP_REG_CNA, 0, CNA_PAD_CON0)                     # no padding for mode 6
    ops[26] = npuop(OP_REG_CNA, input_dma & 0xFFFFFFFF, CNA_FEATURE_DATA_ADDR)
    ops[27] = npuop(OP_REG_CNA, 0, CNA_FC_CON2)
    ops[28] = npuop(OP_REG_CNA, 0x000F000F, CNA_DMA_CON0)            # burst_len=15
    ops[29] = npuop(OP_REG_CNA, 0x20, CNA_DMA_CON1)                  # trace-frozen: line stride
    ops[30] = npuop(OP_REG_CNA, 0x3E0, CNA_DMA_CON2)                 # trace-frozen: surf stride
    ops[31] = npuop(OP_REG_CNA, data_size0, CNA_FC_DATA_SIZE0)        # mirrors DATA_SIZE0
    ops[32] = npuop(OP_REG_CNA, 0x08, CNA_FC_DATA_SIZE1)              # trace-frozen: channel=8
    ops[33] = npuop(OP_REG_CNA, 0, CNA_DCOMP_CTRL)                    # repeat
    ops[34] = npuop(OP_REG_CNA, 0, CNA_DCOMP_REGNUM)                  # repeat
    ops[35] = npuop(OP_REG_CNA, weights_dma & 0xFFFFFFFF, CNA_DCOMP_ADDR0)
    for i in range(36, 52):
        ops[i] = npuop(OP_REG_CNA, 0, CNA_DCOMP_AMOUNT + (i - 36) * 4)
    ops[52] = npuop(OP_REG_CNA, 0, CNA_CVT_CON5)
    ops[53] = npuop(OP_REG_CNA, 0, CNA_PAD_CON1)

    # CORE registers
    ops[54] = npuop(OP_REG_CORE, 0x200, CORE_MISC_CFG)                # trace-frozen: mode 6 core config
    ops[55] = npuop(OP_REG_CORE, ((H_out - 1) << 16) | (H_out - 1), CORE_DATAOUT_SIZE_0)
    ops[56] = npuop(OP_REG_CORE, N_aligned - 1, CORE_DATAOUT_SIZE_1)
    ops[57] = npuop(OP_REG_CORE, 0, CORE_CLIP_TRUNCATE)
    ops[58] = npuop(OP_REG_CORE, 0, CORE_3030)

    # DPU registers
    ops[59] = npuop(OP_REG_DPU, 0x1E4, DPU_FEATURE_MODE_CFG)         # conv_mode=0, flying=1, output_mode=2, fp16
    ops[60] = npuop(OP_REG_DPU, 0x48000002, DPU_DATA_FORMAT)          # FP16 output format
    ops[61] = npuop(OP_REG_DPU, 0, DPU_OFFSET_PEND)
    ops[62] = npuop(OP_REG_DPU, output_dma & 0xFFFFFFFF, DPU_DST_BASE_ADD)
    ops[63] = npuop(OP_REG_DPU, M_padded * 16, DPU_DST_SURF_STRIDE)
    ops[64] = npuop(OP_REG_DPU, W_out - 1, DPU_DATA_CUBE_WIDTH)
    ops[65] = npuop(OP_REG_DPU, H_out - 1, DPU_DATA_CUBE_HEIGHT)
    ops[66] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_NOTCH)
    ops[67] = npuop(OP_REG_DPU, ((N - 1) << 16) | (N_aligned - 1), DPU_DATA_CUBE_CHANNEL)

    # BS config — ReLU fusion
    if relu:
        ops[68] = npuop(OP_REG_DPU, 0x00020110, DPU_BS_CFG)          # BS active + ReLU
    else:
        ops[68] = npuop(OP_REG_DPU, 0x00020150, DPU_BS_CFG)          # BS bypassed

    ops[69] = npuop(OP_REG_DPU, 0, DPU_BS_ALU_CFG)
    ops[70] = npuop(OP_REG_DPU, 0, DPU_BS_MUL_CFG)
    ops[71] = npuop(OP_REG_DPU, 0, DPU_BS_RELUX_CMP)
    ops[72] = npuop(OP_REG_DPU, 0x126, DPU_BS_OW_CFG)                # trace-frozen: size_e + od_bypass
    ops[73] = npuop(OP_REG_DPU, 0, DPU_BS_OW_OP)
    ops[74] = npuop(OP_REG_DPU, N_aligned - 1, DPU_WDMA_SIZE_0)
    ops[75] = npuop(OP_REG_DPU, ((H_out - 1) << 16) | (H_out - 1), DPU_WDMA_SIZE_1)
    ops[76] = npuop(OP_REG_DPU, 0x53, DPU_BN_CFG)                    # BN bypassed
    ops[77] = npuop(OP_REG_DPU, 0, DPU_BN_ALU_CFG)
    ops[78] = npuop(OP_REG_DPU, 0, DPU_BN_MUL_CFG)
    ops[79] = npuop(OP_REG_DPU, 0, DPU_BN_RELUX_CMP)
    ops[80] = npuop(OP_REG_DPU, 0x383, DPU_EW_CFG)                   # EW bypassed
    ops[81] = npuop(OP_REG_DPU, 0, DPU_EW_CVT_OFFSET)
    ops[82] = npuop(OP_REG_DPU, 1, DPU_EW_CVT_SCALE)
    ops[83] = npuop(OP_REG_DPU, 0, DPU_EW_RELUX_CMP)
    ops[84] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_OFFSET)
    ops[85] = npuop(OP_REG_DPU, 0x00010001, DPU_OUT_CVT_SCALE)       # fp32tofp16_en=1, scale=1
    ops[86] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_SHIFT)
    # EW operand values (all zero)
    for i, reg in enumerate(range(DPU_EW_OP_VALUE_0, DPU_EW_OP_VALUE_7 + 4, 4)):
        ops[87 + i] = npuop(OP_REG_DPU, 0, reg)
    ops[95] = npuop(OP_REG_DPU, M_padded * 32, DPU_SURFACE_ADD)      # C_aligned=32
    ops[96] = npuop(OP_REG_DPU, 0, DPU_40C4)
    # LUT configuration (all zero)
    for i, reg in enumerate(_LUT_REGS):
        ops[97 + i] = npuop(OP_REG_DPU, 0, reg)

    # DPU_RDMA registers
    ops[109] = npuop(OP_REG_DPU_RDMA, W_out - 1, DPU_RDMA_500C)
    ops[110] = npuop(OP_REG_DPU_RDMA, H_out - 1, DPU_RDMA_5010)
    ops[111] = npuop(OP_REG_DPU_RDMA, N_aligned - 1, DPU_RDMA_5014)
    ops[112] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_SRC_A_ADDR)
    ops[113] = npuop(OP_REG_DPU_RDMA, 0x02, DPU_RDMA_501C)
    _rdma = (rdma_dma if rdma_dma is not None else output_dma) & 0xFFFFFFFF
    ops[114] = npuop(OP_REG_DPU_RDMA, _rdma, DPU_RDMA_5020)
    ops[115] = npuop(OP_REG_DPU_RDMA, M_padded * 16, DPU_RDMA_5028)  # line stride
    ops[116] = npuop(OP_REG_DPU_RDMA, M_padded * 16, DPU_RDMA_502C)  # surf stride
    ops[117] = npuop(OP_REG_DPU_RDMA, 0x01, DPU_RDMA_5034)           # trace-frozen
    ops[118] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_EW_SRC_ADDR)
    ops[119] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_EW_SURF_STRIDE)
    ops[120] = npuop(OP_REG_DPU_RDMA, 0x17850, DPU_RDMA_LINE_STRIDE) # trace-frozen
    ops[121] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_SURF_STRIDE)
    ops[122] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_504C)
    ops[123] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5064)
    ops[124] = npuop(OP_REG_DPU_RDMA, 0x01010101, DPU_RDMA_BURST_CFG)
    ops[125] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_506C)

    # PC control tail
    _write_pc_tail(ops, 126, TASK_ENABLE_MASK_MODE6)


# =============================================================================
# Bias/Residual RDMA Block
# =============================================================================

def gen_bias_rdma_block(ops, start_idx, W_out, H_out, N_aligned,
                        bias_dma=0, has_bias=True,
                        has_residual=False, residual_dma=0, ew_surf_stride=0,
                        conv_mode: int = 0):
    """Write 17 DPU_RDMA registers for bias and/or fused residual reading."""
    rdma_5034 = 0x40000008 if has_residual else 0x01

    M_tile = H_out * W_out
    ew_skip = (2 * ew_surf_stride - M_tile * 16) if has_residual else 0

    i = start_idx
    ops[i] = npuop(OP_REG_DPU_RDMA, W_out - 1, DPU_RDMA_500C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, H_out - 1, DPU_RDMA_5010); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, N_aligned - 1, DPU_RDMA_5014); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_SRC_A_ADDR); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x02 if has_bias else 0, DPU_RDMA_501C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, bias_dma if has_bias else 0, DPU_RDMA_5020); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5028); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_502C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, rdma_5034, DPU_RDMA_5034); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, residual_dma if has_residual else 0, DPU_RDMA_EW_SRC_ADDR); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, ew_surf_stride if has_residual else 0, DPU_RDMA_EW_SURF_STRIDE); i += 1
    rdma_line_stride = 0x17856 if conv_mode == 3 else 0x17850
    ops[i] = npuop(OP_REG_DPU_RDMA, rdma_line_stride, DPU_RDMA_LINE_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_SURF_STRIDE); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, ew_skip, DPU_RDMA_504C); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0, DPU_RDMA_5064); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, 0x01010101, DPU_RDMA_BURST_CFG); i += 1
    ops[i] = npuop(OP_REG_DPU_RDMA, ew_skip, DPU_RDMA_506C); i += 1
    return i
