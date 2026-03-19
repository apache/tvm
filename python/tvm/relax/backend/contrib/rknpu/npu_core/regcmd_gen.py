# SPDX-License-Identifier: MIT
"""NPU register command generation — task orchestration layer.

This module is the entry point for NPU register command generation.
Implementation details are in _cna_regcfg.py (field computation)
and _dpu_regcfg.py (register encoding).
"""

from dataclasses import dataclass
import os
import numpy as np
from typing import TYPE_CHECKING

from ._cna_regcfg import (
    PrecisionConfig, PREC_FP16_CFG, PREC_I8_CFG, PREC_BF16_CFG, PREC_INT16_CFG,
    CnaDesc, CoreDesc, DpuDesc,
    fill_matmul_descriptors, fill_conv_mode0_descriptors,
)
from ._dpu_regcfg import (
    npuop, gen_cna_dpu_regcmds, gen_ew_op, gen_conv_fp16_mode6, gen_bias_rdma_block,
    gen_lut_upload, gen_lut_eval, gen_lut_combined, REGCMD_COUNT_LUT_COMBINED,
    OP_NONE, OP_40, OP_ENABLE,
    OP_REG_CNA, OP_REG_CORE, OP_REG_DPU, OP_REG_DPU_RDMA, OP_REG_PC,
    OP_REG_PPU, OP_REG_PPU_RDMA,
    _write_pc_tail,
)
from .hardware import (
    REGCMD_COUNT, REGCFG_AMOUNT,
    TASK_ENABLE_MASK, TASK_INT_MASK, TASK_INT_CLEAR,
    REGCMD_COUNT_EW, REGCMD_COUNT_LUT_UPLOAD,
    TASK_EW_ENABLE_MASK, TASK_EW_INT_MASK, TASK_EW_INT_CLEAR,
    REGCMD_COUNT_MODE6, TASK_ENABLE_MASK_MODE6, TASK_INT_MASK_MODE6, TASK_INT_CLEAR_MODE6,
    REGCMD_COUNT_PPU, TASK_PPU_ENABLE_MASK, TASK_PPU_INT_MASK, TASK_PPU_INT_CLEAR,
    NPU_MAX_N_PER_SUBMIT, NPU_CBUF_BANK_SIZE, NPU_CBUF_BANKS,
    NPU_MAX_DPU_DIM,
    PC_BASE_ADDRESS, PC_REGISTER_AMOUNTS, PC_OPERATION_ENABLE,
    CNA_CONV_CON2, CNA_WEIGHT_SIZE0, CNA_WEIGHT_SIZE2,
    CORE_DATAOUT_SIZE_1,
    DPU_DATA_CUBE_CHANNEL, DPU_WDMA_SIZE_0,
    DPU_RDMA_S_POINTER, DPU_RDMA_5014, DPU_EW_CFG,
    DPU_DST_BASE_ADD, DPU_RDMA_EW_SRC_ADDR,
    BlockType, RegCmd, Task,
)

if TYPE_CHECKING:
    from .handles import TensorHandle

from .alignment import align_up, pad_m


# =============================================================================
# Task Construction
# =============================================================================

_TASK_MASKS = {
    'standard': (TASK_ENABLE_MASK, TASK_INT_MASK, TASK_INT_CLEAR),
    'mode6':    (TASK_ENABLE_MASK_MODE6, TASK_INT_MASK_MODE6, TASK_INT_CLEAR_MODE6),
    'ew':       (TASK_EW_ENABLE_MASK, TASK_EW_INT_MASK, TASK_EW_INT_CLEAR),
    'ppu':      (TASK_PPU_ENABLE_MASK, TASK_PPU_INT_MASK, TASK_PPU_INT_CLEAR),
}


def _make_task(regcmds, mode='standard'):
    en, im, ic = _TASK_MASKS[mode]
    return Task(regcmds=regcmds, enable_mask=en, int_mask=im, int_clear=ic)


def _emit_task(n_cmds: int, kind: str, fill_fn) -> Task:
    """Allocate ops array, call fill_fn(ops), convert to Task."""
    ops = np.zeros(n_cmds, dtype=np.uint64)
    fill_fn(ops)
    return _make_task(_numpy_to_regcmds(ops, n_cmds), kind)


def _numpy_to_regcmds(ops: np.ndarray, count: int) -> list:
    """Convert numpy uint64 register command array to list of RegCmd objects.

    Shared helper for all generate_* methods to avoid duplicated decode logic.
    """
    regcmds = []
    for i in range(count):
        op_val = int(ops[i])
        op = (op_val >> 48) & 0xFFFF
        value = (op_val >> 16) & 0xFFFFFFFF
        reg = op_val & 0xFFFF

        if op == OP_REG_CNA:
            block = BlockType.CNA
        elif op == OP_REG_CORE:
            block = BlockType.CORE
        elif op == OP_REG_DPU:
            block = BlockType.DPU
        elif op == OP_REG_DPU_RDMA:
            block = BlockType.DPU_RDMA
        elif op == OP_REG_PPU:
            block = BlockType.PPU
        elif op == OP_REG_PPU_RDMA:
            block = BlockType.PPU_RDMA
        elif op == OP_REG_PC:
            block = BlockType.PC
        elif op == OP_NONE or op == OP_40 or op == OP_ENABLE:
            block = op
        else:
            block = BlockType(op & 0xFF00)

        regcmds.append(RegCmd(block, value, reg))
    return regcmds


# =============================================================================
# Task Splitting (Ping-Pong Pipeline)
# =============================================================================
#
# Rockchip always generates 3 tasks per conv: two CNA+DPU tasks (ping/pong)
# plus a PPU post-processing task.  This matches the hardware's ping-pong
# pipeline where Task 0 computes full N, Task 1 computes N/2 (pong buffer),
# and PPU handles FP32->FP16 output conversion.

# N-split formula from hardware traces:
#   N1 = N // 2   when N % 32 == 0
#   N1 = N        when N % 32 != 0
_REGCMD_COUNT_TASK1 = 126  # 130 - 4 init regcmds


def _compute_n1(N: int) -> int:
    """Compute N for Task 1 (pong) based on N alignment."""
    if N % 32 == 0:
        return N // 2
    return N


def generate_ppu_task(scratch_dma: int = 0) -> Task:
    """Generate a no-op PPU task (26 data regcmds + 4 PC tail).

    The PPU task exists to satisfy the hardware's 3-task ping-pong pipeline
    requirement.  Both CNA+DPU tasks (Task 0 and Task 1) write directly to
    the final output buffer; the PPU reads from and writes to scratch memory
    (typically within the weight buffer) and has no effect on results.

    Parameters
    ----------
    scratch_dma : int
        DMA address of scratch memory for PPU src/dst.  PPU_DST is placed
        at scratch_dma, PPU_RDMA_SRC at scratch_dma + 0x400.
    """
    PPU = BlockType.PPU
    PPU_RDMA = BlockType.PPU_RDMA
    PC = BlockType.PC

    ppu_dst = scratch_dma & 0xFFFFFFFF
    ppu_src = (scratch_dma + 0x400) & 0xFFFFFFFF

    data_cmds = [
        RegCmd(PPU,      0x0000000E, 0x6004),
        RegCmd(PPU_RDMA, 0x0000000E, 0x7004),
        RegCmd(PPU,      0x00000000, 0x600C),
        RegCmd(PPU,      0x00000000, 0x6010),
        RegCmd(PPU,      0x0000001F, 0x6014),
        RegCmd(PPU,      0x00000000, 0x6018),
        RegCmd(PPU,      0x00000000, 0x601C),
        RegCmd(PPU,      0x0000001F, 0x6020),
        RegCmd(PPU,      0x00000011, 0x6024),
        RegCmd(PPU,      0x00000000, 0x6034),
        RegCmd(PPU,      0x00000000, 0x6038),
        RegCmd(PPU,      0x00000000, 0x603C),
        RegCmd(PPU,      0x00000000, 0x6040),
        RegCmd(PPU,      0x00000000, 0x6044),
        RegCmd(PPU,      0x00000000, 0x6048),
        RegCmd(PPU,      ppu_dst,    0x6070),
        RegCmd(PPU,      0x00000010, 0x607C),
        RegCmd(PPU,      0x00000010, 0x6084),
        RegCmd(PPU,      0x00000003, 0x60DC),
        RegCmd(PPU_RDMA, 0x00000000, 0x700C),
        RegCmd(PPU_RDMA, 0x00000000, 0x7010),
        RegCmd(PPU_RDMA, 0x0000001F, 0x7014),
        RegCmd(PPU_RDMA, ppu_src,    0x701C),
        RegCmd(PPU_RDMA, 0x00000010, 0x7024),
        RegCmd(PPU_RDMA, 0x00000010, 0x7028),
        RegCmd(PPU_RDMA, 0x00000001, 0x7030),
    ]
    pc_tail = [
        RegCmd(OP_NONE,   0, 0),
        RegCmd(PC,        0, PC_REGISTER_AMOUNTS),
        RegCmd(OP_40,     0, 0),
        RegCmd(OP_ENABLE, TASK_PPU_ENABLE_MASK, PC_OPERATION_ENABLE),
    ]

    return Task(
        regcmds=data_cmds + pc_tail,
        enable_mask=TASK_PPU_ENABLE_MASK,
        int_mask=TASK_PPU_INT_MASK,
        int_clear=TASK_PPU_INT_CLEAR,
    )


def derive_task1(task0: Task, N: int, wbpk: int,
                 scratch_dma: int = 0,
                 has_residual: bool = False,
                 ew_surf_stride: int = 0) -> Task:
    """Derive Task 1 (pong/delta) from Task 0 by removing init regcmds and
    adjusting N-related registers.

    Task 1 has 126 regcmds (4 fewer than Task 0's 130): the first 4 init
    regcmds (CBUF_CON0, DCOMP_REGNUM, DCOMP_CTRL, CONV_CON1) are removed,
    and 7 output-channel registers are patched for N1 = N/2 (or N if N%32!=0).

    Note: residual (EW) ops currently use single-task mode to avoid EW
    surface counter contamination.  If has_residual is set, Task 1's WDMA
    output is redirected to scratch memory as a safety measure.

    Parameters
    ----------
    task0 : Task
        The full Mode 6 task (130 regcmds).
    N : int
        Total output channels.
    wbpk : int
        Weight bytes per kernel (C_cna * kH * kW * elem_size).
    scratch_dma : int
        DMA address of scratch memory (PPU no-op target).
    has_residual : bool
        Whether the op has fused residual addition (triggers WDMA redirect).
    ew_surf_stride : int
        Unused, kept for API compatibility.
    """
    N1 = _compute_n1(N)
    # Hardware registers use aligned N (multiple of 16)
    from .alignment import align_up as _au
    N1_al = _au(N1, 16)

    # Drop first 4 init regcmds and last 4 PC tail entries
    body = list(task0.regcmds[4:-4])

    # Patch N-related registers by matching on offset.
    # Bit[30] of CNA_CONV_CON2 is a "skip CNA DMA reload" flag -- the trace
    # sets it on Task 1 in 99/102 models.  It requires the full 3-task
    # ping-pong pipeline (Task 0 + Task 1 + PPU) to avoid hardware hangs.
    # CNA weight registers use raw N1; CORE/DPU registers use N1_aligned.
    for i, cmd in enumerate(body):
        off = cmd.offset
        if off == CNA_CONV_CON2:
            # bit[30] = skip CNA DMA reload.  Disabled until fg/CBUF diffs
            # are fixed -- wrong CBUF config causes the CNA to stall when
            # it tries to reuse stale buffer contents.
            # body[i] = RegCmd(cmd.block, cmd.value | 0x40000000, off)
            pass
        # N-split register patching:
        elif off == CNA_WEIGHT_SIZE0:
            body[i] = RegCmd(cmd.block, wbpk * N1, off)
        elif off == CNA_WEIGHT_SIZE2:
            body[i] = RegCmd(cmd.block, (cmd.value & ~0x3FFF) | (N1 & 0x3FFF), off)
        elif off == CORE_DATAOUT_SIZE_1:
            body[i] = RegCmd(cmd.block, N1_al - 1, off)
        elif off == DPU_DATA_CUBE_CHANNEL:
            body[i] = RegCmd(cmd.block, ((N1_al - 1) << 16) | (N1_al - 1), off)
        elif off == DPU_WDMA_SIZE_0:
            body[i] = RegCmd(cmd.block, N1_al - 1, off)
        elif off == DPU_RDMA_5014:
            body[i] = RegCmd(cmd.block, N1_al - 1, off)
        # Note: Task 1's EW reads are correct because the DPU EW
        # surface counter auto-advances from Task 0.  Task 0 processes
        # surfaces 0..N0/16-1, counter advances to N0/16.  Task 1
        # continues from N0/16..N/16-1, reading correct residual data.

    # Append PC tail
    pc_tail = [
        RegCmd(OP_NONE,   0, 0),
        RegCmd(BlockType.PC, 0, PC_REGISTER_AMOUNTS),
        RegCmd(OP_40,     0, 0),
        RegCmd(OP_ENABLE, TASK_ENABLE_MASK_MODE6, PC_OPERATION_ENABLE),
    ]

    regcmds = body + pc_tail
    assert len(regcmds) == _REGCMD_COUNT_TASK1, \
        f"Task 1 should have {_REGCMD_COUNT_TASK1} regcmds, got {len(regcmds)}"

    return Task(
        regcmds=regcmds,
        enable_mask=TASK_ENABLE_MASK_MODE6,
        int_mask=TASK_INT_MASK_MODE6,
        int_clear=TASK_INT_CLEAR_MODE6,
    )


# =============================================================================
# PC Chaining and Tiling
# =============================================================================

def patch_pc_chain(ops: np.ndarray, next_dma_addr: int, next_n_cmds: int = REGCMD_COUNT) -> None:
    """Patch PC tail of a CNA+DPU task to chain to the next task.

    The PC_BASE_ADDRESS regcmd (ops[108]) directly updates the hardware's
    PC data fetch address -- it must be an absolute DMA address.
    PC_REGISTER_AMOUNTS (ops[109]) encodes the command count as
    next_n_cmds // 2 - 1 (matching kernel PC_DATA_AMOUNT formula).

    Parameters
    ----------
    ops : numpy uint64 array (length REGCMD_COUNT)
        Register commands for this task (already generated).
    next_dma_addr : int
        Absolute DMA address of the next task's regcmd blob.
    next_n_cmds : int
        Number of 64-bit register commands in the next task (default 112).
    """
    ops[108] = npuop(OP_REG_PC, next_dma_addr & 0xFFFFFFFF, PC_BASE_ADDRESS)
    ops[109] = npuop(OP_REG_PC, next_n_cmds // 2 - 1, PC_REGISTER_AMOUNTS)


def patch_pc_chain_mode6(ops: np.ndarray, next_dma_addr: int, next_n_cmds: int = REGCMD_COUNT_MODE6) -> None:
    """Patch PC tail of a Mode 6 task to chain to the next task.

    Mode 6 tasks have 130 regcmds with PC tail at ops[126..129].
    """
    ops[126] = npuop(OP_REG_PC, next_dma_addr & 0xFFFFFFFF, PC_BASE_ADDRESS)
    ops[127] = npuop(OP_REG_PC, next_n_cmds // 2 - 1, PC_REGISTER_AMOUNTS)


def patch_pc_chain_task1(ops: np.ndarray, next_dma_addr: int, next_n_cmds: int = REGCMD_COUNT_PPU) -> None:
    """Patch PC tail of a Task 1 (126 regcmds) to chain to the next task.

    Task 1 has 126 regcmds with PC tail at ops[122..125].
    """
    ops[122] = npuop(OP_REG_PC, next_dma_addr & 0xFFFFFFFF, PC_BASE_ADDRESS)
    ops[123] = npuop(OP_REG_PC, next_n_cmds // 2 - 1, PC_REGISTER_AMOUNTS)


def patch_pc_chain_ppu(ops: np.ndarray, next_dma_addr: int, next_n_cmds: int = REGCMD_COUNT_MODE6) -> None:
    """Patch PC tail of a PPU task (30 regcmds) to chain to the next task.

    PPU tasks have 30 regcmds with PC tail at ops[26..29].
    """
    ops[26] = npuop(OP_REG_PC, next_dma_addr & 0xFFFFFFFF, PC_BASE_ADDRESS)
    ops[27] = npuop(OP_REG_PC, next_n_cmds // 2 - 1, PC_REGISTER_AMOUNTS)


def patch_pc_chain_ew(ops: np.ndarray, next_dma_addr: int, next_n_cmds: int = REGCMD_COUNT_MODE6) -> None:
    """Patch PC tail of an EW task (74 regcmds) to chain to the next task.

    EW tasks have 74 regcmds with PC tail at ops[70..73].
    """
    ops[70] = npuop(OP_REG_PC, next_dma_addr & 0xFFFFFFFF, PC_BASE_ADDRESS)
    ops[71] = npuop(OP_REG_PC, next_n_cmds // 2 - 1, PC_REGISTER_AMOUNTS)


# =============================================================================
# CBUF Tiling
# =============================================================================

def compute_m_tile(K_aligned: int, Mp: int | None = None, elem_size: int = 2,
                   N: int = 0) -> int:
    """Compute maximum M_tile that fits in CBUF for given K_aligned.

    This uses the hardware-specific formula from the reverse-engineered driver.
    M-tiling is needed when the batch/sequence dimension (M) is too large to
    fit in the NPU's Coefficient Buffer (CBUF) simultaneously.

    Parameters
    ----------
    K_aligned : int
        K dimension aligned to precision-specific alignment (32 for FP16, 64 for INT8).
    Mp : int, optional
        Padded M dimension (M if M<=1, else next multiple of 4).
        If None, returned tile_size is not capped.
    elem_size : int
        Bytes per element (2 for FP16, 1 for INT8).
    N : int
        Output channels. When N is large, more CBUF banks are needed for
        weights, reducing the banks available for feature data and thus the
        maximum M tile size.

    Returns
    -------
    int
        Maximum tile size for M dimension (multiple of 4).

    Formula
    -------
    The CBUF has 12 banks of 32KB each. Feature data (input) needs:
        fd_bytes = Mp * K_aligned * elem_size
        fd_banks = ceil(fd_bytes / 32768)

    We reserve banks for weights based on actual N requirements (matching
    _allocate_cbuf logic). When N * wbpk exceeds the initial weight capacity
    and weight_banks < MIN_WEIGHT_BANKS_MULTIPASS (3), weight banks are
    increased to 3, reducing available feature data banks.
    """
    wbpk = K_aligned * elem_size

    # Minimum weight banks (32 * K_aligned is the weight read stride)
    min_wb = ((32 * K_aligned + NPU_CBUF_BANK_SIZE // 2)
              + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE
    if min_wb < 1:
        min_wb = 1

    # Max feature data banks
    max_fd_banks = NPU_CBUF_BANKS - min_wb
    if max_fd_banks > NPU_CBUF_BANKS - 1:
        max_fd_banks = NPU_CBUF_BANKS - 1
    if max_fd_banks < 1:
        max_fd_banks = 1

    # Account for multi-pass weight banks: when the initial weight bank
    # allocation is too small for all N kernels, _allocate_cbuf increases
    # weight_banks to MIN_WEIGHT_BANKS_MULTIPASS (3).  We must mirror that
    # logic here so the M-tile fits the actual available feature banks.
    MIN_WEIGHT_BANKS_MULTIPASS = 3
    if N > 0:
        weight_banks = NPU_CBUF_BANKS - max_fd_banks
        weight_capacity = weight_banks * NPU_CBUF_BANK_SIZE
        if weight_banks < MIN_WEIGHT_BANKS_MULTIPASS and N * wbpk > weight_capacity:
            max_fd_banks = NPU_CBUF_BANKS - MIN_WEIGHT_BANKS_MULTIPASS
            if max_fd_banks < 1:
                max_fd_banks = 1

    # Max padded M that fits in available banks
    max_pad_m = (max_fd_banks * NPU_CBUF_BANK_SIZE) // (K_aligned * elem_size)

    # Round down to multiple of 4
    m_tile = (max_pad_m // 4) * 4
    if m_tile < 4:
        m_tile = 4

    # Cap at hardware register max: feature_grains = Mp + 1 must fit in 10 bits
    # (max 1023), so Mp <= 1022.  Round down to multiple of 4 → 1020.
    MAX_MP_REG = 1020
    if m_tile > MAX_MP_REG:
        m_tile = MAX_MP_REG

    # Cap at Mp if provided
    if Mp is not None and m_tile > Mp:
        m_tile = Mp

    return m_tile


def compute_n_tile(N: int) -> int:
    """Compute N tile size for output channel dimension.

    N-tiling is needed when N > NPU_MAX_N_PER_SUBMIT (8192).
    Each tile can have at most 8192 output channels.

    Parameters
    ----------
    N : int
        Total N dimension (output channels).

    Returns
    -------
    int
        Tile size for N dimension (min(N, 8192)).
    """
    return min(N, NPU_MAX_N_PER_SUBMIT)


# =============================================================================
# RegCmdGenerator
# =============================================================================

from .abstract import AbstractMatmulTask, AbstractElementwiseTask, AbstractConv2DTask, AbstractMaxPoolTask


@dataclass
class RegCmdGenerator:
    """Generate RegCmds for AbstractTasks using real NPU register format.

    This is the correct implementation ported from rknpu-py.
    """
    base_addr: int = 0xDEAD0000

    def generate_conv_mode0(
        self,
        task: AbstractConv2DTask,
        input_handle: 'TensorHandle',
        weight_handle: 'TensorHandle',
        output_handle: 'TensorHandle',
        bias_handle: 'TensorHandle | None' = None,
        residual_handle: 'TensorHandle | None' = None,
    ) -> Task:
        """Generate RegCmds for conv2d Mode 0 (im2col + MAC).

        Uses the same 112-regcmd template as matmul (gen_cna_dpu_regcmds).
        The CNA performs im2col internally based on the spatial dimension registers.

        When bias or residual is present, uses the 130-regcmd template with
        DPU_RDMA block for BS (bias) and/or EW (residual) pipelines.

        Args:
            task: AbstractConv2DTask with mode=0
            input_handle: Input tensor (shape: [H, W, C])
            weight_handle: Weight tensor (packed as matmul weights)
            output_handle: Output tensor (shape: [H_out, W_out, N])
            bias_handle: Optional bias tensor (per-channel)
            residual_handle: Optional residual tensor (fused EW add)

        Returns:
            Task with 112 or 130 RegCmds programmed
        """
        input_dma = input_handle.dma_addr
        weights_dma = weight_handle.dma_addr
        output_dma = output_handle.dma_addr
        bias_dma = bias_handle.dma_addr if bias_handle else 0
        residual_dma = residual_handle.dma_addr if residual_handle else 0
        # M-tiling: offset residual DMA address by m_offset positions
        if task.is_mtile and task.m_offset and residual_handle:
            residual_dma += task.m_offset * 8 * 2  # C2=8 channels x 2 bytes per FP16
        has_bias = task.has_bias and bias_handle is not None
        has_residual = task.has_residual and residual_handle is not None

        cna, core, dpu = fill_conv_mode0_descriptors(
            C=task.C, H=task.H, W=task.W, N=task.N,
            kH=task.kH, kW=task.kW, stride=task.stride,
            pad_top=task.pad_top, pad_bottom=task.pad_bottom,
            pad_left=task.pad_left, pad_right=task.pad_right,
            input_dma=input_dma,
            weights_dma=weights_dma,
            output_dma=output_dma,
            relu=task.relu,
            clip_max=task.clip_max,
            has_bias=has_bias,
            bias_dma=bias_dma,
            has_residual=has_residual,
            groups=task.groups,
            H_tile_out=task.H_tile_out if task.is_mtile else None,
            m_offset=task.m_offset if task.is_mtile else 0,
            is_depthwise=task.is_depthwise,
            conv_mode=task.conv_mode,
            mode3_n_idx=getattr(task, 'mode3_n_idx', 0),
            c_tile_offset=task.c_tile_offset,
            c_tile_channels=task.c_tile_channels,
        )

        # 130-regcmd template for bias/residual (DPU_RDMA block carries LINE_STRIDE)
        # Mode 3 without bias uses 112-regcmd template (DPU_RDMA LINE_STRIDE not needed)
        if has_bias or has_residual:
            # DPU_RDMA cube dimensions MUST match the DPU cube dimensions.
            # When internal H-tiling kicks in (fill_conv_mode0_descriptors
            # reduces H_out_eff to fit CBUF), the DPU processes fewer rows
            # than task.H_out.  Using task.H_out for DPU_RDMA would cause a
            # deadlock: the DPU_RDMA tries to synchronise with more output
            # rows than the DPU actually produces.
            # Read the effective dimensions directly from the DPU descriptor.
            H_out_eff = dpu.height + 1  # dpu.height is 0-based
            W_out = dpu.width + 1
            # DW mode 3 uses C_aligned (32-aligned) for all channel-related fields
            N_aligned = align_up(task.C_aligned, 16) if task.is_depthwise else align_up(task.N, 16)
            # ew_surf_stride is the DRAM surface stride of the residual tensor,
            # which uses the FULL output layout (not the H-tiled subset).
            H_out_full = task.H_out
            Mp = pad_m(H_out_full * task.W_out)
            ew_surf_stride = Mp * 16  # 8 channels x 2 bytes per FP16 surface atom

            def fill_130(ops):
                # Generate 112-regcmd template first
                temp_ops = np.zeros(REGCMD_COUNT, dtype=np.uint64)
                gen_cna_dpu_regcmds(temp_ops, cna, core, dpu)

                # Build 130-regcmd array: insert DPU_RDMA S_POINTER at [5]
                # Layout: [0..4] preamble, [5] RDMA S_PTR, [6..108] CNA/CORE/DPU,
                # [109..125] RDMA block, [126..129] PC tail.
                ops[0:5] = temp_ops[0:5]
                ops[5] = npuop(OP_REG_DPU_RDMA, 0x0E, DPU_RDMA_S_POINTER)
                ops[6:109] = temp_ops[5:108]

                # Apply EW_CFG override at ops[80] (position 79+1 due to RDMA S_PTR insertion)
                if has_residual and dpu.ew_cfg_override is not None:
                    ops[80] = npuop(OP_REG_DPU, dpu.ew_cfg_override, DPU_EW_CFG)

                gen_bias_rdma_block(ops, 109, W_out, H_out_eff, N_aligned,
                                    bias_dma=bias_dma, has_bias=has_bias,
                                    has_residual=has_residual,
                                    residual_dma=residual_dma,
                                    ew_surf_stride=ew_surf_stride,
                                    conv_mode=task.conv_mode)

                _write_pc_tail(ops, 126, TASK_ENABLE_MASK_MODE6)

            return _emit_task(REGCMD_COUNT_MODE6, 'mode6', fill_130)
        else:
            # No bias or residual: existing 112-regcmd path
            def fill_112(ops):
                gen_cna_dpu_regcmds(ops, cna, core, dpu)
            return _emit_task(REGCMD_COUNT, 'standard', fill_112)

    def generate_matmul(
        self,
        task: AbstractMatmulTask,
        input_a: 'TensorHandle',
        input_b: 'TensorHandle',
        output_c: 'TensorHandle',
        bias_handle: 'TensorHandle | None' = None,
    ) -> Task:
        """Generate RegCmds for matrix multiplication.

        Args:
            task: AbstractMatmulTask with dimensions (M, K, N)
            input_a: Left operand tensor (shape: [M, K])
            input_b: Right operand tensor (shape: [K, N])
            output_c: Output tensor (shape: [M, N])
            bias_handle: Optional bias tensor (shape: [N], FP32 in DMA).
                When provided, uses 130-regcmd template with DPU_RDMA for bias.

        Returns:
            Task with RegCmds programmed
        """
        # Map precision to PrecisionConfig
        precision_map = {
            "float16": PREC_FP16_CFG,
            "int8": PREC_I8_CFG,
            "bfloat16": PREC_BF16_CFG,
            "int16": PREC_INT16_CFG,
        }
        pc = precision_map.get(task.precision, PREC_FP16_CFG)

        # Get DMA addresses from tensor handles
        input_dma = input_a.dma_addr
        weights_dma = input_b.dma_addr
        output_dma = output_c.dma_addr

        # Check tiling flags (Phase 4 — attributes may not exist yet on base task)
        is_mtile = getattr(task, 'is_mtile', False)
        is_ntile = getattr(task, 'is_ntile', False)

        # Compute actual dimensions (tiled or full)
        M_actual = getattr(task, 'M_tile', task.M) if is_mtile else task.M
        N_actual = getattr(task, 'N_tile', task.N) if is_ntile else task.N

        # Align K and N for hardware using precision config
        K_aligned = align_up(task.K, pc.k_align)
        N_aligned = align_up(N_actual, pc.n_align)

        # M-tiling: pass m_offset and M_full so fill_matmul_descriptors can
        # offset DMA addresses and use the full tensor's surf_stride.
        m_offset = getattr(task, 'm_offset', 0) if is_mtile else 0
        M_full = getattr(task, 'M_full', task.M) if is_mtile else None

        has_bias = bias_handle is not None
        bias_dma = bias_handle.dma_addr if has_bias else 0

        # Fill descriptors with ALIGNED dimensions (critical for hardware correctness)
        cna, core, dpu = fill_matmul_descriptors(
            M_actual, K_aligned, N_aligned,
            input_dma, weights_dma, output_dma,
            task.relu, pc,
            output_fp16=getattr(task, 'output_fp16', False),
            m_offset=m_offset,
            M_full=M_full,
            has_bias=has_bias,
            bias_dma=bias_dma,
        )

        if has_bias:
            Mp = pad_m(M_actual)

            def fill_130(ops):
                temp_ops = np.zeros(REGCMD_COUNT, dtype=np.uint64)
                gen_cna_dpu_regcmds(temp_ops, cna, core, dpu)

                ops[0:5] = temp_ops[0:5]
                ops[5] = npuop(OP_REG_DPU_RDMA, 0x0E, DPU_RDMA_S_POINTER)
                ops[6:109] = temp_ops[5:108]

                gen_bias_rdma_block(ops, 109, 1, Mp, N_aligned,
                                    bias_dma=bias_dma, has_bias=True,
                                    has_residual=False, residual_dma=0,
                                    ew_surf_stride=0, conv_mode=0)

                _write_pc_tail(ops, 126, TASK_ENABLE_MASK_MODE6)

            return _emit_task(REGCMD_COUNT_MODE6, 'mode6', fill_130)
        else:
            # Generate register commands
            def fill(ops):
                gen_cna_dpu_regcmds(ops, cna, core, dpu)
            return _emit_task(REGCMD_COUNT, 'standard', fill)

    def generate_elementwise(
        self,
        task: 'AbstractElementwiseTask',
        input_a: 'TensorHandle',
        input_b: 'TensorHandle',
        output: 'TensorHandle',
    ) -> Task:
        """Generate RegCmds for elementwise operation (e.g., Add, Mul).

        Args:
            task: AbstractElementwiseTask with operation type and shape
            input_a: First input tensor
            input_b: Second input tensor (for binary ops)
            output: Output tensor

        Returns:
            Task with RegCmds programmed
        """
        # Get DMA addresses
        src_a_dma = input_a.dma_addr
        src_b_dma = input_b.dma_addr
        dst_dma = output.dma_addr

        # Compute cube dimensions from shape
        # Elementwise uses matmul-style [C_al/8, Mp, 1, 8] layout
        # matching rknpu-py reference for all 2D ops
        shape = task.shape
        if len(shape) == 2:
            H, C = shape
        elif len(shape) == 1:
            H, C = 1, shape[0]
        else:
            raise ValueError(f"Elementwise expects 1D or 2D tensor, got shape {shape}")

        C_aligned = align_up(C, 16)
        M_pad = pad_m(H)

        if C_aligned > NPU_MAX_DPU_DIM:
            raise ValueError(f"C_aligned={C_aligned}: max {NPU_MAX_DPU_DIM}")

        # Cube dimensions are 0-based (value - 1).
        # Default to canonical layout for correctness; transposed EW layout
        # can be re-enabled explicitly for performance experiments.
        C2 = 8
        use_transposed = os.getenv("TVM_RKNPU_EW_TRANSPOSED_LAYOUT", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if use_transposed and M_pad % 8 == 0:
            cube_w = C - 1
            cube_h = 0
            cube_c = M_pad - 1
            surf_stride = C * C2 * 2
        else:
            cube_w = 0
            cube_h = M_pad - 1
            cube_c = C_aligned - 1
            surf_stride = M_pad * C2 * 2

        # Map op_type to EW operation
        op_type = task.op_type.lower()
        if op_type not in ('add', 'mul'):
            raise ValueError(f"Unsupported elementwise op '{task.op_type}', expected 'Add' or 'Mul'")

        # Broadcast mode: set EW_SURF_STRIDE=0 so source B channel group 0
        # is repeated for every output channel group.
        ew_surf = 0 if getattr(task, 'broadcast_b', False) else None

        # Generate register commands
        def fill(ops):
            relu = "relu" in getattr(task, "op_name", "").lower()
            gen_ew_op(ops, op_type, cube_w, cube_h, cube_c,
                      src_a_dma, src_b_dma, dst_dma, surf_stride,
                      ew_surf_stride=ew_surf, relu=relu)
        return _emit_task(REGCMD_COUNT_EW, 'ew', fill)

    def generate_pool(
        self,
        task: AbstractMaxPoolTask,
        input_handle: 'TensorHandle',
        output_handle: 'TensorHandle',
    ) -> Task:
        """Generate RegCmds for pooling via PPU hardware.

        Supports both max pooling (task.method=1) and average pooling
        (task.method=0). Average pooling additionally encodes reciprocal
        kernel dimensions in PPU registers 0x6038 and 0x603C.

        Ported from rknpu-py gen_pool() (regcmd.py:1368). Generates 30
        regcmds (26 data + 4 PC tail) for the PPU+PPU_RDMA pipeline.

        Args:
            task: AbstractMaxPoolTask with pool dimensions and method
            input_handle: Input tensor (shape: [H, W, C])
            output_handle: Output tensor (shape: [H_out, W_out, C])

        Returns:
            Task with 30 RegCmds (PPU enable mask 0x0060)
        """
        input_dma = input_handle.dma_addr
        output_dma = output_handle.dma_addr

        C = task.C
        H = task.H
        W = task.W
        kH = task.kH
        kW = task.kW
        stride_h = task.stride_h
        stride_w = task.stride_w
        pad_top = task.pad_top
        pad_bottom = task.pad_bottom
        pad_left = task.pad_left
        pad_right = task.pad_right
        H_out = task.H_out
        W_out = task.W_out

        c2 = 8   # FP16 channel group
        elem = 2  # FP16 bytes per element
        method = task.method  # 1=max, 2=min, 3=avg

        # Reciprocal kernel dims for avg pooling: ceil(2^16 / kernel_dim)
        if method == 3:
            recip_kH = (0x10000 + kH - 1) // kH
            recip_kW = (0x10000 + kW - 1) // kW
        else:
            recip_kH = 0
            recip_kW = 0

        def fill(ops):
            i = 0
            # S_POINTER registers
            ops[i] = npuop(OP_REG_PPU, 0x0E, 0x6004); i += 1
            ops[i] = npuop(OP_REG_PPU_RDMA, 0x0E, 0x7004); i += 1

            # PPU: input/output cube dimensions
            ops[i] = npuop(OP_REG_PPU, (W - 1) & 0x1FFF, 0x600C); i += 1
            ops[i] = npuop(OP_REG_PPU, (H - 1) & 0x1FFF, 0x6010); i += 1
            ops[i] = npuop(OP_REG_PPU, (C - 1) & 0x1FFF, 0x6014); i += 1
            ops[i] = npuop(OP_REG_PPU, (W_out - 1) & 0x1FFF, 0x6018); i += 1
            ops[i] = npuop(OP_REG_PPU, (H_out - 1) & 0x1FFF, 0x601C); i += 1
            ops[i] = npuop(OP_REG_PPU, (C - 1) & 0x1FFF, 0x6020); i += 1

            # Operation mode: method (1=max, 2=min, 3=avg), bit4=flying mode
            mode = (method & 0x3) | (1 << 4)
            ops[i] = npuop(OP_REG_PPU, mode, 0x6024); i += 1

            # Kernel config
            kernel_cfg = (
                (((stride_h - 1) & 0xF) << 20) | (((stride_w - 1) & 0xF) << 16)
                | (((kH - 1) & 0xFF) << 8) | ((kW - 1) & 0xFF)
            )
            ops[i] = npuop(OP_REG_PPU, kernel_cfg, 0x6034); i += 1

            # Reciprocal kernel dims (0 for max, ceil(2^16/k) for avg)
            ops[i] = npuop(OP_REG_PPU, recip_kH, 0x6038); i += 1
            ops[i] = npuop(OP_REG_PPU, recip_kW, 0x603C); i += 1

            # Padding
            pad_cfg = (
                ((pad_bottom & 0xF) << 12) | ((pad_top & 0xF) << 8)
                | ((pad_right & 0xF) << 4) | (pad_left & 0xF)
            )
            ops[i] = npuop(OP_REG_PPU, pad_cfg, 0x6040); i += 1
            ops[i] = npuop(OP_REG_PPU, 0, 0x6044); i += 1
            ops[i] = npuop(OP_REG_PPU, 0, 0x6048); i += 1

            # Output destination
            ops[i] = npuop(OP_REG_PPU, output_dma & 0xFFFFFFFF, 0x6070); i += 1
            out_surf = H_out * W_out * c2 * elem
            ops[i] = npuop(OP_REG_PPU, out_surf, 0x607C); i += 1

            # Data format
            prec = 3  # PREC_FLOAT16
            data_fmt = out_surf | (prec & 0x7)
            ops[i] = npuop(OP_REG_PPU, data_fmt, 0x6084); i += 1
            ops[i] = npuop(OP_REG_PPU, 0x3, 0x60DC); i += 1

            # PPU_RDMA: source read DMA
            ops[i] = npuop(OP_REG_PPU_RDMA, (W - 1) & 0x1FFF, 0x700C); i += 1
            ops[i] = npuop(OP_REG_PPU_RDMA, (H - 1) & 0x1FFF, 0x7010); i += 1
            ops[i] = npuop(OP_REG_PPU_RDMA, (C - 1) & 0x1FFF, 0x7014); i += 1
            ops[i] = npuop(OP_REG_PPU_RDMA, input_dma & 0xFFFFFFFF, 0x701C); i += 1

            in_line_stride = W * c2 * elem
            in_surf_stride = in_line_stride * H
            ops[i] = npuop(OP_REG_PPU_RDMA, in_line_stride, 0x7024); i += 1
            ops[i] = npuop(OP_REG_PPU_RDMA, in_surf_stride, 0x7028); i += 1
            ops[i] = npuop(OP_REG_PPU_RDMA, prec & 0x7, 0x7030); i += 1

            # PC control tail
            _write_pc_tail(ops, 26, TASK_PPU_ENABLE_MASK)

        return _emit_task(REGCMD_COUNT_PPU, 'ppu', fill)

    def generate_maxpool(self, task, input_handle, output_handle):
        """Backward-compatible alias for generate_pool."""
        return self.generate_pool(task, input_handle, output_handle)

    def generate_lut_upload_task(
        self,
        le_table: list[int],
        lo_table: list[int],
    ) -> Task:
        """Generate LUT upload task (uploads LE + LO tables to DPU SRAM).

        Args:
            le_table: 513 Q0.15 entries for Linear/Exponent table
            lo_table: 513 Q0.15 entries for Linear Offset table

        Returns:
            Task with 1101 RegCmds (EW enable mask 0x0018)
        """
        def fill(ops):
            gen_lut_upload(ops, le_table, lo_table)
        return _emit_task(REGCMD_COUNT_LUT_UPLOAD, 'ew', fill)

    def generate_lut_eval_task(
        self,
        shape: tuple[int, ...],
        src_dma: int,
        dst_dma: int,
        bn_mul_cfg: int,
    ) -> Task:
        """Generate LUT evaluation task (applies LUT to input data).

        Uses the same feature layout as elementwise ops: [C/8, M_pad, 1, 8].

        Args:
            shape: Input/output shape (1D or 2D)
            src_dma: DMA address for input
            dst_dma: DMA address for output
            bn_mul_cfg: BN prescale value for input→LUT index mapping

        Returns:
            Task with 74 RegCmds (EW enable mask 0x0018)
        """
        if len(shape) == 2:
            H, C = shape
        elif len(shape) == 1:
            H, C = 1, shape[0]
        else:
            raise ValueError(f"LUT eval expects 1D or 2D tensor, got shape {shape}")

        C_aligned = align_up(C, 16)
        M_pad = pad_m(H)

        if C_aligned > NPU_MAX_DPU_DIM:
            raise ValueError(f"C_aligned={C_aligned}: max {NPU_MAX_DPU_DIM}")

        # Use same transposed layout optimization as elementwise ops
        C2 = 8
        # C==1 (e.g. reciprocal Mx1) is correctness-sensitive with the
        # transposed LUT layout; use canonical feature layout instead.
        if M_pad % 8 == 0 and C > 1:
            cube_w = C - 1
            cube_h = 0
            cube_c = M_pad - 1
            surf_stride = C * C2 * 2
        else:
            cube_w = 0
            cube_h = M_pad - 1
            cube_c = C_aligned - 1
            surf_stride = M_pad * C2 * 2

        def fill(ops):
            gen_lut_eval(ops, cube_w, cube_h, cube_c,
                         src_dma, dst_dma, surf_stride, bn_mul_cfg)
        return _emit_task(REGCMD_COUNT_EW, 'ew', fill)

    def generate_lut_combined_task(
        self,
        le_table: list[int],
        lo_table: list[int],
        shape: tuple[int, ...],
        src_dma: int,
        dst_dma: int,
        lut_params: dict,
    ) -> Task:
        """Generate combined LUT upload + eval as a single task.

        Structure: 69 eval config + 1028 LUT data + 4 PC tail = 1101 total.
        The eval config sets real DMA addresses and active LUT parameters,
        then LUT tables are uploaded to DPU SRAM, then OP_ENABLE triggers
        data processing through the loaded LUT.

        Args:
            le_table: 513 entries for Linear/Exponent table
            lo_table: 513 entries for Linear Offset table
            shape: Input/output shape (1D or 2D)
            src_dma: DMA address for input
            dst_dma: DMA address for output
            lut_params: Activation-specific params dict with keys:
                bn_mul_cfg, ew_cfg, out_cvt_offset, out_cvt_shift,
                lut_le_slope_scale, lut_le_slope_shift,
                lut_lo_slope_scale, lut_lo_slope_shift

        Returns:
            Task with 1101 RegCmds (EW enable mask 0x0018)
        """
        if len(shape) == 2:
            H, C = shape
        elif len(shape) == 1:
            H, C = 1, shape[0]
        else:
            raise ValueError(f"LUT combined expects 1D or 2D tensor, got shape {shape}")

        C_aligned = align_up(C, 16)
        M_pad = pad_m(H)

        if C_aligned > NPU_MAX_DPU_DIM:
            raise ValueError(f"C_aligned={C_aligned}: max {NPU_MAX_DPU_DIM}")

        C2 = 8
        # C==1 (e.g. reciprocal Mx1) is correctness-sensitive with the
        # transposed LUT layout; use canonical feature layout instead.
        if M_pad % 8 == 0 and C > 1:
            cube_w = C - 1
            cube_h = 0
            cube_c = M_pad - 1
            surf_stride = C * C2 * 2
        else:
            cube_w = 0
            cube_h = M_pad - 1
            cube_c = C_aligned - 1
            surf_stride = M_pad * C2 * 2

        def fill(ops):
            gen_lut_combined(ops, le_table, lo_table,
                             cube_w, cube_h, cube_c,
                             src_dma, dst_dma, surf_stride,
                             **lut_params)
        return _emit_task(REGCMD_COUNT_LUT_COMBINED, 'ew', fill)
