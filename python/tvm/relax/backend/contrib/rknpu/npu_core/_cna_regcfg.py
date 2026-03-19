# SPDX-License-Identifier: MIT
"""CNA pipeline configuration — computes register values for each operation.

Descriptor dataclasses (CnaDesc, CoreDesc, DpuDesc), precision configuration,
CBUF allocation, and fill functions that construct descriptor tuples.
"""

from dataclasses import dataclass
import struct as _struct

from .hardware import NPU_CBUF_BANK_SIZE, NPU_CBUF_BANKS
from .alignment import align_up, pad_m


# =============================================================================
# Precision Codes
# =============================================================================

PREC_INT8 = 0
PREC_INT16 = 1
PREC_FLOAT16 = 2
PREC_BFLOAT16 = 3
PREC_INT32 = 4
PREC_FLOAT32 = 5

DIRECT_CONV = 0

DW_MODE3 = 3  # Hardware conv_mode for depthwise convolution


# =============================================================================
# Precision Configuration
# =============================================================================

@dataclass
class PrecisionConfig:
    cna_precision: int
    out_precision: int
    elem_size: int      # bytes per element (2 for FP16, 1 for INT8)
    k_align: int        # K alignment (32 for FP16, 64 for INT8)
    n_align: int        # N alignment (16 for FP16, 32 for INT8)
    c2_divisor: int     # for data_entries (32 for FP16, 64 for INT8)
    core_qd_en: int     # quantize-disable (1 for FP16, 0 for INT8)
    size_e: int         # output element code (3=FP32, 7=INT32, 1=FP16)
    surf_multiplier: int  # 4 for FP32 out, 2 for FP16, 8 for INT32


PREC_FP16_CFG = PrecisionConfig(
    cna_precision=PREC_FLOAT16, out_precision=PREC_FLOAT32,
    elem_size=2, k_align=32, n_align=16, c2_divisor=32,
    core_qd_en=0, size_e=3, surf_multiplier=4,
)

PREC_I8_CFG = PrecisionConfig(
    cna_precision=PREC_INT8, out_precision=PREC_INT32,
    elem_size=1, k_align=64, n_align=32, c2_divisor=64,
    core_qd_en=0, size_e=7, surf_multiplier=8,
)

PREC_BF16_CFG = PrecisionConfig(
    cna_precision=PREC_BFLOAT16, out_precision=PREC_FLOAT32,
    elem_size=2, k_align=32, n_align=16, c2_divisor=32,
    core_qd_en=1, size_e=3, surf_multiplier=4,
)

PREC_INT16_CFG = PrecisionConfig(
    cna_precision=PREC_INT16, out_precision=PREC_FLOAT32,
    elem_size=2, k_align=32, n_align=16, c2_divisor=32,
    core_qd_en=1, size_e=3, surf_multiplier=4,
)


# =============================================================================
# Internal Descriptors
# =============================================================================

@dataclass
class CnaDesc:
    conv_mode: int
    in_precision: int
    proc_precision: int
    kernel_groups: int
    feature_grains: int
    conv_x_stride: int
    conv_y_stride: int
    datain_width: int
    datain_height: int
    datain_channel: int
    dataout_width: int
    dataout_height: int
    dataout_atomics: int
    weight_bytes: int
    weight_bytes_per_kernel: int
    weight_width: int
    weight_height: int
    weight_kernels: int
    weight_bank: int
    data_bank: int
    data_entries: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    feature_base_addr: int
    line_stride: int
    surf_stride: int
    dma_width: int
    dma_height: int
    dma_channel: int
    decompress_addr0: int
    datain_channel_upper: int | None = None


@dataclass
class CoreDesc:
    proc_precision: int
    qd_en: int
    dw_flag: int
    dataout_height: int
    dataout_width: int
    dataout_channel: int


@dataclass
class DpuDesc:
    conv_mode: int
    out_precision: int
    in_precision: int
    proc_precision: int
    dst_base_addr: int
    dst_surf_stride: int
    width: int
    height: int
    channel: int
    bs_bypass: int
    bs_alu_bypass: int
    bs_mul_bypass: int
    bs_relu_bypass: int
    size_e_2: int
    size_e_1: int
    size_e_0: int
    channel_wdma: int
    height_wdma: int
    width_wdma: int
    bn_relu_bypass: int
    bn_relux_compare_en: int
    bn_mul_bypass: int
    bn_alu_bypass: int
    bn_bypass: int
    bn_relux_cmp: int
    fp32tofp16_en: int
    surf_add: int
    ew_cfg_override: int | None = None
    bn_dw_mode: bool = False


# =============================================================================
# Helper Functions
# =============================================================================

def _bs_stage_flags(*, has_bias: bool = False, relu_only: bool = False) -> dict:
    """Return BS stage field values."""
    if relu_only:
        return dict(bs_bypass=0, bs_alu_bypass=1, bs_mul_bypass=1, bs_relu_bypass=0)
    elif has_bias:
        return dict(bs_bypass=0, bs_alu_bypass=0, bs_mul_bypass=1, bs_relu_bypass=1)
    return dict(bs_bypass=1, bs_alu_bypass=1, bs_mul_bypass=1, bs_relu_bypass=1)


def _bn_stage_flags(*, relu: bool = False, clip_max: float = 0.0,
                    dw_mode: bool = False,
                    force_clip6: bool = False) -> dict:
    """Return BN stage field values."""
    bn_dw_mode = dw_mode
    if force_clip6:
        return dict(bn_bypass=0, bn_relu_bypass=0, bn_relux_compare_en=1,
                    bn_mul_bypass=1, bn_alu_bypass=1,
                    bn_relux_cmp=0x40c00000, bn_dw_mode=bn_dw_mode)  # float32(6.0)
    elif relu or clip_max > 0:
        bn_relux_compare_en = 1 if clip_max > 0 else 0
        bn_relux_cmp = (_struct.unpack('<I', _struct.pack('<f', clip_max))[0]
                        if clip_max > 0 else 0)
        return dict(bn_bypass=0, bn_relu_bypass=0, bn_mul_bypass=1,
                    bn_alu_bypass=1, bn_relux_compare_en=bn_relux_compare_en,
                    bn_relux_cmp=bn_relux_cmp, bn_dw_mode=bn_dw_mode)
    return dict(bn_bypass=1, bn_relu_bypass=1, bn_mul_bypass=1,
                bn_alu_bypass=1, bn_relux_compare_en=0,
                bn_relux_cmp=0, bn_dw_mode=bn_dw_mode)


def _ew_override(*, has_residual: bool = False) -> int | None:
    """Return ew_cfg_override value or None."""
    return 0x108202C0 if has_residual else None


# =============================================================================
# CBUF Allocation
# =============================================================================

def _allocate_cbuf(fd_bytes: int, wbpk: int, N: int,
                   fixed_groups: int | None = None,
                   dbl_buf_K: int = 0) -> tuple[int, int, int]:
    """Allocate CBUF banks between feature data and weights.

    Returns (data_banks, weight_banks, kernel_groups).
    """
    fd_banks = (fd_bytes + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE
    if fd_banks > NPU_CBUF_BANKS - 1:
        wb_all = (N * wbpk + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE
        if wb_all <= NPU_CBUF_BANKS - 1:
            fd_banks = NPU_CBUF_BANKS - wb_all
        else:
            fd_banks = NPU_CBUF_BANKS - 1

    weight_banks = NPU_CBUF_BANKS - fd_banks

    MIN_WEIGHT_BANKS_MULTIPASS = 3
    if weight_banks < MIN_WEIGHT_BANKS_MULTIPASS:
        weight_capacity = weight_banks * NPU_CBUF_BANK_SIZE
        if N * wbpk > weight_capacity:
            fd_banks = NPU_CBUF_BANKS - MIN_WEIGHT_BANKS_MULTIPASS
            weight_banks = MIN_WEIGHT_BANKS_MULTIPASS

    if fixed_groups is not None:
        return fd_banks, weight_banks, fixed_groups

    if wbpk > NPU_CBUF_BANK_SIZE:
        raise ValueError("Weight bytes per kernel exceeds one CBUF bank")

    weight_capacity = weight_banks * NPU_CBUF_BANK_SIZE
    kernels_per_group = weight_capacity // wbpk
    if kernels_per_group == 0:
        raise ValueError("Zero kernels per group — N too large for CBUF")
    num_groups = (N + kernels_per_group - 1) // kernels_per_group
    kernel_groups = num_groups - 1

    if num_groups > 1:
        if 32 * dbl_buf_K + NPU_CBUF_BANK_SIZE / 2 > weight_banks * NPU_CBUF_BANK_SIZE:
            raise ValueError("Multi-pass double-buffering constraint violated")

    return fd_banks, weight_banks, kernel_groups


# =============================================================================
# fill_matmul_descriptors
# =============================================================================

def fill_matmul_descriptors(M: int, K: int, N: int,
                            input_dma: int, weights_dma: int, output_dma: int,
                            relu: bool, pc: PrecisionConfig,
                            output_fp16: bool = False,
                            m_offset: int = 0,
                            M_full: int | None = None,
                            has_bias: bool = False,
                            bias_dma: int = 0):
    """Compute CNA/CORE/DPU descriptors for a matmul task.

    Returns (cna, core, dpu) or raises ValueError on CBUF overflow.
    """
    Mp = pad_m(M)

    if Mp + 1 > 1023:
        raise ValueError(
            f"M_padded={Mp}: feature_grains={Mp+1} exceeds 10-bit register max (1023)"
        )

    Mp_full = pad_m(M_full) if M_full is not None else Mp
    C2 = 8
    elem_size = pc.elem_size
    dma_row_bytes = C2 * elem_size

    wbpk = K * elem_size
    fd_bytes = Mp * K * elem_size
    data_bank, weight_bank, kernel_groups = _allocate_cbuf(
        fd_bytes, wbpk, N, dbl_buf_K=K)

    surf = 1 * 4 * (Mp_full // 4 - 1)
    if surf < 0:
        surf += 1

    cna = CnaDesc(
        conv_mode=DIRECT_CONV,
        in_precision=pc.cna_precision,
        proc_precision=pc.cna_precision,
        feature_grains=Mp + 1,
        conv_x_stride=1,
        conv_y_stride=1,
        datain_width=1,
        datain_height=Mp,
        datain_channel=K,
        dataout_width=1,
        dataout_height=Mp,
        dataout_atomics=1 * Mp,
        weight_width=1,
        weight_height=1,
        weight_kernels=N,
        weight_bytes_per_kernel=wbpk,
        weight_bytes=wbpk * N,
        data_bank=data_bank,
        weight_bank=weight_bank,
        kernel_groups=kernel_groups,
        data_entries=(K + pc.c2_divisor - 1) // pc.c2_divisor,
        pad_left=0, pad_top=0, pad_right=0, pad_bottom=0,
        feature_base_addr=input_dma + m_offset * dma_row_bytes,
        line_stride=1 * 4,
        surf_stride=surf,
        dma_width=1,
        dma_height=Mp,
        dma_channel=K,
        decompress_addr0=weights_dma,
    )

    core = CoreDesc(
        proc_precision=pc.cna_precision,
        qd_en=pc.core_qd_en,
        dw_flag=0,
        dataout_height=Mp - 1,
        dataout_width=0,
        dataout_channel=N - 1,
    )

    dst_surf_stride = Mp_full * 1
    if output_fp16:
        out_prec = PREC_FLOAT16
        fp16_en = 1
        se = 1
        surf_add = dst_surf_stride * 2
    else:
        out_prec = pc.out_precision
        fp16_en = 0
        se = pc.size_e
        surf_add = dst_surf_stride * pc.surf_multiplier

    dpu = DpuDesc(
        conv_mode=DIRECT_CONV,
        out_precision=out_prec,
        in_precision=pc.cna_precision,
        proc_precision=pc.cna_precision,
        dst_base_addr=output_dma + m_offset * dma_row_bytes,
        dst_surf_stride=dst_surf_stride,
        width=core.dataout_width,
        height=core.dataout_height,
        channel=core.dataout_channel,
        **(_bs_stage_flags(has_bias=True) if has_bias
           else _bs_stage_flags(relu_only=relu)),
        size_e_2=se, size_e_1=se, size_e_0=se,
        channel_wdma=core.dataout_channel,
        height_wdma=core.dataout_height,
        width_wdma=core.dataout_width,
        **(_bn_stage_flags(relu=relu) if has_bias
           else _bn_stage_flags()),
        fp32tofp16_en=fp16_en,
        surf_add=surf_add,
    )

    return cna, core, dpu


# =============================================================================
# Depthwise conv_mode=3 descriptors
# =============================================================================

def _fill_dw_mode3_descriptors(
    C: int, H: int, W: int, N: int,
    kH: int, kW: int, stride: int,
    pad_top: int, pad_bottom: int, pad_left: int, pad_right: int,
    input_dma: int, weights_dma: int, output_dma: int,
    relu: bool = False, clip_max: float = 0.0,
    has_bias: bool = False, bias_dma: int = 0,
    H_tile_out: int | None = None,
    m_offset: int = 0,
    c_tile_offset: int = 0,
    c_tile_channels: int | None = None,
):
    """Compute CNA/CORE/DPU descriptors for depthwise conv_mode=3.

    For M-tiling (H_tile_out is not None), the caller passes the FULL input
    dimensions (H, pad_top, pad_bottom) and m_offset = h_start * W_out.
    This function computes the per-tile input window, per-tile padding,
    and address offsets internally.

    For channel tiling (c_tile_channels is not None), the caller passes the
    FULL C/N and c_tile_offset/c_tile_channels.  This function offsets DMA
    addresses and uses c_tile_channels for all dimension/CBUF calculations.
    """
    pc = PREC_FP16_CFG
    elem_size = pc.elem_size
    C2 = 8  # FP16 channel group size

    assert N == C, f"Depthwise requires N==C, got N={N}, C={C}"

    # Channel tiling: offset DMA addresses and reduce C/N to tile size.
    # Must be done before any dimension calculations.
    if c_tile_channels is not None:
        # Compute M_padded_full from FULL spatial dims (for output stride)
        H_out_full_global = (H + pad_top + pad_bottom - kH) // stride + 1
        W_out_global = (W + pad_left + pad_right - kW) // stride + 1
        M_padded_full_global = pad_m(H_out_full_global * W_out_global)

        # Input [C/8, H, W, 8]: skip c_tile_offset/8 surfaces
        input_dma += c_tile_offset * H * W * elem_size
        # Weight flat [C_al32 * kH * kW]: skip c_tile_offset channels
        weights_dma += c_tile_offset * kH * kW * elem_size
        # Output [N_al/8, M_pad, 1, 8]: skip c_tile_offset/8 surfaces
        output_dma += c_tile_offset * M_padded_full_global * elem_size
        # Bias [N] FP32: skip c_tile_offset values
        if has_bias:
            bias_dma += c_tile_offset * 4

        C = c_tile_channels
        N = c_tile_channels

    # DW mode 3 requires C to be 32-aligned for CNA channel group processing.
    # Non-32-aligned C (e.g. 144) causes hardware timeout.
    C_al32 = align_up(C, 32)

    H_out_full = (H + pad_top + pad_bottom - kH) // stride + 1
    W_out = (W + pad_left + pad_right - kW) // stride + 1
    M_padded_full = pad_m(H_out_full * W_out)
    N_aligned = align_up(C_al32, 16)

    if H_tile_out is not None:
        # M-tiling: compute per-tile input window and padding
        H_out_eff = H_tile_out
        h_start = m_offset // W_out  # starting output row

        # Input row range needed for output rows [h_start, h_start + H_tile_out)
        in_top = h_start * stride - pad_top
        in_bot = (h_start + H_tile_out - 1) * stride + kH - 1 - pad_top

        tile_pad_top = max(0, -in_top)
        tile_pad_bot = max(0, in_bot - (H - 1))
        actual_in_start = max(0, in_top)
        actual_in_end = min(H - 1, in_bot)
        tile_H = actual_in_end - actual_in_start + 1

        # Address offsets: input rows start at actual_in_start
        # In [C/C2, M_pad, 1, C2] layout, row offset within each surface
        # is actual_in_start * W positions * C2 * elem_size bytes
        input_addr = input_dma + actual_in_start * W * C2 * elem_size
        output_addr = output_dma + m_offset * C2 * elem_size
    else:
        H_out_eff = H_out_full
        tile_H = H
        tile_pad_top = pad_top
        tile_pad_bot = pad_bottom
        input_addr = input_dma
        output_addr = output_dma

    M = H_out_eff * W_out

    K_eff = C_al32 * kH * kW
    wbpk = K_eff * elem_size

    fd_bytes = tile_H * W * C_al32 * elem_size
    data_bank, weight_bank, kernel_groups = _allocate_cbuf(
        fd_bytes, wbpk, 1, fixed_groups=0)

    # line_stride and surf_stride use FULL H (DRAM layout unchanged by tiling)
    line_stride = W * 4
    surf_stride = max(line_stride * (H - 4) // 4, (H - 1) * 4)

    cna = CnaDesc(
        conv_mode=DW_MODE3,
        in_precision=pc.cna_precision,
        proc_precision=pc.cna_precision,
        datain_width=W,
        datain_height=tile_H,
        datain_channel=C_al32,
        datain_channel_upper=(C_al32 - 1) % 64 if C_al32 > 64 else None,
        dataout_width=W_out,
        dataout_height=H_out_eff,
        dataout_atomics=M,
        weight_width=kW,
        weight_height=kH,
        weight_kernels=1,
        weight_bytes_per_kernel=wbpk,
        weight_bytes=wbpk,
        conv_x_stride=stride,
        conv_y_stride=stride,
        pad_left=pad_left, pad_top=tile_pad_top,
        pad_right=pad_right, pad_bottom=tile_pad_bot,
        feature_grains=C_al32 // 32,
        data_bank=data_bank,
        weight_bank=weight_bank,
        kernel_groups=kernel_groups,
        data_entries=W * (C_al32 // 32),
        feature_base_addr=input_addr,
        line_stride=line_stride,
        surf_stride=surf_stride,
        dma_width=W,
        dma_height=tile_H,
        dma_channel=C_al32,
        decompress_addr0=weights_dma,
    )

    core = CoreDesc(
        proc_precision=pc.cna_precision,
        qd_en=0,
        dw_flag=1,
        dataout_height=H_out_eff - 1,
        dataout_width=W_out - 1,
        dataout_channel=N_aligned - 1,
    )

    dpu = DpuDesc(
        conv_mode=DW_MODE3,
        out_precision=PREC_FLOAT16,
        in_precision=pc.cna_precision,
        proc_precision=pc.cna_precision,
        fp32tofp16_en=1,
        dst_base_addr=output_addr,
        dst_surf_stride=M_padded_full,
        width=W_out - 1,
        height=H_out_eff - 1,
        channel=N_aligned - 1,
        size_e_2=3, size_e_1=3, size_e_0=3,
        surf_add=M_padded_full * 4,
        width_wdma=core.dataout_width,
        height_wdma=core.dataout_height,
        channel_wdma=core.dataout_channel,
        **_bn_stage_flags(relu=relu, clip_max=clip_max, dw_mode=True),
        **_bs_stage_flags(has_bias=has_bias),
    )

    return cna, core, dpu


# =============================================================================
# fill_conv_mode0_descriptors (and mode 3 helper)
# =============================================================================

def _fill_conv_mode3_descriptors(
    C: int, H: int, W: int, N: int,
    stride: int,
    input_dma: int, weights_dma: int, output_dma: int,
    relu: bool, clip_max: float,
    has_bias: bool, bias_dma: int,
    has_residual: bool,
    H_tile_out: int | None,
    m_offset: int,
    mode3_n_idx: int,
):
    """Compute CNA/CORE/DPU descriptors for Mode 3 (CHANNEL_EXT) convolution."""
    pc = PREC_FP16_CFG
    elem_size = pc.elem_size
    C2 = 8

    H_out = (H + 1 + 0 - 3) // stride + 1
    W_out = (W + 1 + 0 - 3) // stride + 1
    M_padded_full = pad_m(H_out * W_out)
    H_out_eff = H_tile_out if H_tile_out is not None else H_out

    wbpk = C * elem_size * 9
    wt_banks = max(1, (wbpk + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE)

    line_stride = W * 4
    surf = line_stride * (H // 4 - 1)
    if surf < 0:
        surf += 1

    cna = CnaDesc(
        conv_mode=3,
        in_precision=pc.cna_precision,
        proc_precision=pc.cna_precision,
        datain_width=W,
        datain_height=H,
        datain_channel=C,
        datain_channel_upper=min(C, 64) - 1,
        dataout_width=W_out,
        dataout_height=H_out_eff,
        dataout_atomics=H_out_eff * W_out,
        weight_width=3,
        weight_height=3,
        weight_kernels=1,
        weight_bytes_per_kernel=wbpk,
        weight_bytes=wbpk,
        conv_x_stride=stride,
        conv_y_stride=stride,
        pad_left=1, pad_top=1, pad_right=0, pad_bottom=0,
        feature_grains=N - 1,
        data_bank=NPU_CBUF_BANKS - wt_banks,
        weight_bank=wt_banks,
        kernel_groups=0,
        data_entries=W * (C // 32),
        feature_base_addr=input_dma + m_offset * C2 * elem_size,
        line_stride=line_stride,
        surf_stride=surf,
        dma_width=W,
        dma_height=H,
        dma_channel=align_up(C, 32),
        decompress_addr0=weights_dma + mode3_n_idx * C * elem_size * 9,
    )

    core = CoreDesc(
        proc_precision=pc.cna_precision,
        qd_en=0,
        dw_flag=0,
        dataout_channel=C - 1,
        dataout_height=H_out_eff - 1,
        dataout_width=W_out - 1,
    )

    dpu = DpuDesc(
        conv_mode=3,
        out_precision=PREC_FLOAT16,
        in_precision=pc.cna_precision,
        proc_precision=pc.cna_precision,
        fp32tofp16_en=1,
        dst_base_addr=output_dma + m_offset * C2 * elem_size,
        dst_surf_stride=M_padded_full,
        width=W_out - 1,
        height=H_out_eff - 1,
        channel=C - 1,
        size_e_2=3, size_e_1=3, size_e_0=3,
        surf_add=M_padded_full,
        channel_wdma=C - 1,
        width_wdma=W_out - 1,
        height_wdma=H_out_eff - 1,
        **_bn_stage_flags(relu=relu, clip_max=clip_max, force_clip6=True),
        **_bs_stage_flags(has_bias=has_bias),
        ew_cfg_override=_ew_override(has_residual=has_residual),
    )

    return cna, core, dpu


def fill_conv_mode0_descriptors(
    C: int, H: int, W: int, N: int,
    kH: int, kW: int, stride: int,
    pad_top: int, pad_bottom: int, pad_left: int, pad_right: int,
    input_dma: int, weights_dma: int, output_dma: int,
    relu: bool = False, clip_max: float = 0.0,
    has_bias: bool = False, bias_dma: int = 0,
    has_residual: bool = False,
    groups: int = 1,
    H_tile_out: int | None = None,
    m_offset: int = 0,
    is_depthwise: bool = False,
    conv_mode: int = 0,
    mode3_n_idx: int = 0,
    c_tile_offset: int = 0,
    c_tile_channels: int | None = None,
):
    """Compute CNA/CORE/DPU descriptors for Mode 0 or Mode 3 convolution.

    Mode 0 uses the same 112-regcmd template as matmul (gen_cna_dpu_regcmds).
    The CNA performs im2col internally based on spatial dimension registers.

    For depthwise (is_depthwise=True), delegates to conv_mode=3 native DW
    where the CNA does depthwise im2col with weight_kernels=1 and K_eff=C*kH*kW.

    Mode 3 (CHANNEL_EXT) uses a virtual 3x3 kernel over channel-extended
    spatial positions, with weight_kernels=1 and FP32 DPU pipeline.
    mode3_n_idx selects which output channel this task computes.

    Returns (cna, core, dpu) or raises ValueError on CBUF overflow.
    """
    pc = PREC_FP16_CFG
    elem_size = pc.elem_size  # 2 for FP16

    if is_depthwise:
        return _fill_dw_mode3_descriptors(
            C=C, H=H, W=W, N=N,
            kH=kH, kW=kW, stride=stride,
            pad_top=pad_top, pad_bottom=pad_bottom,
            pad_left=pad_left, pad_right=pad_right,
            input_dma=input_dma, weights_dma=weights_dma, output_dma=output_dma,
            relu=relu, clip_max=clip_max,
            has_bias=has_bias, bias_dma=bias_dma,
            H_tile_out=H_tile_out, m_offset=m_offset,
            c_tile_offset=c_tile_offset, c_tile_channels=c_tile_channels,
        )

    if conv_mode == 3:
        return _fill_conv_mode3_descriptors(
            C=C, H=H, W=W, N=N, stride=stride,
            input_dma=input_dma, weights_dma=weights_dma, output_dma=output_dma,
            relu=relu, clip_max=clip_max,
            has_bias=has_bias, bias_dma=bias_dma,
            has_residual=has_residual,
            H_tile_out=H_tile_out, m_offset=m_offset,
            mode3_n_idx=mode3_n_idx,
        )

    C2 = 8  # FP16 channel group size
    C_aligned = align_up(C, 32)
    N_aligned = align_up(N, 16)

    H_out = (H + pad_top + pad_bottom - kH) // stride + 1
    W_out = (W + pad_left + pad_right - kW) // stride + 1
    M_padded_full = pad_m(H_out * W_out)

    H_out_eff = H_tile_out if H_tile_out is not None else H_out
    M = H_out_eff * W_out
    Mp = pad_m(M)

    # K_eff_al32 retained for double-buffering constraint in _allocate_cbuf
    if kH == 1 and kW == 1:
        K_eff = C_aligned
    else:
        K_eff = C_aligned * kH * kW
    K_eff_al32 = align_up(K_eff, 32)

    # wbpk must match the WeightIndexFP16 tile stride (32 elements per kernel).
    # Use K_eff (based on C_aligned) so the CNA reads the correct per-kernel
    # stride for any channel count.
    wbpk = K_eff * elem_size

    # ---- Per-tile source addressing for M-tiled convolutions ----
    # For 1x1 convs: each tile reads exactly H_tile_out rows (no overlap).
    # For kH>1 convs (3x3 etc.): each tile needs extra rows for the kernel
    # window.  Adjust feature_base_addr, H_in_eff, and padding per tile
    # so the CNA DMA doesn't read beyond the source tensor.
    tile_pad_top = pad_top
    tile_pad_bottom = pad_bottom
    src_addr_offset = m_offset * C2 * elem_size  # default source address offset

    if H_tile_out is not None and kH == 1 and kW == 1:
        H_in_eff = H_tile_out
    elif H_tile_out is not None and (kH > 1 or kW > 1):
        # M-tiled spatial conv: compute per-tile input region
        h_out_start = m_offset // W_out  # starting output row for this tile

        # Input row range needed (in original unpadded coords, can be <0 or >=H)
        h_in_first = h_out_start * stride - pad_top
        h_in_last = (h_out_start + H_tile_out - 1) * stride + kH - 1 - pad_top

        # Clamp to valid data rows and compute per-tile padding
        data_start = max(0, h_in_first)
        data_end = min(H - 1, h_in_last)
        tile_pad_top = max(0, -h_in_first)
        tile_pad_bottom = max(0, h_in_last - (H - 1))

        H_in_eff = data_end - data_start + 1
        src_addr_offset = data_start * W * C2 * elem_size
    else:
        H_in_eff = H

    fd_bytes = H_in_eff * W * C_aligned * elem_size  # CBUF allocation uses C_aligned
    # Trace always uses kernel_groups=0; hardware handles weight multi-pass
    # automatically when N > weight_bank_capacity / wbpk.
    data_bank, weight_bank, kernel_groups = _allocate_cbuf(
        fd_bytes, wbpk, N, fixed_groups=0)

    # data_entries (epl) = entries per line, using C_aligned to match K_eff.
    data_entries = max(1, -(-W * C_aligned // 32))  # ceil(W * C_aligned / 32)

    # H-tiling: when full spatial input exceeds CBUF data capacity,
    # reduce effective H to the max rows fitting in data banks.
    # Confirmed by 9 hardware traces: H_tile = floor(db * 512 / epl).
    if H_tile_out is None and data_entries > 0:
        max_h_tile = data_bank * 512 // data_entries
        if H > max_h_tile:
            H_in_eff = max_h_tile
            # First H-tile: has pad_top but no pad_bottom.
            # General formula works for all kernel sizes and strides.
            H_out_eff = (max_h_tile + pad_top - kH) // stride + 1
            M = H_out_eff * W_out
            Mp = pad_m(M)

    # feature_grains: H-dimension tiling in CBUF.
    # Reverse-engineered from 102 hardware traces (sweeps 1-13).
    # Three regimes based on CBUF data bank pressure:
    #   R1 (unconstrained): fg = fg_max when data fits comfortably
    #   R2 (slightly constrained): fg = fg_max - 1
    #   R3 (heavily constrained): fg = max(9, ceil(1120/epl))
    fg_max = H_in_eff + kH + tile_pad_top
    if data_bank <= 1:
        fg = fg_max
    elif fg_max * data_entries <= 768:
        fg = fg_max  # R1: data small enough for unconstrained operation
    else:
        cap = max(9, -(-1120 // data_entries))  # ceil(1120/epl)
        fg = min(fg_max - 1, cap)
    fg = min(fg, 1023)

    line_stride = W * 4
    surf = W * (H - 4)
    if surf < 0:
        surf = 0

    cna = CnaDesc(
        conv_mode=0,
        in_precision=pc.cna_precision,
        proc_precision=pc.cna_precision,
        datain_width=W,
        datain_height=H_in_eff,
        datain_channel=C_aligned,  # 32-aligned to match weight tile format
        datain_channel_upper=(C_aligned - 1) % 64,  # matches C_aligned
        dataout_width=W_out,
        dataout_height=H_out_eff,
        dataout_atomics=M,
        weight_width=kW,
        weight_height=kH,
        weight_kernels=N_aligned,
        weight_bytes_per_kernel=wbpk,
        weight_bytes=wbpk * N_aligned,
        conv_x_stride=stride,
        conv_y_stride=stride,
        pad_left=pad_left, pad_top=tile_pad_top,
        pad_right=pad_right, pad_bottom=tile_pad_bottom,
        feature_grains=fg,
        data_bank=data_bank,
        weight_bank=weight_bank,
        kernel_groups=kernel_groups,
        data_entries=data_entries,
        feature_base_addr=input_dma + src_addr_offset,
        line_stride=line_stride,
        surf_stride=surf,
        dma_width=W,
        dma_height=H_in_eff,
        dma_channel=C_aligned,   # 32-aligned to match weight tile format
        decompress_addr0=weights_dma,
    )

    core = CoreDesc(
        proc_precision=pc.cna_precision,
        qd_en=pc.core_qd_en,
        dw_flag=0,
        dataout_channel=N_aligned - 1,
        dataout_height=H_out_eff - 1,
        dataout_width=W_out - 1,
    )

    dpu = DpuDesc(
        conv_mode=0,
        out_precision=PREC_FLOAT16,
        in_precision=pc.cna_precision,
        proc_precision=pc.cna_precision,
        fp32tofp16_en=1,
        dst_base_addr=output_dma + m_offset * C2 * elem_size,
        dst_surf_stride=M_padded_full,
        width=W_out - 1,
        height=H_out_eff - 1,
        channel=N_aligned - 1,
        size_e_2=1, size_e_1=1, size_e_0=1,
        surf_add=M_padded_full * 2,
        channel_wdma=core.dataout_channel,
        width_wdma=core.dataout_width,
        height_wdma=core.dataout_height,
        **_bn_stage_flags(relu=relu, clip_max=clip_max),
        **_bs_stage_flags(has_bias=has_bias),
        ew_cfg_override=_ew_override(has_residual=has_residual),
    )

    return cna, core, dpu
