# SPDX-License-Identifier: MIT
"""Rockchip RK3588 NPU hardware definitions.

This module defines the hardware model for the RK3588 NPU, using Rockchip's
terminology and register offsets from the reverse-engineered driver.

Hardware Architecture:
    - NPU has 3 cores that can execute in parallel
    - Each core has multiple sub-units (Blocks): PC, CNA, CORE, DPU
    - Tasks are sequences of register commands (RegCmds) submitted via IOCTL
    - Multiple tasks can be chained via PC (Program Counter)

References:
    - Linux driver: rockchip-linux-kernel-drivers-rknpu-6.1
"""

from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Block Types (NPU Sub-units)
# =============================================================================

class BlockType(Enum):
    """NPU hardware block IDs.

    Each block corresponds to a functional unit in the NPU pipeline.
    The block ID is encoded in the upper 16 bits of each RegCmd.

    From driver: rknpu_drv.h (struct rknpu_oper_config)
    """
    PC      = 0x0101   # Program Counter (fetches regcmds from memory)
    CNA     = 0x0201   # Convolution/Network Accelerator (matmul/conv)
    CORE    = 0x0801   # Core (MAC array configuration)
    DPU     = 0x1001   # Data Processing Unit (post-processing, bias, activation)
    DPU_RDMA = 0x2001  # DPU DMA (reads bias/residual data)
    PPU      = 0x4001  # Post-Processing Unit (pooling, etc.)
    PPU_RDMA = 0x8001  # PPU DMA

    def __int__(self) -> int:
        return self.value


# =============================================================================
# Register Offsets (from rknpu driver)
# =============================================================================

# PC (Program Counter) registers
PC_OPERATION_ENABLE   = 0x0008  # Enable PC operation
PC_BASE_ADDRESS       = 0x0010  # Base address of regcmd blob
PC_REGISTER_AMOUNTS   = 0x0014  # Number of regcmds to fetch

# CNA (Convolution/Network Accelerator) registers
CNA_S_POINTER         = 0x1004  # Source pointer
CNA_CONV_CON1         = 0x100C  # Convolution config 1
CNA_CONV_CON2         = 0x1010  # Convolution config 2
CNA_CONV_CON3         = 0x1014  # Convolution config 3
CNA_DATA_SIZE0        = 0x1020  # Input data size 0
CNA_DATA_SIZE1        = 0x1024  # Input data size 1
CNA_DATA_SIZE2        = 0x1028  # Input data size 2
CNA_DATA_SIZE3        = 0x102C  # Input data size 3
CNA_WEIGHT_SIZE0      = 0x1030  # Weight size 0
CNA_WEIGHT_SIZE1      = 0x1034  # Weight size 1
CNA_WEIGHT_SIZE2      = 0x1038  # Weight size 2
CNA_CBUF_CON0         = 0x1040  # Coefficient buffer config 0
CNA_CBUF_CON1         = 0x1044  # Coefficient buffer config 1
CNA_CVT_CON0          = 0x104C  # Conversion config 0
CNA_CVT_CON1          = 0x1050  # Conversion config 1
CNA_CVT_CON2          = 0x1054  # Conversion config 2
CNA_CVT_CON3          = 0x1058  # Conversion config 3
CNA_CVT_CON4          = 0x105C  # Conversion config 4
CNA_FC_CON0           = 0x1060  # Fully-connected config 0
CNA_FC_CON1           = 0x1064  # Fully-connected config 1
CNA_PAD_CON0          = 0x1068  # Padding config 0
CNA_FEATURE_DATA_ADDR = 0x1070  # Feature data address (INPUT)
CNA_FC_CON2           = 0x1074  # Fully-connected config 2
CNA_DMA_CON0          = 0x1078  # DMA config 0
CNA_DMA_CON1          = 0x107C  # DMA config 1
CNA_DMA_CON2          = 0x1080  # DMA config 2
CNA_FC_DATA_SIZE0     = 0x1084  # FC data size 0
CNA_FC_DATA_SIZE1     = 0x1088  # FC data size 1
CNA_DCOMP_CTRL        = 0x1100  # Decompression control
CNA_DCOMP_REGNUM      = 0x1104  # Decompression register num
CNA_DCOMP_ADDR0       = 0x1110  # Decompression address 0
CNA_DCOMP_AMOUNT      = 0x1140  # Decompression amount
CNA_DCOMP_AMOUNT15    = 0x117C  # Decompression amount (last of 16)
CNA_CVT_CON5          = 0x1180  # Conversion config 5
CNA_PAD_CON1          = 0x1184  # Padding config 1

# CORE (MAC Array) registers
CORE_S_POINTER        = 0x3004  # Source pointer
CORE_MISC_CFG         = 0x3010  # Miscellaneous config
CORE_DATAOUT_SIZE_0   = 0x3014  # Data output size 0
CORE_DATAOUT_SIZE_1   = 0x3018  # Data output size 1
CORE_CLIP_TRUNCATE    = 0x301C  # Clip/truncate config
CORE_3030             = 0x3030  # Unknown config register

# DPU (Data Processing Unit) registers
DPU_S_POINTER            = 0x4004  # Source pointer
DPU_FEATURE_MODE_CFG     = 0x400C  # Feature mode config
DPU_DATA_FORMAT          = 0x4010  # Data format
DPU_OFFSET_PEND          = 0x4014  # Offset pending
DPU_DST_BASE_ADD         = 0x4020  # Destination base address (OUTPUT)
DPU_DST_SURF_STRIDE      = 0x4024  # Destination surface stride
DPU_DATA_CUBE_WIDTH      = 0x4030  # Data cube width
DPU_DATA_CUBE_HEIGHT     = 0x4034  # Data cube height
DPU_DATA_CUBE_NOTCH      = 0x4038  # Data cube notch
DPU_DATA_CUBE_CHANNEL    = 0x403C  # Data cube channel
DPU_BS_CFG               = 0x4040  # Bias/scale config
DPU_BS_ALU_CFG           = 0x4044  # Bias/scale ALU config
DPU_BS_MUL_CFG           = 0x4048  # Bias/scale multiply config
DPU_BS_RELUX_CMP         = 0x404C  # Bias/scale ReLU compare
DPU_BS_OW_CFG            = 0x4050  # Bias/scale output weight config
DPU_BS_OW_OP             = 0x4054  # Bias/scale output weight operation
DPU_WDMA_SIZE_0          = 0x4058  # Write DMA size 0
DPU_WDMA_SIZE_1          = 0x405C  # Write DMA size 1
DPU_BN_CFG               = 0x4060  # Batch-norm config
DPU_BN_ALU_CFG           = 0x4064  # Batch-norm ALU config
DPU_BN_MUL_CFG           = 0x4068  # Batch-norm multiply config
DPU_BN_RELUX_CMP         = 0x406C  # Batch-norm ReLU compare
DPU_EW_CFG               = 0x4070  # Elementwise config
DPU_EW_CVT_OFFSET        = 0x4074  # Elementwise conversion offset
DPU_EW_CVT_SCALE         = 0x4078  # Elementwise conversion scale
DPU_EW_RELUX_CMP         = 0x407C  # Elementwise ReLU compare
DPU_OUT_CVT_OFFSET       = 0x4080  # Output conversion offset
DPU_OUT_CVT_SCALE        = 0x4084  # Output conversion scale
DPU_OUT_CVT_SHIFT        = 0x4088  # Output conversion shift
DPU_EW_OP_VALUE_0        = 0x4090  # Elementwise operation value 0
DPU_EW_OP_VALUE_7        = 0x40AC  # Elementwise operation value 7
DPU_SURFACE_ADD          = 0x40C0  # Surface address
DPU_40C4                 = 0x40C4  # Unknown DPU register
DPU_LUT_ACCESS_CFG       = 0x4100  # LUT access config
DPU_LUT_ACCESS_DATA      = 0x4104  # LUT access data
DPU_LUT_CFG              = 0x4108  # LUT config
DPU_LUT_INFO             = 0x410C  # LUT info
DPU_LUT_LE_START         = 0x4110  # LUT linear/exponent start
DPU_LUT_LE_END           = 0x4114  # LUT linear/exponent end
DPU_LUT_LO_START         = 0x4118  # LUT log start
DPU_LUT_LO_END           = 0x411C  # LUT log end
DPU_LUT_LE_SLOPE_SCALE   = 0x4120  # LUT LE slope scale
DPU_LUT_LE_SLOPE_SHIFT   = 0x4124  # LUT LE slope shift
DPU_LUT_LO_SLOPE_SCALE   = 0x4128  # LUT LO slope scale
DPU_LUT_LO_SLOPE_SHIFT   = 0x412C  # LUT LO slope shift

# DPU_RDMA registers (for bias/residual reads)
DPU_RDMA_S_POINTER       = 0x5004  # DPU_RDMA s_pointer
DPU_RDMA_500C            = 0x500C  # cube width (add mode)
DPU_RDMA_5010            = 0x5010  # cube height
DPU_RDMA_5014            = 0x5014  # cube channel
DPU_RDMA_SRC_A_ADDR      = 0x5018  # source A DMA address
DPU_RDMA_501C            = 0x501C  # Unknown
DPU_RDMA_5020            = 0x5020  # Unknown
DPU_RDMA_5028            = 0x5028  # Unknown
DPU_RDMA_502C            = 0x502C  # Unknown
DPU_RDMA_5034            = 0x5034  # RDMA config / enable flags
DPU_RDMA_EW_SRC_ADDR     = 0x5038  # EW operand (source B) DMA address
DPU_RDMA_EW_SURF_STRIDE  = 0x5040  # EW surface stride
DPU_RDMA_LINE_STRIDE     = 0x5044  # Line stride
DPU_RDMA_SURF_STRIDE     = 0x5048  # Surface stride
DPU_RDMA_504C            = 0x504C  # Unknown
DPU_RDMA_5064            = 0x5064  # Unknown
DPU_RDMA_BURST_CFG       = 0x5068  # Burst config
DPU_RDMA_506C            = 0x506C  # Unknown


# =============================================================================
# Task Constants
# =============================================================================

# CNA+DPU task (matmul without bias)
REGCMD_COUNT          = 112    # Number of 64-bit regcmds per task
REGCFG_AMOUNT         = 108    # Number of data commands (kernel adds 4)
TASK_ENABLE_MASK      = 0x000D # PC + CNA + DPU
TASK_INT_MASK         = 0x0300 # Interrupt mask (DPU group 0+1)
TASK_INT_CLEAR        = 0x1FFFF # Clear all interrupts

# CNA+DPU+DPU_RDMA task (matmul with bias via DPU_RDMA)
REGCMD_COUNT_MODE6    = 130    # Regcmds per bias task
REGCFG_AMOUNT_MODE6   = 126    # Data commands for bias task
TASK_ENABLE_MASK_MODE6 = 0x001D  # CNA + DPU + DPU_RDMA
TASK_INT_MASK_MODE6   = 0x0300 # DPU interrupt
TASK_INT_CLEAR_MODE6  = 0x1FFFF # Clear all interrupts

# EW task (elementwise: DPU + DPU_RDMA only)
REGCMD_COUNT_EW       = 74     # Regcmds per elementwise task
TASK_EW_ENABLE_MASK   = 0x0018 # DPU + DPU_RDMA
TASK_EW_INT_MASK      = 0x0300 # DPU interrupt
TASK_EW_INT_CLEAR     = 0x1FFFF # Clear all interrupts

# PPU task (pooling: PPU + PPU_RDMA)
REGCMD_COUNT_PPU      = 30     # Regcmds per PPU task
TASK_PPU_ENABLE_MASK  = 0x0060 # PPU + PPU_RDMA
TASK_PPU_INT_MASK     = 0x0C00 # PPU interrupt
TASK_PPU_INT_CLEAR    = 0x1FFFF # Clear all interrupts

# LUT upload task (DPU + DPU_RDMA, uploads 513 LE + 513 LO table entries)
# Padded to even count for PC chain data-amount alignment (scale=2 on RK3588).
REGCMD_COUNT_LUT_UPLOAD   = 1102   # 69 preamble + 1028 LUT data + 1 NOP + 4 PC tail
REGCFG_AMOUNT_LUT_UPLOAD  = 1098   # Data commands (kernel adds 4 extra)
REGCFG_AMOUNT_EW          = 69     # EW data commands (kernel adds 4 for 73 total)
# Combined LUT: 69 eval config + 1028 LUT data + 1 NOP + 4 PC tail = 1102
REGCMD_COUNT_LUT_COMBINED = 1102
REGCFG_AMOUNT_LUT_COMBINED = 1098  # Data commands for combined task


# =============================================================================
# Hardware Limits
# =============================================================================

NPU_CBUF_BANK_SIZE    = 32768  # 32 KB per coefficient buffer bank
NPU_CBUF_BANKS        = 12     # 12 banks per core
NPU_MAX_N_PER_SUBMIT  = 8192   # Max output channels per submit
NPU_MAX_DPU_DIM       = 8192   # Max DPU dimension


# =============================================================================
# Register Command
# =============================================================================

@dataclass(frozen=True)
class RegCmd:
    """A single NPU register command.

    Register commands are 64-bit values written to NPU registers to configure
    the hardware for a single operation. Each task consists of a sequence of
    regcmds that are fetched and executed by the PC (Program Counter) unit.

    Format (little-endian uint64):
        [block_id:16][value:32][reg_offset:16]

    The hardware reads this format as:
        - Upper 16 bits: which block to configure (PC, CNA, DPU, etc.)
        - Middle 32 bits: value to write
        - Lower 16 bits: which register within the block

    Attributes:
        block: Which hardware block to configure (BlockType enum or raw int for special opcodes)
        value: 32-bit value to write to register
        offset: Register offset within the block
    """
    block: BlockType | int  # Allow raw int for special opcodes (OP_NONE=0, OP_40=0x41, OP_ENABLE=0x81)
    value: int       # 32-bit register value
    offset: int      # 16-bit register offset

    def __post_init__(self):
        if not 0 <= self.value <= 0xFFFFFFFF:
            raise ValueError(f"value must be 32-bit, got {self.value:#x}")
        if not 0 <= self.offset <= 0xFFFF:
            raise ValueError(f"offset must be 16-bit, got {self.offset:#x}")

    def to_u64(self) -> int:
        """Encode to 64-bit integer in hardware format."""
        return (int(self.block) << 48) | ((self.value & 0xFFFFFFFF) << 16) | (self.offset & 0xFFFF)

    def patch_value(self, new_value: int) -> 'RegCmd':
        """Create new RegCmd with patched value (for runtime DMA address patching)."""
        return RegCmd(self.block, new_value, self.offset)


# =============================================================================
# Hardware Task
# =============================================================================

@dataclass
class Task:
    """A hardware task: sequence of regcmds + metadata.

    This corresponds to the rknpu_task struct (40 bytes) defined in the
    Rockchip NPU driver. Tasks are submitted to the driver via IOCTL and
    executed by the NPU hardware.

    Attributes:
        regcmds: List of register commands
        enable_mask: Bitmask of which blocks to enable
        int_mask: Interrupt mask (which interrupts to wait for)
        int_clear: Interrupt clear bits
        regcmd_dma_addr: DMA address of regcmd blob (set during lowering)
        core_mask: Which NPU cores to use (0x1 = core 0, 0x7 = all cores)
    """
    regcmds: list[RegCmd]
    enable_mask: int
    int_mask: int
    int_clear: int

    # Runtime fields (set during lowering/execution)
    regcmd_dma_addr: int | None = None
    core_mask: int | None = None

    @property
    def regcmd_count(self) -> int:
        """Number of regcmds in this task."""
        return len(self.regcmds)
