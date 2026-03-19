# SPDX-License-Identifier: MIT
"""Abstract tasks: NPU-specific but address-agnostic.

AbstractTasks represent NPU operations without concrete memory addresses.
They can be optimized, fused, and transformed before lowering to concrete Tasks.
"""

from dataclasses import dataclass
from abc import ABC
from typing import TYPE_CHECKING
import numpy as np

from .alignment import align_up, pad_m


class Precision:
    """Data type precision for operations."""
    FP32 = "float32"
    FP16 = "float16"
    INT8 = "int8"
    INT16 = "int16"

if TYPE_CHECKING:
    from .handles import TensorHandle


# =============================================================================
# Abstract Task Base Class
# Dtype byte sizes used by fix_tensor_sizes across task types.
_DTYPE_SIZES = {"float32": 4, "float16": 2, "int32": 4, "int16": 2, "int8": 1, "uint8": 1}


def _ensure_min_size(tensor: 'TensorHandle', needed: int) -> None:
    """Set tensor.size_bytes = max(current, needed) — never shrink a shared buffer."""
    tensor.size_bytes = max(tensor.size_bytes, needed)


# =============================================================================

class AbstractTask(ABC):
    """Abstract NPU task without concrete addresses.

    Attributes:
        op_name: Name of the operation this task executes
        inputs: List of input tensor handles
        outputs: List of output tensor handles
        allow_fuse: Whether this task can fuse with successor
    """

    def __init__(
        self,
        op_name: str,
        inputs: list['TensorHandle'],
        outputs: list['TensorHandle'],
        allow_fuse: bool = True,
    ):
        self.op_name = op_name
        self.inputs = inputs
        self.outputs = outputs
        self.allow_fuse = allow_fuse

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.op_name})"


# =============================================================================
# Abstract MatMul Task
# =============================================================================

@dataclass
class AbstractMatmulTask(AbstractTask):
    """Abstract matrix multiplication task.

    Represents C = A @ B (+ bias) with optional ReLU activation.
    All dimensions are known but DMA addresses are not yet assigned.

    Attributes:
        op_name: Operation name
        M: Batch/sequence dimension (output rows)
        K: Inner dimension (must match A.columns == B.rows)
        N: Output/features dimension (output columns)
        precision: Data type precision (FP16, INT8, etc.)
        relu: Whether to apply ReLU activation
        has_bias: Whether bias addition is fused
        output_fp16: DPU fp32->fp16 conversion (always on for unified FP16 layout)
    """
    op_name: str
    M: int
    K: int
    N: int
    precision: str = Precision.FP16
    relu: bool = False
    has_bias: bool = False
    output_fp16: bool = True

    allow_fuse: bool = True
    inputs: list['TensorHandle'] = None
    outputs: list['TensorHandle'] = None

    def __post_init__(self):
        AbstractTask.__init__(
            self,
            op_name=self.op_name,
            inputs=[],
            outputs=[],
            allow_fuse=self.allow_fuse,
        )

    def compute_m_tile_size(self) -> int:
        """Compute optimal M tile size based on CBUF constraints."""
        from .regcmd_gen import compute_m_tile

        k_align = 64 if self.precision == "int8" else 32
        elem_size = 1 if self.precision == "int8" else 2
        K_aligned = align_up(self.K, k_align)
        N_aligned = align_up(self.N, 16)
        Mp = pad_m(self.M)

        return compute_m_tile(K_aligned, Mp, elem_size=elem_size, N=N_aligned)

    def __repr__(self) -> str:
        act = "ReLU" if self.relu else "linear"
        return f"AbstractMatmulTask({self.op_name}, M={self.M}, K={self.K}, N={self.N}, {act})"


# =============================================================================
# Abstract Elementwise Task
# =============================================================================

@dataclass
class AbstractElementwiseTask(AbstractTask):
    """Abstract elementwise operation task.

    Represents elementwise operations (Add, Mul, Sigmoid, etc.) applied
    pointwise to tensor(s).

    Attributes:
        op_name: Operation name
        op_type: Type of elementwise operation
        n_inputs: Number of inputs (1 for unary, 2 for binary)
        shape: Shape of input/output tensors
        core_mask: Which NPU cores to use (0x1=core0, 0x3=cores0+1, 0x7=all)
    """
    op_name: str
    op_type: str  # e.g., "Add", "Mul", "Sigmoid", "Tanh", "ReLU"
    n_inputs: int
    shape: tuple[int, ...]
    broadcast_b: bool = False  # True: set EW_SURF_STRIDE=0 to broadcast source B group 0
    core_mask: int | None = None  # None = use executor default (0x1)

    # Elementwise ops can usually be fused
    allow_fuse: bool = True

    # Don't include in dataclass fields
    inputs: list['TensorHandle'] = None
    outputs: list['TensorHandle'] = None

    def __post_init__(self):
        AbstractTask.__init__(
            self,
            op_name=self.op_name,
            inputs=[],
            outputs=[],
            allow_fuse=self.allow_fuse,
        )

    def fix_tensor_sizes(self):
        """Fix tensor sizes for matmul-style layout.

        This must be called after TensorHandles are connected to the task.
        Elementwise uses the same [C_al/8, Mp, 1, 8] FP16 layout as matmul,
        where C_aligned = align_up(C, 16) and Mp = pad_m(H).

        When broadcast_b=True, the second input (source B) is smaller than
        source A / output — its physical size is based on its own shape, not
        the output shape.
        """
        if len(self.shape) == 2:  # [H, C] format
            H, C = self.shape
            C_aligned = align_up(C, 16)  # match matmul N alignment
            M_padded = pad_m(H)
            size = C_aligned * M_padded * 2  # FP16

            for tensor in self.inputs + self.outputs:
                if tensor.shape == self.shape:
                    _ensure_min_size(tensor, size)

    @property
    def is_unary(self) -> bool:
        """Whether this is a unary operation."""
        return self.n_inputs == 1

    @property
    def is_binary(self) -> bool:
        """Whether this is a binary operation."""
        return self.n_inputs == 2

    def __repr__(self) -> str:
        """String representation with op type and shape."""
        shape_str = "\u00d7".join(str(d) for d in self.shape)
        return f"AbstractElementwiseTask({self.op_name}, {self.op_type}, shape=[{shape_str}])"


# =============================================================================
# Abstract Conv2D Task
# =============================================================================

@dataclass
class AbstractConv2DTask(AbstractTask):
    """Abstract 2D convolution task.

    Supports two modes:
    - Mode 0 (im2col + MAC): General convolution via CNA im2col pipeline.
      Uses the same 112-regcmd template as matmul. Supports padding, arbitrary
      C, bias in DPU BS stage, and activation in DPU BN stage.
    - Mode 6 (direct spatial): Legacy mode for C <= 32, no padding.
      Uses 130-regcmd template with DPU_RDMA.

    Input layout: [C/8, H, W, 8] spatial FP16 (both modes).
    Output layout: [N/8, M_padded, 1, 8] FP16 (both modes).

    Attributes:
        op_name: Operation name
        C: Input channels
        H: Input height
        W: Input width
        N: Output channels (filters)
        kH: Kernel height
        kW: Kernel width
        stride: Stride (H and W, must be equal)
        mode: 0=im2col (general), 6=direct spatial (C<=32 legacy)
        groups: 1=standard conv, C=depthwise conv
        padding: Legacy padding field (Mode 6)
        pad_top, pad_bottom, pad_left, pad_right: Per-side padding (Mode 0)
        relu: Whether to apply ReLU activation
        has_bias: Per-channel bias handled in DPU BS stage (Mode 0)
        has_residual: Fused residual addition via DPU EW pipeline (Mode 0)
        clip_max: 0=no clip, 6.0=Clip(0,6) via DPU BN_RELUX_CMP (Mode 0)
        core_mask: Which NPU cores to use
    """
    op_name: str
    C: int
    H: int
    W: int
    N: int
    kH: int
    kW: int
    stride: int
    mode: int = 0
    groups: int = 1
    padding: int = 0
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    relu: bool = False
    has_bias: bool = False
    has_residual: bool = False
    clip_max: float = 0.0
    is_depthwise: bool = False
    conv_mode: int = 0  # 0=standard, 3=CHANNEL_EXT (projection pw1x1 where N < C)
    core_mask: int | None = None
    allow_fuse: bool = True
    inputs: list['TensorHandle'] = None
    outputs: list['TensorHandle'] = None

    # M-tiling (spatial tiling along H)
    is_mtile: bool = False
    m_offset: int = 0               # offset in linearized M dimension (= h_start * W_out)
    H_tile_out: int | None = None   # output rows for this tile (None = use full H_out)

    # Channel tiling (depthwise only — each channel is independent)
    c_tile_offset: int = 0              # starting channel index (0 = no offset)
    c_tile_channels: int | None = None  # channels for this tile (None = use full C)

    # Mode 3 multi-task: which output channel this task computes (0-indexed)
    mode3_n_idx: int = 0

    def __post_init__(self):
        AbstractTask.__init__(
            self,
            op_name=self.op_name,
            inputs=[],
            outputs=[],
            allow_fuse=self.allow_fuse,
        )

    @property
    def H_out(self) -> int:
        return (self.H + self.pad_top + self.pad_bottom - self.kH) // self.stride + 1

    @property
    def W_out(self) -> int:
        return (self.W + self.pad_left + self.pad_right - self.kW) // self.stride + 1

    @property
    def M(self) -> int:
        """Output spatial size: H_out * W_out."""
        return self.H_out * self.W_out

    @property
    def H_out_effective(self) -> int:
        """H_out for this tile (or full H_out if not tiled)."""
        return self.H_tile_out if self.H_tile_out is not None else self.H_out

    @property
    def M_effective(self) -> int:
        """M for this tile."""
        return self.H_out_effective * self.W_out

    @property
    def M_padded_effective(self) -> int:
        """M_padded for this tile."""
        M = self.M_effective
        return pad_m(M)

    @property
    def M_tile(self) -> int | None:
        """M tile size for compatibility with exec_plan metadata extraction."""
        if self.H_tile_out is not None:
            return self.H_tile_out * self.W_out
        return None

    @property
    def C_aligned(self) -> int:
        """C aligned up to 32 (for register formulas)."""
        return align_up(self.C, 32)

    @property
    def C_aligned8(self) -> int:
        """C aligned up to 8 (for weight layout)."""
        return align_up(self.C, 8)

    @property
    def N_aligned(self) -> int:
        return align_up(self.N, 16)

    @property
    def K_eff(self) -> int:
        """Effective K dimension for weight packing.

        Mode 0: C_aligned32 * kH * kW (CNA processes channels in groups of 32)
        Mode 0 DW (conv_mode=3): C * kH * kW (no alignment!)
        Mode 6: C_aligned32 * kH * kW (legacy)
        """
        if self.mode == 0:
            if self.is_depthwise:
                # conv_mode=3 DW: K_eff uses C aligned to 32 (CNA requires
                # 32-aligned channel groups for DW mode 3)
                return self.C_aligned * self.kH * self.kW
            elif self.kH == 1 and self.kW == 1:
                # 1x1 PW: K = C (aligned to 32 like matmul)
                return self.C_aligned
            else:
                # Spatial: CNA im2col uses 32-channel groups (c2_divisor)
                return self.C_aligned * self.kH * self.kW
        else:
            return self.C_aligned * self.kH * self.kW

    @property
    def M_padded(self) -> int:
        M = self.M
        if M <= 1:
            return M
        return pad_m(M)

    def fix_tensor_sizes(self):
        """Fix tensor sizes for NPU layout alignment.

        Mode 0:
        - Input: C_aligned * H * W * 2 bytes (spatial FP16 [C/8, H, W, 8])
        - Weight: K_eff_aligned * N_aligned * 2 bytes (matmul-style packed)
        - Output: N_aligned * M_padded * 2 bytes (FP16 [N/8, M_padded, 1, 8])

        Mode 6:
        - Input: C_aligned * H * W * 2 bytes (spatial FP16 [C/8, H, W, 8])
        - Weight: kH * kW * N_aligned * C_aligned8 * 2 bytes (mode 6 layout)
        - Output: N_aligned * M_padded * 2 bytes (FP16 [N/8, M_padded, 1, 8])
        """
        N_al = self.N_aligned
        Mp = self.M_padded

        # Input: spatial layout [C_aligned/8, H, W, 8], FP16 (same for both modes)
        if len(self.inputs) >= 1:
            _ensure_min_size(self.inputs[0], self.C_aligned * self.H * self.W * 2)

        # Weight layout depends on mode
        if self.mode == 0:
            if self.is_depthwise:
                weight_size = align_up(self.K_eff, 32) * 2
            elif self.conv_mode == 3:
                weight_size = self.N * self.C * 2 * 9
            else:
                weight_size = align_up(self.K_eff, 32) * N_al * 2
        else:
            weight_size = self.kH * self.kW * N_al * self.C_aligned8 * 2

        if len(self.inputs) >= 2:
            self.inputs[1].size_bytes = weight_size

        # Output: FP16 feature layout [N_aligned/8, M_padded, 1, 8]
        # DW output uses C_aligned (32-aligned) channels since the CNA/DPU
        # write C_aligned channels even if only C are meaningful.
        N_out = self.C_aligned if self.is_depthwise else N_al
        if self.outputs:
            _ensure_min_size(self.outputs[0], N_out * Mp * 2)

        # Bias: FP32 per-channel, N_aligned values
        # DW bias needs C_aligned values to match register configuration.
        N_bias = self.C_aligned if self.is_depthwise else N_al
        if self.has_bias and len(self.inputs) >= 3:
            self.inputs[2].size_bytes = N_bias * 4

        # Residual: FP16 spatial, same layout as output
        residual_idx = 3 if self.has_bias else 2
        if self.has_residual and len(self.inputs) > residual_idx:
            _ensure_min_size(self.inputs[residual_idx], N_al * Mp * 2)

    def __repr__(self) -> str:
        act = "ReLU" if self.relu else "linear"
        if self.clip_max > 0:
            act = f"Clip(0,{self.clip_max})"
        pad = ""
        if self.pad_top or self.pad_bottom or self.pad_left or self.pad_right:
            pad = f", pad=({self.pad_top},{self.pad_bottom},{self.pad_left},{self.pad_right})"
        return (f"AbstractConv2DTask({self.op_name}, mode={self.mode}, C={self.C}, "
                f"{self.H}\u00d7{self.W}, N={self.N}, k={self.kH}\u00d7{self.kW}, "
                f"s={self.stride}{pad}, {act})")


# =============================================================================
# Abstract MaxPool Task
# =============================================================================

@dataclass
class AbstractMaxPoolTask(AbstractTask):
    """Abstract max pooling task via PPU hardware.

    Uses the PPU (Post-Processing Unit) for hardware-accelerated max pooling.
    Input/output use spatial [C/8, H, W, 8] FP16 layout.

    Attributes:
        op_name: Operation name
        C: Number of channels
        H: Input height
        W: Input width
        kH: Kernel height
        kW: Kernel width
        stride_h: Stride height
        stride_w: Stride width
        pad_top, pad_bottom, pad_left, pad_right: Padding
    """
    op_name: str
    C: int
    H: int
    W: int
    kH: int
    kW: int
    stride_h: int
    stride_w: int
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    method: int = 1  # 1=max, 2=min, 3=avg

    core_mask: int | None = None
    allow_fuse: bool = True
    inputs: list['TensorHandle'] = None
    outputs: list['TensorHandle'] = None

    def __post_init__(self):
        AbstractTask.__init__(
            self,
            op_name=self.op_name,
            inputs=[],
            outputs=[],
            allow_fuse=self.allow_fuse,
        )

    @property
    def H_out(self) -> int:
        return (self.H + self.pad_top + self.pad_bottom - self.kH) // self.stride_h + 1

    @property
    def W_out(self) -> int:
        return (self.W + self.pad_left + self.pad_right - self.kW) // self.stride_w + 1

    @property
    def C_aligned(self) -> int:
        return align_up(self.C, 8)

    def fix_tensor_sizes(self):
        """Fix tensor sizes for NPU spatial layout.

        Input:  C_aligned * H * W * 2 bytes (FP16 [C/8, H, W, 8])
        Output: C_aligned * H_out * W_out * 2 bytes (FP16 [C/8, H_out, W_out, 8])
        """
        C_al = self.C_aligned
        if len(self.inputs) >= 1:
            _ensure_min_size(self.inputs[0], C_al * self.H * self.W * 2)
        if self.outputs:
            _ensure_min_size(self.outputs[0], C_al * self.H_out * self.W_out * 2)

    def __repr__(self) -> str:
        return (f"AbstractMaxPoolTask({self.op_name}, C={self.C}, "
                f"{self.H}x{self.W}, k={self.kH}x{self.kW}, "
                f"s={self.stride_h}x{self.stride_w}, "
                f"out={self.H_out}x{self.W_out})")
