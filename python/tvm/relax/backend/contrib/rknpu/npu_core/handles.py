# SPDX-License-Identifier: MIT
"""Tensor handles: buffer references for allocation.

TensorHandles represent tensors that will be allocated in DMA buffers.
"""

from dataclasses import dataclass


@dataclass
class TensorHandle:
    """Reference to a tensor in a DMA buffer.

    Attributes:
        name: Tensor name
        shape: Tensor shape
        dtype: Data type
        size_bytes: Total size in bytes
        dma_addr: DMA address (placeholder at codegen time, patched at runtime)
    """
    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    dma_addr: int | None = None

    def __repr__(self) -> str:
        if self.dma_addr is not None:
            return f"TensorHandle({self.name}, size={self.size_bytes}B, addr=0x{self.dma_addr:x})"
        return f"TensorHandle({self.name}, size={self.size_bytes}B)"
