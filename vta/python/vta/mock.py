"""Mock interface for skip part of compute """
from .intrin import intrin_gevm, intrin_gemm

GEMM = intrin_gemm(True)
GEVM = intrin_gevm(True)
DMA_COPY = "skip_dma_copy"
ALU = "skip_alu"
