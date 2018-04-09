"""VTA configuration constants (should match hw_spec.h"""
from __future__ import absolute_import as _abs

#  Log of input/activation width in bits (default 3 -> 8 bits)
VTA_LOG_INP_WIDTH = 3
#  Log of kernel weight width in bits (default 3 -> 8 bits)
VTA_LOG_WGT_WIDTH = 3
#  Log of accum width in bits (default 5 -> 32 bits)
VTA_LOG_ACC_WIDTH = 5
#  Log of tensor batch size (A in (A,B)x(B,C) matrix multiplication)
VTA_LOG_BATCH = 0
#  Log of tensor inner block size (B in (A,B)x(B,C) matrix multiplication)
VTA_LOG_BLOCK_IN = 4
#  Log of tensor outer block size (C in (A,B)x(B,C) matrix multiplication)
VTA_LOG_BLOCK_OUT = 4
VTA_LOG_OUT_WIDTH = VTA_LOG_INP_WIDTH
#  Log of uop buffer size in Bytes
VTA_LOG_UOP_BUFF_SIZE = 15
#  Log of acc buffer size in Bytes
VTA_LOG_ACC_BUFF_SIZE = 17

# The Constants
VTA_WGT_WIDTH = 8
VTA_INP_WIDTH = VTA_WGT_WIDTH
VTA_OUT_WIDTH = 32

VTA_TARGET = "VTA_PYNQ_TARGET"

# Dimensions of the GEMM unit
# (BATCH,BLOCK_IN) x (BLOCK_IN,BLOCK_OUT)
VTA_BATCH = 1
VTA_BLOCK_IN = 16
VTA_BLOCK_OUT = 16

# log-2 On-chip wgt buffer size in Bytes
VTA_LOG_WGT_BUFF_SIZE = 15
# log-2 On-chip input buffer size in Bytes
VTA_LOG_INP_BUFF_SIZE = 15
# log-2 On-chip output buffer size in Bytes
VTA_LOG_OUT_BUFF_SIZE = 17
# On-chip wgt buffer size in Bytes
VTA_WGT_BUFF_SIZE = 1 << VTA_LOG_WGT_BUFF_SIZE
# Input buffer size
VTA_INP_BUFF_SIZE = 1 << VTA_LOG_INP_BUFF_SIZE
# Output buffer size.
VTA_OUT_BUFF_SIZE = 1 << VTA_LOG_OUT_BUFF_SIZE

# Number of bytes per buffer
VTA_INP_ELEM_BYTES = (VTA_BATCH*VTA_BLOCK_IN*VTA_INP_WIDTH//8)
VTA_WGT_ELEM_BYTES = (VTA_BLOCK_OUT*VTA_BLOCK_IN*VTA_WGT_WIDTH//8)
VTA_OUT_ELEM_BYTES = (VTA_BATCH*VTA_BLOCK_OUT*VTA_OUT_WIDTH//8)

# Maximum external buffer size in bytes
VTA_MAX_XFER = 1 << 22

# Number of elements
VTA_INP_BUFF_DEPTH = VTA_INP_BUFF_SIZE//VTA_INP_ELEM_BYTES
VTA_WGT_BUFF_DEPTH = VTA_WGT_BUFF_SIZE//VTA_WGT_ELEM_BYTES
VTA_OUT_BUFF_DEPTH = VTA_OUT_BUFF_SIZE//VTA_OUT_ELEM_BYTES

# Memory id for DMA
VTA_MEM_ID_UOP = 0
VTA_MEM_ID_WGT = 1
VTA_MEM_ID_INP = 2
VTA_MEM_ID_ACC = 3
VTA_MEM_ID_OUT = 4

# VTA ALU Opcodes
VTA_ALU_OPCODE_MIN = 0
VTA_ALU_OPCODE_MAX = 1
VTA_ALU_OPCODE_ADD = 2
VTA_ALU_OPCODE_SUB = 3
VTA_ALU_OPCODE_MUL = 4
VTA_ALU_OPCODE_SHL = 5
VTA_ALU_OPCODE_SHR = 6
VTA_ALU_OPCODE_UNSET = 7

# Task queue id (pipeline stage)
VTA_QID_LOAD_INP = 1
VTA_QID_LOAD_WGT = 1
VTA_QID_LOAD_OUT = 2
VTA_QID_STORE_OUT = 3
VTA_QID_COMPUTE = 2
VTA_QID_STORE_INP = 3

# Debug flags
DEBUG_DUMP_INSN = (1 << 1)
DEBUG_DUMP_UOP = (1 << 2)
DEBUG_SKIP_READ_BARRIER = (1 << 3)
DEBUG_SKIP_WRITE_BARRIER = (1 << 4)
