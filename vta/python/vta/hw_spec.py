"""VTA configuration constants (should match hw_spec.h"""
from __future__ import absolute_import as _abs

import os
python_vta_dir = os.path.dirname(__file__)
print python_vta_dir
filename = os.path.join(python_vta_dir, '../../config.mk')

VTA_PYNQ_BRAM_W = 32
VTA_PYNQ_BRAM_D = 1024
VTA_PYNQ_NUM_BRAM = 124

keys = ["VTA_LOG_INP_WIDTH", "VTA_LOG_WGT_WIDTH", "VTA_LOG_ACC_WIDTH",
        "VTA_LOG_BATCH", "VTA_LOG_BLOCK_IN", "VTA_LOG_BLOCK_OUT",
        "VTA_LOG_UOP_BUFF_SIZE", "VTA_LOG_INP_BUFF_SIZE",
        "VTA_LOG_WGT_BUFF_SIZE", "VTA_LOG_ACC_BUFF_SIZE"]

params = {}

if os.path.isfile(filename):
  with open(filename) as f:
    for line in f:
      for k in keys:
        if k+" =" in line:
          val = line.split("=")[1].strip()
          params[k] = int(val)
  # print params
else:
  print "Error: {} not found. Please make sure you have config.mk in your vta root".format(filename)
  exit()

# The Constants
VTA_INP_WIDTH = 1 << params["VTA_LOG_INP_WIDTH"]
VTA_WGT_WIDTH = 1 << params["VTA_LOG_WGT_WIDTH"]
VTA_OUT_WIDTH = 1 << params["VTA_LOG_ACC_WIDTH"]

VTA_TARGET = "VTA_PYNQ_TARGET"

# Dimensions of the GEMM unit
# (BATCH,BLOCK_IN) x (BLOCK_IN,BLOCK_OUT)
VTA_BATCH = 1 << params["VTA_LOG_BATCH"]
VTA_BLOCK_IN = 1 << params["VTA_LOG_BLOCK_IN"]
VTA_BLOCK_OUT = 1 << params["VTA_LOG_BLOCK_OUT"]

# log-2 On-chip uop buffer size in Bytes
VTA_LOG_UOP_BUFF_SIZE = params["VTA_LOG_UOP_BUFF_SIZE"]
# log-2 On-chip input buffer size in Bytes
VTA_LOG_INP_BUFF_SIZE = params["VTA_LOG_INP_BUFF_SIZE"]
# log-2 On-chip wgt buffer size in Bytes
VTA_LOG_WGT_BUFF_SIZE = params["VTA_LOG_WGT_BUFF_SIZE"]
# log-2 On-chip output buffer size in Bytes
VTA_LOG_OUT_BUFF_SIZE = params["VTA_LOG_ACC_BUFF_SIZE"]
# Uop buffer size
VTA_UOP_BUFF_SIZE = 1 << VTA_LOG_UOP_BUFF_SIZE
# Input buffer size
VTA_INP_BUFF_SIZE = 1 << VTA_LOG_INP_BUFF_SIZE
# On-chip wgt buffer size in Bytes
VTA_WGT_BUFF_SIZE = 1 << VTA_LOG_WGT_BUFF_SIZE
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
VTA_ALU_OPCODE_SHR = 3

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
