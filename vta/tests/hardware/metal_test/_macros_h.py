VTA_LOG_INP_WIDTH=3
VTA_LOG_WGT_WIDTH=3
VTA_LOG_ACC_WIDTH=5
VTA_LOG_BATCH=0
VTA_LOG_BLOCK=4
VTA_LOG_UOP_BUFF_SIZE=15
VTA_LOG_INP_BUFF_SIZE=15
VTA_LOG_WGT_BUFF_SIZE=18
VTA_LOG_ACC_BUFF_SIZE=17
VTA_LOG_BLOCK_IN=4
VTA_LOG_BLOCK_OUT=4
VTA_LOG_OUT_WIDTH=3
VTA_LOG_OUT_BUFF_SIZE=15
VTA_LOG_BUS_WIDTH=6
VTA_FETCH_INSN_COUNT_OFFSET=16
VTA_FETCH_INSN_ADDR_OFFSET=24
VTA_LOAD_INP_ADDR_OFFSET=16
VTA_LOAD_WGT_ADDR_OFFSET=24
VTA_COMPUTE_DONE_WR_OFFSET=16
VTA_COMPUTE_DONE_RD_OFFSET=24
VTA_COMPUTE_UOP_ADDR_OFFSET=32
VTA_COMPUTE_BIAS_ADDR_OFFSET=40
VTA_STORE_OUT_ADDR_OFFSET=16
VTA_CACHED = 1
VTA_NOT_CACHED = 0
VTA_MAX_XFER = (1<<25)
VTA_PAGE_BITS = 12
VTA_PAGE_BYTES = (1 << VTA_PAGE_BITS)
VTA_BUS_WIDTH = (1 << VTA_LOG_BUS_WIDTH)
VTA_LOG_INS_WIDTH = 7
VTA_INS_WIDTH = (1 << VTA_LOG_INS_WIDTH)
VTA_LOG_UOP_WIDTH = 5
VTA_UOP_WIDTH = (1 << VTA_LOG_UOP_WIDTH)
VTA_WGT_WIDTH = (1 << VTA_LOG_WGT_WIDTH)
VTA_INP_WIDTH = (1 << VTA_LOG_INP_WIDTH)
VTA_OUT_WIDTH = (1 << VTA_LOG_OUT_WIDTH)
VTA_ACC_WIDTH = (1 << VTA_LOG_ACC_WIDTH)
VTA_BATCH = (1 << VTA_LOG_BATCH)
VTA_BLOCK_IN = (1 << VTA_LOG_BLOCK_IN)
VTA_BLOCK_OUT = (1 << VTA_LOG_BLOCK_OUT)
VTA_UOP_BUFF_SIZE = (1 << VTA_LOG_UOP_BUFF_SIZE)
VTA_WGT_BUFF_SIZE = (1 << VTA_LOG_WGT_BUFF_SIZE)
VTA_INP_BUFF_SIZE = (1 << VTA_LOG_INP_BUFF_SIZE)
VTA_ACC_BUFF_SIZE = (1 << VTA_LOG_ACC_BUFF_SIZE)
VTA_INP_MATRIX_WIDTH = (VTA_INP_WIDTH * VTA_BATCH * VTA_BLOCK_IN)
VTA_WGT_MATRIX_WIDTH = (VTA_WGT_WIDTH * VTA_BLOCK_OUT * VTA_BLOCK_IN)
VTA_ACC_MATRIX_WIDTH = (VTA_ACC_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)
VTA_OUT_MATRIX_WIDTH = (VTA_OUT_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)
INP_MAT_AXI_RATIO = (VTA_INP_MATRIX_WIDTH / VTA_BUS_WIDTH)
WGT_MAT_AXI_RATIO = (VTA_WGT_MATRIX_WIDTH / VTA_BUS_WIDTH)
ACC_MAT_AXI_RATIO = (VTA_ACC_MATRIX_WIDTH / VTA_BUS_WIDTH)
OUT_MAT_AXI_RATIO = (VTA_OUT_MATRIX_WIDTH / VTA_BUS_WIDTH)
VTA_INS_ELEM_BYTES = (VTA_INS_WIDTH / 8)
VTA_UOP_ELEM_BYTES = (VTA_UOP_WIDTH / 8)
VTA_INP_ELEM_BYTES = (VTA_INP_MATRIX_WIDTH / 8)
VTA_WGT_ELEM_BYTES = (VTA_WGT_MATRIX_WIDTH / 8)
VTA_ACC_ELEM_BYTES = (VTA_ACC_MATRIX_WIDTH / 8)
VTA_OUT_ELEM_BYTES = (VTA_OUT_MATRIX_WIDTH / 8)
VTA_UOP_BUFF_DEPTH = (VTA_UOP_BUFF_SIZE / VTA_UOP_ELEM_BYTES)
VTA_LOG_UOP_BUFF_DEPTH = (VTA_LOG_UOP_BUFF_SIZE - VTA_LOG_UOP_WIDTH + 3)
VTA_WGT_BUFF_DEPTH = (VTA_WGT_BUFF_SIZE / VTA_WGT_ELEM_BYTES)
VTA_LOG_WGT_BUFF_DEPTH = (VTA_LOG_WGT_BUFF_SIZE - VTA_LOG_BLOCK_OUT - VTA_LOG_BLOCK_IN - VTA_LOG_WGT_WIDTH + 3)
VTA_INP_BUFF_DEPTH = (VTA_INP_BUFF_SIZE / VTA_INP_ELEM_BYTES)
VTA_LOG_INP_BUFF_DEPTH = (VTA_LOG_INP_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_IN - VTA_LOG_INP_WIDTH + 3)
VTA_ACC_BUFF_DEPTH = (VTA_ACC_BUFF_SIZE / VTA_ACC_ELEM_BYTES)
VTA_LOG_ACC_BUFF_DEPTH = (VTA_LOG_ACC_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_OUT - VTA_LOG_ACC_WIDTH + 3)
VTA_OPCODE_BIT_WIDTH = 3
VTA_ALU_OPCODE_BIT_WIDTH = 2
VTA_OPCODE_LOAD = 0
VTA_OPCODE_STORE = 1
VTA_OPCODE_GEMM = 2
VTA_OPCODE_FINISH = 3
VTA_OPCODE_ALU = 4
VTA_ALU_OPCODE_MIN = 0
VTA_ALU_OPCODE_MAX = 1
VTA_ALU_OPCODE_ADD = 2
VTA_ALU_OPCODE_SHR = 3
VTA_MEMOP_ID_BIT_WIDTH = 2
VTA_MEMOP_SRAM_ADDR_BIT_WIDTH = 16
VTA_MEMOP_DRAM_ADDR_BIT_WIDTH = 32
VTA_MEMOP_SIZE_BIT_WIDTH = 16
VTA_MEMOP_STRIDE_BIT_WIDTH = 16
VTA_MEMOP_PAD_BIT_WIDTH = 4
VTA_MEMOP_PAD_VAL_BIT_WIDTH = 2
VTA_LOOP_ITER_WIDTH = 14
VTA_ALUOP_IMM_BIT_WIDTH = 16
VTA_SHR_ARG_BIT_WIDTH = (VTA_LOG_ACC_WIDTH)
VTA_MUL_ARG_BIT_WIDTH = 8
VTA_MEM_ID_UOP = 0
VTA_MEM_ID_WGT = 1
VTA_MEM_ID_INP = 2
VTA_MEM_ID_ACC = 3
VTA_MEM_ID_OUT = 4
VTA_UOP_GEM_0_0 = 0
VTA_UOP_GEM_0_1 = (VTA_UOP_GEM_0_0 + VTA_LOG_ACC_BUFF_DEPTH - 1)
VTA_UOP_GEM_1_0 = (VTA_UOP_GEM_0_1 + 1)
VTA_UOP_GEM_1_1 = (VTA_UOP_GEM_1_0 + VTA_LOG_INP_BUFF_DEPTH - 1)
VTA_UOP_GEM_2_0 = (VTA_UOP_GEM_1_1 + 1)
VTA_UOP_GEM_2_1 = (VTA_UOP_GEM_2_0 + VTA_LOG_WGT_BUFF_DEPTH - 1)
VTA_UOP_ALU_0_0 = 0
VTA_UOP_ALU_0_1 = (VTA_UOP_ALU_0_0 + VTA_LOG_ACC_BUFF_DEPTH - 1)
VTA_UOP_ALU_1_0 = (VTA_UOP_ALU_0_1 + 1)
VTA_UOP_ALU_1_1 = (VTA_UOP_ALU_1_0 + VTA_LOG_INP_BUFF_DEPTH - 1)
VTA_START = 0x1
VTA_AUTORESTART = 0x81
VTA_DONE = 0x1
