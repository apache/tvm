#-------------------------------------------------------------------------------
#  Template configuration for compiling VTA runtime.
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of nnvm. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#-------------------------------------------------------------------------------

#---------------------
# choice of compiler
#--------------------

# the additional link flags you want to add
ADD_LDFLAGS=

# the additional compile flags you want to add
ADD_CFLAGS=

# the hardware target
TARGET=PYNQ_TARGET

#---------------------
# VTA hardware parameters
#--------------------

#  Log of input/activation width in bits (default 3 -> 8 bits)
LOG_INP_WIDTH = 3
#  Log of kernel weight width in bits (default 3 -> 8 bits)
LOG_WGT_WIDTH = 3
#  Log of accum width in bits (default 5 -> 32 bits)
LOG_ACC_WIDTH = 5
#  Log of tensor batch size (A in (A,B)x(B,C) matrix multiplication)
LOG_BATCH = 0
#  Log of tensor inner block size (B in (A,B)x(B,C) matrix multiplication)
LOG_BLOCK_IN = 4
#  Log of tensor outer block size (C in (A,B)x(B,C) matrix multiplication)
LOG_BLOCK_OUT = 4
#  Log of uop buffer size in Bytes
LOG_UOP_BUFF_SIZE = 15
#  Log of inp buffer size in Bytes
LOG_INP_BUFF_SIZE = 15
#  Log of wgt buffer size in Bytes
LOG_WGT_BUFF_SIZE = 15
#  Log of acc buffer size in Bytes
LOG_ACC_BUFF_SIZE = 17

#---------------------
# Derived VTA hardware parameters
#--------------------

#  Input width in bits
INP_WIDTH = $(shell echo "$$(( 1 << $(LOG_INP_WIDTH) ))" )
#  Weight width in bits
WGT_WIDTH = $(shell echo "$$(( 1 << $(LOG_WGT_WIDTH) ))" )
#  Log of output width in bits
LOG_OUT_WIDTH = $(LOG_INP_WIDTH)
#  Output width in bits
OUT_WIDTH = $(shell echo "$$(( 1 << $(LOG_OUT_WIDTH) ))" )
#  Tensor batch size
BATCH = $(shell echo "$$(( 1 << $(LOG_BATCH) ))" )
#  Tensor outer block size
IN_BLOCK = $(shell echo "$$(( 1 << $(LOG_BLOCK_IN) ))" )
#  Tensor inner block size
OUT_BLOCK = $(shell echo "$$(( 1 << $(LOG_BLOCK_OUT) ))" )
#  Uop buffer size in Bytes
UOP_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_UOP_BUFF_SIZE) ))" )
#  Inp buffer size in Bytes
INP_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_INP_BUFF_SIZE) ))" )
#  Wgt buffer size in Bytes
WGT_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_WGT_BUFF_SIZE) ))" )
#  Acc buffer size in Bytes
ACC_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_ACC_BUFF_SIZE) ))" )
#  Log of out buffer size in Bytes
LOG_OUT_BUFF_SIZE = $(shell echo "$$(( $(LOG_ACC_BUFF_SIZE)+$(LOG_OUT_WIDTH)-$(LOG_ACC_WIDTH) ))" )
#  Out buffer size in Bytes
OUT_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_OUT_BUFF_SIZE) ))" )

# Update ADD_CFLAGS
ADD_CFLAGS += \
	-D$(TARGET) \
	-DLOG_WGT_WIDTH=$(LOG_WGT_WIDTH) -DLOG_INP_WIDTH=$(LOG_INP_WIDTH) \
	-DLOG_ACC_WIDTH=$(LOG_ACC_WIDTH) -DLOG_OUT_WIDTH=$(LOG_OUT_WIDTH) \
	-DLOG_BATCH=$(LOG_BATCH) -DLOG_BLOCK_IN=$(LOG_BLOCK_IN) -DLOG_BLOCK_OUT=$(LOG_BLOCK_OUT) \
	-DLOG_UOP_BUFF_SIZE=$(LOG_UOP_BUFF_SIZE) -DLOG_INP_BUFF_SIZE=$(LOG_INP_BUFF_SIZE) \
	-DLOG_WGT_BUFF_SIZE=$(LOG_WGT_BUFF_SIZE) -DLOG_ACC_BUFF_SIZE=$(LOG_ACC_BUFF_SIZE) \
	-DLOG_OUT_BUFF_SIZE=$(LOG_OUT_BUFF_SIZE)