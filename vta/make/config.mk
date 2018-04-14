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

# the hardware target, can be [sim, pynq]
VTA_TARGET = pynq

#---------------------
# VTA hardware parameters
#--------------------

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
#  Log of uop buffer size in Bytes
VTA_LOG_UOP_BUFF_SIZE = 15
#  Log of inp buffer size in Bytes
VTA_LOG_INP_BUFF_SIZE = 15
#  Log of wgt buffer size in Bytes
VTA_LOG_WGT_BUFF_SIZE = 15
#  Log of acc buffer size in Bytes
VTA_LOG_ACC_BUFF_SIZE = 17


#---------------------
# Derived VTA hardware parameters
#--------------------

#  Input width in bits
VTA_INP_WIDTH = $(shell echo "$$(( 1 << $(VTA_LOG_INP_WIDTH) ))" )
#  Weight width in bits
VTA_WGT_WIDTH = $(shell echo "$$(( 1 << $(VTA_LOG_WGT_WIDTH) ))" )
#  Log of output width in bits
VTA_LOG_OUT_WIDTH = $(VTA_LOG_INP_WIDTH)
#  Output width in bits
VTA_OUT_WIDTH = $(shell echo "$$(( 1 << $(VTA_LOG_OUT_WIDTH) ))" )
#  Tensor batch size
VTA_BATCH = $(shell echo "$$(( 1 << $(VTA_LOG_BATCH) ))" )
#  Tensor outer block size
VTA_IN_BLOCK = $(shell echo "$$(( 1 << $(VTA_LOG_BLOCK_IN) ))" )
#  Tensor inner block size
VTA_OUT_BLOCK = $(shell echo "$$(( 1 << $(VTA_LOG_BLOCK_OUT) ))" )
#  Uop buffer size in Bytes
VTA_UOP_BUFF_SIZE = $(shell echo "$$(( 1 << $(VTA_LOG_UOP_BUFF_SIZE) ))" )
#  Inp buffer size in Bytes
VTA_INP_BUFF_SIZE = $(shell echo "$$(( 1 << $(VTA_LOG_INP_BUFF_SIZE) ))" )
#  Wgt buffer size in Bytes
VTA_WGT_BUFF_SIZE = $(shell echo "$$(( 1 << $(VTA_LOG_WGT_BUFF_SIZE) ))" )
#  Acc buffer size in Bytes
VTA_ACC_BUFF_SIZE = $(shell echo "$$(( 1 << $(VTA_LOG_ACC_BUFF_SIZE) ))" )
#  Log of out buffer size in Bytes
VTA_LOG_OUT_BUFF_SIZE = \
$(shell echo "$$(( $(VTA_LOG_ACC_BUFF_SIZE) + $(VTA_LOG_OUT_WIDTH) - $(VTA_LOG_ACC_WIDTH) ))" )
#  Out buffer size in Bytes
VTA_OUT_BUFF_SIZE = $(shell echo "$$(( 1 << $(VTA_LOG_OUT_BUFF_SIZE) ))" )

# Update ADD_CFLAGS
ADD_CFLAGS +=
	-DVTA_TARGET=$(VTA_TARGET)\
	-DVTA_LOG_WGT_WIDTH=$(VTA_LOG_WGT_WIDTH) -DVTA_LOG_INP_WIDTH=$(VTA_LOG_INP_WIDTH) \
	-DVTA_LOG_ACC_WIDTH=$(VTA_LOG_ACC_WIDTH) -DVTA_LOG_OUT_WIDTH=$(VTA_LOG_OUT_WIDTH) \
	-DVTA_LOG_BATCH=$(VTA_LOG_BATCH) \
	-DVTA_LOG_BLOCK_IN=$(VTA_LOG_BLOCK_IN) -DVTA_LOG_BLOCK_OUT=$(VTA_LOG_BLOCK_OUT) \
	-DVTA_LOG_UOP_BUFF_SIZE=$(VTA_LOG_UOP_BUFF_SIZE) -DVTA_LOG_INP_BUFF_SIZE=$(VTA_LOG_INP_BUFF_SIZE) \
	-DVTA_LOG_WGT_BUFF_SIZE=$(VTA_LOG_WGT_BUFF_SIZE) -DVTA_LOG_ACC_BUFF_SIZE=$(VTA_LOG_ACC_BUFF_SIZE) \
	-DVTA_LOG_OUT_BUFF_SIZE=$(VTA_LOG_OUT_BUFF_SIZE)
