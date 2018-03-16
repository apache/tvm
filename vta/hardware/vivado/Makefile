# Directories
ROOTDIR = $(CURDIR)
BUILD_DIR = $(ROOTDIR)/build
SCRIPT_DIR = $(ROOTDIR)/scripts
SRC_DIR = $(ROOTDIR)/../../src/hardware/hls
SIM_DIR = $(ROOTDIR)/sim
TEST_DIR = $(ROOTDIR)/../../src/test
INCLUDE_DIR = $(ROOTDIR)/../../include

# Executables
VIVADO_HLS = vivado_hls
VIVADO = vivado
HSI = hsi

# Build parameters:
#  Number of threads during compilation
NUM_THREADS = 8
#  Target Frequency
CLOCK_FREQ = 100
#  Log of input width in bits
LOG_INP_WIDTH = 3
#  Log of weight width in bits
LOG_WGT_WIDTH = 3
#  Log of accum width in bits
LOG_ACC_WIDTH = 5
#  Log of output width in bits
LOG_OUT_WIDTH = $(LOG_INP_WIDTH)
#  Log of tensor batch size (A in (A,B)x(B,C) matrix multiplication)
LOG_BATCH = 0
#  Log of tensor inner block size (B in (A,B)x(B,C) matrix multiplication)
LOG_IN_BLOCK = 4
#  Log of tensor outer block size (C in (A,B)x(B,C) matrix multiplication)
LOG_OUT_BLOCK = 4
#  Log of uop buffer size in Bytes
LOG_UOP_BUFF_SIZE = 15
#  Log of inp buffer size in Bytes
LOG_INP_BUFF_SIZE = 15
#  Log of wgt buffer size in Bytes
LOG_WGT_BUFF_SIZE = 15
#  Log of acc buffer size in Bytes
LOG_ACC_BUFF_SIZE = 17
#  Log of out buffer size in Bytes
LOG_OUT_BUFF_SIZE = $(shell echo "$$(( $(LOG_ACC_BUFF_SIZE)+$(LOG_OUT_WIDTH)-$(LOG_ACC_WIDTH) ))" )

# Derived parameter
#  Input width in bits
INP_WIDTH = $(shell echo "$$(( 1 << $(LOG_INP_WIDTH) ))" )
#  Weight width in bits
WGT_WIDTH = $(shell echo "$$(( 1 << $(LOG_WGT_WIDTH) ))" )
#  Output width in bits
OUT_WIDTH = $(shell echo "$$(( 1 << $(LOG_OUT_WIDTH) ))" )
#  Tensor batch size
BATCH = $(shell echo "$$(( 1 << $(LOG_BATCH) ))" )
#  Tensor outer block size
IN_BLOCK = $(shell echo "$$(( 1 << $(LOG_IN_BLOCK) ))" )
#  Tensor inner block size
OUT_BLOCK = $(shell echo "$$(( 1 << $(LOG_OUT_BLOCK) ))" )
#  Uop buffer size in Bytes
UOP_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_UOP_BUFF_SIZE) ))" )
#  Inp buffer size in Bytes
INP_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_INP_BUFF_SIZE) ))" )
#  Wgt buffer size in Bytes
WGT_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_WGT_BUFF_SIZE) ))" )
#  Acc buffer size in Bytes
ACC_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_ACC_BUFF_SIZE) ))" )
#  Out buffer size in Bytes
OUT_BUFF_SIZE = $(shell echo "$$(( 1 << $(LOG_OUT_BUFF_SIZE) ))" )

# Derive clock target period
TARGET_PER = $(shell echo "$$(( (1000 + $(CLOCK_FREQ) - 1) / $(CLOCK_FREQ) - 0))" )

# Derive config name
CONF = \
	$(BATCH)x$(IN_BLOCK)x$(OUT_BLOCK)_$(INP_WIDTH)bx$(WGT_WIDTH)b_$(CLOCK_FREQ)MHz_$(TARGET_PER)ns
IP_BUILD_PATH = $(BUILD_DIR)/hls/$(CONF)
HW_BUILD_PATH = $(BUILD_DIR)/vivado/$(CONF)

.PHONY: all ip bit driver clean

all: driver

ip: 
	mkdir -p $(IP_BUILD_PATH)
	cd $(IP_BUILD_PATH) && \
		$(VIVADO_HLS) -f $(SCRIPT_DIR)/hls.tcl \
			-tclargs $(SRC_DIR) $(SIM_DIR) $(TEST_DIR) $(INCLUDE_DIR) $(TARGET_PER) \
			$(LOG_INP_WIDTH) $(LOG_WGT_WIDTH) $(LOG_ACC_WIDTH) $(LOG_OUT_WIDTH) \
			$(LOG_BATCH) $(LOG_OUT_BLOCK) $(LOG_IN_BLOCK) \
			$(LOG_UOP_BUFF_SIZE) $(LOG_INP_BUFF_SIZE) $(LOG_WGT_BUFF_SIZE) \
			$(LOG_ACC_BUFF_SIZE) $(LOG_OUT_BUFF_SIZE)

bit: ip
	mkdir -p $(HW_BUILD_PATH)
	cd $(HW_BUILD_PATH) && \
		$(VIVADO) -mode tcl -source $(SCRIPT_DIR)/vivado.tcl \
		-tclargs $(IP_BUILD_PATH) $(NUM_THREADS) $(CLOCK_FREQ) \
		$(INP_WIDTH) $(WGT_WIDTH) $(OUT_WIDTH) \
		$(BATCH) $(IN_BLOCK) $(OUT_BLOCK) \
		$(INP_BUFF_SIZE) $(WGT_BUFF_SIZE) $(OUT_BUFF_SIZE)

driver: bit
	cd $(HW_BUILD_PATH) && $(HSI) -mode tcl -source $(SCRIPT_DIR)/hsi.tcl -nojournal -nolog
	cd $(HW_BUILD_PATH)/bsp && make

clean:
	rm -rf build