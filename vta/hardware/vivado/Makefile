# Directories
ROOTDIR = $(CURDIR)
BUILD_DIR = $(ROOTDIR)/build
SCRIPT_DIR = $(ROOTDIR)/scripts
SRC_DIR = $(ROOTDIR)/src
SIM_DIR = $(ROOTDIR)/sim
TEST_DIR = $(ROOTDIR)/../../tests/hardware/common
INCLUDE_DIR = $(ROOTDIR)/../../include

# Executables
VIVADO_HLS = vivado_hls
VIVADO = vivado
HSI = hsi

# Include top-level config file
ifndef config
ifneq ("$(wildcard ../../config.mk)", "")
	config = ../../config.mk
else
	config = ../../make/config.mk
endif
endif
include $(config)

#---------------------
# Compilation parameters
#--------------------

#  Number of threads during compilation
NUM_THREADS = 8

#  Target Frequency
CLOCK_FREQ = 100

#  Timing closure compensation (0 for none, 3 for highest)
TIMING_CLOSURE_COMP = 0

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
			$(LOG_BATCH) $(LOG_BLOCK_OUT) $(LOG_BLOCK_IN) \
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