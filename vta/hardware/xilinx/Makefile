# Directories
ROOTDIR = $(CURDIR)
BUILD_NAME = build
BUILD_DIR = $(ROOTDIR)/../../$(BUILD_NAME)/hardware/xilinx
SCRIPT_DIR = $(ROOTDIR)/scripts
SRC_DIR = $(ROOTDIR)/src
SIM_DIR = $(ROOTDIR)/sim
TEST_DIR = $(ROOTDIR)/../../tests/hardware/common
INCLUDE_DIR = $(ROOTDIR)/../../include

# Executables
VIVADO_HLS = vivado_hls
VIVADO = vivado
HSI = hsi

# HLS Mode
MODE = all
# SLURM
SLURM = false
# Prevent generation of DSP
NO_DSP = false
# Prevent generation of ALU
NO_ALU = false

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
VTA_HW_COMP_THREADS = 8

#  Target Frequency
VTA_HW_COMP_CLOCK_FREQ = 100

#  Timing closure compensation (0 for none, 3 for highest)
VTA_HW_COMP_TIMING_COMP = 0

# Derive clock target period
TARGET_PER = \
$(shell echo "$$(( (1000 + $(VTA_HW_COMP_CLOCK_FREQ) - 1) / $(VTA_HW_COMP_CLOCK_FREQ) - $(VTA_HW_COMP_TIMING_COMP)))" )

# Derive config name
CONF = \
$(VTA_BATCH)x$(VTA_IN_BLOCK)x$(VTA_OUT_BLOCK)_$(VTA_INP_WIDTH)bx$(VTA_WGT_WIDTH)b_$(VTA_LOG_UOP_BUFF_SIZE)_$(VTA_LOG_INP_BUFF_SIZE)_$(VTA_LOG_WGT_BUFF_SIZE)_$(VTA_LOG_ACC_BUFF_SIZE)_$(VTA_HW_COMP_CLOCK_FREQ)MHz_$(TARGET_PER)ns
IP_BUILD_PATH = $(BUILD_DIR)/hls/$(CONF)
HW_BUILD_PATH = $(BUILD_DIR)/vivado/$(CONF)

ifeq ($(SLURM), true)
	IP_BUILD_PATH = /scratch/hls/$(CONF)
	HW_BUILD_PATH = /scratch/vivado/$(CONF)
endif

.PHONY: all ip bit driver clean clean_all

all: bit

ip: 
	mkdir -p $(IP_BUILD_PATH)
	cd $(IP_BUILD_PATH) && \
		$(VIVADO_HLS) -f $(SCRIPT_DIR)/hls.tcl \
			-tclargs $(SRC_DIR) $(SIM_DIR) $(TEST_DIR) $(INCLUDE_DIR) $(TARGET_PER) \
			$(VTA_LOG_INP_WIDTH) $(VTA_LOG_WGT_WIDTH) $(VTA_LOG_ACC_WIDTH) $(VTA_LOG_OUT_WIDTH) \
			$(VTA_LOG_BATCH) $(VTA_LOG_BLOCK_OUT) $(VTA_LOG_BLOCK_IN) \
			$(VTA_LOG_UOP_BUFF_SIZE) $(VTA_LOG_INP_BUFF_SIZE) $(VTA_LOG_WGT_BUFF_SIZE) \
			$(VTA_LOG_ACC_BUFF_SIZE) $(VTA_LOG_OUT_BUFF_SIZE) \
			$(MODE) $(NO_DSP) $(NO_ALU)
ifeq ($(SLURM), true)
	mkdir -p $(BUILD_DIR)/hls
	mv $(IP_BUILD_PATH) $(BUILD_DIR)/hls/.
endif

bit: ip
	mkdir -p $(HW_BUILD_PATH)
	cd $(HW_BUILD_PATH) && \
		$(VIVADO) -mode tcl -source $(SCRIPT_DIR)/vivado.tcl \
		-tclargs $(BUILD_DIR)/hls/$(CONF) $(VTA_HW_COMP_THREADS) $(VTA_HW_COMP_CLOCK_FREQ) \
		$(VTA_INP_WIDTH) $(VTA_WGT_WIDTH) $(VTA_OUT_WIDTH) \
		$(VTA_BATCH) $(VTA_IN_BLOCK) $(VTA_OUT_BLOCK) \
		$(VTA_INP_BUFF_SIZE) $(VTA_WGT_BUFF_SIZE) $(VTA_OUT_BUFF_SIZE)
ifeq ($(SLURM), true)
	mkdir -p $(BUILD_DIR)/vivado
	mv $(HW_BUILD_PATH) $(BUILD_DIR)/vivado/.
endif

driver: bit
	cd $(HW_BUILD_PATH) && $(HSI) -mode tcl -source $(SCRIPT_DIR)/hsi.tcl -nojournal -nolog
	cd $(HW_BUILD_PATH)/bsp && make

clean:
	rm -rf *.out *.log *.sb figures

clean_all: clean
	rm -rf $(BUILD_DIR)
