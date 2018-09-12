# Directories
ROOTDIR = $(CURDIR)
BUILD_NAME = build
BUILD_DIR = $(ROOTDIR)/../../$(BUILD_NAME)/hardware/intel
SCRIPT_DIR = $(ROOTDIR)/scripts
SRC_DIR = $(ROOTDIR)/src
SIM_DIR = $(ROOTDIR)/sim
TEST_DIR = $(ROOTDIR)/../../tests/hardware/common
INCLUDE_DIR = $(ROOTDIR)/../../include

# Executables
CXX = i++
QSYS = qsys
HSI = hsi

# Device
# DEVICE = 5CSEBA6U23I7
DEVICE = x86-64
HLS_CXX_FLAGS = -I/usr/include/c++/4.8.5 -I/usr/include/x86_64-linux-gnu/c++/4.8 -I$(INCLUDE_DIR)

# HLS mode
MODE = skip_sim
# Debug flag
DEBUG = true
# SLURM
SLURM = false
# Prevent generation of DSP
NO_DSP = false
# Prevent generation of ALU
NO_ALU = false

# Process VTA JSON config
VTA_CONFIG = python $(CURDIR)/../../config/vta_config.py
CFLAGS := $(shell ${VTA_CONFIG} --cflags)
VTA_TARGET := $(shell ${VTA_CONFIG} --target)

#---------------------
# VTA Parameters
#--------------------
VTA_INP_WIDTH := $(shell ${VTA_CONFIG} --get-inpwidth)
VTA_WGT_WIDTH := $(shell ${VTA_CONFIG} --get-wgtwidth)
VTA_ACC_WIDTH := $(shell ${VTA_CONFIG} --get-accwidth)
VTA_OUT_WIDTH := $(shell ${VTA_CONFIG} --get-outwidth)
VTA_BATCH := $(shell ${VTA_CONFIG} --get-batch)
VTA_IN_BLOCK := $(shell ${VTA_CONFIG} --get-blockin)
VTA_OUT_BLOCK := $(shell ${VTA_CONFIG} --get-blockout)
VTA_UOP_BUFF_SIZE := $(shell ${VTA_CONFIG} --get-uopbuffsize)
VTA_INP_BUFF_SIZE := $(shell ${VTA_CONFIG} --get-inpbuffsize)
VTA_WGT_BUFF_SIZE := $(shell ${VTA_CONFIG} --get-wgtbuffsize)
VTA_ACC_BUFF_SIZE := $(shell ${VTA_CONFIG} --get-accbuffsize)
VTA_OUT_BUFF_SIZE := $(shell ${VTA_CONFIG} --get-outbuffsize)

HLS_CXX_FLAGS += -DVTA_LOG_INP_WIDTH=$(VTA_INP_WIDTH)
HLS_CXX_FLAGS += -DVTA_LOG_WGT_WIDTH=$(VTA_WGT_WIDTH)
HLS_CXX_FLAGS += -DVTA_LOG_ACC_WIDTH=$(VTA_ACC_WIDTH)
HLS_CXX_FLAGS += -DVTA_LOG_OUT_WIDTH=$(VTA_OUT_WIDTH)
HLS_CXX_FLAGS += -DVTA_LOG_BATCH=$(VTA_BATCH)
HLS_CXX_FLAGS += -DVTA_LOG_BLOCK_IN=$(VTA_IN_BLOCK)
HLS_CXX_FLAGS += -DVTA_LOG_BLOCK_OUT=$(VTA_OUT_BLOCK)
HLS_CXX_FLAGS += -DVTA_LOG_UOP_BUFF_SIZE=$(VTA_UOP_BUFF_SIZE)
HLS_CXX_FLAGS += -DVTA_LOG_INP_BUFF_SIZE=$(VTA_INP_BUFF_SIZE)
HLS_CXX_FLAGS += -DVTA_LOG_WGT_BUFF_SIZE=$(VTA_WGT_BUFF_SIZE)
HLS_CXX_FLAGS += -DVTA_LOG_ACC_BUFF_SIZE=$(VTA_ACC_BUFF_SIZE)
HLS_CXX_FLAGS += -DVTA_LOG_OUT_BUFF_SIZE=$(VTA_OUT_BUFF_SIZE)

HLS_CXX_FLAGS += --debug-log -v
# HLS_CXX_FLAGS += --fpga-only --debug-log -v
HLS_CXX_FLAGS += --x86-only

ifeq ($(DEBUG), true)
  HLS_CXX_FLAGS += -DVTA_DEBUG=1
endif

#---------------------
# FPGA Parameters
#--------------------
VTA_CLOCK_FREQ = $(shell ${VTA_CONFIG} --get-fpgafreq)
VTA_TARGET_PER = $(shell ${VTA_CONFIG} --get-fpgaper)

#---------------------
# Compilation parameters
#--------------------

#  Number of threads during compilation
VTA_HW_COMP_THREADS = 8

# Derive config name
CONF = $(shell ${VTA_CONFIG} --cfg-str)
IP_BUILD_PATH = $(BUILD_DIR)/hls/$(CONF)
HW_BUILD_PATH = $(BUILD_DIR)/qsys/$(CONF)

ifeq ($(SLURM), true)
	IP_BUILD_PATH = /scratch/hls/$(CONF)
	HW_BUILD_PATH = /scratch/qsys/$(CONF)
endif

# IP file path
IP_PATH = $(BUILD_DIR)/hls/$(CONF)/solution0/impl/ip/intel_fpga_com_hls_vta_1_0.zip

# Bitstream file path
BIT_PATH = $(BUILD_DIR)/qsys/$(CONF)/export/$(CONF).bit

.PHONY: all ip bit bsp clean clean_all

all: ip
ip: $(IP_PATH)
bit: $(BIT_PATH)

$(IP_PATH): $(SRC_DIR)/*
	mkdir -p $(IP_BUILD_PATH)
	cd $(IP_BUILD_PATH) && \
	  $(CXX) $(HLS_CXX_FLAGS) -march=$(DEVICE) $(SRC_DIR)/vta.cc \
    $(SIM_DIR)/vta_test.cc $(TEST_DIR)/test_lib.cc \
    -o $(IP_BUILD_PATH)/vta
# -tclargs $(SRC_DIR) $(SIM_DIR) $(TEST_DIR) $(INCLUDE_DIR) 
# $(MODE) $(DEBUG) $(NO_DSP) $(NO_ALU) $(VTA_TARGET_PER) 
# $(VTA_INP_WIDTH) $(VTA_WGT_WIDTH) $(VTA_ACC_WIDTH) $(VTA_OUT_WIDTH) 
# $(VTA_BATCH) $(VTA_IN_BLOCK) $(VTA_OUT_BLOCK) 
# $(VTA_UOP_BUFF_SIZE) $(VTA_INP_BUFF_SIZE) $(VTA_WGT_BUFF_SIZE) 
# $(VTA_ACC_BUFF_SIZE) $(VTA_OUT_BUFF_SIZE)
ifeq ($(SLURM), true)
	mkdir -p $(BUILD_DIR)/hls
	mv $(IP_BUILD_PATH) $(BUILD_DIR)/hls/.
endif

$(BIT_PATH): $(IP_PATH)
	mkdir -p $(HW_BUILD_PATH)
	cd $(HW_BUILD_PATH) && \
		$(QSYS) -mode tcl -source $(SCRIPT_DIR)/qsys.tcl \
		-tclargs $(BUILD_DIR)/hls/$(CONF) $(VTA_HW_COMP_THREADS) $(VTA_CLOCK_FREQ) \
		$(VTA_INP_WIDTH) $(VTA_WGT_WIDTH) $(VTA_OUT_WIDTH) \
		$(VTA_BATCH) $(VTA_IN_BLOCK) $(VTA_OUT_BLOCK) \
		$(VTA_INP_BUFF_SIZE) $(VTA_WGT_BUFF_SIZE) $(VTA_OUT_BUFF_SIZE)
ifeq ($(SLURM), true)
	mkdir -p $(BUILD_DIR)/qsys
	mv $(HW_BUILD_PATH) $(BUILD_DIR)/qsys/.
endif

bsp: $(BIT_PATH)
	cd $(HW_BUILD_PATH) && $(HSI) -mode tcl -source $(SCRIPT_DIR)/hsi.tcl -nojournal -nolog
	cd $(HW_BUILD_PATH)/bsp && make

clean:
	rm -rf *.out *.log *.sb figures

cleanall: clean
	rm -rf $(BUILD_DIR)
