# Directories
ROOTDIR = $(CURDIR)
BUILD_NAME = build
BUILD_DIR = $(ROOTDIR)/../../$(BUILD_NAME)/hardware/intel
SCRIPT_DIR = $(ROOTDIR)/scripts
SRC_DIR = $(ROOTDIR)/ip
SIM_DIR = $(ROOTDIR)/sim
TEST_DIR = $(ROOTDIR)/../../tests/hardware/common
INCLUDE_DIR = $(ROOTDIR)/../../include

# Executables
TCLSH = quartus_sh

# HLS mode, options: skip_sim, sim
MODE = skip_sim
# Debug flag
DEBUG = false
# SLURM
SLURM = false
# Prevent generation of DSP
NO_DSP = false
# Prevent generation of ALU
NO_ALU = false

# Device
ifeq ($(MODE), sim)
  DEVICE = x86-64
else
  DEVICE = 5CSEBA6U23I7
endif

# Add header files into cxx flags
HLS_CXX_FLAGS = -I/usr/include/c++/4.8.5 -I/usr/include/x86_64-linux-gnu/c++/4.8 -I$(INCLUDE_DIR)

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

ifeq ($(MODE), skip_sim)
  HLS_CXX_FLAGS += --fpga-only
endif

ifeq ($(MODE), sim)
  HLS_CXX_FLAGS += --x86-only -g
endif

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
IP_BUILD_PATH = $(BUILD_DIR)/chisel/$(CONF)
HW_BUILD_PATH = $(BUILD_DIR)/quartus/$(CONF)

# IP file path
IP_PATH = $(BUILD_DIR)/chisel/$(CONF)/solution0/impl/ip/intel_fpga_com_chisel_vta_1_0.tgz

# Bitstream file path
BIT_PATH = $(BUILD_DIR)/quartus/$(CONF)/export/$(CONF).rbf

.PHONY: all ip bit bsp clean clean_all

all: bit
ip: $(IP_PATH)
bit: $(BIT_PATH)

$(IP_PATH): $(SRC_DIR)
	mkdir -p $(IP_BUILD_PATH)
	cd $(IP_BUILD_PATH) && \
    mkdir -p $(shell dirname $(IP_PATH)) && \
    cd $(SRC_DIR)/.. && \
    tar zcvf $(IP_PATH) ip

$(BIT_PATH): $(IP_PATH)
	mkdir -p $(HW_BUILD_PATH)
	cd $(HW_BUILD_PATH) && \
    cp -r $(ROOTDIR)/ip $(HW_BUILD_PATH)/.. && \
    cp $(ROOTDIR)/scripts/* $(HW_BUILD_PATH) && \
    $(TCLSH) -t $(SCRIPT_DIR)/DE10_NANO_SoC_GHRD.tcl

clean:
	rm -rf *.out *.log *.sb figures

cleanall: clean
	rm -rf $(BUILD_DIR)
