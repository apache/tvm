# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Directories
ROOTDIR = $(CURDIR)
BUILD_NAME = build
BUILD_DIR = $(ROOTDIR)/../../$(BUILD_NAME)/hardware/intel
SCRIPT_DIR = $(ROOTDIR)/scripts
SRC_DIR = $(ROOTDIR)/../chisel

# Process VTA JSON config
VTA_CONFIG = python $(CURDIR)/../../config/vta_config.py

# Debug flag
DEBUG = false
# Prevent generation of DSP
NO_DSP = true
# Device
DEVICE = $(shell $(VTA_CONFIG) --get-fpga-dev)
# Device family
DEVICE_FAMILY = $(shell $(VTA_CONFIG) --get-fpga-family)
# Project name
PROJECT = de10_nano_top
# Frequency in MHz
FREQ_MHZ = $(shell $(VTA_CONFIG) --get-fpga-freq)

#---------------------
# Compilation parameters
#--------------------

# Derive config name
CONF = $(shell ${VTA_CONFIG} --cfg-str)
IP_BUILD_PATH = $(BUILD_DIR)/chisel/$(CONF)
HW_BUILD_PATH = $(BUILD_DIR)/quartus/$(CONF)

ifeq ($(NO_DSP), true)
  DSP_FLAG =
else
  DSP_FLAG = --dsp
endif

# IP file path
IP_PATH = $(IP_BUILD_PATH)/VTA.DefaultDe10Config.v

# Bitstream file path
BIT_PATH = $(HW_BUILD_PATH)/export/vta_$(FREQ_MHZ)MHz.rbf
CPF_OPT := -o bitstream_compression=on

# System design file path
QSYS_PATH = $(HW_BUILD_PATH)/soc_system.qsys

.PHONY: all ip bit qsys clean

all: bit
ip: $(IP_PATH)
bit: $(BIT_PATH)
qsys: $(QSYS_PATH)

$(IP_PATH): $(SRC_DIR)
	mkdir -p $(IP_BUILD_PATH)
	cd $(SRC_DIR) && \
    make CONFIG=DefaultDe10Config chisel_build_dir=$(IP_BUILD_PATH) verilog

$(QSYS_PATH): $(IP_PATH)
	mkdir -p $(HW_BUILD_PATH)
	cd $(HW_BUILD_PATH) && \
    cp -r $(SCRIPT_DIR)/* $(HW_BUILD_PATH) && \
    python3 $(SCRIPT_DIR)/set_attrs.py -i $(IP_PATH) -o $(HW_BUILD_PATH)/ip/vta/VTAShell.v $(DSP_FLAG) && \
    qsys-script --script=soc_system.tcl $(DEVICE) $(DEVICE_FAMILY) $(FREQ_MHZ)

$(BIT_PATH): $(QSYS_PATH)
	cd $(HW_BUILD_PATH) && \
    quartus_sh -t $(SCRIPT_DIR)/compile_design.tcl $(DEVICE) $(PROJECT) && \
    mkdir -p $(shell dirname $(BIT_PATH)) && \
    quartus_cpf $(CPF_OPT) -c $(HW_BUILD_PATH)/$(PROJECT).sof $(BIT_PATH)

clean:
	rm -rf $(BUILD_DIR)

clean-qsys:
	rm -rf $(QSYS_PATH)
