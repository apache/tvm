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
VTA_DIR = $(CURDIR)/../..
BUILD_DIR = $(VTA_DIR)/build/hardware/xilinx
SCRIPT_DIR = $(CURDIR)/scripts
SRC_DIR = $(CURDIR)/src

# Executables
VIVADO_HLS = vivado_hls
VIVADO = vivado

# Process VTA JSON config
VTA_CONFIG := $(CURDIR)/../../config/vta_config.py

# Derive config name
CONF := $(shell python ${VTA_CONFIG} --cfg-str)
IP_BUILD_PATH := $(BUILD_DIR)/hls/$(CONF)
HW_BUILD_PATH := $(BUILD_DIR)/vivado/$(CONF)

# IP file path
IP_PATH := $(BUILD_DIR)/hls/$(CONF)/vta_compute/soln/impl/ip/xilinx_com_hls_compute_1_0.zip

# Bitstream file path
BIT_PATH := $(BUILD_DIR)/vivado/$(CONF)/export/$(CONF).bit

.PHONY: all ip bit clean clean_all

all: bit
ip: $(IP_PATH)
bit: $(BIT_PATH)

$(IP_PATH): $(SRC_DIR)/*
	mkdir -p $(IP_BUILD_PATH)
	cd $(IP_BUILD_PATH) && \
		$(VIVADO_HLS) \
		-f $(SCRIPT_DIR)/hls.tcl \
		-tclargs \
			$(VTA_DIR) \
			${VTA_CONFIG}

$(BIT_PATH): $(IP_PATH)
	mkdir -p $(HW_BUILD_PATH)
	cd $(HW_BUILD_PATH) && \
		$(VIVADO) \
		-mode tcl \
		-source $(SCRIPT_DIR)/vivado.tcl \
		-tclargs \
			$(BUILD_DIR)/hls/$(CONF) \
			${VTA_CONFIG}

clean:
	rm -rf *.out *.log

cleanall: clean
	rm -rf $(BUILD_DIR)
