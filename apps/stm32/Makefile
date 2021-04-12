
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

# Top level Makefile for the STM32 microcontroller project.

#
# Set the path to your ARM Toolset.
# Only gcc is supported at this time.
#
#export ARM_PATH ?= 
ifndef ARM_PATH
$(error ARM_PATH must be set and point at your GCC ARM compiler installation)
endif

#
# Set the path to your TVM compiler installation
#
#export TVM_PATH ?= 
ifndef TVM_PATH
$(error TVM_PATH must be set and point at your TVM stack installation)
endif

#
# Set the path to your CubeMX installation
#
#export X_CUBE_PATH ?= 
ifndef X_CUBE_PATH
$(error X_CUBE_PATH must be set and point at your STM32 Cube Firmware installation)
endif

#export X_CUBE_TOOL_PATH ?= 
ifndef X_CUBE_TOOL_PATH
$(error X_CUBE_TOOL_PATH must be set and point at your STM32 Cube Programmer installation)
endif

#
# Setup path to your network generated files
#
#export MODEL_PATH = 
ifndef MODEL_PATH
$(error MODEL_PATH must be set and point at your model implementation generated with the TVM compiler)
endif

BOARD ?= h747idisco
LDCONF ?= 4

all: perf valid

check_model:
	@if [ ! -d "$(MODEL_PATH)" ]; then \
		echo "$(MODEL_PATH) does not exist! check Makefile!"; \
		exit 1; \
	fi

perf: check_model
	@echo " === Perf: Building from TVM sources ..."
	$(MAKE) -f $(BOARD)/makefile.base CONFIG=$(LDCONF)

valid: check_model
	@echo " === Validation: Building from TVM sources ..."
	$(MAKE) -f $(BOARD)/makefile.base VALID=1 CONFIG=$(LDCONF)

flash-perf:
	$(MAKE) -f $(BOARD)/makefile.base flash

flash-valid:
	$(MAKE) -f $(BOARD)/makefile.base VALID=1 flash

clean: clean-perf clean-valid

clean-perf:
	$(MAKE) -f $(BOARD)/makefile.base clean

clean-valid:
	$(MAKE) -f $(BOARD)/makefile.base VALID=1 clean
