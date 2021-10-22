
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
#ifndef X_CUBE_PATH
#$(error X_CUBE_PATH must be set and point at your STM32 Cube Firmware installation)
#endif

#export X_CUBE_TOOL_PATH ?= 
ifndef X_CUBE_TOOL_PATH
$(error X_CUBE_TOOL_PATH must be set and point at your STM32 Cube Programmer installation)
endif

#
# Choose a board from the list below
#

# Disco
BOARD = disco-h747i
#VID=0x0483
#PID=0x374e
SERIAL_NUMBER = 003500193137511439383538
#
# Setup the firmware path
#
X_CUBE_PATH = /prj/p_mcd_ais/users/stoutchi/tvm-apps-stm32/STM32Cube_FW_H7_V1.8.0

# Nucleao
#BOARD = nucleo-f412zg
#VID=0x0483
#PID=0x374b
#SERIAL_NUMBER = 0674FF535752877167073836
#
# Setup the firmware path
#
#X_CUBE_PATH = /prj/p_mcd_ais/users/stoutchi/tvm-apps-stm32/STM32Cube_FW_F4_V1.26.0

# Indicate the revision of the board (if necessary)
#  xyy with x:1 for A, 2 for B ...
#           yy:01 ...
# ex: MB1361-L552ZEQ-C02	-> 302
#     MB1361-L552ZEQ-A02    -> 102
# BOARD_VERSION ?= 102

#ifeq ($(BOARD),$(filter $(BOARD), "stm32h743" "stm32h747"))
#  BOARD_NAME = stm32h7xx
#endif

ifeq ($(BOARD), disco-h747i)
  EXTLOAD = "${X_CUBE_TOOL_PATH}/ExternalLoader/MT25TL01G_STM32H747I-DISCO.stldr"
#else ifeq ($(BOARD), nucleo-f412)
#  EXTLOAD = "${X_CUBE_TOOL_PATH}/ExternalLoader/N25Q128A_STM32412G-DISCO.stldr" 
endif

# 115200
# BAUDRATE = 921600

STM32_PROG ?= ${X_CUBE_TOOL_PATH}/STM32_Programmer_CLI

ifdef EXTLOAD
#STM32_PROG += --extload "${X_CUBE_TOOL_PATH}/ExternalLoader/MT25TL01G_STM32H747I-DISCO.stldr"
STM32_PROG += --extload $(EXTLOAD)
endif

BUILD_DIR = ./build/$(BOARD)
BOARDS_DIR = ./Boards

APP_NAME = Project

debug:
	@echo " === Board: $(BOARD) ..."

check:
#	$(if $(strip $(TARGET)), @echo " === Target: $(TARGET) ...", $(error BOARD $(BOARD) not supported))
	$(if $(strip $(MODEL_PATH)), @echo " === Model path: $(MODEL_PATH)", $(error MODEL_PATH must be set and point at your model implementation generated with the TVM compiler))
	@if [ ! -d "$(MODEL_PATH)" ]; then \
		echo "$(MODEL_PATH) does not exist!"; \
		exit 1; \
	fi

info:
	$(MAKE) -f $(BOARDS_DIR)/makefile.$(BOARD) info BUILD_DIR=$(BUILD_DIR) APP_NAME=$(APP_NAME)

perf: check
	@echo " === Perf: Building from TVM sources ..."
	$(MAKE) -f $(BOARDS_DIR)/makefile.$(BOARD) CONFIG=$(LDCONF) BUILD_DIR=$(BUILD_DIR) APP_NAME=$(APP_NAME)

valid: debug check
	@echo " === Validation: Building from TVM sources ..."
	$(MAKE) -f $(BOARDS_DIR)/makefile.$(BOARD) VALID=1 CONFIG=$(LDCONF) BUILD_DIR=$(BUILD_DIR) APP_NAME=$(APP_NAME)

#######################################
# clean up
#######################################

clean: app.clean

app.clean:
	@echo -e "I: Cleaning APP build directory (${BUILD_DIR}/*)..."
	${q}rm -rf $(BUILD_DIR)

clean_nn:
	-rm $(BUILD_DIR)/network.*	
	-rm $(BUILD_DIR)/network_data.*
	-rm $(BUILD_DIR)/network_lib.*

#######################################
# flash
#######################################

flash: $(BUILD_DIR)/$(APP_NAME).elf 
	${STM32_PROG} -c port=swd sn=${SERIAL_NUMBER} mode=UR reset=HWrst -d $(BUILD_DIR)/$(APP_NAME).elf -s
