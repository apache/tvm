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

# Makefile to build demo

# Setup build environment
BUILD_DIR := build

ARM_CPU = ARMCM55
ETHOSU_PATH = /opt/arm/ethosu
CMSIS_PATH ?= ${ETHOSU_PATH}/cmsis
ETHOSU_PLATFORM_PATH ?= ${ETHOSU_PATH}/core_platform
STANDALONE_CRT_PATH := $(abspath $(BUILD_DIR))/runtime
CORSTONE_300_PATH = ${ETHOSU_PLATFORM_PATH}/targets/corstone-300
PKG_COMPILE_OPTS = -g -Wall -O2 -Wno-incompatible-pointer-types -Wno-format -mcpu=cortex-m55 -mthumb -mfloat-abi=hard -std=gnu99
CMAKE ?= cmake
CC = arm-none-eabi-gcc
AR = arm-none-eabi-ar
RANLIB = arm-none-eabi-ranlib
PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	-I${STANDALONE_CRT_PATH}/include \
	-I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
	-I${PWD}/${BUILD_DIR}/crt_config \
	-I${PWD}/include \
	-I${ETHOSU_PLATFORM_PATH}/drivers/uart/include \
	-I${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Include/ \
	-I${CMSIS_PATH}/CMSIS/Core/Include \
	-I${CMSIS_PATH}/CMSIS-NN/Include \
	-I${CMSIS_PATH}/CMSIS/DSP/Include \
	-I$(abspath $(BUILD_DIR))/codegen/host/include
CMAKE_FLAGS = -DCMAKE_TOOLCHAIN_FILE=$(abspath $(BUILD_DIR))/../arm-none-eabi-gcc.cmake \
	-DTARGET_CPU=cortex-m55
PKG_LDFLAGS = -lm -specs=nosys.specs -static -T corstone300.ld

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

DEMO_MAIN = src/demo_bare_metal.c
CODEGEN_SRCS = $(wildcard $(abspath $(BUILD_DIR))/codegen/host/src/*.c)
CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS))
CMSIS_STARTUP_SRCS = $(wildcard ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/*.c)
CMSIS_NN_SRCS = $(shell find ${CMSIS_PATH}/CMSIS-NN/Source/*/*.c)
CORSTONE_300_SRCS = $(wildcard ${CORSTONE_300_PATH}/*.c)

demo: $(BUILD_DIR)/demo

$(BUILD_DIR)/stack_allocator.o: $(STANDALONE_CRT_PATH)/src/runtime/crt/memory/stack_allocator.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^

$(BUILD_DIR)/crt_backend_api.o: $(STANDALONE_CRT_PATH)/src/runtime/crt/common/crt_backend_api.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^

# Build generated code
$(BUILD_DIR)/libcodegen.a: $(CODEGEN_SRCS)
	$(QUIET)cd $(abspath $(BUILD_DIR)/codegen/host/src) && $(CC) -c $(PKG_CFLAGS) $(CODEGEN_SRCS)
	$(QUIET)$(AR) -cr $(abspath $(BUILD_DIR)/libcodegen.a) $(CODEGEN_OBJS)
	$(QUIET)$(RANLIB) $(abspath $(BUILD_DIR)/libcodegen.a)

# Build CMSIS startup code
${BUILD_DIR}/libcmsis_startup.a: $(CMSIS_STARTUP_SRCS)
	$(QUIET)mkdir -p $(abspath $(BUILD_DIR)/libcmsis_startup)
	$(QUIET)cd $(abspath $(BUILD_DIR)/libcmsis_startup) && $(CC) -c $(PKG_CFLAGS) -D${ARM_CPU} $^
	$(QUIET)$(AR) -cr $(abspath $(BUILD_DIR)/libcmsis_startup.a) $(abspath $(BUILD_DIR))/libcmsis_startup/*.o
	$(QUIET)$(RANLIB) $(abspath $(BUILD_DIR)/libcmsis_startup.a)

# Build CMSIS-NN code
${BUILD_DIR}/libcmsis_nn.a: $(CMSIS_NN_SRCS)
	$(QUIET)mkdir -p $(abspath $(BUILD_DIR)/libcmsis_nn)
	$(QUIET)cd $(abspath $(BUILD_DIR)/libcmsis_nn) && $(CC) -c $(PKG_CFLAGS) -D${ARM_CPU} $^
	$(QUIET)$(AR) -cr $(abspath $(BUILD_DIR)/libcmsis_nn.a) $(abspath $(BUILD_DIR))/libcmsis_nn/*.o
	$(QUIET)$(RANLIB) $(abspath $(BUILD_DIR)/libcmsis_nn.a)

# Build UART driver
${BUILD_DIR}/ethosu_core_platform/libethosu_uart_cmsdk_apb.a:
	$(QUIET)mkdir -p $(@D)
	$(QUIET)cd ${ETHOSU_PLATFORM_PATH}/drivers/uart && $(CMAKE) -B $(abspath $(BUILD_DIR)/ethosu_core_platform) $(CMAKE_FLAGS)
	$(QUIET)cd $(abspath $(BUILD_DIR)/ethosu_core_platform) && $(MAKE)

# Build demo application
$(BUILD_DIR)/demo: $(DEMO_MAIN) $(CORSTONE_300_SRCS) $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o \
	${BUILD_DIR}/libcodegen.a ${BUILD_DIR}/libcmsis_startup.a ${BUILD_DIR}/libcmsis_nn.a ${BUILD_DIR}/ethosu_core_platform/libethosu_uart_cmsdk_apb.a
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(PKG_CFLAGS) $(FREERTOS_FLAGS) -o $@ -Wl,--whole-archive $^ -Wl,--no-whole-archive $(PKG_LDFLAGS)

clean:
	$(QUIET)rm -rf $(BUILD_DIR)/codegen

cleanall:
	$(QUIET)rm -rf $(BUILD_DIR)

.SUFFIXES:

.DEFAULT: demo
