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
build_dir := build
STANDALONE_CRT_PATH := $(shell python3 -c "import tvm.micro; print(tvm.micro.get_standalone_crt_dir())")

ARM_CPU=ARMCM55
ETHOSU_PATH=/opt/arm/ethosu
DRIVER_PATH=${ETHOSU_PATH}/core_driver
CMSIS_PATH=${ETHOSU_PATH}/cmsis
PLATFORM_PATH=${ETHOSU_PATH}/core_platform/targets/corstone-300
PKG_COMPILE_OPTS = -g -Wall -O2 -Wno-incompatible-pointer-types -Wno-format -mcpu=cortex-m55 -mthumb -mfloat-abi=hard -std=gnu99
CMAKE = /opt/arm/cmake/bin/cmake
CC = arm-none-eabi-gcc
AR = arm-none-eabi-ar
RANLIB = arm-none-eabi-ranlib
CC_OPTS = CC=$(CC) AR=$(AR) RANLIB=$(RANLIB)
PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	-I${STANDALONE_CRT_PATH}/include \
	-I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
	-Iinclude \
	-I${PLATFORM_PATH} \
	-I${ETHOSU_PATH}/core_driver/include \
	-I${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Include/ \
	-I${CMSIS_PATH}/CMSIS/Core/Include \
	-I$(abspath $(build_dir))/codegen/host/include \
	-DETHOSU_TEST_RUNNER_TOL=${ETHOSU_TEST_RUNNER_TOL}
DRIVER_CMAKE_FLAGS = -DCMAKE_TOOLCHAIN_FILE=$(abspath $(build_dir))/../arm-none-eabi-gcc.cmake \
	-DETHOSU_LOG_SEVERITY=debug \
	-DCMAKE_SYSTEM_PROCESSOR=cortex-m55
PKG_LDFLAGS = -lm -specs=nosys.specs -static -T corstone300.ld

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

CODEGEN_SRCS = $(shell find $(abspath $(build_dir))/codegen/host/src/*.c)
CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS))
CMSIS_STARTUP_SRCS = $(shell find ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/*.c)
UART_SRCS = $(shell find ${PLATFORM_PATH}/*.c)

demo: $(build_dir)/demo

$(build_dir)/stack_allocator.o: $(STANDALONE_CRT_PATH)/src/runtime/crt/memory/stack_allocator.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^

$(build_dir)/crt_backend_api.o: $(STANDALONE_CRT_PATH)/src/runtime/crt/common/crt_backend_api.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^

# Build generated code
$(build_dir)/libcodegen.a: $(CODEGEN_SRCS)
	$(QUIET)cd $(abspath $(build_dir)/codegen/host/src) && $(CC) -c $(PKG_CFLAGS) $(CODEGEN_SRCS)
	$(QUIET)$(AR) -cr $(abspath $(build_dir)/libcodegen.a) $(CODEGEN_OBJS)
	$(QUIET)$(RANLIB) $(abspath $(build_dir)/libcodegen.a)

# Build CMSIS startup code
${build_dir}/libcmsis_startup.a: $(CMSIS_STARTUP_SRCS)
	$(QUIET)mkdir -p $(abspath $(build_dir)/libcmsis_startup)
	$(QUIET)cd $(abspath $(build_dir)/libcmsis_startup) && $(CC) -c $(PKG_CFLAGS) -D${ARM_CPU} $^
	$(QUIET)$(AR) -cr $(abspath $(build_dir)/libcmsis_startup.a) $(abspath $(build_dir))/libcmsis_startup/*.o
	$(QUIET)$(RANLIB) $(abspath $(build_dir)/libcmsis_startup.a)

# Build UART code
${build_dir}/libuart.a: $(UART_SRCS)
	$(QUIET)mkdir -p $(abspath $(build_dir)/libuart)
	$(QUIET)cd $(abspath $(build_dir)/libuart) && $(CC) -c $(PKG_CFLAGS) $^
	$(QUIET)$(AR) -cr $(abspath $(build_dir)/libuart.a) $(abspath $(build_dir))/libuart/*.o
	$(QUIET)$(RANLIB) $(abspath $(build_dir)/libuart.a)

# Build Arm(R) Ethos(TM)-U core driver
${build_dir}/ethosu_core_driver/libethosu_core_driver.a:
	$(QUIET)mkdir -p $(@D)
	$(QUIET)cd $(DRIVER_PATH) && $(CMAKE) -B $(abspath $(build_dir)/ethosu_core_driver) $(DRIVER_CMAKE_FLAGS)
	$(QUIET)cd $(abspath $(build_dir)/ethosu_core_driver) && $(MAKE)

# Build demo application
$(build_dir)/demo: src/demo.c $(build_dir)/stack_allocator.o $(build_dir)/crt_backend_api.o ${build_dir}/libcodegen.a ${build_dir}/libcmsis_startup.a ${build_dir}/ethosu_core_driver/libethosu_core_driver.a ${build_dir}/libuart.a
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(PKG_CFLAGS) -o $@ $^ $(PKG_LDFLAGS)

clean:
	$(QUIET)rm -rf $(build_dir)/crt

cleanall:
	$(QUIET)rm -rf $(build_dir)

.SUFFIXES:

.DEFAULT: demo
