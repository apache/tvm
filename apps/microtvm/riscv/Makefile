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

# Makefile to build app

# Setup build environment
BUILD_DIR := build

STANDALONE_CRT_PATH := $(abspath $(BUILD_DIR))/runtime
PKG_COMPILE_OPTS = $(COMPILE_OPTS) -O3

ifeq ($(origin CC),default)
CC = $(TOOLCHAIN_PREFIX)gcc
endif

AR = $(TOOLCHAIN_PREFIX)ar
RANLIB = $(TOOLCHAIN_PREFIX)ranlib
LD = $(TOOLCHAIN_PREFIX)gcc
PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	-I${STANDALONE_CRT_PATH}/include \
	-I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
	-I${PWD}/include \
	-I$(abspath $(BUILD_DIR))/codegen/host/include \
	$(MODEL_FLAGS)
PKG_LDFLAGS = -lm -static $(LDFLAGS)

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

MAIN = src/main.c
CODEGEN_SRCS = $(wildcard $(abspath $(BUILD_DIR))/codegen/host/src/*.c)
CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS))

model: $(BUILD_DIR)/model

$(BUILD_DIR)/main.o: $(MAIN)
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) -c $(PKG_CFLAGS) -o $@  $^

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

$(BUILD_DIR)/model: $(BUILD_DIR)/main.o $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o \
	${BUILD_DIR}/libcodegen.a
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(LD) -o $@ -Wl,--whole-archive $^ -Wl,--no-whole-archive $(PKG_LDFLAGS)

clean:
	$(QUIET)rm -rf $(BUILD_DIR)/codegen

cleanall:
	$(QUIET)rm -rf $(BUILD_DIR)

.SUFFIXES:

.DEFAULT: model
