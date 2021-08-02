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
# Setup build environment
#
AOT_ROOT ?= $(TVM_ROOT)/src/runtime/crt/aot

ENABLE_TVM_PLATFORM_ABORT_BACKTRACE = 0
DMLC_CORE=$(TVM_ROOT)/3rdparty/dmlc-core
PKG_COMPILE_OPTS = -g 
CC = gcc
AR = ar
RANLIB = ranlib
CC_OPTS = CC=$(CC) AR=$(AR) RANLIB=$(RANLIB)


PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	-I$(TVM_ROOT)/src/runtime/crt/include \
	-I$(TVM_ROOT)/src/runtime/crt/host \
	-I$(TVM_ROOT)/include \
	-I$(DMLC_CORE)/include \
	-I$(TVM_ROOT)/3rdparty/dlpack/include \
	-I$(AOT_ROOT)\
	-I$(build_dir) \
	-I$(CODEGEN_ROOT)/host/include

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

CRT_SRCS = $(shell find $(CRT_ROOT))

aot_test_runner: $(build_dir)/aot_test_runner

source_libs= $(wildcard $(build_dir)/../codegen/host/src/*.c)
lib_objs =$(source_libs:.c=.o) 

$(build_dir)/aot_test_runner: $(build_dir)/test.c  $(build_dir)/aot_executor.o  $(source_libs) $(build_dir)/stack_allocator.o $(build_dir)/crt_backend_api.o
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(CFLAGS) $(PKG_CFLAGS) -o $@ $^ $(PKG_LDFLAGS) $(BACKTRACE_LDFLAGS) $(BACKTRACE_CFLAGS) -lm

$(build_dir)/%.o: $(build_dir)/../codegen/host/src/%.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(CFLAGS) -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

$(build_dir)/aot_executor.o: $(TVM_ROOT)/src/runtime/crt/aot_executor/aot_executor.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(CFLAGS) -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

$(build_dir)/stack_allocator.o: $(TVM_ROOT)/src/runtime/crt/memory/stack_allocator.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(CFLAGS) -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

$(build_dir)/crt_backend_api.o: $(TVM_ROOT)/src/runtime/crt/common/crt_backend_api.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)$(CC) $(CFLAGS) -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

clean:
	$(QUIET)rm -rf $(build_dir)/crt
cleanall:
	$(QUIET)rm -rf $(build_dir)
# Don't define implicit rules; they tend to match on logical target names that aren't targets (i.e. bundle_static)
.SUFFIXES:
