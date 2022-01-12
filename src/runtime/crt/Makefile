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

# NOTE: Although this Makefile contains build commands for the C runtime, it isn't intended to be
# used directly in the TVM source tree. Instead, build the "standalone_crt" target, which produces a
# directory tree suitable for this Makefile. If this Makefile looks like it's the top-level of a
# source tree, you can probably ignore this message.

# NOTE: If files appear to be missing in the generated standalone_crt target, consult the copy job
# specs listed in the TVM repo in cmake/modules/StandaloneCrt.cmake.

ifeq ($(CRT_CONFIG),)
$(error "Must supply path to crt_config.h: CRT_CONFIG=...")
endif

ifneq ($(wildcard .gitignore),)
$(error "detected building inside tvm source tree.")
$(error "build the standalone_crt target, and re-invoke makefile in build/standalone_crt")
endif

BUILD_DIR ?= build
PREFIX ?=

AR ?= ${PREFIX}ar
CC ?= ${PREFIX}gcc
CXX ?= ${PREFIX}g++
RANLIB ?= ${PREFIX}ranlib

QUIET ?= @

CRT_PREFIX = $(wildcard src/crt)

INCLUDES ?= -isystem include -iquote $(dir ${CRT_CONFIG})
CFLAGS += ${INCLUDES} -Werror -g $(EXTRA_CFLAGS) -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>
CXXFLAGS += ${INCLUDES} -std=c++11 -Werror -g $(EXTRA_CXXFLAGS) -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>
LDFLAGS += -Werror -g $(EXTRA_LDFLAGS)

${BUILD_DIR}/%.o: src/%.c $(CRT_CONFIG)
	${QUIET}mkdir -p $(dir $@)
	${QUIET}${CC} ${CFLAGS} -c -o "$@" "$<"

${BUILD_DIR}/%.o: src/%.cc $(CRT_CONFIG)
	${QUIET}mkdir -p $(dir $@)
	${QUIET}${CXX} ${CXXFLAGS} -c -o "$@" "$<"

define LIB_template
$${BUILD_DIR}/lib$(notdir $(1)).a: $$(patsubst src/%.c,$${BUILD_DIR}/%.o,$$(wildcard src/$(1:src/%=%)/*.c)) $$(patsubst src/%.cc,${BUILD_DIR}/%.o,$$(wildcard src/$(1:src/%=%)/*.cc))
	$${QUIET}$${AR} -cr "$$@" $$^
	$${QUIET}$${RANLIB} $${RANLIBFLAGS} "$$@"
$(notdir $(1)): $${BUILD_DIR}/lib$(notdir $(1)).a

endef

LIBS = \
	src/runtime/crt/common \
	src/runtime/crt/graph_executor \
	src/runtime/crt/graph_executor_module \
	src/runtime/crt/memory \
	src/runtime/crt/microtvm_rpc_common \
	src/runtime/crt/microtvm_rpc_server

$(foreach lib,$(LIBS),$(eval $(call LIB_template,$(lib))))

all: $(notdir $(LIBS))
clean:
	rm -rf "${BUILD_DIR}"

.PHONY: all $(notdir $(LIBS))
.DEFAULT_GOAL: all
