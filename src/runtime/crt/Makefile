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

ifeq ($(CRT_CONFIG),)
$(error "Must supply path to crt_config.h: CRT_CONFIG=...")
endif
DLPACK_INCLUDE_DIR ?= ../../../3rdparty/dlpack/include
TVM_INCLUDE_DIR ?= ../../../include

BUILD_DIR ?= build
PREFIX ?=

AR ?= ${PREFIX}ar
CC ?= ${PREFIX}gcc
RANLIB ?= ${PREFIX}ranlib

QUIET ?= @

CFLAGS += -isystem "${TVM_INCLUDE_DIR}" -isystem "${DLPACK_INCLUDE_DIR}" -I include -I $(dir ${CRT_CONFIG})
CFLAGS += -Werror -g
LDFLAGS += -Werror -g

${BUILD_DIR}/%.o: %.c
	${QUIET}mkdir -p $(dir $@)
	${QUIET}${CC} ${CFLAGS} -c -o "$@" "$<"

${BUILD_DIR}/common/libcommon.a: $(patsubst %.c,${BUILD_DIR}/%.o,$(wildcard common/*.c))
	${QUIET}${AR} -cr "$@" $^
	${QUIET}${RANLIB} ${RANLIBFLAGS} "$@"

${BUILD_DIR}/graph_runtime/libgraph_runtime.a: $(patsubst %.c,${BUILD_DIR}/%.o,$(wildcard graph_runtime/*.c))
	${QUIET}${AR} -cr "$@" $^
	${QUIET}${RANLIB} ${RANLIBFLAGS} "$@"

common: ${BUILD_DIR}/common/libcommon.a
graph_runtime: ${BUILD_DIR}/graph_runtime/libgraph_runtime.a

all: common graph_runtime
clean:
	rm -rf "${BUILD_DIR}"

.PHONY: all common graph_runtime
.DEFAULT_GOAL: all
