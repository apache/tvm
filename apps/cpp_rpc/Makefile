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

# Makefile to compile RPC Server.
TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
TVM_RUNTIME_DIR?=
OS?=

# Android can not link pthrad, but Linux need.
ifeq ($(OS), Linux)
LINK_PTHREAD=-lpthread
else
LINK_PTHREAD=
endif

PKG_CFLAGS = -std=c++14 -O2 -fPIC -Wall\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include

PKG_LDFLAGS = -L$(TVM_RUNTIME_DIR) $(LINK_PTHREAD) -ltvm_runtime -ldl -Wl,-R$(TVM_RUNTIME_DIR)

ifeq ($(USE_GLOG), 1)
        PKG_CFLAGS += -DDMLC_USE_GLOG=1
        PKG_LDFLAGS += -lglog
endif

.PHONY: clean all

all: tvm_rpc

# Build rule for all in one TVM package library
tvm_rpc: *.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@ $(filter %.cc %.o %.a, $^) $(PKG_LDFLAGS)

clean:
	-rm -f tvm_rpc