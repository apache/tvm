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

# Makefile Example to deploy TVM modules.
ROCM_PATH=/opt/rocm

TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++14 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I${ROCM_PATH}/include

PKG_LDFLAGS = -L${ROCM_PATH}/lib -L${TVM_ROOT}/lib -ldl -lpthread -lhip_hcc -lMIOpen

.PHONY: clean all

all: lib/libtvm_runtime_rocm.so

# Build rule for all in one TVM package library
lib/libtvm_runtime_rocm.so: rocm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -shared -o $@ $(filter %.cc %.o %.a, $^) $(PKG_LDFLAGS)
