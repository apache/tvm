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

# Makefile Example to bundle TVM modules.

# Setup build environment
TVM_ROOT=$(shell cd ../..; pwd)
CRT_ROOT ?= ../../build/standalone_crt
ifeq ($(shell ls -lhd $(CRT_ROOT)),)
$(error "CRT not found. Ensure you have built the standalone_crt target and try again")
endif

ENABLE_TVM_PLATFORM_ABORT_BACKTRACE ?= 1

build_dir := build

DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
PKG_COMPILE_OPTS = -g -Wall -O2 -fPIC
PKG_CXXFLAGS = ${PKG_COMPILE_OPTS} -std=c++17 \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include \
	-I${build_dir}/crt_config \
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>
PKG_CFLAGS = ${PKG_COMPILE_OPTS} \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include \
	-I${build_dir}/crt_config \
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -pthread -lm



BACKTRACE_SRCS =
BACKTRACE_LDFLAGS =
BACKTRACE_CFLAGS =
$(ifeq ENABLE_TVM_PLATFORM_ABORT_BACKTRACE,1)
BACKTRACE_SRCS += backtrace.c
BACKTRACE_LDFLAGS += -ldl
BACKTRACE_CFLAGS += -DENABLE_TVM_PLATFORM_ABORT_BACKTRACE
$(endif)

BACKTRACE_OBJS = $(patsubst %.c,$(build_dir)/%.o,$(BACKTRACE_SRCS))

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

MODEL_OBJ = $(build_dir)/model_c/devc.o $(build_dir)/model_c/lib0.o $(build_dir)/model_c/lib1.o
TEST_MODEL_OBJ = $(build_dir)/test_model_c/devc.o $(build_dir)/test_model_c/lib0.o $(build_dir)/test_model_c/lib1.o

demo_dynamic: $(build_dir)/demo_dynamic $(build_dir)/bundle.so $(build_dir)/bundle_c.so $(build_dir)/bundle.so $(build_dir)/graph_cpp.json $(build_dir)/graph_c.json $(build_dir)/params_cpp.bin $(build_dir)/params_c.bin $(build_dir)/cat.bin
	$(QUIET)TVM_NUM_THREADS=1 $(build_dir)/demo_dynamic $(build_dir)/bundle.so $(build_dir)/graph_cpp.json $(build_dir)/params_cpp.bin $(build_dir)/cat.bin
	$(QUIET)TVM_NUM_THREADS=1 $(build_dir)/demo_dynamic $(build_dir)/bundle_c.so $(build_dir)/graph_c.json $(build_dir)/params_c.bin $(build_dir)/cat.bin

test_dynamic: $(build_dir)/test_dynamic $(build_dir)/test_bundle.so $(build_dir)/test_bundle_c.so $(build_dir)/test_data_c.bin $(build_dir)/test_output_c.bin $(build_dir)/test_data_cpp.bin $(build_dir)/test_output_cpp.bin
	$(QUIET)TVM_NUM_THREADS=1 $(build_dir)/test_dynamic $(build_dir)/test_bundle.so $(build_dir)/test_data_cpp.bin $(build_dir)/test_output_cpp.bin $(build_dir)/test_graph_cpp.json $(build_dir)/test_params_cpp.bin
	$(QUIET)TVM_NUM_THREADS=1 $(build_dir)/test_dynamic $(build_dir)/test_bundle_c.so $(build_dir)/test_data_c.bin $(build_dir)/test_output_c.bin $(build_dir)/test_graph_c.json $(build_dir)/test_params_c.bin

demo_static: $(build_dir)/demo_static $(build_dir)/cat.bin
	$(QUIET)TVM_NUM_THREADS=1 $(build_dir)/demo_static $(build_dir)/cat.bin

test_static: $(build_dir)/crt_config/crt_config.h $(build_dir)/test_static $(build_dir)/test_data_c.bin $(build_dir)/test_output_c.bin
	$(QUIET)TVM_NUM_THREADS=1 $(build_dir)/test_static $(build_dir)/test_data_c.bin $(build_dir)/test_output_c.bin $(build_dir)/test_graph_c.json $(build_dir)/test_params_c.bin

$(build_dir)/crt_config/crt_config.h:
	$(QUIET)mkdir --parents $(build_dir)/crt_config && cp ../../build/microtvm_template_projects/crt/crt_config/crt_config.h $(build_dir)/crt_config/crt_config.h

$(build_dir)/crt:
	$(QUIET)mkdir --parents $(build_dir)/crt && cp -r $(CRT_ROOT)/* $(build_dir)/crt

$(build_dir)/libcommon.a: $(build_dir)/crt $(build_dir)/crt_config/crt_config.h
	$(QUIET)cd $(build_dir)/crt && mkdir -p build && cd build && cmake -DCRT_CONFIG_PATH=$(abspath $(build_dir)/crt_config) -DCMAKE_C_FLAGS="${PKG_COMPILE_OPTS}" .. && make && cp libcommon.a $(abspath $(build_dir)/libcommon.a)

$(build_dir)/libgraph_executor.a: $(build_dir)/crt $(build_dir)/crt_config/crt_config.h
	$(QUIET)cd $(build_dir)/crt && mkdir -p build && cd build && cmake -DCRT_CONFIG_PATH=$(abspath $(build_dir)/crt_config) -DCMAKE_C_FLAGS="${PKG_COMPILE_OPTS}" .. && make && cp libgraph_executor.a $(abspath $(build_dir)/libgraph_executor.a)

$(build_dir)/libmemory.a: $(build_dir)/crt $(build_dir)/crt_config/crt_config.h
	$(QUIET)cd $(build_dir)/crt && mkdir -p build && cd build && cmake -DCRT_CONFIG_PATH=$(abspath $(build_dir)/crt_config) -DCMAKE_C_FLAGS="${PKG_COMPILE_OPTS}" .. && make && cp libmemory.a $(abspath $(build_dir)/libmemory.a)

$(build_dir)/demo_dynamic: demo.cc
	$(QUIET)mkdir -p $(@D)
	$(QUIET)g++ $(PKG_CXXFLAGS) -o $@ demo.cc $(BACKTRACE_LDFLAGS)

$(build_dir)/test_dynamic: test.cc ${build_dir}/test_graph_c.json ${build_dir}/test_params_c.bin $(BACKTRACE_OBJS)
	$(QUIET)mkdir -p $(@D)
	$(QUIET)g++ $(PKG_CXXFLAGS) -o $@ test.cc $(BACKTRACE_OBJS) $(BACKTRACE_LDFLAGS)

$(build_dir)/demo_static: demo_static.c ${build_dir}/bundle_static.o $(MODEL_OBJ) ${build_dir}/libmemory.a ${build_dir}/libgraph_executor.a ${build_dir}/libcommon.a ${build_dir}/graph_c.json.c ${build_dir}/params_c.bin.c $(BACKTRACE_OBJS)
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc $(PKG_CFLAGS) -o $@ $^ $(PKG_LDFLAGS) $(BACKTRACE_LDFLAGS) $(BACKTRACE_CFLAGS)

$(build_dir)/test_static: test_static.c ${build_dir}/bundle_static.o $(TEST_MODEL_OBJ) ${build_dir}/libmemory.a ${build_dir}/libgraph_executor.a ${build_dir}/libcommon.a $(BACKTRACE_OBJS)
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc $(PKG_CFLAGS) -o $@ $^ $(BACKTRACE_LDFLAGS)

$(build_dir)/backtrace.o: backtrace.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc -c $(PKG_CFLAGS) -o $@ $^ $(BACKTRACE_CFLAGS)

# Serialize our graph.json file.
$(build_dir)/graph_cpp.json.c: $(build_dir)/graph_cpp.json
	$(QUIET)xxd -i $^  > $@

$(build_dir)/graph_c.json.c: $(build_dir)/graph_c.json
	$(QUIET)xxd -i $^  > $@

# Serialize our params.bin file.
$(build_dir)/params_c.bin.c: $(build_dir)/params_c.bin
	$(QUIET)xxd -i $^  > $@

$(build_dir)/params_cpp.bin.c: $(build_dir)/params_cpp.bin
	$(QUIET)xxd -i $^  > $@

$(MODEL_OBJ) $(build_dir)/graph_c.json $(build_dir)/model_cpp.o $(build_dir)/graph_cpp.json $(build_dir)/params.bin $(build_dir)/cat.bin: build_model.py
	$(QUIET)python3 $< -o $(build_dir)
	$(QUIET)mkdir -p build/model_c
	$(QUIET)tar -C build/model_c -xvf build/model_c.tar

$(TEST_MODEL_OBJ) $(build_dir)/test_graph_c.json $(build_dir)/test_params_c.bin $(build_dir)/test_data_c.bin $(build_dir)/test_output_c.bin $(build_dir)/test_model_cpp.o $(build_dir)/test_graph_cpp.json $(build_dir)/test_params_cpp.bin $(build_dir)/test_data_cpp.bin $(build_dir)/test_output_cpp.bin: build_model.py
	$(QUIET)python3 $< -o $(build_dir) --test
	$(QUIET)mkdir -p build/test_model_c
	$(QUIET)tar -C build/test_model_c -xvf build/test_model_c.tar

# Build our bundle against the serialized bundle.c API, the runtime.cc API, and
# the serialized graph.json and params.bin
$(build_dir)/bundle.so: bundle.cc runtime.cc $(build_dir)/model_cpp.o
	$(QUIET)mkdir -p $(@D)
	$(QUIET)g++ -shared $(PKG_CXXFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/bundle_c.so: bundle.c $(MODEL_OBJ) ${build_dir}/libmemory.a ${build_dir}/libgraph_executor.a ${build_dir}/libcommon.a $(BACKTRACE_OBJS)
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc -shared $(PKG_CFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS) $(BACKTRACE_LDFLAGS) $(BACKTRACE_CFLAGS)

$(build_dir)/test_bundle.so: bundle.cc runtime.cc $(build_dir)/test_model_cpp.o
	$(QUIET)mkdir -p $(@D)
	$(QUIET)g++ -shared $(PKG_CXXFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/test_bundle_c.so: bundle.c $(TEST_MODEL_OBJ) ${build_dir}/libmemory.a ${build_dir}/libgraph_executor.a ${build_dir}/libcommon.a $(BACKTRACE_OBJS)
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc -shared $(PKG_CFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS) $(BACKTRACE_LDFLAGS) $(BACKTRACE_CFLAGS)

$(build_dir)/bundle_static.o: bundle_static.c
	$(QUIET)mkdir -p $(@D)
	$(QUIET)gcc -c $(PKG_CFLAGS) -o $@  $^ $(BACKTRACE_CFLAGS)

clean:
	$(QUIET)rm -rf $(build_dir)/bundle.so $(build_dir)/bundle_c.so $(build_dir)/test_bundle.so $(build_dir)/test_bundle_c.so $(build_dir)/crt

cleanall:
	$(QUIET)rm -rf $(build_dir)

# Don't define implicit rules; they tend to match on logical target names that aren't targets (i.e. bundle_static)
.SUFFIXES:

.DEFAULT: demo_static demo_dynamic

test: test_static test_dynamic
.PHONY: test
