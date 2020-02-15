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

# Configure runtime
DEBUG = 0
MAX_NDIM = 6
MAX_ARGS = 10
GRAPH_RUNTIME_NODE_MAX_INPUTS = 300
GRAPH_RUNTIME_MAX_CONTEXTS = 1
GRAPH_RUNTIME_MAX_NODES = 400
GRAPH_RUNTIME_MAX_INPUT_NODES = 300
GRAPH_RUNTIME_MAX_NODE_ROW_PTR = 300
GRAPH_RUNTIME_MAX_OUTPUTS = 300

# Setup build environment
TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
PKG_CXXFLAGS = -std=c++14 -O0 -g -fPIC \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include
PKG_CFLAGS = -std=c99 -O0 -g -fPIC \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include \
    -DTVM_CRT_DEBUG=${DEBUG} \
    -DTVM_CRT_MAX_NDIM=${MAX_NDIM} \
    -DTVM_CRT_MAX_ARGS=${MAX_ARGS} \
    -DGRAPH_RUNTIME_NODE_MAX_INPUTS=${GRAPH_RUNTIME_NODE_MAX_INPUTS} \
    -DGRAPH_RUNTIME_MAX_CONTEXTS=${GRAPH_RUNTIME_MAX_CONTEXTS} \
    -DGRAPH_RUNTIME_MAX_NODES=${GRAPH_RUNTIME_MAX_NODES} \
    -DGRAPH_RUNTIME_MAX_INPUT_NODES=${GRAPH_RUNTIME_MAX_INPUT_NODES} \
    -DGRAPH_RUNTIME_MAX_NODE_ROW_PTR=${GRAPH_RUNTIME_MAX_NODE_ROW_PTR} \
    -DGRAPH_RUNTIME_MAX_OUTPUTS=${GRAPH_RUNTIME_MAX_OUTPUTS}

PKG_LDFLAGS = -pthread

build_dir := build

test: $(build_dir)/demo $(build_dir)/bundle.so
	$(build_dir)/demo $(build_dir)/bundle.so

$(build_dir)/demo: demo.cc
	@mkdir -p $(@D)
	g++ $(PKG_CXXFLAGS) -o $@  $^ -ldl

# Serialize our graph.json file.
$(build_dir)/graph.json.c: $(build_dir)/graph.json
	xxd -i $^  > $@

# Serialize our params.bin file.
$(build_dir)/params.bin.c: $(build_dir)/params.bin
	xxd -i $^  > $@

$(build_dir)/model.o $(build_dir)/graph.json $(build_dir)/params.bin $(build_dir)/bridge.c: build_model.py
	python3 $< -o $(build_dir)

# Build our bundle against the serialized bundle.cc API, the runtime.cc API, and
# the serialized graph.json and params.bin
$(build_dir)/bundle.so: bundle.c runtime.c $(build_dir)/model.o $(build_dir)/graph.json.o $(build_dir)/params.bin.o $(build_dir)/bridge.c
	@mkdir -p $(@D)
	gcc -shared $(PKG_CFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/graph.json.o: $(build_dir)/graph.json.c
	gcc -c ${PKG_CFLAGS} -o $@ $^

$(build_dir)/params.bin.o: $(build_dir)/params.bin.c
	gcc -c ${PKG_CFLAGS} -o $@ $^

clean:
	rm -rf $(build_dir)/bundle.so

cleanall:
	rm -rf $(build_dir)
