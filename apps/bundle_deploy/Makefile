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
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
PKG_CXXFLAGS = -Wall -std=c++14 -O2 -fPIC \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include
PKG_CFLAGS = -Wall -std=c99 -O2 -fPIC \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include

PKG_LDFLAGS = -pthread

build_dir := build

demo_dynamic: $(build_dir)/demo_dynamic $(build_dir)/bundle.so $(build_dir)/bundle_c.so $(build_dir)/cat.bin
	TVM_NUM_THREADS=1 $(build_dir)/demo_dynamic $(build_dir)/bundle.so $(build_dir)/cat.bin
	TVM_NUM_THREADS=1 $(build_dir)/demo_dynamic $(build_dir)/bundle_c.so $(build_dir)/cat.bin

test_dynamic: $(build_dir)/test_dynamic $(build_dir)/test_bundle.so $(build_dir)/test_bundle_c.so $(build_dir)/test_data.bin $(build_dir)/test_output.bin
	TVM_NUM_THREADS=1 $(build_dir)/test_dynamic $(build_dir)/test_bundle.so $(build_dir)/test_data.bin $(build_dir)/test_output.bin $(build_dir)/test_graph.json $(build_dir)/test_params.bin
	TVM_NUM_THREADS=1 $(build_dir)/test_dynamic $(build_dir)/test_bundle_c.so $(build_dir)/test_data.bin $(build_dir)/test_output.bin $(build_dir)/test_graph.json $(build_dir)/test_params.bin

demo_static: $(build_dir)/demo_static $(build_dir)/cat.bin
	TVM_NUM_THREADS=1 $(build_dir)/demo_static $(build_dir)/cat.bin

test_static: $(build_dir)/test_static $(build_dir)/test_data.bin $(build_dir)/test_output.bin
	TVM_NUM_THREADS=1 $(build_dir)/test_static $(build_dir)/test_data.bin $(build_dir)/test_output.bin $(build_dir)/test_graph.json $(build_dir)/test_params.bin

$(build_dir)/demo_dynamic: demo.cc ${build_dir}/graph.json.c ${build_dir}/params.bin.c
	@mkdir -p $(@D)
	g++ $(PKG_CXXFLAGS) -o $@  demo.cc -ldl

$(build_dir)/test_dynamic: test.cc ${build_dir}/test_graph.json ${build_dir}/test_params.bin
	@mkdir -p $(@D)
	g++ $(PKG_CXXFLAGS) -o $@  test.cc -ldl

$(build_dir)/demo_static: demo_static.c ${build_dir}/bundle_static.o ${build_dir}/model.o ${build_dir}/graph.json.c ${build_dir}/params.bin.c
	@mkdir -p $(@D)
	gcc $(PKG_CFLAGS) -o $@ demo_static.c ${build_dir}/bundle_static.o ${build_dir}/model.o -lm

$(build_dir)/test_static: test_static.c ${build_dir}/bundle_static.o ${build_dir}/test_model.o
	@mkdir -p $(@D)
	gcc $(PKG_CFLAGS) -o $@ $^

# Serialize our graph.json file.
$(build_dir)/graph.json.c: $(build_dir)/graph.json
	xxd -i $^  > $@

# Serialize our params.bin file.
$(build_dir)/params.bin.c: $(build_dir)/params.bin
	xxd -i $^  > $@

$(build_dir)/model.o $(build_dir)/graph.json $(build_dir)/params.bin $(build_dir)/cat.bin: build_model.py
	python3 $< -o $(build_dir)

$(build_dir)/test_model.o $(build_dir)/test_graph.json $(build_dir)/test_params.bin $(build_dir)/test_data.bin $(build_dir)/test_output.bin: build_model.py
	python3 $< -o $(build_dir) --test

# Build our bundle against the serialized bundle.c API, the runtime.cc API, and
# the serialized graph.json and params.bin
$(build_dir)/bundle.so: bundle.cc runtime.cc $(build_dir)/model.o
	@mkdir -p $(@D)
	g++ -shared $(PKG_CXXFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/bundle_c.so: bundle.c runtime.c $(build_dir)/model.o
	@mkdir -p $(@D)
	gcc -shared $(PKG_CFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/test_bundle.so: bundle.cc runtime.cc $(build_dir)/test_model.o
	@mkdir -p $(@D)
	g++ -shared $(PKG_CXXFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/test_bundle_c.so: bundle.c runtime.c $(build_dir)/test_model.o
	@mkdir -p $(@D)
	gcc -shared $(PKG_CFLAGS) -fvisibility=hidden -o $@  $^ $(PKG_LDFLAGS)

$(build_dir)/bundle_static.o: bundle_static.c
	@mkdir -p $(@D)
	gcc -c $(PKG_CFLAGS) -o $@  $^

clean:
	rm -rf $(build_dir)/bundle.so $(build_dir)/bundle_c.so $(build_dir)/test_bundle.so $(build_dir)/test_bundle_c.so

cleanall:
	rm -rf $(build_dir)
