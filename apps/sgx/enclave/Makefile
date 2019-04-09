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

MODEL ?= resnet
NUM_THREADS ?= 4
BATCH_SIZE ?= 64
TRAINING ?= true
DEBUG ?= false

build_dir := ../build

ifeq ($(DEBUG), false)
	debug := release
	xargo_args := --release
else
	debug := debug
endif

target=target/x86_64-unknown-linux-sgx/$(debug)/libmodel-enclave.a

$(target): $(build_dir)/libmodel.a **/* $(TVM_DIR)/rust/patched.txt
	RUST_TARGET_PATH=$(shell pwd) \
		RUST_TARGET_DIR=$(shell pwd)/target \
		RUSTFLAGS="-Z force-unstable-if-unmarked" \
		TVM_NUM_THREADS=$(NUM_THREADS) \
		BUILD_DIR=../build \
		xargo build --target x86_64-unknown-linux-sgx $(xargo_args) -q

$(TVM_DIR)/rust/patched.txt: $(shell pwd)/sgx-deps.diff
	echo $(TVM_DIR)
	cd $(TVM_DIR) && git apply $<
	touch $@

$(build_dir)/libmodel.a: $(build_dir)/model.o
	$(AR) cr $@ $^

$(build_dir)/model.o: $(build_dir)/model.bc
	$(CC) -c $< -o $@ -fPIC -O3
	objcopy --globalize-symbol __tvm_module_startup $@

$(build_dir)/model.bc: src/build_model.py
	python3 $< -o $(build_dir)

clean:
	xargo clean
