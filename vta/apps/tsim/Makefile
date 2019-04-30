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

export PYTHONPATH:=$(PWD)/python:$(PYTHONPATH)

BUILD_DIR = $(shell python python/tsim/config.py --get-build-name)

TVM_DIR = $(abspath ../../../)

TSIM_TARGET = verilog
TSIM_TOP_NAME = TestAccel
TSIM_BUILD_NAME = build

# optional
TSIM_TRACE_NAME = trace.vcd

default: cmake run

.PHONY: cmake

cmake: | $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

$(BUILD_DIR):
	mkdir -p $@

run:
	python3 tests/python/test_tsim.py | grep PASS

clean:
	-rm -rf $(BUILD_DIR)
