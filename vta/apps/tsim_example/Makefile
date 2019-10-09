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

BUILD_NAME = build
build_dir = $(abspath .)/$(BUILD_NAME)

default: verilog driver
	python3 tests/python/verilog_accel.py

run_chisel: chisel driver
	python3 tests/python/chisel_accel.py

driver: | $(build_dir)
	cd $(build_dir) && cmake .. && make

$(build_dir):
	mkdir -p $@

verilog:
	make -C hardware/verilog

chisel:
	make -C hardware/chisel

clean:
	-rm -rf $(build_dir)
	make -C hardware/chisel clean
	make -C hardware/verilog clean
