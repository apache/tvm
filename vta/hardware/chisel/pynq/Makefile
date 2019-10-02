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

TOP = VTA
CONFIG = DefaultPynqConfig
VTA_TARGET = hw_emu
TEST_NAME = simple_load_store

root_dir = $(abspath ../../../)
script_dir = $(abspath .)/scripts
sim_dir = $(abspath .)/sim
out_dir = $(abspath .)/out
chisel = $(abspath ../)

num_threads = 12

verilog_dir = $(root_dir)/build/chisel
generated_verilog = $(verilog_dir)/$(TOP).$(CONFIG).v

testbench_file = $(sim_dir)/$(TEST_NAME)/tb.v

default: $(out_dir)/system.bit

$(out_dir)/system.bit: $(out_dir)/ip/component.xml
	vivado -mode batch -source $(script_dir)/vivado.tcl \
	-tclargs $(out_dir) $(num_threads) $(VTA_TARGET) $(testbench_file) \
	$(sim_dir) $(TEST_NAME)

$(out_dir)/ip/component.xml: $(generated_verilog)
	vivado -mode batch -source $(script_dir)/package_rtl.tcl \
	-tclargs $(verilog_dir) $(out_dir)

$(generated_verilog): | $(out_dir)
	make -C $(chisel) verilog

$(out_dir):
	mkdir -p $@

clean:
	-rm -rf $(out_dir) *.log *.jou *.hdf NA
	make -C $(chisel) cleanall
