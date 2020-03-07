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

ifeq (, $(shell which verilator))
 $(error "No Verilator in $(PATH), consider doing apt-get install verilator")
endif

# Change VERILATOR_INC_DIR if Verilator is installed on a different location
ifeq (, $(VERILATOR_INC_DIR))
  ifeq (, $(wildcard /usr/local/share/verilator/include/*))
    ifeq (, $(wildcard /usr/share/verilator/include/*))
      $(error "Verilator include directory is not set properly")
    else
      VERILATOR_INC_DIR := /usr/share/verilator/include
    endif
  else
      VERILATOR_INC_DIR := /usr/local/share/verilator/include
  endif
endif

CONFIG = DefaultPynqConfig
TOP = VTA
TOP_TEST = Test
BUILD_NAME = build
USE_TRACE = 0
USE_THREADS = $(shell nproc)
VTA_LIBNAME = libvta_hw
UNITTEST_NAME = all
CXX = g++
DEBUG = 0

config_test = $(TOP_TEST)$(CONFIG)
vta_dir = $(abspath ../../)
tvm_dir = $(abspath ../../../)
verilator_build_dir = $(vta_dir)/$(BUILD_NAME)/verilator
chisel_build_dir = $(vta_dir)/$(BUILD_NAME)/chisel

verilator_opt = --cc
verilator_opt += +define+RANDOMIZE_GARBAGE_ASSIGN
verilator_opt += +define+RANDOMIZE_REG_INIT
verilator_opt += +define+RANDOMIZE_MEM_INIT
verilator_opt += --x-assign unique
verilator_opt += --output-split 20000
verilator_opt += --output-split-cfuncs 20000
verilator_opt += --top-module ${TOP_TEST}
verilator_opt += -Mdir ${verilator_build_dir}
verilator_opt += -I$(chisel_build_dir)

ifeq ($(DEBUG), 0)
  cxx_flags = -O2 -Wall
else
  cxx_flags = -O0 -g -Wall
endif
cxx_flags += -fvisibility=hidden -std=c++11
cxx_flags += -DVL_TSIM_NAME=V$(TOP_TEST)
cxx_flags += -DVL_PRINTF=printf
cxx_flags += -DVL_USER_FINISH
cxx_flags += -DVM_COVERAGE=0
cxx_flags += -DVM_SC=0
cxx_flags += -Wno-sign-compare
cxx_flags += -include V$(TOP_TEST).h
cxx_flags += -I$(verilator_build_dir)
cxx_flags += -I$(VERILATOR_INC_DIR)
cxx_flags += -I$(VERILATOR_INC_DIR)/vltstd
cxx_flags += -I$(vta_dir)/include
cxx_flags += -I$(tvm_dir)/include
cxx_flags += -I$(tvm_dir)/3rdparty/dlpack/include

ld_flags = -fPIC -shared

cxx_objs = $(verilator_build_dir)/verilated.o $(verilator_build_dir)/verilated_dpi.o $(verilator_build_dir)/tsim_device.o

ifneq ($(USE_TRACE), 0)
  verilator_opt += --trace
  cxx_flags += -DVM_TRACE=1
  cxx_flags += -DTSIM_TRACE_FILE=$(verilator_build_dir)/$(TOP_TEST).vcd
  cxx_objs += $(verilator_build_dir)/verilated_vcd_c.o
else
  cxx_flags += -DVM_TRACE=0
endif

ifneq ($(USE_THREADS), 0)
  verilator_opt += --threads $(USE_THREADS)
  cxx_flags += -DVL_THREADED
  cxx_objs += $(verilator_build_dir)/verilated_threads.o
endif

VPATH = $(VERILATOR_INC_DIR):$(verilator_build_dir):$(vta_dir)/hardware/dpi

# The following is to be consistent with cmake
ifeq ($(shell uname), Darwin)
  lib_path = $(vta_dir)/$(BUILD_NAME)/$(VTA_LIBNAME).dylib
  cxx_flags += -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
else
  lib_path = $(vta_dir)/$(BUILD_NAME)/$(VTA_LIBNAME).so
endif

default: lint lib

lint:
	sbt scalastyle

lib: $(lib_path)

$(verilator_build_dir)/%.o: %.cpp
	$(CXX) -fPIC $(cxx_flags) -c $^ -o $@

$(verilator_build_dir)/tsim_device.o: tsim_device.cc
	$(CXX) -fPIC $(cxx_flags) -c $^ -o $@

$(lib_path): $(verilator_build_dir)/V$(TOP_TEST).cpp $(cxx_objs)
	for f in $(shell find $(verilator_build_dir)/*.cpp); do \
		$(CXX) -fPIC $(cxx_flags) -c $${f} -o $${f}.o ; \
	done
	$(CXX) $(ld_flags) $(cxx_flags) $(cxx_objs) $(patsubst %.cpp,%.cpp.o,$(shell find $(verilator_build_dir)/*.cpp)) -o $@

verilator: $(verilator_build_dir)/V$(TOP_TEST).cpp
$(verilator_build_dir)/V$(TOP_TEST).cpp: $(chisel_build_dir)/$(TOP_TEST).$(CONFIG).v
	verilator $(verilator_opt) $<

verilog: $(chisel_build_dir)/$(TOP).$(CONFIG).v
$(chisel_build_dir)/$(TOP).$(CONFIG).v:
	sbt 'runMain vta.$(CONFIG) --target-dir $(chisel_build_dir) --top-name $(TOP).$(CONFIG)'

verilog_test: $(chisel_build_dir)/$(TOP_TEST).$(CONFIG).v
$(chisel_build_dir)/$(TOP_TEST).$(CONFIG).v:
	sbt 'runMain vta.$(config_test) --target-dir $(chisel_build_dir) --top-name $(TOP_TEST).$(CONFIG)'

unittest:
	sbt 'test:runMain unittest.Launcher $(UNITTEST_NAME)'

clean:
	-rm -rf target project/target project/project test_run_dir

cleanall:
	-rm -rf $(vta_dir)/$(BUILD_NAME)/chisel
	-rm -rf $(vta_dir)/$(BUILD_NAME)/libvta_hw.so
	-rm -rf $(vta_dir)/$(BUILD_NAME)/verilator
