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

CONFIG = DefaultDe10Config
TOP = VTA
TOP_TEST = Test
BUILD_NAME = build
# Set USE_TRACE = 1 to generate a trace during simulation.
USE_TRACE = 0
# With USE_TRACE = 1, default trace format is VCD.
# Set USE_TRACE_FST = 1 to use the FST format.
# Note that although FST is around two orders of magnitude smaller than VCD
# it is also currently much slower to produce (verilator limitation). But if
# you are low on disk space it may be your only option.
USE_TRACE_FST = 0
# With USE_TRACE = 1, USE_TRACE_DETAILED = 1 will generate traces that also
# include non-interface internal signal names starting with an underscore.
# This will significantly increase the trace size and should only be used
# on a per need basis for difficult debug problems.
USE_TRACE_DETAILED = 0
USE_THREADS = 0
VTA_LIBNAME = libvta_hw
UNITTEST_NAME = all
CXX = g++
# A debug build with DEBUG = 1 is useful to trace the simulation with a
# debugger.
DEBUG = 0
# With DEBUG = 1, SANITIZE = 1 turns on address sanitizing to verify that
# the verilator build is sane. To be used if you know what you are doing.
SANITIZE = 0

CXX_MAJOR := $(shell $(CXX) -dumpversion | sed 's/\..*//')
CXX_HAS_ALIGN_NEW := $(shell [ $(CXX_MAJOR) -ge 7 ] && echo true)

config_test = $(TOP_TEST)$(CONFIG)


ifndef TVM_PATH
   TVM_PATH := $(abspath ../../../../)
endif

ifndef VTA_HW_PATH
   VTA_HW_PATH := $(abspath ../../)
endif

verilator_build_dir = $(VTA_HW_PATH)/$(BUILD_NAME)/verilator
chisel_build_dir = $(VTA_HW_PATH)/$(BUILD_NAME)/chisel

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
	cxx_flags = -O2 -Wall -fvisibility=hidden
else
	cxx_flags = -O0 -g -Wall
endif

cxx_flags += -std=c++11 -Wno-maybe-uninitialized
ifeq ($(CXX_HAS_ALIGN_NEW),true)
	cxx_flags += -faligned-new
endif
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
cxx_flags += -I$(VTA_HW_PATH)/include
cxx_flags += -I$(TVM_PATH)/include
cxx_flags += -I$(TVM_PATH)/3rdparty/dlpack/include

ld_flags = -fPIC -shared

ifeq ($(SANITIZE), 1)
	ifeq ($(DEBUG), 1)
		cxx_flags += -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address
		ld_flags += -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address
	endif
endif

cxx_objs = $(verilator_build_dir)/verilated.o $(verilator_build_dir)/verilated_dpi.o $(verilator_build_dir)/tsim_device.o

ifneq ($(USE_TRACE), 0)
	cxx_flags += -DVM_TRACE=1
	ifeq ($(USE_TRACE_FST), 1)
		cxx_flags += -DVM_TRACE_FST
		verilator_opt += --trace-fst
	else
		verilator_opt += --trace
	endif
	ifeq ($(USE_TRACE_DETAILED), 1)
		verilator_opt += --trace-underscore --trace-structs
	endif
	ifeq ($(USE_TRACE_FST), 1)
		cxx_flags += -DTSIM_TRACE_FILE=$(verilator_build_dir)/$(TOP_TEST).fst
		cxx_objs += $(verilator_build_dir)/verilated_fst_c.o
	else
		cxx_flags += -DTSIM_TRACE_FILE=$(verilator_build_dir)/$(TOP_TEST).vcd
		cxx_objs += $(verilator_build_dir)/verilated_vcd_c.o
	endif
else
	cxx_flags += -DVM_TRACE=0
endif

ifneq ($(USE_THREADS), 0)
	verilator_opt += --threads $(USE_THREADS)
	cxx_flags += -DVL_THREADED
	cxx_objs += $(verilator_build_dir)/verilated_threads.o
endif

VPATH = $(VERILATOR_INC_DIR):$(verilator_build_dir):$(VTA_HW_PATH)/hardware/dpi

# The following is to be consistent with cmake
ifeq ($(shell uname), Darwin)
	lib_path = $(VTA_HW_PATH)/$(BUILD_NAME)/$(VTA_LIBNAME).dylib
	cxx_flags += -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
else
	lib_path = $(VTA_HW_PATH)/$(BUILD_NAME)/$(VTA_LIBNAME).so
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
	-rm -rf $(VTA_HW_PATH)/$(BUILD_NAME)/chisel
	-rm -rf $(VTA_HW_PATH)/$(BUILD_NAME)/libvta_hw.so
	-rm -rf $(VTA_HW_PATH)/$(BUILD_NAME)/libvta_hw.dylib
	-rm -rf $(VTA_HW_PATH)/$(BUILD_NAME)/verilator
