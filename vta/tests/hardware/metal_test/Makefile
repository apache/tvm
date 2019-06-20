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

CC ?= g++
CFLAGS = -Wall -O3 -std=c++11 -I/usr/include
LDFLAGS = -L/usr/lib -L/opt/python3.6/lib/python3.6/site-packages/pynq/lib/
LIBS = -l:libcma.so -lstdc++ -pthread
INCLUDE_DIR = ../../../include
DRIVER_DIR = ../../../src/pynq
TESTLIB_DIR = ../common
VPATH = $(DRIVER_DIR):$(TESTLIB_DIR)
SOURCES = pynq_driver.cc test_lib.cc
OBJECTS = pynq_driver.o test_lib.o metal_test.o
EXECUTABLE = vta

# Include VTA config
VTA_CONFIG = python ../../../config/vta_config.py
CFLAGS += `${VTA_CONFIG} --cflags`
LDFLAGS += `${VTA_CONFIG} --ldflags`
VTA_TARGET := $(shell ${VTA_CONFIG} --target)

# Include bitstream
VTA_PROGRAM = python3 ../../../python/vta/program_bitstream.py
VTA_BIT = "vta.bit"

# Define flags
CFLAGS += -I $(INCLUDE_DIR) -DNO_SIM -DVTA_DEBUG=0

# All Target
all: vtainstall $(EXECUTABLE)

%.o: %.cc $(SOURCES)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

vtainstall:
	${VTA_PROGRAM} ${VTA_TARGET} ${VTA_BIT}
clean:
	rm -rf *.o $(EXECUTABLE)
