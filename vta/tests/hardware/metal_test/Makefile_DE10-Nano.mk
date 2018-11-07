CC ?= g++
CFLAGS = -Wall -g -O0 -std=c++11 -I/usr/include
LDFLAGS = -L/usr/lib
LIBS = -lstdc++
INCLUDE_DIR = -I../../../include
DRIVER_DIR = ../../../src/de10-nano
TESTLIB_DIR = ../common
VPATH = $(DRIVER_DIR):$(TESTLIB_DIR)
SOURCES = de10-nano_driver.cc cma_api.cc test_lib.cc
OBJECTS = de10-nano_driver.o cma_api.o test_lib.o metal_test.o
EXECUTABLE = vta_metal_test

# Include VTA config
VTA_CONFIG = python ../../../config/vta_config.py
CFLAGS += `${VTA_CONFIG} --cflags`
LDFLAGS += `${VTA_CONFIG} --ldflags`
VTA_TARGET := $(shell ${VTA_CONFIG} --target)

# Define flags
CFLAGS += $(INCLUDE_DIR) -DNO_SIM -DVTA_DEBUG=0 -DVTA_TARGET_DE10_NANO

# All Target
all: $(EXECUTABLE)

%.o: %.cc $(SOURCES)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

clean:
	rm -rf *.o $(EXECUTABLE)
