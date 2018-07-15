CC ?= g++
CFLAGS = -Wall -O3 -std=c++11 -I/usr/include
LDFLAGS = -L/usr/lib -L/opt/python3.6/lib/python3.6/site-packages/pynq/lib/
LIBS = -l:libsds_lib.so -l:libdma.so -lstdc++
INCLUDE_DIR = ../../../include
DRIVER_DIR = ../../../src/pynq
TESTLIB_DIR = ../common
VPATH = $(DRIVER_DIR):$(TESTLIB_DIR)
SOURCES = pynq_driver.cc test_lib.cc
OBJECTS = pynq_driver.o test_lib.o metal_test.o
EXECUTABLE = vta

# Include VTA config
VTA_CONFIG = python ../../../make/vta_config.py
CFLAGS += `${VTA_CONFIG} --cflags`
LDFLAGS += `${VTA_CONFIG} --ldflags`
VTA_TARGET := $(shell ${VTA_CONFIG} --target)

# Define flags
CFLAGS += -I $(INCLUDE_DIR) -DNO_SIM -DVTA_DEBUG=0

# All Target
all: $(EXECUTABLE)

%.o: %.cc $(SOURCES)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

clean:
	rm -rf *.o $(EXECUTABLE)
