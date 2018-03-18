CC ?= g++
CFLAGS = -Wall -O3 -std=c++11 -I/usr/include
LDFLAGS = -L/usr/lib -L/home/xilinx/pynq/drivers
LIBS = -l:libsds_lib.so -l:libdma.so
INCLUDE_DIR = ../../../include
DRIVER_DIR = ../../../src/pynq
TESTLIB_DIR = ../common
VPATH = $(DRIVER_DIR):$(TESTLIB_DIR)
SOURCES = pynq_driver.cc test_lib.cc
OBJECTS = pynq_driver.o test_lib.o metal_test.o
EXECUTABLE = vta

# Include top-level config file
ifndef config
ifneq ("$(wildcard ../../../config.mk)", "")
	config = ../../../config.mk
else
	config = ../../../make/config.mk
endif
endif
include $(config)

# Define flags
CFLAGS += -I $(INCLUDE_DIR) -DNO_SIM -DDEBUG=0
CFLAGS += $(ADD_CFLAGS)

# All Target
all: $(EXECUTABLE)

%.o: %.cc $(SOURCES)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

clean:
	rm -rf *.o $(EXECUTABLE)
