CC ?= g++
CFLAGS = -Wall -O3 -std=c++11 -I/usr/include
LDFLAGS = -L/usr/lib -L/home/xilinx/pynq/drivers
LIBS = -l:libsds_lib.so -l:libdma.so
SRC_DIR = ../../src
INCLUDE_DIR = ../../include
DRIVER_DIR = $(SRC_DIR)/driver/pynq
TESTLIB_DIR = $(SRC_DIR)/test
VPATH = $(DRIVER_DIR):$(TESTLIB_DIR)
SOURCES = vta_pynq_driver.c vta_test_lib.cc
OBJECTS = vta_pynq_driver.o vta_test_lib.o driver_test.o
EXECUTABLE = vta

# VTA Parameters
#  Log of input width in bits
LOG_INP_WIDTH = 3
#  Log of weight width in bits
LOG_WGT_WIDTH = 3
#  Log of accum width in bits
LOG_ACC_WIDTH = 5
#  Log of output width in bits
LOG_OUT_WIDTH = $(LOG_INP_WIDTH)
#  Log of tensor batch size (A in (A,B)x(B,C) matrix multiplication)
LOG_BATCH = 0
#  Log of tensor inner block size (B in (A,B)x(B,C) matrix multiplication)
LOG_IN_BLOCK = 4
#  Log of tensor outer block size (C in (A,B)x(B,C) matrix multiplication)
LOG_OUT_BLOCK = 4
#  Log of uop buffer size in Bytes
LOG_UOP_BUFF_SIZE = 15
#  Log of inp buffer size in Bytes
LOG_INP_BUFF_SIZE = 15
#  Log of wgt buffer size in Bytes
LOG_WGT_BUFF_SIZE = 15
#  Log of acc buffer size in Bytes
LOG_ACC_BUFF_SIZE = 17
#  Log of out buffer size in Bytes
LOG_OUT_BUFF_SIZE = $(shell echo "$$(( $(LOG_ACC_BUFF_SIZE)+$(LOG_OUT_WIDTH)-$(LOG_ACC_WIDTH) ))" )

# Define flags
CFLAGS += -I $(INCLUDE_DIR) -DNO_SIM \
	-DDEBUG=0 -DLOG_WGT_WIDTH=$(LOG_WGT_WIDTH) -DLOG_INP_WIDTH=$(LOG_INP_WIDTH) \
	-DLOG_ACC_WIDTH=$(LOG_ACC_WIDTH) -DLOG_OUT_WIDTH=$(LOG_OUT_WIDTH) \
	-DLOG_BATCH=$(LOG_BATCH) -DLOG_BLOCK_IN=$(LOG_IN_BLOCK) -DLOG_BLOCK_OUT=$(LOG_OUT_BLOCK) \
	-DLOG_UOP_BUFF_SIZE=$(LOG_UOP_BUFF_SIZE) -DLOG_INP_BUFF_SIZE=$(LOG_INP_BUFF_SIZE) \
	-DLOG_WGT_BUFF_SIZE=$(LOG_WGT_BUFF_SIZE) -DLOG_ACC_BUFF_SIZE=$(LOG_ACC_BUFF_SIZE) \
	-DLOG_OUT_BUFF_SIZE=$(LOG_OUT_BUFF_SIZE)

# All Target
all: $(EXECUTABLE)

%.o: %.cc $(SOURCES)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

clean:
	rm -rf *.o $(EXECUTABLE)
