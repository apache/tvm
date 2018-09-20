# Makefile to compile RPC Server.
#g++ -fPIC -g -Wall rpc_server.cc -ltvm -lpthread -ldl -L../../build/  -I../../include -I../../dlpack/include -Wl,-R../../build/
#g++ -std=c++11 -O2 -fPIC -g -Wall -I/media/siju/DataDisk/work/rpc/tvm/include -I/media/siju/DataDisk/work/rpc/tvm/dmlc-core/include -I/media/siju/DataDisk/work/rpc/tvm/dlpack/include tvm_rpc rpc_server.cc -L/media/siju/DataDisk/work/rpc/tvm/build -ldl -lpthread -ltvm -Wl,-R/media/siju/DataDisk/work/rpc/tvm/build

#TVM_ROOT=$(shell cd ../..; pwd)
TVM_ROOT=../..
DMLC_CORE=${TVM_ROOT}/dmlc-core
BUILD=${TVM_ROOT}/build

PKG_CFLAGS = -std=c++11 -O2 -fPIC -g -Wall\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/dlpack/include

PKG_LDFLAGS = -L${BUILD} -ldl -lpthread -ltvm -Wl,-R${BUILD}

.PHONY: clean all

all: tvm_rpc

# Build rule for all in one TVM package library
tvm_rpc: rpc_server.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@ $(filter %.cc %.o %.a, $^) $(PKG_LDFLAGS)
