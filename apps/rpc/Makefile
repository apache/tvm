# Makefile to compile RPC Server.
TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
BUILD=${TVM_ROOT}/build

PKG_CFLAGS = -std=c++11 -O2 -fPIC -g -Wall\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include

PKG_LDFLAGS = -L${BUILD} -ldl -lpthread -ltvm -Wl,-R${BUILD}

.PHONY: clean all

all: tvm_rpc

# Build rule for all in one TVM package library
tvm_rpc: rpc_server.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@ $(filter %.cc %.o %.a, $^) $(PKG_LDFLAGS)
