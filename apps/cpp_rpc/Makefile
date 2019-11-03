# Makefile to compile RPC Server.
TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
TVM_RUNTIME_DIR?=

PKG_CFLAGS = -std=c++11 -O2 -fPIC -Wall\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include

PKG_LDFLAGS = -L$(TVM_RUNTIME_DIR) -ltvm_runtime -ldl -Wl,-R$(TVM_RUNTIME_DIR)

ifeq ($(USE_GLOG), 1)
        PKG_CFLAGS += -DDMLC_USE_GLOG=1
        PKG_LDFLAGS += -lglog
endif

.PHONY: clean all

all: tvm_rpc

# Build rule for all in one TVM package library
tvm_rpc: *.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@ $(filter %.cc %.o %.a, $^) $(PKG_LDFLAGS)

clean:
	-rm -f tvm_rpc