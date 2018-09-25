# Makefile Example to deploy TVM modules.
TVM_ROOT=$(shell cd ../..; pwd)
NNVM_PATH=nnvm
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -lpthread

.PHONY: clean all

all: lib/cpp_deploy_pack lib/cpp_deploy_normal

# Build rule for all in one TVM package library
lib/libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

# The code library built by TVM
lib/test_addone_sys.o: prepare_test_libs.py
	python prepare_test_libs.py

# Deploy using the all in one TVM package library
lib/cpp_deploy_pack: cpp_deploy.cc lib/test_addone_sys.o lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

# Deploy using pre-built libtvm_runtime.so
lib/cpp_deploy_normal: cpp_deploy.cc lib/test_addone_sys.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -ltvm_runtime
