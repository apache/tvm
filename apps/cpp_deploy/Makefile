# Minimum Makefile for the extension package
TVM_ROOT=$(shell cd ../..; pwd)
NNVM_PATH=nnvm
DMLC_CORE=${TVM_ROOT}/dmlc-core

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/dlpack/include\

PKG_LDFLAGS = -L${TVM_ROOT}/lib -ltvm_runtime


lib/cpp_deploy: cpp_deploy.cc lib/test_addone_sys.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

lib/test_addone_sys.o: prepare_test_libs.py
	python prepare_test_libs.py
