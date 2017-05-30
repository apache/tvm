# Minimum Makefile for the extension package
TVM_ROOT=$(shell cd ../..; pwd)
PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}/dmlc-core/include\
	-I${TVM_ROOT}/dlpack/include\
	-I${TVM_ROOT}/HalideIR/src

PKG_LDFLAGS =-L${TVM_ROOT}/lib

lib/libtvm_ext.so: src/tvm_ext.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS) -ltvm
