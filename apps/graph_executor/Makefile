# Minimum Makefile for the extension package
TVM_ROOT=$(shell cd ../..; pwd)
NNVM_PATH=nnvm
DMLC_CORE=${TVM_ROOT}/dmlc-core

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/dlpack/include\
	-I${TVM_ROOT}/HalideIR/src

PKG_LDFLAGS =
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	PKG_LDFLAGS += -undefined dynamic_lookup
	WHOLE_ARCH= -all_load
	NO_WHOLE_ARCH= -noall_load
else
	WHOLE_ARCH= --whole-archive
	NO_WHOLE_ARCH= --no-whole-archive
endif

NNVM_CONTRIB_SRC = $(wildcard src/*.cc)
NNVM_CONTRIB_OBJ = $(patsubst src/%.cc, build/%.o, $(NNVM_CONTRIB_SRC))

include $(DMLC_CORE)/make/dmlc.mk

ALL_DEP = $(NNVM_CONTRIB_OBJ)

PKG_CFLAGS += -I${NNVM_PATH}/include
ALL_DEP += ${DMLC_CORE}/libdmlc.a ${NNVM_PATH}/lib/libnnvm.a

.PHONY: clean all

all: lib/libtvm_graph_exec.so

nnvm:
	git clone https://github.com/dmlc/nnvm --recursive

nnvm/lib/libnnvm.a: | nnvm
	+	cd nnvm; make ; cd -

$(DMLC_CORE)/libdmlc.a:
	+	cd $(DMLC_CORE); make libdmlc.a; cd $(TVM_ROOT)

build/%.o: src/%.cc | nnvm
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(PKG_CFLAGS) -c $< -o $@

lib/libtvm_graph_exec.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -shared -o $@ $(filter %.o, $^) $(PKG_LDFLAGS) \
	-Wl,${WHOLE_ARCH} $(filter %.a, $^) -Wl,${NO_WHOLE_ARCH} $(PKG_LDFLAGS)

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o */*.d */*/*.d */*/*/*.d

-include build/*.d
-include build/*/*.d
