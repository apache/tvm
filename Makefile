ROOTDIR = $(CURDIR)

ifndef config
ifneq ("$(wildcard ./config.mk)","")
	config ?= config.mk
else
	config ?= make/config.mk
endif
endif

include $(config)

# specify tensor path
.PHONY: clean all test doc pylint cpplint lint verilog cython cython2 cython3

all: lib/libtvm.so lib/libtvm_runtime.so lib/libtvm.a

LIB_HALIDE_IR = HalideIR/lib/libHalideIR.a

SRC = $(wildcard src/*.cc src/*/*.cc src/*/*/*.cc)
METAL_SRC = $(wildcard src/runtime/metal/*.mm)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))
METAL_OBJ = $(patsubst src/%.mm, build/%.o, $(METAL_SRC))
ALL_DEP = $(ALL_OBJ) $(LIB_HALIDE_IR)

RUNTIME_SRC = $(wildcard src/runtime/*.cc src/runtime/*/*.cc)
RUNTIME_DEP = $(patsubst src/%.cc, build/%.o, $(RUNTIME_SRC))

ALL_DEP = $(ALL_OBJ) $(LIB_HALIDE_IR)

export LDFLAGS = -pthread -lm
export CFLAGS =  -std=c++11 -Wall -O2 -fno-rtti\
	 -Iinclude -Idlpack/include -Idmlc-core/include -IHalideIR/src  -fPIC -DDMLC_ENABLE_RTTI=0
export OBJCFLAGS= -fobjc-arc

ifdef CUDA_PATH
	NVCC=$(CUDA_PATH)/bin/nvcc
	CFLAGS += -I$(CUDA_PATH)/include
	LDFLAGS += -L$(CUDA_PATH)/lib64
endif

ifeq ($(ENABLE_CUDA), 1)
	CFLAGS += -DTVM_CUDA_RUNTIME=1
	LDFLAGS += -lcuda -lcudart -lnvrtc
else
	CFLAGS += -DTVM_CUDA_RUNTIME=0
endif

FRAMEWORKS=

UNAME_S := $(shell uname -s)

ifeq ($(ENABLE_OPENCL), 1)
	CFLAGS += -DTVM_OPENCL_RUNTIME=1
	ifeq ($(UNAME_S), Darwin)
		FRAMEWORKS += -framework OpenCL
	else
		LDFLAGS += -lOpenCL
	endif
else
	CFLAGS += -DTVM_OPENCL_RUNTIME=0
endif

ifeq ($(ENABLE_METAL), 1)
	CFLAGS += -DTVM_METAL_RUNTIME=1
	LDFLAGS += -lObjc
	ALL_DEP += $(METAL_OBJ)
	RUNTIME_DEP += $(METAL_OBJ)
	FRAMEWORKS += -framework Metal -framework Foundation
else
	CFLAGS += -DTVM_METAL_RUNTIME=0
endif

# llvm configuration
LLVM_CONFIG=llvm-config

ifeq ($(ENABLE_LLVM), 1)
	LLVM_VERSION=$(shell $(LLVM_CONFIG) --version| cut -b 1,3)
	LLVM_INCLUDE=$(filter -I%, $(shell $(LLVM_CONFIG) --cxxflags))
	LDFLAGS += $(shell $(LLVM_CONFIG) --ldflags --libs --system-libs)
	CFLAGS += $(LLVM_INCLUDE) -DTVM_LLVM_VERSION=$(LLVM_VERSION)
endif

ifdef ADD_CFLAGS
	CFLAGS += $(ADD_CFLAGS)
endif

ifdef ADD_LDFLAGS
	LDFLAGS += $(ADD_LDFLAGS)
endif

include tests/cpp/unittest.mk

test: $(TEST)

include verilog/verilog.mk
verilog: $(VER_LIBS)

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

build/%.o: src/%.mm
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(OBJCFLAGS) -c $(CFLAGS) -c $< -o $@

lib/libtvm.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(FRAMEWORKS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libtvm_runtime.so: $(RUNTIME_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(FRAMEWORKS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libtvm.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

$(LIB_HALIDE_IR): LIBHALIDEIR

LIBHALIDEIR:
	+ cd HalideIR; make lib/libHalideIR.a ; cd $(ROOTDIR)

cpplint:
	python dmlc-core/scripts/lint.py tvm cpp include src verilog

pylint:
	pylint python/tvm --rcfile=$(ROOTDIR)/tests/lint/pylintrc

lint: cpplint pylint

doc:
	doxygen docs/Doxyfile

# Cython build
cython:
	cd python; python setup.py build_ext --inplace

cython2:
	cd python; python2 setup.py build_ext --inplace

cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/tvm/*/*/*.so python/tvm/*/*/*.cpp

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o */*.d */*/*.d */*/*/*.d

-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
