ROOTDIR = $(CURDIR)

ifndef config
ifneq ("$(wildcard ./config.mk)","")
	config ?= config.mk
else
	config ?= make/config.mk
endif
endif

include $(config)

.PHONY: clean install installdev all test doc pylint cpplint lint verilog cython cython2 cython3 web runtime

ifndef DMLC_CORE_PATH
  DMLC_CORE_PATH = $(ROOTDIR)/dmlc-core
endif

ifndef DLPACK_PATH
  DLPACK_PATH = $(ROOTDIR)/dlpack
endif

UNAME_S := $(shell uname -s)

# The flags
LLVM_CFLAGS= -fno-rtti -DDMLC_ENABLE_RTTI=0 -DDMLC_USE_FOPEN64=0
LDFLAGS = -pthread -lm -ldl
INCLUDE_FLAGS = -Iinclude -I$(DLPACK_PATH)/include -I$(DMLC_CORE_PATH)/include -IHalideIR/src -Itopi/include
CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
FRAMEWORKS =
OBJCFLAGS = -fno-objc-arc
EMCC_FLAGS= -s RESERVED_FUNCTION_POINTERS=2 -s NO_EXIT_RUNTIME=1 -s MAIN_MODULE=1 -DDMLC_LOG_STACK_TRACE=0\
	 -std=c++11 -Oz $(INCLUDE_FLAGS)

# llvm configuration
ifdef LLVM_CONFIG
	LLVM_VERSION=$(shell $(LLVM_CONFIG) --version| cut -b 1,3)
	LLVM_INCLUDE=$(filter -I%, $(shell $(LLVM_CONFIG) --cxxflags))
	LDFLAGS += $(shell $(LLVM_CONFIG) --ldflags --libs --system-libs)
	LLVM_CFLAGS += $(LLVM_INCLUDE) -DTVM_LLVM_VERSION=$(LLVM_VERSION)
else
	LLVM_VERSION=00
endif

# The source code dependencies
LIB_HALIDEIR = HalideIR/lib/libHalideIR.a

CC_SRC = $(filter-out src/contrib/%.cc src/runtime/%.cc src/codgen/llvm/%.cc,\
             $(wildcard src/*/*.cc src/*/*/*.cc))
LLVM_SRC = $(wildcard src/codegen/llvm/*.cc src/codegen/llvm/*/*.cc)
METAL_SRC = $(wildcard src/runtime/metal/*.mm)
CUDA_SRC = $(wildcard src/runtime/cuda/*.cc)
ROCM_SRC = $(wildcard src/runtime/rocm/*.cc)
OPENCL_SRC = $(wildcard src/runtime/opencl/*.cc)
RPC_SRC = $(wildcard src/runtime/rpc/*.cc)
GRAPH_SRC = $(wildcard src/runtime/graph/*.cc)
RUNTIME_SRC = $(wildcard src/runtime/*.cc)

# Objectives
LLVM_BUILD = build/llvm${LLVM_VERSION}
LLVM_OBJ = $(patsubst src/%.cc, ${LLVM_BUILD}/%.o, $(LLVM_SRC))
METAL_OBJ = $(patsubst src/%.mm, build/%.o, $(METAL_SRC))
CUDA_OBJ = $(patsubst src/%.cc, build/%.o, $(CUDA_SRC))
ROCM_OBJ = $(patsubst src/%.cc, build/%.o, $(ROCM_SRC))
OPENCL_OBJ = $(patsubst src/%.cc, build/%.o, $(OPENCL_SRC))
RPC_OBJ = $(patsubst src/%.cc, build/%.o, $(RPC_SRC))
GRAPH_OBJ = $(patsubst src/%.cc, build/%.o, $(GRAPH_SRC))
CC_OBJ = $(patsubst src/%.cc, build/%.o, $(CC_SRC)) $(LLVM_OBJ)
RUNTIME_OBJ = $(patsubst src/%.cc, build/%.o, $(RUNTIME_SRC))
CONTRIB_OBJ =

# Deps
ALL_DEP = $(CC_OBJ) $(CONTRIB_OBJ) $(LIB_HALIDEIR)
RUNTIME_DEP = $(RUNTIME_OBJ)

# Dependency specific rules
ifdef CUDA_PATH
	NVCC=$(CUDA_PATH)/bin/nvcc
	CFLAGS += -I$(CUDA_PATH)/include
	LDFLAGS += -L$(CUDA_PATH)/lib64
endif

ifeq ($(USE_CUDA), 1)
	CFLAGS += -DTVM_CUDA_RUNTIME=1
	LDFLAGS += -lcuda -lcudart -lnvrtc
	RUNTIME_DEP += $(CUDA_OBJ)
else
	CFLAGS += -DTVM_CUDA_RUNTIME=0
endif

ifdef ROCM_PATH
	CFLAGS += -I$(ROCM_PATH)/include
	LDFLAGS += -L$(ROCM_PATH)/lib
endif

ifeq ($(USE_ROCM), 1)
	CFLAGS += -DTVM_ROCM_RUNTIME=1 -D__HIP_PLATFORM_HCC__=1
	LDFLAGS += -lhip_hcc
	RUNTIME_DEP += $(ROCM_OBJ)
else
	CFLAGS += -DTVM_ROCM_RUNTIME=0
endif

ifeq ($(USE_OPENCL), 1)
	CFLAGS += -DTVM_OPENCL_RUNTIME=1
	ifeq ($(UNAME_S), Darwin)
		FRAMEWORKS += -framework OpenCL
	else
		LDFLAGS += -lOpenCL
	endif
	RUNTIME_DEP += $(OPENCL_OBJ)
else
	CFLAGS += -DTVM_OPENCL_RUNTIME=0
endif

ifeq ($(USE_METAL), 1)
	CFLAGS += -DTVM_METAL_RUNTIME=1
	LDFLAGS += -lObjc
	RUNTIME_DEP += $(METAL_OBJ)
	FRAMEWORKS += -framework Metal -framework Foundation
else
	CFLAGS += -DTVM_METAL_RUNTIME=0
endif

ifeq ($(USE_RPC), 1)
	RUNTIME_DEP += $(RPC_OBJ)
endif

ifeq ($(USE_GRAPH_RUNTIME), 1)
	RUNTIME_DEP += $(GRAPH_OBJ)
endif

include make/contrib/cblas.mk
include make/contrib/nnpack.mk
include make/contrib/cudnn.mk

ifdef ADD_CFLAGS
	CFLAGS += $(ADD_CFLAGS)
endif

ifdef ADD_LDFLAGS
	LDFLAGS += $(ADD_LDFLAGS)
endif

ifeq ($(OS),Windows_NT)
	JVM_PKG_PROFILE := windows
	SHARED_LIBRARY_SUFFIX := dll
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S), Darwin)
		JVM_PKG_PROFILE := osx-x86_64
		SHARED_LIBRARY_SUFFIX := dylib
	else
		JVM_PKG_PROFILE := linux-x86_64
		SHARED_LIBRARY_SUFFIX := so
	endif
endif

JVM_TEST_ARGS := $(if $(JVM_TEST_ARGS),$(JVM_TEST_ARGS),-DskipTests -Dcheckstyle.skip=true)

ifeq ($(USE_CUDA), 1)
	JVM_PKG_PROFILE := $(JVM_PKG_PROFILE)-gpu
else ifeq ($(USE_OPENCL), 1)
	JVM_PKG_PROFILE := $(JVM_PKG_PROFILE)-gpu
else ifeq ($(USE_METAL), 1)
	JVM_PKG_PROFILE := $(JVM_PKG_PROFILE)-gpu
else
	JVM_PKG_PROFILE := $(JVM_PKG_PROFILE)-cpu
endif

BUILD_TARGETS ?= lib/libtvm.$(SHARED_LIBRARY_SUFFIX) lib/libtvm_runtime.$(SHARED_LIBRARY_SUFFIX)
all: ${BUILD_TARGETS}
runtime: lib/libtvm_runtime.$(SHARED_LIBRARY_SUFFIX)
web: lib/libtvm_web_runtime.js lib/libtvm_web_runtime.bc

include tests/cpp/unittest.mk

test: $(TEST)

include verilog/verilog.mk
verilog: $(VER_LIBS)

# Special rules for LLVM related modules.
${LLVM_BUILD}/codegen/llvm/%.o: src/codegen/llvm/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(LLVM_CFLAGS) -MM -MT ${LLVM_BUILD}/codegen/llvm/$*.o $< >${LLVM_BUILD}/codegen/llvm/$*.d
	$(CXX) -c $(CFLAGS) $(LLVM_CFLAGS) -c $< -o $@

build/runtime/metal/%.o: src/runtime/metal/%.mm
	@mkdir -p $(@D)
	$(CXX) $(OBJCFLAGS) $(CFLAGS) -MM -MT build/runtime/metal/$*.o $< >build/runtime/metal/$*.d
	$(CXX) $(OBJCFLAGS) -c $(CFLAGS) -c $< -o $@

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libtvm.dylib: $(ALL_DEP) $(RUNTIME_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(FRAMEWORKS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libtvm_runtime.dylib: $(RUNTIME_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(FRAMEWORKS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libtvm.so: $(ALL_DEP) $(RUNTIME_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(FRAMEWORKS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libtvm_runtime.so: $(RUNTIME_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(FRAMEWORKS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libtvm_web_runtime.bc: web/web_runtime.cc
	@mkdir -p build/web
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT lib/libtvm_web_runtime.bc $< >build/web/web_runtime.d
	emcc $(EMCC_FLAGS) -o $@ web/web_runtime.cc

lib/libtvm_web_runtime.js: lib/libtvm_web_runtime.bc
	@mkdir -p $(@D)
	emcc $(EMCC_FLAGS) -o $@ lib/libtvm_web_runtime.bc

$(LIB_HALIDEIR): LIBHALIDEIR

LIBHALIDEIR:
	+ cd HalideIR; make lib/libHalideIR.a ; cd $(ROOTDIR)

cpplint:
	python dmlc-core/scripts/lint.py topi cpp topi/include;
	python dmlc-core/scripts/lint.py tvm cpp include src verilog\
	 examples/extension/src examples/graph_executor/src

pylint:
	pylint python/tvm --rcfile=$(ROOTDIR)/tests/lint/pylintrc
	pylint topi/python/topi --rcfile=$(ROOTDIR)/tests/lint/pylintrc

jnilint:
	python dmlc-core/scripts/lint.py tvm4j-jni cpp jvm/native/src

lint: cpplint pylint jnilint

doc:
	doxygen docs/Doxyfile

install: lib/libtvm_runtime.$(SHARED_LIBRARY_SUFFIX)
	mkdir -p $(DESTDIR)$(PREFIX)/include/tvm/runtime
	cp -R include/tvm/runtime/. $(DESTDIR)$(PREFIX)/include/tvm/runtime
	cp lib/libtvm_runtime.$(SHARED_LIBRARY_SUFFIX) $(DESTDIR)$(PREFIX)/lib

installdev: lib/libtvm.$(SHARED_LIBRARY_SUFFIX) lib/libtvm_runtime.$(SHARED_LIBRARY_SUFFIX) lib/libtvm.a
	mkdir -p $(DESTDIR)$(PREFIX)/include
	cp -R include/tvm $(DESTDIR)$(PREFIX)/include
	cp lib/libtvm.$(SHARED_LIBRARY_SUFFIX) $(DESTDIR)$(PREFIX)/lib
	cp lib/libtvm_runtime.$(SHARED_LIBRARY_SUFFIX) $(DESTDIR)$(PREFIX)/lib
	cp lib/libtvm.a $(DESTDIR)$(PREFIX)/lib

# Cython build
cython:
	cd python; python setup.py build_ext --inplace

cython2:
	cd python; python2 setup.py build_ext --inplace

cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/tvm/*/*/*.so python/tvm/*/*/*.dylib python/tvm/*/*/*.cpp

jvmpkg:
	(cd $(ROOTDIR)/jvm; \
		mvn clean package -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/lib" $(JVM_TEST_ARGS))
jvminstall:
	(cd $(ROOTDIR)/jvm; \
		mvn install -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(CFLAGS)" -Dldflags="$(LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/lib" $(JVM_TEST_ARGS))

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o */*.d */*/*.d */*/*/*.d
	cd HalideIR; make clean; cd $(ROOTDIR)

-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
-include build/*/*/*/*.d
