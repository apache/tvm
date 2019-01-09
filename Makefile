ROOTDIR = $(CURDIR)

.PHONY: clean all test doc pylint cpplint lint\
	 cython cython2 cython3 web runtime vta

ifndef DMLC_CORE_PATH
  DMLC_CORE_PATH = $(ROOTDIR)/3rdparty/dmlc-core
endif

ifndef DLPACK_PATH
  DLPACK_PATH = $(ROOTDIR)/3rdparty/dlpack
endif

INCLUDE_FLAGS = -Iinclude -I$(DLPACK_PATH)/include -I$(DMLC_CORE_PATH)/include
PKG_CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
PKG_LDFLAGS =


all:
	@mkdir -p build && cd build && cmake .. && $(MAKE)

runtime:
	@mkdir -p build && cd build && cmake .. && $(MAKE) runtime

vta:
	@mkdir -p build && cd build && cmake .. && $(MAKE) vta

cpptest:
	@mkdir -p build && cd build && cmake .. && $(MAKE) cpptest

# EMCC; Web related scripts
EMCC_FLAGS= -std=c++11 -DDMLC_LOG_STACK_TRACE=0\
	-Oz -s RESERVED_FUNCTION_POINTERS=2 -s MAIN_MODULE=1 -s NO_EXIT_RUNTIME=1\
	-s TOTAL_MEMORY=1073741824\
	-s EXTRA_EXPORTED_RUNTIME_METHODS="['addFunction','cwrap','getValue','setValue']"\
	-s USE_GLFW=3 -s USE_WEBGL2=1 -lglfw\
	$(INCLUDE_FLAGS)

web: build/libtvm_web_runtime.js build/libtvm_web_runtime.bc

build/libtvm_web_runtime.bc: web/web_runtime.cc
	@mkdir -p build/web
	@mkdir -p $(@D)
	emcc $(EMCC_FLAGS) -MM -MT build/libtvm_web_runtime.bc $< >build/web/web_runtime.d
	emcc $(EMCC_FLAGS) -o $@ web/web_runtime.cc

build/libtvm_web_runtime.js: build/libtvm_web_runtime.bc
	@mkdir -p $(@D)
	emcc $(EMCC_FLAGS) -o $@ build/libtvm_web_runtime.bc

# Lint scripts
cpplint:
	python3 3rdparty/dmlc-core/scripts/lint.py vta cpp vta/include vta/src
	python3 3rdparty/dmlc-core/scripts/lint.py topi cpp topi/include;
	python3 3rdparty/dmlc-core/scripts/lint.py nnvm cpp nnvm/include nnvm/src;
	python3 3rdparty/dmlc-core/scripts/lint.py tvm cpp include src verilog\
	 examples/extension/src examples/graph_executor/src

pylint:
	python3 -m pylint python/tvm --rcfile=$(ROOTDIR)/tests/lint/pylintrc
	python3 -m pylint topi/python/topi --rcfile=$(ROOTDIR)/tests/lint/pylintrc
	python3 -m pylint nnvm/python/nnvm --rcfile=$(ROOTDIR)/tests/lint/pylintrc
	python3 -m pylint vta/python/vta --rcfile=$(ROOTDIR)/tests/lint/pylintrc

jnilint:
	python3 3rdparty/dmlc-core/scripts/lint.py tvm4j-jni cpp jvm/native/src

lint: cpplint pylint jnilint

doc:
	doxygen docs/Doxyfile

javadoc:
	# build artifact is in jvm/core/target/site/apidocs
	cd jvm && mvn javadoc:javadoc

# Cython build
cython:
	cd python; python setup.py build_ext --inplace

cython2:
	cd python; python2 setup.py build_ext --inplace

cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/tvm/*/*/*.so python/tvm/*/*/*.dylib python/tvm/*/*/*.cpp

# JVM build rules
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

jvmpkg:
	(cd $(ROOTDIR)/jvm; \
		mvn clean package -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(PKG_CFLAGS)" -Dldflags="$(PKG_LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/build" $(JVM_TEST_ARGS))
jvminstall:
	(cd $(ROOTDIR)/jvm; \
		mvn install -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(PKG_CFLAGS)" -Dldflags="$(PKG_LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/build" $(JVM_TEST_ARGS))

# clean rule
clean:
	@mkdir -p build && cd build && cmake .. && $(MAKE) clean
