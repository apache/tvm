# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

ROOTDIR = $(CURDIR)
# Specify an alternate output directory relative to ROOTDIR. Default build
OUTPUTDIR = $(if $(OUTDIR), $(OUTDIR), build)

.PHONY: clean all test doc pylint cpplint scalalint lint\
	 cython cython2 cython3 web runtime vta

ifndef DMLC_CORE_PATH
  DMLC_CORE_PATH = $(ROOTDIR)/3rdparty/dmlc-core
endif

ifndef DLPACK_PATH
  DLPACK_PATH = $(ROOTDIR)/3rdparty/dlpack
endif

ifndef VTA_HW_PATH
  VTA_HW_PATH = $(ROOTDIR)/3rdparty/vta-hw
endif

INCLUDE_FLAGS = -Iinclude -I$(DLPACK_PATH)/include -I$(DMLC_CORE_PATH)/include
PKG_CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
PKG_LDFLAGS =


all:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE)

runtime:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE) runtime

vta:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE) vta

cpptest:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE) cpptest

crttest:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE) crttest

# EMCC; Web related scripts
EMCC_FLAGS= -std=c++11\
	-Oz -s RESERVED_FUNCTION_POINTERS=2 -s MAIN_MODULE=1 -s NO_EXIT_RUNTIME=1\
	-s TOTAL_MEMORY=1073741824\
	-s EXTRA_EXPORTED_RUNTIME_METHODS="['addFunction','cwrap','getValue','setValue']"\
	-s USE_GLFW=3 -s USE_WEBGL2=1 -lglfw\
	$(INCLUDE_FLAGS)

web: $(OUTPUTDIR)/libtvm_web_runtime.js $(OUTPUTDIR)/libtvm_web_runtime.bc

$(OUTPUTDIR)/libtvm_web_runtime.bc: web/web_runtime.cc
	@mkdir -p $(OUTPUTDIR)/web
	@mkdir -p $(@D)
	emcc $(EMCC_FLAGS) -MM -MT $(OUTPUTDIR)/libtvm_web_runtime.bc $< >$(OUTPUTDIR)/web/web_runtime.d
	emcc $(EMCC_FLAGS) -o $@ web/web_runtime.cc

$(OUTPUTDIR)/libtvm_web_runtime.js: $(OUTPUTDIR)/libtvm_web_runtime.bc
	@mkdir -p $(@D)
	emcc $(EMCC_FLAGS) -o $@ $(OUTPUTDIR)/libtvm_web_runtime.bc

# Lint scripts
# NOTE: lint scripts that are executed in the CI should be in tests/lint. This allows docker/lint.sh
# to behave similarly to the CI.
cpplint:
	tests/lint/cpplint.sh

pylint:
	tests/lint/pylint.sh

jnilint:
	python3 3rdparty/dmlc-core/scripts/lint.py tvm4j-jni cpp jvm/native/src

scalalint:
	make -C $(VTA_HW_PATH)/hardware/chisel lint

lint: cpplint pylint jnilint

doc:
	doxygen docs/Doxyfile

javadoc:
	# build artifact is in jvm/core/target/site/apidocs
	cd jvm && mvn javadoc:javadoc -Dnotimestamp=true

# Cython build
cython:
	cd python; python3 setup.py build_ext --inplace

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
			-Dcurrent_libdir="$(ROOTDIR)/$(OUTPUTDIR)" $(JVM_TEST_ARGS))
jvminstall:
	(cd $(ROOTDIR)/jvm; \
		mvn install -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(PKG_CFLAGS)" -Dldflags="$(PKG_LDFLAGS)" \
			-Dcurrent_libdir="$(ROOTDIR)/$(OUTPUTDIR)" $(JVM_TEST_ARGS))
format:
	./tests/lint/git-clang-format.sh -i origin/main
	black .
	cd rust; which cargo && cargo fmt --all; cd ..


# clean rule
clean:
	@mkdir -p $(OUTPUTDIR) && cd $(OUTPUTDIR) && cmake .. && $(MAKE) clean
