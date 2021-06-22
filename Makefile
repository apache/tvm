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


.PHONY: all \
        runtime vta cpptest crttest \
        lint pylint cpplint scalalint \
	doc \
	web webclean \
	cython cython3 cyclean \
        clean

# Remember the root directory, to be usable by submake invocation.
ROOTDIR = $(CURDIR)

# Specify an alternate output directory relative to ROOTDIR.  Defaults
# to "build"
TVM_BUILD_PATH ?= build
TVM_BUILD_PATH := $(abspath $(TVM_BUILD_PATH))

# Allow environment variables for 3rd-party libraries, default to
# packaged version.
DMLC_CORE_PATH ?= $(ROOTDIR)/3rdparty/dmlc-core
DLPACK_PATH ?= $(ROOTDIR)/3rdparty/dlpack
VTA_HW_PATH ?= $(ROOTDIR)/3rdparty/vta-hw




all: cmake_all


# Delegate to the cmake build system, with a few aliases for backwards
# compatibility.
$(TVM_BUILD_PATH)/config.cmake: | $(ROOTDIR)/cmake/config.cmake
	@echo "No config.cmake found in $(TVM_BUILD_PATH), using default config.cmake"
	@mkdir -p $(TVM_BUILD_PATH)
	@cp $| $@

$(TVM_BUILD_PATH)/CMakeCache.txt: $(TVM_BUILD_PATH)/config.cmake
	@cd $(TVM_BUILD_PATH) && cmake $(ROOTDIR)

# Cannot use .PHONY here as that disables the implicit rule.
FORCE:
cmake_%: $(TVM_BUILD_PATH)/CMakeCache.txt FORCE
	@$(MAKE) -C $(TVM_BUILD_PATH) $*

runtime: cmake_runtime
vta: cmake_vta
cpptest: cmake_cpptest
crttest: cmake_crttest


# Dev tools for formatting, linting, and documenting.  NOTE: lint
# scripts that are executed in the CI should be in tests/lint. This
# allows docker/lint.sh to behave similarly to the CI.
format:
	./tests/lint/git-clang-format.sh -i origin/main
	black .
	cd rust && which cargo && cargo fmt --all

lint: cpplint pylint jnilint

cpplint:
	tests/lint/cpplint.sh

pylint:
	tests/lint/pylint.sh

jnilint:
	python3 3rdparty/dmlc-core/scripts/lint.py tvm4j-jni cpp jvm/native/src

scalalint:
	make -C $(VTA_HW_PATH)/hardware/chisel lint


doc:
	doxygen docs/Doxyfile


# Cython build
cython cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/tvm/*/*/*.so python/tvm/*/*/*.dylib python/tvm/*/*/*.cpp



# EMCC; Web related scripts
web:
	$(MAKE) -C $(ROOTDIR)/web

webclean:
	$(MAKE) -C $(ROOTDIR)/web clean


# JVM build rules
INCLUDE_FLAGS = -Iinclude -I$(DLPACK_PATH)/include -I$(DMLC_CORE_PATH)/include
PKG_CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
PKG_LDFLAGS =

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

JVM_TEST_ARGS ?= -DskipTests -Dcheckstyle.skip=true

# Built java docs are in jvm/core/target/site/apidocs
javadoc:
	(cd $(ROOTDIR)/jvm; \
		mvn "javadoc:javadoc" -Dnotimestamp=true)

jvmpkg:
	(cd $(ROOTDIR)/jvm; \
		mvn clean package -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(PKG_CFLAGS)" -Dldflags="$(PKG_LDFLAGS)" \
			-Dcurrent_libdir="$(TVM_BUILD_PATH)" $(JVM_TEST_ARGS))

jvminstall:
	(cd $(ROOTDIR)/jvm; \
		mvn install -P$(JVM_PKG_PROFILE) -Dcxx="$(CXX)" \
			-Dcflags="$(PKG_CFLAGS)" -Dldflags="$(PKG_LDFLAGS)" \
			-Dcurrent_libdir="$(TVM_BUILD_PATH)" $(JVM_TEST_ARGS))

# Final cleanup rules, delegate to more specific rules.
clean: cmake_clean cyclean webclean
