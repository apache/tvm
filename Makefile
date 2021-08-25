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

.SECONDEXPANSION:

# Remember the root directory, to be usable by submake invocation.
ROOTDIR = $(CURDIR)

# Specify an alternate output directory relative to ROOTDIR.  Defaults
# to "build".  Can also be a space-separated list of build
# directories, each with a different configuation.
TVM_BUILD_PATH ?= build
TVM_BUILD_PATH := $(abspath $(TVM_BUILD_PATH))

# Allow environment variables for 3rd-party libraries, default to
# packaged version.
DMLC_CORE_PATH ?= $(ROOTDIR)/3rdparty/dmlc-core
DLPACK_PATH ?= $(ROOTDIR)/3rdparty/dlpack
VTA_HW_PATH ?= $(ROOTDIR)/3rdparty/vta-hw




all: $(addsuffix /all,$(TVM_BUILD_PATH))

runtime: $(addsuffix /runtime,$(TVM_BUILD_PATH))
vta: $(addsuffix /vta,$(TVM_BUILD_PATH))
cpptest: $(addsuffix /cpptest,$(TVM_BUILD_PATH))
crttest: $(addsuffix /crttest,$(TVM_BUILD_PATH))

# If there is a config.cmake in the tvm directory, preferentially use
# it.  Otherwise, copy the default cmake/config.cmake.
ifeq ($(wildcard config.cmake),config.cmake)
%/config.cmake: | config.cmake
	@echo "No config.cmake found in $(TVM_BUILD_PATH), using config.cmake in root tvm directory"
	@mkdir -p $(@D)
else
# filter-out used to avoid circular dependency
%/config.cmake: | $$(filter-out %/config.cmake,$(ROOTDIR)/cmake/config.cmake)
	@echo "No config.cmake found in $(TVM_BUILD_PATH), using default config.cmake"
	@mkdir -p $(@D)
	@cp $| $@
endif


# Cannot use .PHONY with a pattern rule, using FORCE instead.  For
# now, force cmake to be re-run with each compile to mimic previous
# behavior.  This may be relaxed in the future with the
# CONFIGURE_DEPENDS option for GLOB (requres cmake >= 3.12).
FORCE:
%/CMakeCache.txt: %/config.cmake FORCE
	@cd $(@D) && cmake $(ROOTDIR)


# Since the pattern stem is already being used for the directory name,
# cannot also have it refer to the command passed to cmake.
# Therefore, explicitly listing out the delegated.
CMAKE_TARGETS = all runtime vta cpptest crttest

define GEN_CMAKE_RULE
%/$(CMAKE_TARGET): %/CMakeCache.txt FORCE
	@$$(MAKE) -C $$(@D) $(CMAKE_TARGET)
endef
$(foreach CMAKE_TARGET,$(CMAKE_TARGETS),$(eval $(GEN_CMAKE_RULE)))



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


mypy:
	tests/scripts/task_mypy.sh

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
