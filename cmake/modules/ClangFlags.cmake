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

# If we are running clang >= 10.0 then enable more checking. Some of these warnings may not exist
# in older versions of clang so we limit the use of older clang for these checks.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version)
  string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION ${clang_full_version})
  message(STATUS "CLANG_VERSION ${CLANG_VERSION}")
  # cmake 3.2 does not support VERSION_GREATER_EQUAL
  set(CLANG_MINIMUM_VERSION 10.0)
  if ((CLANG_VERSION VERSION_GREATER ${CLANG_MINIMUM_VERSION})
      OR
      (CLANG_VERSION VERSION_GREATER ${CLANG_MINIMUM_VERSION}))
    message(STATUS "Setting enhanced clang warning flags")

    set(warning_opts
      # These warnings are only enabled when clang's -Weverything flag is enabled
      # but there is no harm in turning them off for all cases.
      -Wno-c++98-compat
      -Wno-c++98-compat-extra-semi
      -Wno-c++98-compat-pedantic
      -Wno-padded
      -Wno-extra-semi
      -Wno-extra-semi-stmt
      -Wno-unused-parameter
      -Wno-sign-conversion
      -Wno-weak-vtables
      -Wno-deprecated-copy-dtor
      -Wno-global-constructors
      -Wno-double-promotion
      -Wno-float-equal
      -Wno-missing-prototypes
      -Wno-implicit-int-float-conversion
      -Wno-implicit-float-conversion
      -Wno-implicit-int-conversion
      -Wno-float-conversion
      -Wno-shorten-64-to-32
      -Wno-covered-switch-default
      -Wno-unused-exception-parameter
      -Wno-return-std-move
      -Wno-over-aligned
      -Wno-undef
      -Wno-inconsistent-missing-destructor-override
      -Wno-unreachable-code
      -Wno-deprecated-copy
      -Wno-implicit-fallthrough
      -Wno-unreachable-code-return
      -Wno-non-virtual-dtor
      # Here we have non-standard warnings that clang has available and are useful
      # so enable them if we are using clang.
      -Wreserved-id-macro
      -Wused-but-marked-unused
      -Wdocumentation-unknown-command
      -Wcast-qual
      -Wzero-as-null-pointer-constant
      # These warnings should be enabled one at a time and fixed.
      # To enable one of these warnings remove the `no-` after -W so
      # -Wno-documentation -> -Wdocumentation
      -Wno-documentation
      -Wno-shadow-uncaptured-local
      -Wno-shadow-field-in-constructor
      -Wno-shadow
      -Wno-shadow-field
      -Wno-exit-time-destructors
      -Wno-switch-enum
      -Wno-old-style-cast
      -Wno-gnu-anonymous-struct
      -Wno-nested-anon-types
    )
  target_compile_options(tvm_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>: ${warning_opts}>)
  target_compile_options(tvm_runtime_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>: ${warning_opts}>)


  endif ()
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
