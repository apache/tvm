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

# CMake Build rules for VTA
find_program(PYTHON NAMES python python3 python3.6)

if(MSVC)
  message(STATUS "VTA build is skipped in Windows..")
elseif(PYTHON)
  set(VTA_CONFIG ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/vta/config/vta_config.py)

  if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
    message(STATUS "Use VTA config " ${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
    set(VTA_CONFIG ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/vta/config/vta_config.py
      --use-cfg=${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
  endif()

  execute_process(COMMAND ${VTA_CONFIG} --target OUTPUT_VARIABLE VTA_TARGET OUTPUT_STRIP_TRAILING_WHITESPACE)

  message(STATUS "Build VTA runtime with target: " ${VTA_TARGET})

  execute_process(COMMAND ${VTA_CONFIG} --defs OUTPUT_VARIABLE __vta_defs)

  string(REGEX MATCHALL "(^| )-D[A-Za-z0-9_=.]*" VTA_DEFINITIONS "${__vta_defs}")

  file(GLOB VTA_RUNTIME_SRCS vta/src/*.cc)
  # Add sim driver sources
  if(${VTA_TARGET} STREQUAL "sim")
    file(GLOB __vta_target_srcs vta/src/sim/*.cc)
  endif()
  # Add tsim driver sources
  if(${VTA_TARGET} STREQUAL "tsim")
    file(GLOB __vta_target_srcs vta/src/tsim/*.cc)
    file(GLOB RUNTIME_DPI_SRCS vta/src/dpi/module.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_DPI_SRCS})
  endif()
  # Add pynq driver sources
  if(${VTA_TARGET} STREQUAL "pynq" OR ${VTA_TARGET} STREQUAL "ultra96")
    file(GLOB __vta_target_srcs vta/src/pynq/*.cc)
  endif()
  list(APPEND VTA_RUNTIME_SRCS ${__vta_target_srcs})

  add_library(vta SHARED ${VTA_RUNTIME_SRCS})

  target_include_directories(vta PUBLIC vta/include)

  foreach(__def ${VTA_DEFINITIONS})
    string(SUBSTRING ${__def} 3 -1 __strip_def)
    target_compile_definitions(vta PUBLIC ${__strip_def})
  endforeach()

  # Enable tsim macro
  if(${VTA_TARGET} STREQUAL "tsim")
    include_directories("vta/include")
    target_compile_definitions(vta PUBLIC USE_TSIM)
  endif()

  if(APPLE)
    set_target_properties(vta PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(APPLE)

  # PYNQ rules for Pynq v2.4
  if(${VTA_TARGET} STREQUAL "pynq" OR ${VTA_TARGET} STREQUAL "ultra96")
    find_library(__cma_lib NAMES cma PATH /usr/lib)
    target_link_libraries(vta ${__cma_lib})
  endif()

else()
  message(STATUS "Cannot found python in env, VTA build is skipped..")
endif()
