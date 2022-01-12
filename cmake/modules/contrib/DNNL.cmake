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

macro(find_dnnl)
  # 1. Try to find via dnnl-config.cmake
  find_package(dnnl CONFIG)

  if (NOT dnnl_FOUND)
    # 2. Try to find dnnl like a lib + headers distribution
    find_library(EXTERN_LIBRARY_DNNL dnnl NO_CACHE)
    if (EXTERN_LIBRARY_DNNL)
      get_filename_component(DNNL_LIB_DIR ${EXTERN_LIBRARY_DNNL} DIRECTORY)
      get_filename_component(DNNL_HDR_DIR ${DNNL_LIB_DIR} DIRECTORY)
      string(APPEND DNNL_HDR_DIR "/include")

      find_file(DNNL_CONFIG_HDR dnnl_config.h PATHS ${DNNL_HDR_DIR} NO_CACHEEEE)
      if (DNNL_CONFIG_HDR)
        file(READ ${DNNL_CONFIG_HDR} DNNL_CONFIG)
        string(REGEX MATCH "DNNL_CPU_RUNTIME DNNL_RUNTIME_(OMP|SEQ|TBB)" DNNL_CPU_RUNTIME "${DNNL_CONFIG}")
        string(REGEX MATCH "(OMP|SEQ|TBB)" DNNL_CPU_RUNTIME "${DNNL_CPU_RUNTIME}")

        if (DNNL_CPU_RUNTIME)
          add_library(DNNL::dnnl SHARED IMPORTED)
          set_target_properties(DNNL::dnnl PROPERTIES
                  INTERFACE_INCLUDE_DIRECTORIES "${DNNL_HDR_DIR}"
                  IMPORTED_LOCATION "${EXTERN_LIBRARY_DNNL}"
                  )

          set(dnnl_FOUND TRUE)
          set(dnnl_DIR "${DNNL_LIB_DIR}")
        endif()
      endif()

      # because find_file put this value to cache
      unset(EXTERN_LIBRARY_DNNL CACHE)
      unset(DNNL_CONFIG_HDR CACHE)
    endif()
  endif()

  if (NOT dnnl_FOUND)
    message(FATAL_ERROR
            "Cannot detect DNNL package. Please make sure that you have it properly installed "
            "and corresponding variables are set (CMAKE_PREFIX_PATH or CMAKE_LIBRARY_PATH).")
  endif()
endmacro(find_dnnl)


if (USE_DNNL_CODEGEN STREQUAL "ON" )
  find_dnnl()

  if (DNNL_CPU_RUNTIME STREQUAL "OMP" AND NOT USE_OPENMP)
    message(WARNING
            "DNNL and TVM are using different threading runtimes. Mixing of thread "
            "pools may lead to significant performance penalty. Suggestion is to "
            "switch TVM to use OpenMP (cmake flag: -DUSE_OPENMP=ON).")
  endif()
endif()

if((USE_DNNL_CODEGEN STREQUAL "ON") OR (USE_DNNL_CODEGEN STREQUAL "JSON"))
  tvm_file_glob(GLOB DNNL_RELAY_CONTRIB_SRC src/relay/backend/contrib/dnnl/*.cc)
  tvm_file_glob(GLOB DNNL_CONTRIB_SRC src/runtime/contrib/dnnl/*.cc)

  list(APPEND COMPILER_SRCS ${DNNL_RELAY_CONTRIB_SRC})
  list(APPEND RUNTIME_SRCS ${DNNL_CONTRIB_SRC})
  list(APPEND TVM_RUNTIME_LINKER_LIBS DNNL::dnnl)
  # WA. Have to use system include path while TVM doesn't use targets to describe dependencies
  include_directories(SYSTEM $<TARGET_PROPERTY:DNNL::dnnl,INTERFACE_INCLUDE_DIRECTORIES>)
  add_definitions(-DUSE_JSON_RUNTIME=1)

  message(STATUS "Build with DNNL JSON runtime: ${dnnl_DIR} (${DNNL_CPU_RUNTIME})"  )
elseif(USE_DNNL_CODEGEN STREQUAL "C_SRC")
  tvm_file_glob(GLOB DNNL_RELAY_CONTRIB_SRC src/relay/backend/contrib/dnnl/*.cc)
  list(APPEND COMPILER_SRCS ${DNNL_RELAY_CONTRIB_SRC})

  find_library(EXTERN_LIBRARY_DNNL dnnl)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_DNNL})
  tvm_file_glob(GLOB DNNL_CONTRIB_SRC src/runtime/contrib/dnnl/dnnl.cc)
  list(APPEND RUNTIME_SRCS ${DNNL_CONTRIB_SRC})
  message(STATUS "Build with DNNL C source module: " ${EXTERN_LIBRARY_DNNL})
endif()

