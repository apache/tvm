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

if(NOT USE_TFLITE STREQUAL "OFF")
  message(STATUS "Build with contrib.tflite")
  if (USE_TENSORFLOW_PATH STREQUAL "none")
    set(USE_TENSORFLOW_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)
  endif()

  tvm_file_glob(GLOB TFLITE_CONTRIB_SRC src/runtime/contrib/tflite/*.cc)
  list(APPEND RUNTIME_SRCS ${TFLITE_CONTRIB_SRC})
  include_directories(${USE_TENSORFLOW_PATH})

  # Additional EdgeTPU libs
  if (NOT USE_EDGETPU STREQUAL "OFF")
    message(STATUS "Build with contrib.edgetpu")
    tvm_file_glob(GLOB EDGETPU_CONTRIB_SRC src/runtime/contrib/edgetpu/*.cc)
    list(APPEND RUNTIME_SRCS ${EDGETPU_CONTRIB_SRC})
    include_directories(${USE_EDGETPU}/libedgetpu)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${USE_EDGETPU}/libedgetpu/direct/aarch64/libedgetpu.so.1)
  endif()

  if (USE_TFLITE STREQUAL "ON")
    set(USE_TFLITE ${USE_TENSORFLOW_PATH}/tensorflow/lite/tools/make/gen/*/lib)
  endif()
  find_library(TFLITE_CONTRIB_LIB libtensorflow-lite.a ${USE_TFLITE})
  file(GLOB_RECURSE TFLITE_DEPS "${USE_TFLITE}/*.a")

  list(APPEND TVM_RUNTIME_LINKER_LIBS ${TFLITE_CONTRIB_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${TFLITE_DEPS})

  if (NOT USE_FLATBUFFERS_PATH STREQUAL "none")
    include_directories(${USE_FLATBUFFERS_PATH}/include)
  endif()
endif()
