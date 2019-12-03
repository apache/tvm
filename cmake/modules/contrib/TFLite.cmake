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
  message("tensorflow path: ${USE_TENSORFLOW_PATH}")
  if (USE_TENSORFLOW_PATH STREQUAL "none") 
    set(USE_TENSORFLOW_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)
  endif()
  message("tensorflow path: ${USE_TENSORFLOW_PATH}")

  file(GLOB TFLITE_CONTRIB_SRC src/runtime/contrib/tflite/*.cc)
  list(APPEND RUNTIME_SRCS ${TFLITE_CONTRIB_SRC})
  include_directories(${USE_TENSORFLOW_PATH})

  if (USE_TFLITE STREQUAL "ON")
    set(USE_TFLITE ${USE_TENSORFLOW_PATH}/tensorflow/lite/tools/make/gen/*/lib)
  endif()
  find_library(TFLITE_CONTRIB_LIB libtensorflow-lite.a ${USE_TFLITE})
  message("tflite lib path: ${TFLITE_CONTRIB_LIB}")

  list(APPEND TVM_RUNTIME_LINKER_LIBS ${TFLITE_CONTRIB_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS rt dl flatbuffers)
endif()
