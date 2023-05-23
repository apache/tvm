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

if(NOT USE_TF_TVMDSOOP STREQUAL "OFF")
  find_package(Python3 COMPONENTS Interpreter)

  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS_STR
    RESULT_VARIABLE TF_STATUS)
  if (NOT ${TF_STATUS} EQUAL 0)
    message(FATAL_ERROR "Fail to get TensorFlow compile flags")
  endif()

  if(NOT USE_CUDA STREQUAL "OFF")
    add_definitions(-DTF_TVMDSOOP_ENABLE_GPU)
  endif()

  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS_STR
    RESULT_VARIABLE TF_STATUS)
  if (NOT ${TF_STATUS} EQUAL 0)
    message(FATAL_ERROR "Fail to get TensorFlow link flags")
  endif()

  string(REGEX REPLACE "\n" " " TF_FLAGS "${TF_COMPILE_FLAGS} ${TF_LINK_FLAGS}")
  separate_arguments(TF_COMPILE_FLAGS UNIX_COMMAND ${TF_COMPILE_FLAGS_STR})
  separate_arguments(TF_LINK_FLAGS UNIX_COMMAND ${TF_LINK_FLAGS_STR})


  set(OP_LIBRARY_NAME tvm_dso_op)
  tvm_file_glob(GLOB_RECURSE TFTVM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/tf_op/*.cc)
  add_library(${OP_LIBRARY_NAME} SHARED ${TFTVM_SRCS})
  set(TFTVM_LINK_FLAGS  -ltvm -L${CMAKE_CURRENT_BINARY_DIR})

  if (NOT BUILD_TVMDSOOP_ONLY STREQUAL "ON")
      add_dependencies(${OP_LIBRARY_NAME} tvm)
  endif()

  target_compile_options(${OP_LIBRARY_NAME} PUBLIC ${TFTVM_COMPILE_FLAGS} ${TF_COMPILE_FLAGS})
  target_link_libraries(${OP_LIBRARY_NAME} PUBLIC ${TFTVM_LINK_FLAGS} ${TF_LINK_FLAGS})

endif()
