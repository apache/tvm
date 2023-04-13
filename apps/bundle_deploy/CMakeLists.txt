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

cmake_minimum_required(VERSION 3.18)
project(bundle_deploy C CXX)

set(TVM_ROOT ${CMAKE_SOURCE_DIR}/../../)
set(DMLC_CORE ${TVM_ROOT}/3rdparty/dmlc-core)

include_directories(
    ${TVM_ROOT}/include
    ${DMLC_CORE}/include
    ${TVM_ROOT}/3rdparty/dlpack/include
    ${CMAKE_CURRENT_SOURCE_DIR}/crt_config
)
add_definitions("-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>")

# find system python3 executable
find_package(Python 3 REQUIRED)

set(MODEL_OBJ model_c/devc.o model_c/lib0.o model_c/lib1.o)
set(TEST_MODEL_OBJ test_model_c/devc.o test_model_c/lib0.o test_model_c/lib1.o)

add_custom_command(OUTPUT
        model_cpp.o graph_c.json graph_cpp.json params_c.bin params_cpp.bin cat.bin
        ${MODEL_OBJ}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/build_model.py -o ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/model_c
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR}/model_c
            ${CMAKE_COMMAND} -E tar xvf ${CMAKE_CURRENT_BINARY_DIR}/model_c.tar
)

add_custom_command(OUTPUT
        test_graph_c.json test_params_c.bin test_data_c.bin test_output_c.bin test_model_cpp.o test_graph_cpp.json test_params_cpp.bin test_data_cpp.bin test_output_cpp.bin
        ${TEST_MODEL_OBJ}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/build_model.py -o ${CMAKE_CURRENT_BINARY_DIR} --test
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/test_model_c
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR}/test_model_c
            ${CMAKE_COMMAND} -E tar xvf ${CMAKE_CURRENT_BINARY_DIR}/test_model_c.tar
)

add_library(bundle SHARED bundle.cc runtime.cc model_cpp.o)
target_compile_features(bundle PRIVATE cxx_std_17)
# ignore compile warnings
if(MSVC)
    target_compile_definitions(bundle PRIVATE "TVM_EXPORTS")
    target_compile_options(bundle PRIVATE "/wd4068")  # unknown annotation "GCC"
    target_compile_options(bundle PRIVATE "/wd4244")  # narrow casting
endif()

add_library(test_bundle SHARED bundle.cc runtime.cc test_model_cpp.o)
target_compile_features(test_bundle PRIVATE cxx_std_17)
# ignore compile warnings
if(MSVC)
    target_compile_definitions(test_bundle PRIVATE "TVM_EXPORTS")
    target_compile_options(test_bundle PRIVATE "/wd4068")  # unknown annotation "GCC"
    target_compile_options(test_bundle PRIVATE "/wd4244")  # narrow casting
endif()

add_executable(demo_dynamic demo.cc)
target_compile_features(demo_dynamic PRIVATE cxx_std_17)
if(MSVC)
    target_compile_definitions(demo_dynamic PRIVATE "_CRT_SECURE_NO_WARNINGS")
endif()

add_executable(test_dynamic test.cc)
target_compile_features(test_dynamic PRIVATE cxx_std_17)
if(MSVC)
    target_compile_definitions(test_dynamic PRIVATE "_CRT_SECURE_NO_WARNINGS")
endif()
