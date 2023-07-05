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

cmake_minimum_required(VERSION 3.20.0)
set(BOARD mps3_an547)
set(EMU_PLATFORM armfvp)

find_package(Zephyr)
project(my_zephyr_app)

if (NOT DEFINED CMSIS_PATH)
    set(CMSIS_PATH /opt/arm/ethosu/cmsis/)
endif()

set(TVMC_COMMAND python3 -m tvm.driver.tvmc)
set(TVMC_ARGS
    --target="cmsis-nn -mcpu=cortex-m55, c" # CMSIS-NN and C targets
    --runtime=crt # C Runtime
    --executor=aot # Ahead-of-Time Executor
    --executor-aot-unpacked-api=1 # Direct internal calls to operators
    --executor-aot-interface-api=c # Expose C interface to the model
    --pass-config=tir.disable_vectorize=1 # Disable vectorizer for C output
    --output-format=mlf # Output Model Library Format tarball
)

set(TVM_RUNTIME
    ${CMAKE_CURRENT_BINARY_DIR}/runtime/src/runtime/crt/common/crt_backend_api.c
    ${CMAKE_CURRENT_BINARY_DIR}/runtime/src/runtime/crt/memory/stack_allocator.c
)
set(CODEGEN_OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/codegen/host/src/default_lib0.c
    ${CMAKE_CURRENT_BINARY_DIR}/codegen/host/src/default_lib1.c
    ${CMAKE_CURRENT_BINARY_DIR}/codegen/host/src/default_lib2.c
)
set(DATA_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/inputs.c
    ${CMAKE_CURRENT_BINARY_DIR}/outputs.c
    ${CMAKE_CURRENT_BINARY_DIR}/labels.c
)
set(CMSIS_SOURCES
    ${CMSIS_PATH}/CMSIS-NN/Source/SoftmaxFunctions/arm_softmax_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c
    ${CMSIS_PATH}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
    ${CMSIS_PATH}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c
)

add_custom_command(
    OUTPUT ${TVM_RUNTIME}
    OUTPUT ${CODEGEN_OUTPUT}
    COMMAND ${TVMC_COMMAND} compile ${TVMC_ARGS} ${CMAKE_CURRENT_SOURCE_DIR}/model/cnn_s_quantized.tflite
    COMMAND tar xf ${CMAKE_CURRENT_BINARY_DIR}/module.tar
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_command(
    OUTPUT ${DATA_FILES}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/model/convert_input.py ${CMAKE_CURRENT_SOURCE_DIR}/model/input.txt ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/model/convert_labels.py ${CMAKE_CURRENT_SOURCE_DIR}/model/labels.txt ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

target_sources(app PRIVATE
    src/main.c
    ${TVM_RUNTIME}
    ${CODEGEN_OUTPUT}
    ${DATA_FILES}
    ${CMSIS_SOURCES}
)
target_include_directories(app
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/crt_config
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/runtime/include ${CMAKE_CURRENT_BINARY_DIR}/codegen/host/include
    PUBLIC ${CMSIS_PATH}/CMSIS-NN/Include/ ${CMSIS_PATH}/CMSIS/DSP/Include
)
