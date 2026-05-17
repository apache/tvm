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

if(USE_XNNPACK STREQUAL "OFF")
  return()
endif()

if(IS_DIRECTORY "${USE_XNNPACK}")
  set(XNNPACK_ROOT "${USE_XNNPACK}")
  set(XNNPACK_FIND_ARGS HINTS "${XNNPACK_ROOT}" PATH_SUFFIXES include lib lib64 NO_DEFAULT_PATH)
elseif(USE_XNNPACK STREQUAL "ON")
  set(XNNPACK_FIND_ARGS)
else()
  message(FATAL_ERROR "Invalid option: USE_XNNPACK=${USE_XNNPACK}")
endif()

find_path(XNNPACK_INCLUDE_DIR xnnpack.h ${XNNPACK_FIND_ARGS})
find_library(XNNPACK_LIBRARY NAMES XNNPACK xnnpack ${XNNPACK_FIND_ARGS})
find_library(XNNPACK_MICROKERNELS_LIBRARY NAMES xnnpack-microkernels-prod
             ${XNNPACK_FIND_ARGS})
find_library(PTHREADPOOL_LIBRARY NAMES pthreadpool ${XNNPACK_FIND_ARGS})
find_library(CPUINFO_LIBRARY NAMES cpuinfo ${XNNPACK_FIND_ARGS})
find_library(KLEIDIAI_LIBRARY NAMES kleidiai ${XNNPACK_FIND_ARGS})

if(NOT XNNPACK_INCLUDE_DIR OR NOT XNNPACK_LIBRARY)
  message(FATAL_ERROR "USE_XNNPACK is enabled, but xnnpack.h or the XNNPACK library was not found")
endif()

message(STATUS "Build with XNNPACK support: ${XNNPACK_LIBRARY}")

include_directories(SYSTEM ${XNNPACK_INCLUDE_DIR})
add_definitions(-DTVM_USE_XNNPACK=1)
add_definitions(-DUSE_JSON_RUNTIME=1)

include(CheckCXXSourceCompiles)
set(_XNNPACK_PREV_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(_XNNPACK_PREV_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
set(CMAKE_REQUIRED_INCLUDES "${XNNPACK_INCLUDE_DIR}")
set(CMAKE_REQUIRED_LIBRARIES ${XNNPACK_LIBRARY})
foreach(_lib ${XNNPACK_MICROKERNELS_LIBRARY} ${PTHREADPOOL_LIBRARY} ${CPUINFO_LIBRARY}
             ${KLEIDIAI_LIBRARY})
  if(_lib)
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${_lib})
  endif()
endforeach()

foreach(_feature
    INITIALIZE
    CREATE_SUBGRAPH
    RUNTIME_V2
    RUNTIME_V4
    RUNTIME_V3
    DEFINE_TENSOR_VALUE
    DEFINE_UNARY
    DEFINE_BINARY
    DEFINE_CONVOLUTION_2D
    DEFINE_MAX_POOLING_2D
    DEFINE_AVERAGE_POOLING_2D
    WEIGHTS_CACHE
    WORKSPACE
    PROFILING
    BASIC_PROFILING_FLAG
    HINT_FP16_INFERENCE_FLAG
    FORCE_FP16_INFERENCE_FLAG
    FP32_STATIC_WEIGHTS_FLAG
    FP32_STATIC_BIASES_FLAG
    DATATYPE_FP16
    DATATYPE_QINT8
    DATATYPE_QUINT8
    DATATYPE_QINT32
    DATATYPE_QCINT8
    DATATYPE_QCINT32
    DATATYPE_QDINT8
    DATATYPE_QDUINT8
    DATATYPE_QPINT8
    EXTRA_QUANTIZATION_PARAMS
    DEFINE_CONVERT
    DEFINE_QUANTIZED_TENSOR_VALUE
    DEFINE_DYNAMICALLY_QUANTIZED_TENSOR_VALUE
    DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE
    DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE_V2
    VALIDATE_QUANTIZED_TENSOR
    VALIDATE_CHANNELWISE_QUANTIZED_TENSOR
    FULLY_CONNECTED
    DEPTHWISE_CONVOLUTION_2D
    UNARY_GELU
    UNARY_APPROXGELU
    DEFINE_SOFTMAX
    DYNAMIC_RANGE_QD8_OPS
    DYNAMIC_RANGE_FULLY_CONNECTED_SUBGRAPH
    DYNAMIC_RANGE_CONV2D_SUBGRAPH
    TRANSPOSE_WEIGHTS_FLAG
    STATIC_RESHAPE
    COPY
    RUNTIME_RESHAPE
    RESHAPE_EXTERNAL_VALUE
    SETUP_RUNTIME_V2
    GET_EXTERNAL_VALUE_SHAPE
    DYNAMIC_BATCH_RUNTIME
    DONT_SPIN_WORKERS_FLAG
    TRANSIENT_INDIRECTION_BUFFER_FLAG
    PTHREADPOOL_CREATE
    FP16_FLAGS
    QS8_DATATYPES
    QS8_SUBGRAPH_OPS
    DYNAMIC_QUANT_DATATYPES
    DYNAMIC_RANGE_SUBGRAPH_OPS)
  unset(TVM_XNNPACK_HAS_${_feature} CACHE)
endforeach()

check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_initialize(nullptr);
    return 0;
  }" TVM_XNNPACK_HAS_INITIALIZE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    xnn_subgraph_t subgraph = nullptr;
    (void)xnn_create_subgraph(0, 0, &subgraph);
    return 0;
  }" TVM_XNNPACK_HAS_CREATE_SUBGRAPH)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    xnn_runtime_t runtime = nullptr;
    (void)xnn_create_runtime_v2(nullptr, nullptr, 0, &runtime);
    return 0;
  }" TVM_XNNPACK_HAS_RUNTIME_V2)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    uint32_t id = 0;
    const size_t dims[1] = {1};
    (void)xnn_define_tensor_value(nullptr, xnn_datatype_fp32, 1, dims, nullptr,
                                  XNN_INVALID_VALUE_ID, 0, &id);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_TENSOR_VALUE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_unary(nullptr, xnn_unary_clamp, nullptr, 0, 1, 0);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_UNARY)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_binary(nullptr, xnn_binary_add, nullptr, 0, 1, 2, 0);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_BINARY)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_convolution_2d(nullptr, 0, 0, 0, 0, 3, 3, 1, 1, 1, 1, 1, 1, 1,
                                   -1.0f, 1.0f, 0, 1, 2, 3, 0);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_CONVOLUTION_2D)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_max_pooling_2d(nullptr, 0, 0, 0, 0, 2, 2, 1, 1, 1, 1,
                                   -1.0f, 1.0f, 0, 1, 0);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_MAX_POOLING_2D)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_average_pooling_2d(nullptr, 0, 0, 0, 0, 2, 2, 1, 1,
                                       -1.0f, 1.0f, 0, 1, 0);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_AVERAGE_POOLING_2D)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    xnn_runtime_t runtime = nullptr;
    (void)xnn_create_runtime_v4(nullptr, nullptr, nullptr, nullptr, 0, &runtime);
    return 0;
  }" TVM_XNNPACK_HAS_RUNTIME_V4)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    xnn_runtime_t runtime = nullptr;
    (void)xnn_create_runtime_v3(nullptr, nullptr, nullptr, 0, &runtime);
    return 0;
  }" TVM_XNNPACK_HAS_RUNTIME_V3)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    xnn_weights_cache_t cache = nullptr;
    (void)xnn_create_weights_cache(&cache);
    (void)xnn_finalize_weights_cache(cache, xnn_weights_cache_finalization_kind_soft);
    (void)xnn_delete_weights_cache(cache);
    return 0;
  }" TVM_XNNPACK_HAS_WEIGHTS_CACHE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    xnn_workspace_t workspace = nullptr;
    (void)xnn_create_workspace(&workspace);
    (void)xnn_release_workspace(workspace);
    return 0;
  }" TVM_XNNPACK_HAS_WORKSPACE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    size_t size = 0;
    (void)xnn_get_runtime_profiling_info(nullptr, xnn_profile_info_num_operators, 0, nullptr, &size);
    return 0;
  }" TVM_XNNPACK_HAS_PROFILING)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_FLAG_BASIC_PROFILING == 0; }" TVM_XNNPACK_HAS_BASIC_PROFILING_FLAG)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_FLAG_HINT_FP16_INFERENCE == 0; }"
  TVM_XNNPACK_HAS_HINT_FP16_INFERENCE_FLAG)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_FLAG_FORCE_FP16_INFERENCE == 0; }"
  TVM_XNNPACK_HAS_FORCE_FP16_INFERENCE_FLAG)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_FLAG_FP32_STATIC_WEIGHTS == 0; }"
  TVM_XNNPACK_HAS_FP32_STATIC_WEIGHTS_FLAG)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_FLAG_FP32_STATIC_BIASES == 0; }"
  TVM_XNNPACK_HAS_FP32_STATIC_BIASES_FLAG)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_fp16 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_FP16)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_qint8 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_QINT8)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_quint8 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_QUINT8)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_qint32 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_QINT32)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_qcint8 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_QCINT8)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_qcint32 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_QCINT32)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_qdint8 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_QDINT8)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_qduint8 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_QDUINT8)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_datatype_qpint8 == xnn_datatype_invalid; }" TVM_XNNPACK_HAS_DATATYPE_QPINT8)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_EXTRA_QUANTIZATION_PARAMS == 0; }" TVM_XNNPACK_HAS_EXTRA_QUANTIZATION_PARAMS)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_convert(nullptr, 0, 1, 0);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_CONVERT)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    uint32_t id = 0;
    const size_t dims[1] = {1};
    (void)xnn_define_quantized_tensor_value(nullptr, xnn_datatype_qint8, 0, 1.0f, 1,
                                            dims, nullptr, XNN_INVALID_VALUE_ID, 0, &id);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_QUANTIZED_TENSOR_VALUE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    uint32_t id = 0;
    const size_t dims[1] = {1};
    (void)xnn_define_dynamically_quantized_tensor_value(nullptr, xnn_datatype_qdint8, 1, 1, dims,
                                                        XNN_INVALID_VALUE_ID, 0, &id);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_DYNAMICALLY_QUANTIZED_TENSOR_VALUE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    uint32_t id = 0;
    const size_t dims[1] = {1};
    const float scale[1] = {1.0f};
    (void)xnn_define_channelwise_quantized_tensor_value(nullptr, xnn_datatype_qcint8, scale, 1,
                                                        0, dims, nullptr,
                                                        XNN_INVALID_VALUE_ID, 0, &id);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    uint32_t id = 0;
    const size_t dims[1] = {1};
    const float scale[1] = {1.0f};
    (void)xnn_define_channelwise_quantized_tensor_value_v2(nullptr, xnn_datatype_qcint8, 0, scale,
                                                           1, 0, dims, nullptr,
                                                           XNN_INVALID_VALUE_ID, 0, &id);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE_V2)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    const size_t dims[1] = {1};
    (void)xnn_validate_quantized_tensor(xnn_datatype_qint8, 0, 1.0f, 1, dims);
    return 0;
  }" TVM_XNNPACK_HAS_VALIDATE_QUANTIZED_TENSOR)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    const size_t dims[1] = {1};
    const float scale[1] = {1.0f};
    (void)xnn_validate_channelwise_quantized_tensor(xnn_datatype_qcint8, 0, scale, 1, 0, dims);
    return 0;
  }" TVM_XNNPACK_HAS_VALIDATE_CHANNELWISE_QUANTIZED_TENSOR)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_fully_connected(nullptr, -1.0f, 1.0f, 0, 1, XNN_INVALID_VALUE_ID, 2, 0);
    return 0;
  }" TVM_XNNPACK_HAS_FULLY_CONNECTED)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_depthwise_convolution_2d(nullptr, 0, 0, 0, 0, 3, 3, 1, 1, 1, 1, 1, 1,
                                             -1.0f, 1.0f, 0, 1, XNN_INVALID_VALUE_ID, 2, 0);
    return 0;
  }" TVM_XNNPACK_HAS_DEPTHWISE_CONVOLUTION_2D)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_unary_gelu == xnn_unary_invalid; }" TVM_XNNPACK_HAS_UNARY_GELU)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return xnn_unary_approxgelu == xnn_unary_invalid; }"
  TVM_XNNPACK_HAS_UNARY_APPROXGELU)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_softmax(nullptr, 0, 1, 0);
    return 0;
  }" TVM_XNNPACK_HAS_DEFINE_SOFTMAX)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    const size_t shape[2] = {1, 2};
    (void)xnn_define_static_reshape(nullptr, 2, shape, 0, 1, 0);
    return 0;
  }" TVM_XNNPACK_HAS_STATIC_RESHAPE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_define_copy(nullptr, 0, 1, 0);
    return 0;
  }" TVM_XNNPACK_HAS_COPY)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)xnn_reshape_runtime(nullptr);
    return 0;
  }" TVM_XNNPACK_HAS_RUNTIME_RESHAPE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)&xnn_reshape_external_value;
    return 0;
  }" TVM_XNNPACK_HAS_RESHAPE_EXTERNAL_VALUE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)&xnn_setup_runtime_v2;
    return 0;
  }" TVM_XNNPACK_HAS_SETUP_RUNTIME_V2)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)&xnn_get_external_value_shape;
    return 0;
  }" TVM_XNNPACK_HAS_GET_EXTERNAL_VALUE_SHAPE)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    (void)&xnn_create_fully_connected_nc_qd8_f32_qc8w;
    (void)&xnn_create_convolution2d_nhwc_qd8_f32_qc8w;
    return 0;
  }" TVM_XNNPACK_HAS_DYNAMIC_RANGE_QD8_OPS)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() {
    xnn_subgraph_t subgraph = nullptr;
    (void)xnn_create_subgraph(4, 0, &subgraph);
    uint32_t input = 0;
    uint32_t dynamic_input = 0;
    uint32_t weight = 0;
    uint32_t output = 0;
    const size_t input_shape[2] = {1, 4};
    const size_t weight_shape[2] = {3, 4};
    const size_t output_shape[2] = {1, 3};
    const float scales[3] = {0.5f, 0.25f, 0.125f};
    (void)xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 2, input_shape, nullptr, 0,
                                  XNN_VALUE_FLAG_EXTERNAL_INPUT, &input);
    (void)xnn_define_dynamically_quantized_tensor_value(subgraph, xnn_datatype_qdint8, 2, 2,
                                                        input_shape, XNN_INVALID_VALUE_ID, 0,
                                                        &dynamic_input);
    (void)xnn_define_convert(subgraph, input, dynamic_input, 0);
    (void)xnn_define_channelwise_quantized_tensor_value_v2(
        subgraph, xnn_datatype_qcint8, 0, scales, 2, 0, weight_shape, nullptr,
        XNN_INVALID_VALUE_ID, 0, &weight);
    (void)xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 2, output_shape, nullptr, 1,
                                  XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output);
    (void)xnn_define_fully_connected(subgraph, -1.0f, 1.0f, dynamic_input, weight,
                                     XNN_INVALID_VALUE_ID, output, 0);
    (void)xnn_delete_subgraph(subgraph);
    return 0;
  }" TVM_XNNPACK_HAS_DYNAMIC_RANGE_FULLY_CONNECTED_SUBGRAPH)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_FLAG_TRANSPOSE_WEIGHTS == 0; }" TVM_XNNPACK_HAS_TRANSPOSE_WEIGHTS_FLAG)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_FLAG_DONT_SPIN_WORKERS == 0; }" TVM_XNNPACK_HAS_DONT_SPIN_WORKERS_FLAG)
check_cxx_source_compiles("
  #include <xnnpack.h>
  int main() { return XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER == 0; }"
  TVM_XNNPACK_HAS_TRANSIENT_INDIRECTION_BUFFER_FLAG)
check_cxx_source_compiles("
  #include <pthreadpool.h>
  int main() {
    pthreadpool_t pool = pthreadpool_create(2);
    pthreadpool_destroy(pool);
    return 0;
  }" TVM_XNNPACK_HAS_PTHREADPOOL_CREATE)

foreach(_required
    INITIALIZE
    CREATE_SUBGRAPH
    RUNTIME_V2
    DEFINE_TENSOR_VALUE
    DEFINE_UNARY
    DEFINE_BINARY
    DEFINE_CONVOLUTION_2D
    DEFINE_MAX_POOLING_2D
    DEFINE_AVERAGE_POOLING_2D)
  if(NOT TVM_XNNPACK_HAS_${_required})
    message(FATAL_ERROR
            "USE_XNNPACK is enabled, but required XNNPACK baseline feature ${_required} "
            "was not found in the configured header/library")
  endif()
endforeach()

if(TVM_XNNPACK_HAS_HINT_FP16_INFERENCE_FLAG AND TVM_XNNPACK_HAS_FORCE_FP16_INFERENCE_FLAG)
  set(TVM_XNNPACK_HAS_FP16_FLAGS 1)
endif()
if(TVM_XNNPACK_HAS_DATATYPE_QINT8 AND TVM_XNNPACK_HAS_DATATYPE_QINT32 AND
   TVM_XNNPACK_HAS_DATATYPE_QCINT8 AND TVM_XNNPACK_HAS_DATATYPE_QCINT32 AND
   TVM_XNNPACK_HAS_DEFINE_QUANTIZED_TENSOR_VALUE)
  set(TVM_XNNPACK_HAS_QS8_DATATYPES 1)
endif()
if(TVM_XNNPACK_HAS_QS8_DATATYPES AND TVM_XNNPACK_HAS_FULLY_CONNECTED AND
   TVM_XNNPACK_HAS_DEPTHWISE_CONVOLUTION_2D AND TVM_XNNPACK_HAS_STATIC_RESHAPE AND
   TVM_XNNPACK_HAS_COPY)
  set(TVM_XNNPACK_HAS_QS8_SUBGRAPH_OPS 1)
endif()
if(TVM_XNNPACK_HAS_DATATYPE_QDINT8 AND TVM_XNNPACK_HAS_DATATYPE_QDUINT8 AND
   TVM_XNNPACK_HAS_DATATYPE_QPINT8 AND
   TVM_XNNPACK_HAS_DEFINE_DYNAMICALLY_QUANTIZED_TENSOR_VALUE)
  set(TVM_XNNPACK_HAS_DYNAMIC_QUANT_DATATYPES 1)
endif()
if(TVM_XNNPACK_HAS_DYNAMIC_RANGE_FULLY_CONNECTED_SUBGRAPH)
  set(TVM_XNNPACK_HAS_DYNAMIC_RANGE_SUBGRAPH_OPS 1)
endif()
if(TVM_XNNPACK_HAS_RUNTIME_RESHAPE AND TVM_XNNPACK_HAS_RESHAPE_EXTERNAL_VALUE AND
   TVM_XNNPACK_HAS_SETUP_RUNTIME_V2 AND TVM_XNNPACK_HAS_GET_EXTERNAL_VALUE_SHAPE)
  set(TVM_XNNPACK_HAS_DYNAMIC_BATCH_RUNTIME 1)
endif()

set(CMAKE_REQUIRED_INCLUDES "${_XNNPACK_PREV_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_LIBRARIES "${_XNNPACK_PREV_REQUIRED_LIBRARIES}")

foreach(_feature
    INITIALIZE
    CREATE_SUBGRAPH
    RUNTIME_V2
    RUNTIME_V4
    RUNTIME_V3
    DEFINE_TENSOR_VALUE
    DEFINE_UNARY
    DEFINE_BINARY
    DEFINE_CONVOLUTION_2D
    DEFINE_MAX_POOLING_2D
    DEFINE_AVERAGE_POOLING_2D
    WEIGHTS_CACHE
    WORKSPACE
    PROFILING
    BASIC_PROFILING_FLAG
    HINT_FP16_INFERENCE_FLAG
    FORCE_FP16_INFERENCE_FLAG
    FP32_STATIC_WEIGHTS_FLAG
    FP32_STATIC_BIASES_FLAG
    DATATYPE_FP16
    DATATYPE_QINT8
    DATATYPE_QUINT8
    DATATYPE_QINT32
    DATATYPE_QCINT8
    DATATYPE_QCINT32
    DATATYPE_QDINT8
    DATATYPE_QDUINT8
    DATATYPE_QPINT8
    EXTRA_QUANTIZATION_PARAMS
    DEFINE_CONVERT
    DEFINE_QUANTIZED_TENSOR_VALUE
    DEFINE_DYNAMICALLY_QUANTIZED_TENSOR_VALUE
    DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE
    DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE_V2
    VALIDATE_QUANTIZED_TENSOR
    VALIDATE_CHANNELWISE_QUANTIZED_TENSOR
    FULLY_CONNECTED
    DEPTHWISE_CONVOLUTION_2D
    UNARY_GELU
    UNARY_APPROXGELU
    DEFINE_SOFTMAX
    DYNAMIC_RANGE_QD8_OPS
    DYNAMIC_RANGE_FULLY_CONNECTED_SUBGRAPH
    DYNAMIC_RANGE_CONV2D_SUBGRAPH
    STATIC_RESHAPE
    COPY
    RUNTIME_RESHAPE
    RESHAPE_EXTERNAL_VALUE
    SETUP_RUNTIME_V2
    GET_EXTERNAL_VALUE_SHAPE
    DYNAMIC_BATCH_RUNTIME
    TRANSPOSE_WEIGHTS_FLAG
    DONT_SPIN_WORKERS_FLAG
    TRANSIENT_INDIRECTION_BUFFER_FLAG
    PTHREADPOOL_CREATE
    FP16_FLAGS
    QS8_DATATYPES
    QS8_SUBGRAPH_OPS
    DYNAMIC_QUANT_DATATYPES
    DYNAMIC_RANGE_SUBGRAPH_OPS)
  if(TVM_XNNPACK_HAS_${_feature})
    add_definitions(-DTVM_XNNPACK_HAS_${_feature}=1)
  endif()
endforeach()

message(STATUS "XNNPACK baseline: runtime_v2=${TVM_XNNPACK_HAS_RUNTIME_V2}, "
               "fp32_subgraph_ops=${TVM_XNNPACK_HAS_DEFINE_CONVOLUTION_2D}")
message(STATUS "XNNPACK runtime features: v4=${TVM_XNNPACK_HAS_RUNTIME_V4}, "
               "weights_cache=${TVM_XNNPACK_HAS_WEIGHTS_CACHE}, "
               "workspace=${TVM_XNNPACK_HAS_WORKSPACE}, profiling=${TVM_XNNPACK_HAS_PROFILING}")
message(STATUS "XNNPACK precision features: fp16_flags=${TVM_XNNPACK_HAS_FP16_FLAGS}, "
               "datatype_fp16=${TVM_XNNPACK_HAS_DATATYPE_FP16}")
message(STATUS "XNNPACK MLP features: unary_gelu=${TVM_XNNPACK_HAS_UNARY_GELU}, "
               "unary_approxgelu=${TVM_XNNPACK_HAS_UNARY_APPROXGELU}, "
               "softmax=${TVM_XNNPACK_HAS_DEFINE_SOFTMAX}")
message(STATUS "XNNPACK quantization features: qs8_datatypes=${TVM_XNNPACK_HAS_QS8_DATATYPES}, "
               "qs8_subgraph_ops=${TVM_XNNPACK_HAS_QS8_SUBGRAPH_OPS}, "
               "dynamic_quant_datatypes=${TVM_XNNPACK_HAS_DYNAMIC_QUANT_DATATYPES}, "
               "dynamic_range_qd8_ops=${TVM_XNNPACK_HAS_DYNAMIC_RANGE_QD8_OPS}, "
               "dynamic_range_subgraph_ops=${TVM_XNNPACK_HAS_DYNAMIC_RANGE_SUBGRAPH_OPS}")
message(STATUS "XNNPACK reshape/copy features: static_reshape=${TVM_XNNPACK_HAS_STATIC_RESHAPE}, "
               "copy=${TVM_XNNPACK_HAS_COPY}, runtime_reshape=${TVM_XNNPACK_HAS_RUNTIME_RESHAPE}, "
               "dynamic_batch_runtime=${TVM_XNNPACK_HAS_DYNAMIC_BATCH_RUNTIME}")

tvm_file_glob(GLOB XNNPACK_RELAX_CONTRIB_SRC src/relax/backend/contrib/xnnpack/*.cc)
list(APPEND COMPILER_SRCS ${XNNPACK_RELAX_CONTRIB_SRC})

tvm_file_glob(GLOB XNNPACK_RUNTIME_SRC src/runtime/contrib/xnnpack/*.cc)
list(APPEND RUNTIME_SRCS ${XNNPACK_RUNTIME_SRC})

list(APPEND TVM_RUNTIME_LINKER_LIBS ${XNNPACK_LIBRARY})
if(XNNPACK_MICROKERNELS_LIBRARY)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${XNNPACK_MICROKERNELS_LIBRARY})
endif()
if(PTHREADPOOL_LIBRARY)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${PTHREADPOOL_LIBRARY})
endif()
if(CPUINFO_LIBRARY)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CPUINFO_LIBRARY})
endif()
if(KLEIDIAI_LIBRARY)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${KLEIDIAI_LIBRARY})
endif()
