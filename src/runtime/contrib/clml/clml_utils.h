/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/clml/clml_utils.h
 * \brief CLML utilities header
 */
#ifndef TVM_RUNTIME_CONTRIB_CLML_CLML_UTILS_H_
#define TVM_RUNTIME_CONTRIB_CLML_CLML_UTILS_H_
#include <memory>
#include <string>
#include <vector>

#include "clml_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

void CopyDataToCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor, void* data,
                          cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_NCHW_QCOM);

void CopyDataFromCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor, void* data,
                            cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_NCHW_QCOM);

cl_ml_tensor_qcom DeviceMakeCLMLTensor(
    cl_context context, tensor_dims_t dims,
    cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
    cl_channel_type dtype = CL_FLOAT);

cl_mem AllocateOnChipTensorMemory(size_t size, cl_uint on_chip_mem_offset);

cl_mem AllocateDDRTensorMemory(size_t size);

tensor_dims_t GetTensorDims(const JSONGraphNode& node);

cl_channel_type MakeCLDataType(const DLDataType& data_type);

cl_arithmetic_mode_qcom MakeCLArithMode(const cl_channel_type& data_type,
                                        const cl_channel_type& acc_type = CL_FLOAT);

std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensor(const JSONGraphNode& tensor_rep,
                                                              void* data,
                                                              std::vector<size_t> c_shape,
                                                              cl_ml_tensor_layout_qcom layout,
                                                              cl_uint dtype);

std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensorFromJSONNode(
    const JSONGraphNode& node, cl_ml_tensor_layout_qcom layout, cl_uint dtype, void* data = nullptr,
    std::vector<size_t> shape = {});

std::vector<cl_uint> GetVectorValues(const std::vector<std::string>& val);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CLML_CLML_UTILS_H_
