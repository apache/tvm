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
 * \file src/runtime/contrib/clml/clml_utils.cc
 * \brief Utilities.
 */
#ifdef TVM_GRAPH_EXECUTOR_CLML
#include "clml_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

/*!
 * \brief Copy utility to CLML Tensor.
 *
 * \param tensor CLML tensor descriptor
 * \param data pointer to host data
 * \param layout host data layout
 */
void CopyDataToCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor, void* data,
                          cl_ml_tensor_layout_qcom layout) {
  cl_event evt = nullptr;
  CLML_CALL(clEnqueueWriteMLTensorDataQCOM, CLML_QUEUE, data, layout, tensor->tensor,
            tensor->memory, 0, nullptr, &evt);
  ICHECK(evt != nullptr) << "clEnqueueWriteMLTensorDataQCOM";
}

/*!
 * \brief Copy utility from CLML tensor.
 *
 * \param tensor CLML tensor descriptor
 * \param data pointer to host data
 * \param layout expectred host data layout
 */
void CopyDataFromCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor, void* data,
                            cl_ml_tensor_layout_qcom layout) {
  cl_int result = 0;
  cl_event readEvent = nullptr;
  // Read the output tensor
  CLML_CALL(clEnqueueReadMLTensorDataQCOM, CLML_QUEUE, tensor->tensor, tensor->memory, data, layout,
            0, nullptr, &readEvent);
  result = clWaitForEvents(1, &readEvent);
  ICHECK(result == CL_SUCCESS) << "clWaitForEvents:" << result;
}

/*!
 * \brief Make a CLML tensor given it's attributes
 *
 * \param context OpenCL context
 * \param dims Tensor dimensions
 * \param layout CLML tensor layout of tensor
 * \param dtype Tensor data type
 * \return CLML tensor
 */
cl_ml_tensor_qcom DeviceMakeCLMLTensor(cl_context context, tensor_dims_t dims,
                                       cl_ml_tensor_layout_qcom layout, cl_channel_type dtype,
                                       cl_ml_tensor_usage_qcom usage) {
  cl_ml_tensor_qcom tensor;

  cl_ml_tensor_desc_qcom desc = {
      dtype, layout, dims.n, dims.c, dims.h, dims.w, 0, CL_TENSOR_DIMENSIONS_4D_QCOM, {0}};
  CLML_CALL_clCreateMLTensorQCOM(CLML_CTX, nullptr, &desc, usage, &tensor);
  ICHECK(tensor) << "clCreateMLTensorQCOM";
  return tensor;
}

/*!
 * \brief utility that allocates DDR backed memory for the tensor.
 *
 * \param context OpenCL context
 * \param buffer size
 * \return allocated cl_mem object
 */
cl_mem AllocateDDRTensorMemory(size_t size) {
  cl_int result = CL_OUT_OF_HOST_MEMORY;
  cl_mem buffer = nullptr;

  buffer = clCreateBuffer(CLML_CTX, CL_MEM_READ_WRITE, size, nullptr, &result);
  ICHECK(result == CL_SUCCESS) << "clCreateBuffer:" << result;

  return buffer;
}

/*!
 * \brief utility that allocates on chip backed memory for the tensor.
 *
 * \param context OpenCL context
 * \param tensor_desc tensor descriptor
 * \param on_chip_mem_offset on chip memory offset to be used for allocation
 * \return result API status
 */
cl_mem AllocateOnChipTensorMemory(size_t size, cl_uint on_chip_mem_offset) {
  cl_int result = CL_OUT_OF_HOST_MEMORY;
  cl_mem buffer = nullptr;

  cl_mem_properties on_chip_buff_prop[] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1,
                                           CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, on_chip_mem_offset, 0};
  LOG_MEM << "On-Chip Alloc:" << size << " Offset:" << on_chip_mem_offset;
  buffer = clCreateBufferWithProperties(CLML_CTX, on_chip_buff_prop, CL_MEM_READ_WRITE, size,
                                        nullptr, &result);
  ICHECK(result == CL_SUCCESS) << "clCreateBufferWithProperties:" << result;

  return buffer;
}

/*!
 * \brief Utility to extract tensor dimensions from JSON node.
 *
 * \param node JSON graph node
 * \return The CLML tensor dimension
 */
tensor_dims_t GetTensorDims(const JSONGraphNode& node) {
  std::vector<int64_t> shape = node.GetOpShape()[0];
  tensor_dims_t dims;
  dims.n = shape[0];
  dims.c = shape[1];
  dims.h = shape[2];
  dims.w = shape[3];
  return dims;
}

/*!
 * \brief Utility to map TVM data type to OpenCL channel type.
 *
 * \param data_type TVM DType
 * \return OpenCL channel type.
 */
cl_channel_type MakeCLDataType(const DLDataType& data_type) {
  if (data_type.code == DLDataTypeCode::kDLFloat && data_type.bits == 32) {
    return CL_FLOAT;
  } else if (data_type.code == DLDataTypeCode::kDLFloat && data_type.bits == 16) {
    return CL_HALF_FLOAT;
  } else {
    LOG(FATAL) << "Datatype " << data_type << " unsupported by CLML runtime";
  }
}

/*!
 * \brief Utility to map OpenCL types to CLML operator arthematic mode.
 *
 * \param data_type cl data type
 * \param acc_type accumulation type to be used
 * \return the operator arthematic mode
 */
cl_arithmetic_mode_qcom MakeCLArithMode(const cl_channel_type& data_type,
                                        const cl_channel_type& acc_type) {
  if (data_type == CL_FLOAT && acc_type == CL_FLOAT) {
    return CL_ARITHMETIC_MODE_FP32_QCOM;
  } else if (data_type == CL_HALF_FLOAT && acc_type == CL_FLOAT) {
    return CL_ARITHMETIC_MODE_FP16_ACC32_QCOM;
  } else if (data_type == CL_HALF_FLOAT && acc_type == CL_HALF_FLOAT) {
    return CL_ARITHMETIC_MODE_FP16_QCOM;
  } else {
    LOG(FATAL) << "Datatype " << data_type << " unsupported by CLML runtime";
  }
}

/*!
 * \brief Helper to sanity check before tensor creation.
 *
 * \param node The tensor to represent.
 * \param data data pointer to prefill the tensor
 * \param shape shape information of tensor
 * \param layout the tensor layout to be used
 * \param dtype tensor data type
 * \return CLML Tensor descriptor.
 */
std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensor(
    const JSONGraphNode& tensor_rep, void* data, std::vector<size_t> c_shape,
    cl_ml_tensor_layout_qcom layout, cl_uint dtype, cl_ml_tensor_usage_qcom usage) {
  std::vector<int64_t> shape = tensor_rep.GetOpShape()[0];
  std::vector<size_t> clml_shape(shape.begin(), shape.end());
  if (c_shape.size() > 0) {
    clml_shape = c_shape;
  }
  // Make sure the tensors with dimensions less than 4 are padded with 1.
  clml_shape.push_back(1);
  clml_shape.push_back(1);
  clml_shape.push_back(1);

  tensor_dims_t dims;
  dims.n = clml_shape[0];
  dims.c = clml_shape[1];
  dims.h = clml_shape[2];
  dims.w = clml_shape[3];

  auto tensor_dsc = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
  tensor_dsc->tensor = DeviceMakeCLMLTensor(CLML_CTX, dims, layout, dtype, usage);
  return tensor_dsc;
}

/*!
 * \brief Create an CLML tensor given the JSON Node representation.
 *
 * \param node The tensor to represent.
 * \param layout the tensor layout to be used
 * \param dtype tensor data type
 * \param data data pointer to prefill the tensor
 * \param shape shape information of tensor
 * \return CLML Tensor descriptor.
 */
std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensorFromJSONNode(
    const JSONGraphNode& node, cl_ml_tensor_layout_qcom layout, cl_ml_tensor_usage_qcom usage,
    cl_uint dtype, void* data, std::vector<size_t> shape) {
  return MakeCLMLTensor(node, data, shape, layout, dtype, usage);
}

/*!
 * \brief Utility function to extract vector values from string.
 *
 * \param val vector of strings
 * \return vector of cl_uints.
 */
std::vector<cl_uint> GetVectorValues(const std::vector<std::string>& val) {
  std::vector<cl_uint> array;
  for (auto i : val) {
    array.push_back((cl_uint)stoi(i));
  }
  return array;
}

}  //  namespace contrib
}  //  namespace runtime
}  //  namespace tvm
#endif
