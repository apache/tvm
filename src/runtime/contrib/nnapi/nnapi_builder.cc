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

#ifdef TVM_GRAPH_EXECUTOR_NNAPI

#include "nnapi_builder.h"

#include <android/log.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <iterator>
#include <utility>

#include "../json/json_runtime.h"
#include "nnapi_ops.h"

namespace tvm {
namespace runtime {
namespace contrib {

WrappedANeuralNetworksOperandType::WrappedANeuralNetworksOperandType(
    int32_t tensor_type, std::vector<uint32_t> dimensions, float scale, int32_t zero_point)
    : dimensions_(dimensions) {
  ty_.type = tensor_type;
  if (dimensions_.empty()) {
    ty_.dimensions = nullptr;
  } else {
    ty_.dimensions = dimensions_.data();
  }
  ty_.dimensionCount = dimensions_.size();
  ty_.scale = scale;
  ty_.zeroPoint = zero_point;
}

WrappedANeuralNetworksOperandType::WrappedANeuralNetworksOperandType(
    const WrappedANeuralNetworksOperandType& other)
    : dimensions_(other.dimensions_), ty_(other.ty_) {
  if (dimensions_.empty()) {
    ty_.dimensions = nullptr;
  } else {
    ty_.dimensions = dimensions_.data();
  }
}

WrappedANeuralNetworksOperandType& WrappedANeuralNetworksOperandType::operator=(
    const WrappedANeuralNetworksOperandType& other) {
  WrappedANeuralNetworksOperandType temp(other);
  std::swap(*this, temp);
  return *this;
}

const ANeuralNetworksOperandType* WrappedANeuralNetworksOperandType::Get() const { return &ty_; }

NNAPIOperand::NNAPIOperand(uint32_t index, const DLTensor* tensor)
    : index_(index), scalar_(false), dimensions_(tensor->shape, tensor->shape + tensor->ndim) {
  if (dimensions_.size() == 0) {
    dimensions_.push_back(1);
  }

  tensor_type_ = TensorTypeFromDLDataType(tensor->dtype);
  scale_ = 0.0;
  zero_point_ = 0;
}

NNAPIOperand::NNAPIOperand(uint32_t index, const int64_t* shape, int ndim, DLDataType dtype)
    : index_(index), scalar_(false), dimensions_(shape, shape + ndim) {
  if (dimensions_.size() == 0) {
    dimensions_.push_back(1);
  }

  tensor_type_ = TensorTypeFromDLDataType(dtype);
  scale_ = 0.0;
  zero_point_ = 0;
}

NNAPIOperand::NNAPIOperand(uint32_t index, int32_t tensor_type, std::vector<int64_t> dimensions,
                           float scale, int32_t zero_point)
    : index_(index),
      scalar_(false),
      tensor_type_(tensor_type),
      dimensions_(dimensions),
      scale_(scale),
      zero_point_(zero_point) {
  if (dimensions_.size() == 0) {
    dimensions_.push_back(1);
  }
}

NNAPIOperand NNAPIOperand::Scalar(uint32_t index, int32_t tensor_type,
                                  std::vector<int64_t> dimensions, float scale,
                                  int32_t zero_point) {
  NNAPIOperand operand(index, tensor_type, dimensions, scale, zero_point);
  operand.dimensions_.clear();
  operand.scalar_ = true;
  return operand;
}

void NNAPIOperand::SetDimensions(std::vector<int64_t> dimensions) { dimensions_ = dimensions; }

WrappedANeuralNetworksOperandType NNAPIOperand::GetOperandType() const {
  std::vector<uint32_t> dimensions(dimensions_.begin(), dimensions_.end());
  return WrappedANeuralNetworksOperandType(tensor_type_, dimensions, scale_, zero_point_);
}

uint32_t NNAPIOperand::GetOperandIndex() const { return index_; }

const std::vector<int64_t>& NNAPIOperand::GetDimensions() const { return dimensions_; }
const float NNAPIOperand::GetScale() const { return scale_; }
const int32_t NNAPIOperand::GetZeroPoint() const { return zero_point_; }

int32_t NNAPIOperand::GetTensorType() const { return tensor_type_; }
bool NNAPIOperand::IsDynamicShape() const {
  return std::any_of(dimensions_.begin(), dimensions_.end(), [](int64_t dim) { return dim == -1; });
}

NNAPIModelBuilder::NNAPIModelBuilder() {
  ICHECK_EQ(ANeuralNetworksModel_create(&model_), ANEURALNETWORKS_NO_ERROR);
}

NNAPIModelBuilder::~NNAPIModelBuilder() { ANeuralNetworksModel_free(model_); }

NNAPIOperand NNAPIModelBuilder::CreateOperandWithValue(const DLTensor& tensor) {
  NNAPIOperand operand(next_operand_index_++, &tensor);
  const size_t operand_data_size = GetDataSize(tensor);

  ICHECK_EQ(ANeuralNetworksModel_addOperand(model_, operand.GetOperandType().Get()),
            ANEURALNETWORKS_NO_ERROR);
  ICHECK_EQ(ANeuralNetworksModel_setOperandValue(model_, operand.GetOperandIndex(), tensor.data,
                                                 operand_data_size),
            ANEURALNETWORKS_NO_ERROR);

  return operand;
}

NNAPIOperand NNAPIModelBuilder::CreateOperandWithValue(int32_t tensor_type,
                                                       std::vector<int64_t> dimensions, float scale,
                                                       int32_t zero_point, const void* buffer,
                                                       size_t size) {
  NNAPIOperand operand(next_operand_index_++, tensor_type, dimensions, scale, zero_point);
  ICHECK_EQ(ANeuralNetworksModel_addOperand(model_, operand.GetOperandType().Get()),
            ANEURALNETWORKS_NO_ERROR);
  ICHECK_EQ(ANeuralNetworksModel_setOperandValue(model_, operand.GetOperandIndex(), buffer, size),
            ANEURALNETWORKS_NO_ERROR);
  return operand;
}

NNAPIOperand NNAPIModelBuilder::CreateScalarOperandWithValue(OperandCode operand_code,
                                                             const void* buffer, size_t size) {
  NNAPIOperand operand = NNAPIOperand::Scalar(next_operand_index_++, operand_code, {}, 0.0f, 0);

  ICHECK_EQ(ANeuralNetworksModel_addOperand(model_, operand.GetOperandType().Get()),
            ANEURALNETWORKS_NO_ERROR);
  ICHECK_EQ(ANeuralNetworksModel_setOperandValue(model_, operand.GetOperandIndex(), buffer, size),
            ANEURALNETWORKS_NO_ERROR);
  return operand;
}

NNAPIOperand NNAPIModelBuilder::CreateOperand(const DLTensor& tensor) {
  NNAPIOperand operand(next_operand_index_++, tensor.shape, tensor.ndim, tensor.dtype);
  ICHECK_EQ(ANeuralNetworksModel_addOperand(model_, operand.GetOperandType().Get()),
            ANEURALNETWORKS_NO_ERROR);
  return operand;
}

NNAPIOperand NNAPIModelBuilder::CreateOperand(const int64_t* shape, int ndim, DLDataType dtype) {
  NNAPIOperand operand(next_operand_index_++, shape, ndim, dtype);
  ICHECK_EQ(ANeuralNetworksModel_addOperand(model_, operand.GetOperandType().Get()),
            ANEURALNETWORKS_NO_ERROR);
  return operand;
}

NNAPIOperand NNAPIModelBuilder::CreateOperand(int32_t tensor_type, std::vector<int64_t> dimensions,
                                              float scale, int32_t zero_point) {
  NNAPIOperand operand(next_operand_index_++, tensor_type, dimensions, scale, zero_point);
  ICHECK_EQ(ANeuralNetworksModel_addOperand(model_, operand.GetOperandType().Get()),
            ANEURALNETWORKS_NO_ERROR);
  return operand;
}

void NNAPIModelBuilder::AddOperation(ANeuralNetworksOperationType operation,
                                     const std::vector<uint32_t> input_indicies,
                                     const std::vector<uint32_t> output_indicies) {
  ICHECK_EQ(ANeuralNetworksModel_addOperation(model_, operation, input_indicies.size(),
                                              input_indicies.data(), output_indicies.size(),
                                              output_indicies.data()),
            ANEURALNETWORKS_NO_ERROR);
}

void NNAPIModelBuilder::Finish(const std::vector<NNAPIOperand>& model_input_operands,
                               const std::vector<NNAPIOperand>& model_output_operands) {
  const auto model_input_indices = ExtractOperandIndices(model_input_operands);
  const auto model_output_indices = ExtractOperandIndices(model_output_operands);
  ICHECK_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(
                model_, model_input_indices.size(), model_input_indices.data(),
                model_output_indices.size(), model_output_indices.data()),
            ANEURALNETWORKS_NO_ERROR);
  ICHECK_EQ(ANeuralNetworksModel_finish(model_), ANEURALNETWORKS_NO_ERROR);
}

ANeuralNetworksCompilation* NNAPIModelBuilder::Compile() {
  ANeuralNetworksCompilation* compilation;
  ICHECK_EQ(ANeuralNetworksCompilation_create(model_, &compilation), ANEURALNETWORKS_NO_ERROR);
  ICHECK_EQ(ANeuralNetworksCompilation_setPreference(compilation,
                                                     ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER),
            ANEURALNETWORKS_NO_ERROR);
  ICHECK_EQ(ANeuralNetworksCompilation_finish(compilation), ANEURALNETWORKS_NO_ERROR);
  return compilation;
}

int32_t TensorTypeFromDLDataType(DLDataType ty) {
  if (ty.code == kDLInt) {
    if (ty.bits == 32) {
      return ANEURALNETWORKS_TENSOR_INT32;
    } else {
      ICHECK(false) << "Unsupported bit width " << ty.bits << " for NNAPI integer tensor";
    }
  } else if (ty.code == kDLUInt) {
    if (ty.bits == 1) {
      return ANEURALNETWORKS_TENSOR_BOOL8;
    } else {
      ICHECK(false) << "Unsupported bit width " << ty.bits << " for NNAPI unsigned integer tensor";
    }
  } else if (ty.code == kDLFloat) {
    if (ty.bits == 32) {
      return ANEURALNETWORKS_TENSOR_FLOAT32;
    } else if (ty.bits == 16) {
      return ANEURALNETWORKS_TENSOR_FLOAT16;
    } else {
      ICHECK(false) << "Unsupported bit width " << ty.bits << " for NNAPI integer tensor";
    }
  } else {
    ICHECK(false) << "Unsupported DLDataTypeCode for NNAPI: " << ty.code;
  }
}

std::vector<uint32_t> ExtractOperandIndices(const std::vector<NNAPIOperand>& operands) {
  std::vector<uint32_t> indices;
  indices.reserve(operands.size());
  std::transform(operands.begin(), operands.end(), std::back_inserter(indices),
                 [](const NNAPIOperand& operand) { return operand.GetOperandIndex(); });
  return indices;
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_GRAPH_EXECUTOR_NNAPI
