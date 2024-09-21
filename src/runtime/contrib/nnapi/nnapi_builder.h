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

#ifndef TVM_RUNTIME_CONTRIB_NNAPI_NNAPI_BUILDER_H_
#define TVM_RUNTIME_CONTRIB_NNAPI_NNAPI_BUILDER_H_
#ifdef TVM_GRAPH_EXECUTOR_NNAPI

#include <android/NeuralNetworks.h>
#include <dlpack/dlpack.h>

#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

class WrappedANeuralNetworksOperandType {
 public:
  WrappedANeuralNetworksOperandType(int32_t tensor_type, std::vector<uint32_t> dimensions,
                                    float scale, int32_t zero_point);
  WrappedANeuralNetworksOperandType(const WrappedANeuralNetworksOperandType&);
  WrappedANeuralNetworksOperandType& operator=(const WrappedANeuralNetworksOperandType&);

  const ANeuralNetworksOperandType* Get() const;

 private:
  std::vector<uint32_t> dimensions_;
  ANeuralNetworksOperandType ty_;
};

class NNAPIOperand {
 public:
  NNAPIOperand(uint32_t index, const DLTensor* tensor);
  NNAPIOperand(uint32_t index, const int64_t* shape, int ndim, DLDataType dtype);
  NNAPIOperand(uint32_t index, int32_t tensor_type, std::vector<int64_t> dimensions, float scale,
               int32_t zero_point);
  static NNAPIOperand Scalar(uint32_t index, int32_t tensor_type, std::vector<int64_t> dimensions,
                             float scale, int32_t zero_point);
  void SetDimensions(std::vector<int64_t> dimensions);

  WrappedANeuralNetworksOperandType GetOperandType() const;
  uint32_t GetOperandIndex() const;
  const std::vector<int64_t>& GetDimensions() const;
  const float GetScale() const;
  const int32_t GetZeroPoint() const;
  int32_t GetTensorType() const;
  bool IsDynamicShape() const;

 private:
  uint32_t index_;
  bool scalar_;

  // The NNAPI operand type e.g. ANEURALNETWORKS_TENSOR_INT32.
  int32_t tensor_type_;
  std::vector<int64_t> dimensions_;
  float scale_;
  int32_t zero_point_;
};

class NNAPIModelBuilder {
 public:
  NNAPIModelBuilder();
  ~NNAPIModelBuilder();
  NNAPIModelBuilder(const NNAPIModelBuilder&) = delete;
  NNAPIModelBuilder& operator=(const NNAPIModelBuilder&) = delete;
  inline NNAPIModelBuilder(NNAPIModelBuilder&& other) {
    model_ = other.model_;
    other.model_ = nullptr;
    next_operand_index_ = other.next_operand_index_;
    other.next_operand_index_ = 0;
  }
  inline NNAPIModelBuilder& operator=(NNAPIModelBuilder&& other) {
    model_ = other.model_;
    other.model_ = nullptr;
    next_operand_index_ = other.next_operand_index_;
    other.next_operand_index_ = 0;
    return *this;
  }

  NNAPIOperand CreateOperandWithValue(const DLTensor& tensor);
  NNAPIOperand CreateOperandWithValue(int32_t tensor_type, std::vector<int64_t> dimensions,
                                      float scale, int32_t zero_point, const void* buffer,
                                      size_t size);
  NNAPIOperand CreateScalarOperandWithValue(OperandCode operand_code, const void* buffer,
                                            size_t size);

  NNAPIOperand CreateOperand(const DLTensor& tensor);
  NNAPIOperand CreateOperand(const int64_t* shape, int ndim, DLDataType dtype);
  NNAPIOperand CreateOperand(int32_t tensor_type, std::vector<int64_t> dimensions, float scale,
                             int32_t zero_point);

  void AddOperation(ANeuralNetworksOperationType operation,
                    const std::vector<uint32_t> input_indices,
                    const std::vector<uint32_t> output_indices);

  void Finish(const std::vector<NNAPIOperand>& model_input_operands,
              const std::vector<NNAPIOperand>& model_output_operands);
  ANeuralNetworksCompilation* Compile();

 private:
  ANeuralNetworksModel* model_;
  uint32_t next_operand_index_ = 0;
};

/*!
 * \brief Convert a DLDataType to an NNAPI OperandCode.
 */
int32_t TensorTypeFromDLDataType(DLDataType ty);

std::vector<uint32_t> ExtractOperandIndices(const std::vector<NNAPIOperand>& operands);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_GRAPH_EXECUTOR_NNAPI
#endif  // TVM_RUNTIME_CONTRIB_NNAPI_NNAPI_BUILDER_H_
