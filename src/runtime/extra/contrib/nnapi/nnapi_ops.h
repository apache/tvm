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

#ifndef TVM_RUNTIME_CONTRIB_NNAPI_NNAPI_OPS_H_
#define TVM_RUNTIME_CONTRIB_NNAPI_NNAPI_OPS_H_
#ifdef TVM_GRAPH_EXECUTOR_NNAPI

#include <android/NeuralNetworks.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "nnapi_builder.h"

namespace tvm {
namespace runtime {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

struct NNAPIOpConverterParams {
  const JSONGraphNode& node;
  std::vector<NNAPIOperand> inputs;
  std::vector<NNAPIOperand> outputs;
  explicit NNAPIOpConverterParams(const JSONGraphNode& node);
};

class NNAPIOpConverter {
 public:
  std::string op_name_;

  explicit NNAPIOpConverter(std::string op_name);
  virtual ~NNAPIOpConverter() = default;

  virtual void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,  // NOLINT(*)
                       const std::vector<NNAPIOperand>& inputs,
                       std::vector<NNAPIOperand>& outputs) const = 0;  // NOLINT(*)
};

class ElwBinaryOpConverter : public NNAPIOpConverter {
 public:
  inline explicit ElwBinaryOpConverter(std::string op_name) : NNAPIOpConverter(op_name) {}
  ~ElwBinaryOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

class UnaryOpConverter : public NNAPIOpConverter {
 public:
  inline explicit UnaryOpConverter(std::string op_name) : NNAPIOpConverter(op_name) {}
  ~UnaryOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

class SoftmaxOpConverter : public NNAPIOpConverter {
 public:
  inline SoftmaxOpConverter() : NNAPIOpConverter("softmax") {}
  ~SoftmaxOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

class MatmulOpConverter : public NNAPIOpConverter {
 public:
  inline MatmulOpConverter() : NNAPIOpConverter("") {}
  ~MatmulOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

class TransposeOpConverter : public NNAPIOpConverter {
 public:
  inline TransposeOpConverter() : NNAPIOpConverter("") {}
  ~TransposeOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

class CastOpConverter : public NNAPIOpConverter {
 public:
  inline explicit CastOpConverter(std::string op_name) : NNAPIOpConverter(op_name) {}
  ~CastOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};
class Conv2dOpConverter : public NNAPIOpConverter {
 public:
  inline Conv2dOpConverter() : NNAPIOpConverter("") {}
  ~Conv2dOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

class DenseOpConverter : public NNAPIOpConverter {
 public:
  inline DenseOpConverter() : NNAPIOpConverter("") {}
  ~DenseOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

class MaxPool2dOpConverter : public NNAPIOpConverter {
 public:
  inline MaxPool2dOpConverter() : NNAPIOpConverter("") {}
  ~MaxPool2dOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

class MeanOpConverter : public NNAPIOpConverter {
 public:
  inline explicit MeanOpConverter(std::string op_name) : NNAPIOpConverter(op_name) {}
  ~MeanOpConverter() = default;

  void Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
               const std::vector<NNAPIOperand>& inputs,
               std::vector<NNAPIOperand>& outputs) const override;
};

const std::unordered_map<std::string, std::unique_ptr<NNAPIOpConverter>>& GetOpConverters();

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_GRAPH_EXECUTOR_NNAPI
#endif  // TVM_RUNTIME_CONTRIB_NNAPI_NNAPI_OPS_H_
