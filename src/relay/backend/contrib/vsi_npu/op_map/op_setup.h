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
#ifndef TVM_REALY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_OP_SETUP_H_
#define TVM_REALY_BACKEND_CONTRIB_VSI_NPU_OP_MAP_OP_SETUP_H_
#include <tim/vx/graph.h>
#include <tim/vx/operation.h>
#include <tim/vx/tensor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>

#include <memory>

#include "field.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/activations.h"
#include "tim/vx/tensor.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {
namespace op_map {

struct CallbackExpr {
  CallbackExpr(Expr expr) : expr_(expr){};
  CallbackExpr(Expr expr, std::shared_ptr<CallbackExpr> ptr_pre_callback)
      : expr_(expr), ptr_pre_callback_(ptr_pre_callback){};

  Expr expr_;
  std::shared_ptr<CallbackExpr> ptr_pre_callback_ = nullptr;
};

class OpSetup {
 public:
  OpSetup(tim::vx::TensorSpec spec, std::shared_ptr<CallbackExpr> pCallbackexpr)
      : pCallbackexpr_(pCallbackexpr) {
    specs_.push_back(spec);
  }

  OpSetup(tim::vx::TensorSpec spec) { specs_.push_back(spec); };

  OpSetup(std::vector<tim::vx::TensorSpec> specs, std::shared_ptr<CallbackExpr> pCallbackexpr)
      : pCallbackexpr_(pCallbackexpr) {
    specs_ = std::move(specs);
  }

  OpSetup(std::vector<tim::vx::TensorSpec> specs) { specs_ = std::move(specs); };

  void SetSpec(tim::vx::TensorSpec spec) { specs_.push_back(spec); }

  void SetTensor(std::shared_ptr<tim::vx::Tensor> ptensor) { ptensors_.push_back(ptensor); }

  virtual void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                            std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) {
    std::cout << "something wrong in OpSetup::SetupOperand!" << std::endl;
  }

  virtual void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                              std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl);

  virtual std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) {
    std::cout << "something wrong in OpSetup::CreateOperation!" << std::endl;
    return nullptr;
  }

  Call call_;
  Expr expr_key_;
  Expr input_key_;
  std::shared_ptr<CallbackExpr> pCallbackexpr_ = nullptr;
  std::vector<tim::vx::TensorSpec> specs_;
  std::vector<std::shared_ptr<tim::vx::Tensor>> ptensors_;
  // std::shared_ptr<tim::vx::Operation> operation_;
};

void UpdateInputTableInfo(std::map<Expr, std::shared_ptr<OpSetup>>& VxOp_tb, Expr expr,
                          tim::vx::Graph* graph, uint32_t idx = 0);

void UpdateOutputTableInfo(std::map<Expr, std::shared_ptr<OpSetup>>& VxOp_tb, Expr expr,
                           tim::vx::Graph* graph);

class VsiNpuQnnConv2d : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Expr weight_key_;
  Expr bias_key_;
  Call conv_;

 public:
  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class VsiNpuQnnAvgPool : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Call avgpool_;
  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class VsiNpuQnnMean : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Call mean_;
  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class QnnSingleInputOpSetup : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Call op_;
  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class VsiNpuQnnSoftmax : public QnnSingleInputOpSetup {
 public:
  using QnnSingleInputOpSetup::QnnSingleInputOpSetup;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class VsiNpuQnnDeconv : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Expr weight_key_;
  Expr requantize_key_;
  Call conv_;

 public:
  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

template <typename T>
class VsiNpuQnnActivation : public QnnSingleInputOpSetup {
 public:
  using QnnSingleInputOpSetup::QnnSingleInputOpSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override {
    return graph->CreateOperation<T>();
  }
};

// using VsiNpuQnnRelu = VsiNpuQnnActivation<tim::vx::ops::Relu>;
// using VsiNpuQnnRelu6 = VsiNpuQnnActivation<tim::vx::ops::Relu6>;
using VsiNpuQnnSigmoid = VsiNpuQnnActivation<tim::vx::ops::Sigmoid>;
using VsiNpuQnnTanh = VsiNpuQnnActivation<tim::vx::ops::Tanh>;

class VsiNpuQnnLeakyRelu : public QnnSingleInputOpSetup {
 public:
  using QnnSingleInputOpSetup::QnnSingleInputOpSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class VsiNpuConcat : public OpSetup {
 public:
  using OpSetup::OpSetup;

  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class VsiNpuQnnDense : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Expr weight_key_;
  Expr bias_key_;
  Call dense_;

 public:
  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class SingleFloatInputSetup : public OpSetup {
 public:
  using OpSetup::OpSetup;

  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

template <typename T>
class Activation : public SingleFloatInputSetup {
 public:
  using SingleFloatInputSetup::SingleFloatInputSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override {
    return graph->CreateOperation<T>();
  }
};

using Relu = Activation<tim::vx::ops::Relu>;
// using Relu6 = Activation<tim::vx::ops::Relu6>;
using Sigmoid = Activation<tim::vx::ops::Sigmoid>;
using Tanh = Activation<tim::vx::ops::Tanh>;

class Softmax : public SingleFloatInputSetup {
 public:
  using SingleFloatInputSetup::SingleFloatInputSetup;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class AvgPool : public SingleFloatInputSetup {
 public:
  using SingleFloatInputSetup::SingleFloatInputSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class Transpose : public SingleFloatInputSetup {
 public:
  using SingleFloatInputSetup::SingleFloatInputSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class ElementWiseQnnOp : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Expr input2_key_;

 public:
  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl);
};

class QnnAdd : public ElementWiseQnnOp {
 public:
  using ElementWiseQnnOp::ElementWiseQnnOp;

 public:
  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class QnnSubtract : public ElementWiseQnnOp {
 public:
  using ElementWiseQnnOp::ElementWiseQnnOp;

 public:
  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class QnnMul : public ElementWiseQnnOp {
 public:
  using ElementWiseQnnOp::ElementWiseQnnOp;

 public:
  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class ElementWiseNotypeOp : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Expr input2_key_;

 public:
  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl);
};

class Maximum : public ElementWiseNotypeOp {
 public:
  using ElementWiseNotypeOp::ElementWiseNotypeOp;

 public:
  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class Minimum : public ElementWiseNotypeOp {
 public:
  using ElementWiseNotypeOp::ElementWiseNotypeOp;

 public:
  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class TwoBoolInputSetup : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Expr input0_key_;
  Expr input1_key_;

  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class LogicalAnd : public TwoBoolInputSetup {
 public:
  using TwoBoolInputSetup::TwoBoolInputSetup;

 public:
  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class LogicalOr : public TwoBoolInputSetup {
 public:
  using TwoBoolInputSetup::TwoBoolInputSetup;

 public:
  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class TwoFloatInputOpSetup : public OpSetup {
 public:
  using OpSetup::OpSetup;
  Expr input0_key_;
  Expr input1_key_;

  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class Add : public TwoFloatInputOpSetup {
 public:
  using TwoFloatInputOpSetup::TwoFloatInputOpSetup;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class Mean : public SingleFloatInputSetup {
 public:
  using SingleFloatInputSetup::SingleFloatInputSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class Conv : public TwoFloatInputOpSetup {
 public:
  using TwoFloatInputOpSetup::TwoFloatInputOpSetup;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class Quantize : public OpSetup {
 public:
  using OpSetup::OpSetup;

  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class Dequantize : public OpSetup {
 public:
  using OpSetup::OpSetup;

  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class QnnRequantize : public OpSetup {
 public:
  using OpSetup::OpSetup;

  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class NoTypeOpSetup : public OpSetup {
 public:
  using OpSetup::OpSetup;

  void SetupOperand(const CallNode* cn, tim::vx::Quantization& quant_info,
                    std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class VsiNpuQnnClip : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph, Call call,
      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl);
};

class Reshape : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class Squeeze : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class DepthtoSpace : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class ArgMax : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class ArgMin : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

class Resize : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class LeakyRelu : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class MaxPool2d : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  std::shared_ptr<tim::vx::Operation> CreateOperation(
      std::shared_ptr<tim::vx::Graph> graph) override;
};

class Pad : public NoTypeOpSetup {
 public:
  using NoTypeOpSetup::NoTypeOpSetup;

  void SetupOperation(const CallNode* cn, std::shared_ptr<tim::vx::Graph> graph,
                      std::map<Expr, std::shared_ptr<OpSetup>>& vxOpmap_tbl) override;
};

}  // namespace op_map
using VxOpTable = std::map<Expr, std::shared_ptr<op_map::OpSetup>>;
}  // namespace vsi_npu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif
