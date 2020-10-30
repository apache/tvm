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
 *
 * \file mac_count.cc
 * \brief Pass to roughly count the number of MACs (Multiply-Accumulate)
 * operations of a model. Only MACs in CONV and Dense ops are counted.
 * This pass is valid after the type infer pass is called,
 * otherwise the count is 0.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "../transforms/pattern_utils.h"

namespace tvm {
namespace relay {

namespace mac_count {

inline int64_t GetCartesianProd(Array<IndexExpr> arr) {
  int64_t ret = 1;
  for (size_t i = 0; i < arr.size(); i++) {
    const auto* intImm = arr[i].as<IntImmNode>();
    ret *= static_cast<int64_t>(intImm->value);
  }
  return ret;
}

/*
 * \brief Preparation function for MAC count.
 * \param call_node The call node.
 * \return The number of MACs.
 */
using FMacCount = runtime::TypedPackedFunc<int64_t(const Call& call_node)>;

//----------------------------------------------
// Per operator defs for MAC count
//----------------------------------------------

int64_t ConvMacCount(const Call& call_node) {
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the mac count pass";
    return 0;
  }
  Array<Expr> args = call_node->args;
  ICHECK_EQ(args.size(), 2) << "The number of input arguments of a CONV 2D node should be 2.";
  const auto* conv_2d_attr = call_node->attrs.as<Conv2DAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
  std::string data_layout = conv_2d_attr->data_layout;
  int32_t C_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('C'));
  int32_t c_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('c'));
  ICHECK_NE(C_ind, -1) << "There is no input channel dimension.";
  int64_t input_channel = static_cast<int64_t>(data_shape[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1) input_channel *= static_cast<int64_t>(data_shape[c_ind].as<IntImmNode>()->value);
  Array<IndexExpr> kernel_size = conv_2d_attr->kernel_size;
  ICHECK_EQ(kernel_size.size(), 2) << "The dimension of the kernel in Conv 2D should be 2.";
  const auto* expr = call_node->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> output_tensor = expr->shape;
  ICHECK(output_tensor.size() == 4 || output_tensor.size() == 5)
      << "The dimension of the output tensor in Conv 2D should be 4 or 5.";
  int64_t count = GetCartesianProd(output_tensor) * GetCartesianProd(kernel_size);
  ICHECK_EQ(input_channel % conv_2d_attr->groups, 0)
      << "The number of input channels is not divisble by groups.";
  count *= input_channel / conv_2d_attr->groups;
  return count;
}

int64_t Conv2dTransposeMacCount(const Call& call_node) {
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the mac count pass";
    return 0;
  }
  Array<Expr> args = call_node->args;
  ICHECK_EQ(args.size(), 2)
      << "The number of input arguments of a CONV 2D Transpose node should be 2.";
  const auto* conv_2d_transpose_attr = call_node->attrs.as<Conv2DTransposeAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
  std::string data_layout = conv_2d_transpose_attr->data_layout;
  int32_t C_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('C'));
  int32_t c_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('c'));
  ICHECK_NE(C_ind, -1) << "There is no input channel dimension.";
  int64_t input_channel = static_cast<int64_t>(data_shape[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1) input_channel *= static_cast<int64_t>(data_shape[c_ind].as<IntImmNode>()->value);
  Array<IndexExpr> kernel_size = conv_2d_transpose_attr->kernel_size;
  ICHECK_EQ(kernel_size.size(), 2)
      << "The dimension of the kernel in Conv 2D Transpose should be 2.";
  const auto* expr = call_node->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> output_tensor = expr->shape;
  ICHECK(output_tensor.size() == 4 || output_tensor.size() == 5)
      << "The dimension of the output tensor in Conv 2D Transpose should be 4 or 5.";
  int64_t count = GetCartesianProd(output_tensor) * GetCartesianProd(kernel_size);
  ICHECK_EQ(input_channel % conv_2d_transpose_attr->groups, 0)
      << "The number of input channels is not divisble by groups.";
  count *= input_channel / conv_2d_transpose_attr->groups;
  return count;
}

int64_t DenseMacCount(const Call& call_node) {
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the mac count pass";
    return 0;
  }
  Array<Expr> args = call_node->args;
  ICHECK_EQ(args.size(), 2) << "The number of input arguments of a Dense node should be 2.";
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  const auto* weight_type = args[1]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
  Array<IndexExpr> weight_shape = weight_type->shape;
  ICHECK(data_shape.size() == 2 && weight_shape.size() == 2)
      << "The dimension of an input tensor to Dense node should be 2.";
  int64_t d1 = static_cast<int64_t>(data_shape[0].as<IntImmNode>()->value);
  int64_t d2 = static_cast<int64_t>(data_shape[1].as<IntImmNode>()->value);
  int64_t d3 = static_cast<int64_t>(weight_shape[0].as<IntImmNode>()->value);
  int64_t d4 = static_cast<int64_t>(weight_shape[1].as<IntImmNode>()->value);
  ICHECK_EQ(d2, d4) << "The dimensions of input arguments do not match.";
  int64_t count = d1 * d2 * d3;
  return count;
}

int64_t BatchMatmulMacCount(const Call& call_node) {
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the mac count pass";
    return 0;
  }
  Array<Expr> args = call_node->args;
  ICHECK_EQ(args.size(), 2);
  Array<IndexExpr> x_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> y_shape = args[1]->checked_type().as<TensorTypeNode>()->shape;
  int64_t batch = x_shape[0].as<IntImmNode>()->value;
  int64_t m = x_shape[1].as<IntImmNode>()->value;
  int64_t k = x_shape[2].as<IntImmNode>()->value;
  int64_t n = y_shape[1].as<IntImmNode>()->value;
  return batch * m * k * n;
}

RELAY_REGISTER_OP("nn.conv2d").set_attr<FMacCount>("FMacCount", ConvMacCount);

RELAY_REGISTER_OP("nn.conv2d_transpose").set_attr<FMacCount>("FMacCount", Conv2dTransposeMacCount);

RELAY_REGISTER_OP("nn.dense").set_attr<FMacCount>("FMacCount", DenseMacCount);

RELAY_REGISTER_OP("nn.batch_matmul").set_attr<FMacCount>("FMacCount", BatchMatmulMacCount);

class MacCounter : private ExprVisitor {
 public:
  MacCounter() { count_ = 0; }
  static int64_t GetTotalMacNumber(const Expr& expr) {
    LOG(INFO) << "This pass only counts MACs in direct conv2d, "
              << "conv2d_transpose, dense, and batch_matmul ops";
    MacCounter counter;
    counter(expr);
    return counter.count_;
  }

 private:
  void VisitExpr_(const CallNode* call_node) final {
    static const auto& fprep = Op::GetAttrMap<FMacCount>("FMacCount");
    auto f = fprep.get(call_node->op, nullptr);
    if (f != nullptr) count_ += f(GetRef<Call>(call_node));
    ExprVisitor::VisitExpr_(call_node);
  }

  int64_t count_;
};

int64_t GetTotalMacNumber(const Expr& expr) { return MacCounter::GetTotalMacNumber(expr); }

TVM_REGISTER_GLOBAL("relay.analysis.GetTotalMacNumber").set_body_typed(GetTotalMacNumber);

}  // namespace mac_count
}  // namespace relay
}  // namespace tvm
