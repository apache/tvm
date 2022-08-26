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
 * \file src/relay/backend/contrib/ethosn/convert_equivalent.cc
 * \brief Converts operations into a numerically equivalent form
 * that can be understood by the NPU codegen.
 */

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>

#include <unordered_map>

#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"
#include "../../../transforms/simplify_expr.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

/*!
 * \brief Apply constant folding on an expression.
 *
 * \param expr The expression to fold.
 * \param fold_qnn Whether to fold constants for QNN operations.
 * \returns The new folded expression.
 */
Expr FoldConstantExpr(const Expr& expr, bool fold_qnn = true) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::FoldConstant(fold_qnn)(mod);
  auto entry_func = Downcast<Function>(mod->Lookup("main"));
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

/*!
 * \brief Converts qnn.mul to mathematically equivalent
 * qnn.conv2d depthwise operation.
 */
Expr ConvertQnnMultiply(const Expr& expr) {
  Call call = Downcast<Call>(expr);

  Expr input1 = call->args[0];
  Expr input2 = call->args[1];
  Expr input1_scale = call->args[2];
  Expr input1_zero_point = call->args[3];
  Expr input2_scale = call->args[4];
  Expr input2_zero_point = call->args[5];
  // Reverse the inputs if the constant is first input
  if (call->args[0]->IsInstance<ConstantNode>()) {
    input1 = call->args[1];
    input2 = call->args[0];
    input1_scale = call->args[4];
    input1_zero_point = call->args[5];
    input2_scale = call->args[2];
    input2_zero_point = call->args[3];
  }
  Expr output_scale = call->args[6];
  Expr output_zero_point = call->args[7];

  const auto* input_constant = input2.as<ConstantNode>();
  ICHECK(input_constant) << "Expected ConstantNode but got " << input2->GetTypeKey();
  Type input_constant_type = input_constant->checked_type();
  const auto* input_constant_tt = input_constant_type.as<TensorTypeNode>();
  ICHECK(input_constant) << "Expected TensorTypeNode but got " << input_constant_type->GetTypeKey();
  int channels = input_constant_tt->shape.back().as<IntImmNode>()->value;

  runtime::NDArray input_data = input_constant->data;
  runtime::NDArray kernel_data_hwoi =
      runtime::NDArray::Empty({1, 1, channels, 1}, input_data->dtype, input_data->device);
  kernel_data_hwoi.CopyFrom(input_data);
  Constant kernel = Constant(kernel_data_hwoi, input_constant->span);

  Type output_type = expr->checked_type();
  auto output_tt = output_type.as<TensorTypeNode>();
  ICHECK(output_tt) << "Expected TensorTypeNode but got " << output_type->GetTypeKey();
  DataType output_dtype = output_tt->dtype;

  Expr conv2d = qnn::MakeQnnConv2D(
      input1, kernel, input1_zero_point, input2_zero_point, input1_scale, input2_scale, {1, 1},
      {0, 0, 0, 0}, {1, 1}, channels, channels, {1, 1}, "NHWC", "HWOI", "NHWC", DataType::Int(32));
  Constant bias_data = MakeConstantZeros(DataType::Int(32), {channels});
  Expr bias_add = MakeBiasAdd(conv2d, bias_data, 3);
  Expr requantize = qnn::MakeRequantize(bias_add, input1_scale, input1_zero_point, output_scale,
                                        output_zero_point, -1, "None", "None", output_dtype);

  return InferType(requantize);
}

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.ConvertQnnMultiply")
    .set_body_typed(ConvertQnnMultiply);

/*!
 * \brief Converts qnn.add to a mathematically equivalent
 * qnn.conv2d depthwise operation.
 */
Expr ConvertQnnAdd(const Expr& expr) {
  Call call = Downcast<Call>(expr);

  Expr input1 = call->args[0];
  Expr input2 = call->args[1];
  Expr input1_scale = call->args[2];
  Expr input1_zero_point = call->args[3];
  Expr input2_scale = call->args[4];
  Expr input2_zero_point = call->args[5];
  // Reverse the inputs if the constant is first input
  if (call->args[0]->IsInstance<ConstantNode>()) {
    input1 = call->args[1];
    input2 = call->args[0];
    input1_scale = call->args[4];
    input1_zero_point = call->args[5];
    input2_scale = call->args[2];
    input2_zero_point = call->args[3];
  }
  Expr output_scale = call->args[6];
  Expr output_zero_point = call->args[7];

  const auto* input_constant = input2.as<ConstantNode>();
  ICHECK(input_constant) << "Expected ConstantNode but got " << input2->GetTypeKey();
  Type input_constant_type = input_constant->checked_type();
  const auto* input_constant_tt = input_constant_type.as<TensorTypeNode>();
  ICHECK(input_constant) << "Expected TensorTypeNode but got " << input_constant_type->GetTypeKey();
  int channels = input_constant_tt->shape.back().as<IntImmNode>()->value;

  // Create the identity kernel. The kernel data is constructed such that it produces an identity
  // operation in the quantized space. Therefore, the input is not scaled in any way which allows
  // us to later use the bias to perform the addition.
  float input_scale_value = GetScalarFromConstant<float>(input1_scale);
  float output_scale_value = GetScalarFromConstant<float>(output_scale);
  float identity_kernel_scale_ub = std::min(output_scale_value / input_scale_value, 1.f);
  float identity_kernel_scale_lb = (1.f / 255.f);
  float identity_kernel_scale_target = (identity_kernel_scale_ub + identity_kernel_scale_lb) / 2.f;
  float identity_kernel_scale_recip_rounded = std::round(1.f / identity_kernel_scale_target);
  float identity_kernel_scale_value = 1.f / identity_kernel_scale_recip_rounded;
  Constant identity_kernel_scale =
      MakeConstantScalar(DataType::Float(32), identity_kernel_scale_value);
  Constant identity_kernel_zero_point = MakeConstantScalar(DataType::Int(32), 0);
  float identity_kernel_quantized_data = identity_kernel_scale_recip_rounded;
  std::vector<uint8_t> identity_kernel_data(channels,
                                            static_cast<uint8_t>(identity_kernel_quantized_data));
  Constant identity_kernel =
      MakeConstantTensor(input_constant_tt->dtype, {1, 1, channels, 1}, identity_kernel_data);

  // Calculate the bias, this is where the addition happens. The bias values are calculated by
  // scaling the constant input to input_scale * identity_kernel_scale.
  Constant bias_scale =
      MakeConstantScalar(DataType::Float(32), input_scale_value * identity_kernel_scale_value);
  Constant bias_zero_point = MakeConstantScalar(DataType::Int(32), 0);
  Expr requantize_bias =
      qnn::MakeRequantize(input2, input2_scale, input2_zero_point, bias_scale, bias_zero_point, -1,
                          "None", "None", DataType::Int(32));
  Expr reshape_bias = MakeReshape(requantize_bias, {channels});
  Constant bias = Downcast<Constant>(FoldConstantExpr(reshape_bias));

  // Make depthwise conv2d operation
  Expr conv2d =
      qnn::MakeQnnConv2D(input1, identity_kernel, input1_zero_point, identity_kernel_zero_point,
                         input1_scale, identity_kernel_scale, {1, 1}, {0, 0, 0, 0}, {1, 1},
                         channels, channels, {1, 1}, "NHWC", "HWOI", "NHWC", DataType::Int(32));
  Expr bias_add = MakeBiasAdd(conv2d, bias, 3);
  Expr requantize =
      qnn::MakeRequantize(bias_add, input1_scale, input1_zero_point, output_scale,
                          output_zero_point, -1, "None", "None", input_constant_tt->dtype);

  return InferType(requantize);
}

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.ConvertQnnAdd").set_body_typed(ConvertQnnAdd);

class ConvertEquivalentsMutator : public MixedModeMutator {
 public:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    Call call = Downcast<Call>(post);
    if (!call->op->IsInstance<FunctionNode>()) {
      return post;
    }

    Function func = Downcast<Function>(call->op);
    Function new_func = Function(func);
    auto composite_name = func->GetAttr<String>(attr::kComposite);
    if (composite_name == "ethos-n.qnn_mul") {
      Expr new_func_body = ConvertQnnMultiply(func->body);
      new_func = WithFields(func, func->params, new_func_body);
      new_func = WithAttr(std::move(new_func), attr::kComposite, String("ethos-n.qnn_conv2d"));
    } else if (composite_name == "ethos-n.qnn_add" && CheckCanConvertAdd(func->body)) {
      Expr new_func_body = ConvertQnnAdd(func->body);
      new_func = WithFields(func, func->params, new_func_body);
      new_func = WithAttr(std::move(new_func), attr::kComposite, String("ethos-n.qnn_conv2d"));
    }

    Call new_call = WithFields(call, new_func);
    return Downcast<Expr>(new_call);
  }

 private:
  /*!
   * \brief Check whether add can be converted to depthwise, or whether
   * it should be offloaded as a normal add operation.
   */
  bool CheckCanConvertAdd(const Expr& expr) {
    Call call = Downcast<Call>(expr);
    return call->args[0]->IsInstance<ConstantNode>() || call->args[1]->IsInstance<ConstantNode>();
  }
};

tvm::transform::Pass ConvertEquivalents() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule mod, transform::PassContext ctx) {
        for (auto gv : mod->GetGlobalVars()) {
          Function func = Downcast<Function>(mod->Lookup(gv));
          auto compiler_name = func->GetAttr<String>(attr::kCompiler);
          if (compiler_name.defined() && compiler_name == "ethos-n") {
            auto new_body = ConvertEquivalentsMutator().VisitExpr(func->body);
            if (!new_body.same_as(func->body)) {
              Function new_func = WithFields(func, func->params, new_body);
              mod->Update(gv, new_func);
            }
          }
        }
        return mod;
      };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "relay.backend.contrib.ethos-n.ConvertEquivalents", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.ConvertEquivalents")
    .set_body_typed(ConvertEquivalents);

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
