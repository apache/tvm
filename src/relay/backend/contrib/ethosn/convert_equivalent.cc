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
#include "../constant_transforms.h"
#include "ethosn_api.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

/*!
 * \brief Helper class to extract inputs and quantization information from binary
 * elementwise operations ready to convert.
 */
class BinaryElementwiseParams {
 public:
  static BinaryElementwiseParams ExtractBinaryElementwiseParams(const Call& call) {
    auto params = BinaryElementwiseParams();
    params.input1 = call->args[0];
    params.input2 = call->args[1];
    params.input1_scale = call->args[2];
    params.input1_zero_point = call->args[3];
    params.input2_scale = call->args[4];
    params.input2_zero_point = call->args[5];
    // Reverse the inputs if the constant is first input
    if (call->args[0]->IsInstance<ConstantNode>()) {
      params.input1 = call->args[1];
      params.input2 = call->args[0];
      params.input1_scale = call->args[4];
      params.input1_zero_point = call->args[5];
      params.input2_scale = call->args[2];
      params.input2_zero_point = call->args[3];
    }
    params.output_scale = call->args[6];
    params.output_zero_point = call->args[7];
    return params;
  }

  Expr input1;
  Expr input2;
  Expr input1_scale;
  Expr input1_zero_point;
  Expr input2_scale;
  Expr input2_zero_point;
  Expr output_scale;
  Expr output_zero_point;
};

/*!
 * \brief Converts qnn.mul to mathematically equivalent
 * qnn.conv2d depthwise operation.
 *
 * \param expr The expression to attempt to convert.
 *
 * \return Null if conversion is not supported else the converted expression.
 */
Optional<Expr> ConvertQnnMultiplyToDepthwise(const Expr& expr) {
  Call call = Downcast<Call>(expr);
  const auto params = BinaryElementwiseParams::ExtractBinaryElementwiseParams(call);

  Constant input_constant = Downcast<Constant>(params.input2);
  TensorType input_constant_tt = Downcast<TensorType>(input_constant->checked_type());
  TensorType input_tt = Downcast<TensorType>(call->checked_type());
  int channels = Downcast<IntImm>(input_tt->shape.back())->value;
  if (channels != Downcast<IntImm>(input_constant_tt->Size())->value) {
    return NullOpt;
  }

  runtime::NDArray input_data = input_constant->data;
  runtime::NDArray kernel_data_hwoi =
      runtime::NDArray::Empty({1, 1, channels, 1}, input_data->dtype, input_data->device);
  kernel_data_hwoi.CopyFrom(input_data);
  Constant kernel = Constant(kernel_data_hwoi, input_constant->span);

  TensorType output_tt = Downcast<TensorType>(expr->checked_type());
  DataType output_dtype = output_tt->dtype;

  Expr conv2d =
      qnn::MakeQnnConv2D(params.input1, kernel, params.input1_zero_point, params.input2_zero_point,
                         params.input1_scale, params.input2_scale, {1, 1}, {0, 0, 0, 0}, {1, 1},
                         channels, channels, {1, 1}, "NHWC", "HWOI", "NHWC", DataType::Int(32));
  Constant bias_data = MakeConstantZeros(DataType::Int(32), {channels});
  Expr bias_add = MakeBiasAdd(conv2d, bias_data, 3);
  Expr requantize = qnn::MakeRequantize(bias_add, params.input1_scale, params.input1_zero_point,
                                        params.output_scale, params.output_zero_point, -1, "None",
                                        "None", output_dtype);

  try {
    requantize = InferType(requantize);
    return requantize;
  } catch (tvm::Error& e) {
    // Conversion produced an invalid op.
    return NullOpt;
  }
}

/*!
 * \brief Converts qnn.add to a mathematically equivalent
 * qnn.conv2d depthwise operation.
 *
 * \param expr The expression to attempt to convert.
 *
 * \return Null if conversion is not supported else the converted expression.
 */
Optional<Expr> ConvertQnnAddToDepthwise(const Expr& expr) {
  Call call = Downcast<Call>(expr);
  const auto params = BinaryElementwiseParams::ExtractBinaryElementwiseParams(call);

  Constant input_constant = Downcast<Constant>(params.input2);
  TensorType input_constant_tt = Downcast<TensorType>(input_constant->checked_type());
  TensorType input_tt = Downcast<TensorType>(call->checked_type());
  int channels = Downcast<IntImm>(input_tt->shape.back())->value;
  if (channels != Downcast<IntImm>(input_constant_tt->Size())->value) {
    return NullOpt;
  }

  // Create the identity kernel. The kernel data is constructed such that it produces an identity
  // operation in the quantized space. Therefore, the input is not scaled in any way which allows
  // us to later use the bias to perform the addition.
  float input_scale_value = GetScalarFromConstant<float>(params.input1_scale);
  float output_scale_value = GetScalarFromConstant<float>(params.output_scale);
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
      qnn::MakeRequantize(params.input2, params.input2_scale, params.input2_zero_point, bias_scale,
                          bias_zero_point, -1, "None", "None", DataType::Int(32));
  Expr reshape_bias = MakeReshape(requantize_bias, {channels});

  try {
    reshape_bias = FoldConstantExpr(reshape_bias);
  } catch (tvm::Error& e) {
    // Conversion produced an invalid op.
    return NullOpt;
  }
  Constant bias = Downcast<Constant>(reshape_bias);

  // Make depthwise conv2d operation
  Expr conv2d = qnn::MakeQnnConv2D(params.input1, identity_kernel, params.input1_zero_point,
                                   identity_kernel_zero_point, params.input1_scale,
                                   identity_kernel_scale, {1, 1}, {0, 0, 0, 0}, {1, 1}, channels,
                                   channels, {1, 1}, "NHWC", "HWOI", "NHWC", DataType::Int(32));
  Expr bias_add = MakeBiasAdd(conv2d, bias, 3);
  Expr requantize = qnn::MakeRequantize(bias_add, params.input1_scale, params.input1_zero_point,
                                        params.output_scale, params.output_zero_point, -1, "None",
                                        "None", input_constant_tt->dtype);

  try {
    return InferType(requantize);
  } catch (tvm::Error& e) {
    // Conversion produced an invalid op.
    return NullOpt;
  }
}

/*!
 * \brief Converts qnn.mul to a mathematically equivalent qnn.requantize operation.
 * When converting to support library API, a reinterpret quantize operation will be created.
 *
 * \param expr The expression to attempt to convert.
 *
 * \return Null if conversion is not supported else the converted expression.
 */
Optional<Expr> ConvertQnnMultiplyToReinterpretQuantize(const Expr& expr) {
  Call call = Downcast<Call>(expr);
  const auto params = BinaryElementwiseParams::ExtractBinaryElementwiseParams(call);

  Constant input_constant = Downcast<Constant>(params.input2);
  TensorType input_constant_tt = Downcast<TensorType>(input_constant->checked_type());
  if (Downcast<IntImm>(input_constant_tt->Size())->value != 1) {
    return NullOpt;
  }

  float input_scale_value = GetScalarFromConstant<float>(params.input1_scale);
  float constant_scale_value = GetScalarFromConstant<float>(params.input2_scale);
  int constant_zero_point_value = GetScalarFromConstant<int>(params.input2_zero_point);
  float new_output_scale_value = input_scale_value * constant_scale_value *
                                 (ToScalar(input_constant->data) - constant_zero_point_value);
  Constant new_output_scale = MakeConstantScalar(DataType::Float(32), new_output_scale_value);

  if (std::abs(new_output_scale_value - GetScalarFromConstant<float>(params.output_scale)) >
      0.004f) {
    // Multiply does not represent an identity operation so don't convert.
    return NullOpt;
  }

  DataType output_data_type = Downcast<TensorType>(call->checked_type())->dtype;

  // A requantize operation is used to represent the identity reinterperet quantize op in
  // the support library at this stage. That is requantize is used here as a means for
  // passing the quantization information to the API conversion layer.
  Expr requantize = qnn::MakeRequantize(
      params.input1, params.input1_scale, params.input1_zero_point, params.output_scale,
      params.output_zero_point, -1, "None", "None", output_data_type);

  try {
    return InferType(requantize);
  } catch (tvm::Error& e) {
    // Conversion produced an invalid op.
    return NullOpt;
  }
}

/*!
 * \brief Converts qnn.mul to a mathematically equivalent qnn.requantize operation.
 * When converting to support library API, a reinterpret quantize operation will be created.
 *
 * \param expr The expression to attempt to convert.
 *
 * \return Null if conversion is not supported else the converted expression.
 */
Optional<Expr> ConvertQnnAddToReinterpretQuantize(const Expr& expr) {
  Call call = Downcast<Call>(expr);
  const auto params = BinaryElementwiseParams::ExtractBinaryElementwiseParams(call);

  Constant input_constant = Downcast<Constant>(params.input2);
  TensorType input_constant_tt = Downcast<TensorType>(input_constant->checked_type());
  if (Downcast<IntImm>(input_constant_tt->Size())->value != 1) {
    return NullOpt;
  }

  float input_scale = GetScalarFromConstant<float>(params.input1_scale);
  int input_zero_point = GetScalarFromConstant<int>(params.input1_zero_point);
  float scalar_scale = GetScalarFromConstant<float>(params.input2_scale);
  int scalar_zero_point = GetScalarFromConstant<int>(params.input2_zero_point);
  int output_zero_point_value = GetScalarFromConstant<int>(params.output_zero_point);
  float scalar_value = (ToScalar(input_constant->data) - scalar_zero_point) * scalar_scale;

  float new_output_zero_point_value = input_zero_point - (scalar_value / input_scale);
  if (new_output_zero_point_value - output_zero_point_value > 1.0f) {
    // Add does not represent an identity operation so don't convert
    return NullOpt;
  }

  DataType output_data_type = Downcast<TensorType>(call->checked_type())->dtype;

  // A requantize operation is used to represent the identity reinterperet quantize op in
  // the support library at this stage. That is requantize is used here as a means for
  // passing the quantization information to the API conversion layer.
  Expr requantize = qnn::MakeRequantize(
      params.input1, params.input1_scale, params.input1_zero_point, params.output_scale,
      params.output_zero_point, -1, "None", "None", output_data_type);

  try {
    return InferType(requantize);
  } catch (tvm::Error& e) {
    // Conversion produced an invalid op.
    return NullOpt;
  }
}

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

    Optional<Expr> optional_new_func_body;
    String new_composite_name = "";
    if (composite_name == "ethos-n.qnn_mul_to_reinterpret_quantize") {
      optional_new_func_body = ConvertQnnMultiplyToReinterpretQuantize(func->body);
      new_composite_name = "ethos-n.qnn_reinterpret_quantize";
    } else if (composite_name == "ethos-n.qnn_mul_to_depthwise") {
      optional_new_func_body = ConvertQnnMultiplyToDepthwise(func->body);
      new_composite_name = "ethos-n.qnn_conv2d";
    } else if (composite_name == "ethos-n.qnn_add_to_reinterpret_quantize") {
      optional_new_func_body = ConvertQnnAddToReinterpretQuantize(func->body);
      new_composite_name = "ethos-n.qnn_reinterpret_quantize";
    } else if (composite_name == "ethos-n.qnn_add_to_depthwise") {
      optional_new_func_body = ConvertQnnAddToDepthwise(func->body);
      new_composite_name = "ethos-n.qnn_conv2d";
    }

    if (new_composite_name != "") {
      ICHECK(optional_new_func_body)
          << "Operation " << composite_name
          << " was marked as having a valid conversion, but it could not be converted.";
      new_func = WithFields(func, func->params, optional_new_func_body.value());
      new_func = WithAttr(std::move(new_func), attr::kComposite, new_composite_name);
    }

    Call new_call = WithFields(call, new_func);
    return Downcast<Expr>(new_call);
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

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.ConvertQnnMultiplyToDepthwise")
    .set_body_typed(ConvertQnnMultiplyToDepthwise);

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.ConvertQnnAddToDepthwise")
    .set_body_typed(ConvertQnnAddToDepthwise);

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.ConvertQnnMultiplyToReinterpretQuantize")
    .set_body_typed(ConvertQnnMultiplyToReinterpretQuantize);

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.ConvertQnnAddToReinterpretQuantize")
    .set_body_typed(ConvertQnnAddToReinterpretQuantize);

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.ConvertEquivalents")
    .set_body_typed(ConvertEquivalents);

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
