
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
 * \file generate_constant.cc
 * \brief Generates quantization parameters needed by CMSIS-NN
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>

#include "../../../op/make_op.h"
#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"
#include "../constant_transforms.h"
#include "convolutions.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*!
 * \brief This Mutator will find all partitioned functions meant for CMSIS-NN Conv2D.
 * It will substitute original Conv2D's weight zero point and original Requantize's input zero point
 * with CMSIS-NN's quantization parameters.
 * https://github.com/tensorflow/tflite-micro/blob/0f40100fc60276e9f345c23282de3baf19a78059/tensorflow/lite/kernels/internal/quantization_util.cc#L53
 */
class GenerateConstantsMutator : public MixedModeMutator {
 public:
  explicit GenerateConstantsMutator(const IRModule& mod) : mod_(mod) {}

 private:
  /*!  * \brief Converts Kernel layout from HWIO to OHWI to align to CMSIS-NN requirements */
  Expr ConvertKernelLayout(Expr kernel_expr, const Conv2DAttrs* conv2d_attrs, Attrs* new_attrs) {
    auto attrs = make_object<Conv2DAttrs>();
    attrs->strides = std::move(conv2d_attrs->strides);
    attrs->padding = std::move(conv2d_attrs->padding);
    attrs->dilation = std::move(conv2d_attrs->dilation);
    attrs->groups = conv2d_attrs->groups;
    attrs->channels = std::move(conv2d_attrs->channels);
    attrs->kernel_size = std::move(conv2d_attrs->kernel_size);
    attrs->data_layout = std::move(conv2d_attrs->data_layout);
    attrs->kernel_layout = runtime::String("OHWI");
    attrs->out_layout = std::move(conv2d_attrs->out_layout);
    attrs->out_dtype = std::move(conv2d_attrs->out_dtype);
    *new_attrs = tvm::Attrs{attrs};

    Constant conv2d_kernel = Downcast<Constant>(kernel_expr);
    conv2d_kernel = TransposeWeights(conv2d_kernel, conv2d_attrs->kernel_layout, "OHWI");
    return conv2d_kernel;
  }

  /*!  * \brief Performs weight transpose and substitutes existing constants in the composite
   *            function for Conv2D with CMSIS-NN Requantize constants */
  Expr GenerateConv2dRequantConstants(const Expr& expr) {
    const CallNode* clip_call = nullptr;
    const CallNode* requantize_call = nullptr;
    const CallNode* bias_add_call = nullptr;
    const CallNode* conv2d_call = nullptr;
    auto* final_call = expr.as<CallNode>();
    auto* final_op = final_call->op.as<OpNode>();
    if (final_op->name == "clip") {
      clip_call = final_call;
      requantize_call = clip_call->args[0].as<CallNode>();
    } else {
      requantize_call = final_call;
    }
    auto* requantize_input = requantize_call->args[0].as<CallNode>();
    auto* requantize_input_op = requantize_input->op.as<OpNode>();
    if (requantize_input_op->name == "nn.bias_add") {
      bias_add_call = requantize_input;
      conv2d_call = bias_add_call->args[0].as<CallNode>();
    } else {
      conv2d_call = requantize_input;
    }

    auto* conv2d_attrs = conv2d_call->attrs.as<Conv2DAttrs>();
    tvm::Attrs new_conv2d_attrs = conv2d_call->attrs;
    Expr conv2d_kernel = conv2d_call->args[1];

    Array<PrimExpr> input_shape = conv2d_call->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> kernel_shape = conv2d_call->args[1]->type_as<TensorTypeNode>()->shape;
    if (!IsCMSISNNDepthwise(conv2d_attrs, input_shape, kernel_shape)) {
      // Transpose weights: HWIO -> OHWI for Conv2D
      conv2d_kernel = ConvertKernelLayout(conv2d_call->args[1], conv2d_attrs, &new_conv2d_attrs);
    }

    // Obtain input and output scales from Relay's Requantization
    int64_t out_channels = conv2d_attrs->channels.as<IntImmNode>()->value;
    float output_scale = GetScalarFromConstant<float>(requantize_call->args[3]);
    auto input_scale = GetScalarFromConstant<float>(conv2d_call->args[4]);
    auto filter_scales = tvm::relay::qnn::GetFloatVectorFromConstant(conv2d_call->args[5]);

    // Calculate requantization multiplier and shift
    Device dev{DLDeviceType::kDLCPU, 0};
    runtime::NDArray multiplier_nda =
        runtime::NDArray::Empty({out_channels}, DataType::Int(32), dev);
    runtime::NDArray shift_nda = runtime::NDArray::Empty({out_channels}, DataType::Int(32), dev);
    int32_t* multiplier = static_cast<int32_t*>(multiplier_nda->data);
    int32_t* shift = static_cast<int32_t*>(shift_nda->data);
    for (int i = 0; i < out_channels; ++i) {
      double effective_output_scale =
          static_cast<double>(input_scale) * filter_scales[i] / static_cast<double>(output_scale);
      std::tie(*(multiplier + i), *(shift + i)) =
          tvm::relay::qnn::GetFixedPointMultiplierShift(effective_output_scale);
    }

    // Create constants from requantization multiplier and shift
    Constant multiplier_const(multiplier_nda);
    Constant shift_const(shift_nda);

    // Convert scale scalars into Constants
    // Scales are expected as Constants by following passes
    Expr weight_scale = conv2d_call->args[5];
    Expr req_inp_scale = requantize_call->args[1];
    if (out_channels == 1) {
      runtime::NDArray weight_scale_nda =
          runtime::NDArray::Empty({out_channels}, DataType::Float(32), dev);
      float* weight_scale_p = static_cast<float*>(weight_scale_nda->data);
      *weight_scale_p = GetScalarFromConstant<float>(weight_scale);
      weight_scale = Constant(weight_scale_nda);

      runtime::NDArray req_inp_scale_nda =
          runtime::NDArray::Empty({out_channels}, DataType::Float(32), dev);
      float* req_inp_scale_p = static_cast<float*>(req_inp_scale_nda->data);
      *req_inp_scale_p = GetScalarFromConstant<float>(req_inp_scale);
      req_inp_scale = Constant(req_inp_scale_nda);
    }

    // Replace existing weights (HWIO) with the transposed ones (OHWI) for Conv2D
    // Substitute Conv2D weight_zero_point with the CMSIS-NN multiplier
    // Substitute Requantize input_zero_point with CMSIS-NN shift
    // Conv2D arguments: data, weight, input_zp, weight_zp, input_sc, weight_sc
    Array<Expr> conv2d_args = {conv2d_call->args[0], conv2d_kernel,        conv2d_call->args[2],
                               multiplier_const,     conv2d_call->args[4], weight_scale};
    Call ret_call = Call(conv2d_call->op, conv2d_args, new_conv2d_attrs, {});
    if (bias_add_call) {
      ret_call =
          Call(bias_add_call->op, {ret_call, bias_add_call->args[1]}, bias_add_call->attrs, {});
    }
    Array<Expr> requantize_args = {ret_call, req_inp_scale, shift_const, requantize_call->args[3],
                                   requantize_call->args[4]};
    ret_call = Call(requantize_call->op, requantize_args, requantize_call->attrs, {});
    if (clip_call) {
      ret_call = Call(clip_call->op, {ret_call}, clip_call->attrs, {});
    }
    return std::move(ret_call);
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Expr final_call = post;
    auto* post_call = post.as<CallNode>();

    auto* global_var = call->op.as<GlobalVarNode>();
    if (global_var) {
      // Update to global function call needed because the body changes while
      // generating new constants
      Function func = Downcast<Function>(mod_->Lookup(global_var->name_hint));
      Expr new_body = VisitExpr(func->body);
      if (!new_body.same_as(func->body)) {
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        mod_->Update(GetRef<GlobalVar>(global_var), new_func);
        final_call = Call(GetRef<GlobalVar>(global_var), post_call->args);
      }
    }

    // Recreate composite function and corresponding call
    // Updated composite function contains CMSIS-NN quantized multiplier and shift constants
    if (call->op.as<FunctionNode>()) {
      auto* func = call->op.as<FunctionNode>();
      auto func_name = func->GetAttr<String>(attr::kComposite);
      if (func_name.defined() && func_name == "cmsis-nn.qnn_conv2d") {
        Expr new_body = GenerateConv2dRequantConstants(func->body);
        Function new_func = Function(FreeVars(new_body), new_body, func->ret_type,
                                     FreeTypeVars(new_body, mod_), func->attrs);
        final_call = Call(new_func, post_call->args);
      }
    }

    return final_call;
  }

 private:
  IRModule mod_;
};

IRModule GenerateConstants(const IRModule& mod) {
  String func_name;
  Function func;

  // Introduces CMSIS-NN constants before the call to the external Relay function
  auto generate_constants = GenerateConstantsMutator(mod);
  Function main_func = Downcast<Function>(mod->Lookup("main"));
  auto new_main_body = generate_constants.VisitExpr(main_func->body);
  if (!new_main_body.same_as(main_func->body)) {
    auto main_var = mod->GetGlobalVar("main");
    auto new_main_func = Function(main_func->params, new_main_body, main_func->ret_type,
                                  main_func->type_params, main_func->attrs);
    mod->Update(main_var, new_main_func);
  }

  return mod;
}

transform::Pass GenerateCMSISNNConstants() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule m, transform::PassContext pc) { return GenerateConstants(m); };
  return tvm::transform::CreateModulePass(pass_func, 0, "GenerateCMSISNNConstants", {});
}

TVM_REGISTER_GLOBAL("relay.ext.cmsisnn.transform.GenerateCMSISNNConstants")
    .set_body_typed(GenerateCMSISNNConstants);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
