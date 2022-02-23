
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
#include <tvm/ir/transform.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"
#include "buffer_size.h"
#include "compiler_attrs.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

class RelayToTIRVisitor : public MixedModeMutator {
 public:
  explicit RelayToTIRVisitor(IRModule ir_module, Target target)
      : ir_module_(ir_module), target_(target) {
    context_buffer_id_ = 0;
  }

  IRModule Mutate() {
    GlobalVar main_global_var = ir_module_->GetGlobalVar("main");
    Function main = Downcast<Function>(ir_module_->Lookup(main_global_var));
    Function mutated_main = WithFields(main, main->params, VisitExpr(main->body));

    ir_module_->Update(main_global_var, mutated_main);

    return ir_module_;
  }

 private:
  inline IntImm ToArg(int32_t value) { return IntImm(DataType::Int(32), value); }

  void CreatePrimFuncForExtern(const GlobalVar& global_var, Array<tir::Var> func_signature,
                               tvm::Array<PrimExpr> call_extern_args,
                               std::string context_buffer_name = "NULL",
                               int context_buffer_size = 0) {
    Map<String, ObjectRef> dict_attrs;
    dict_attrs.Set(tvm::attr::kGlobalSymbol, global_var->name_hint);
    dict_attrs.Set(tvm::attr::kTarget, target_);
    dict_attrs.Set("tir.noalias", Bool(true));

    tir::Stmt body = tir::Evaluate(
        tvm::tir::Call(DataType::Int(8), tir::builtin::call_extern(), call_extern_args));

    if (context_buffer_size) {
      tir::Var buffer_var(context_buffer_name,
                          PointerType(PrimType(DataType::Int(8)), "global.workspace"));
      body = tir::Allocate(buffer_var, DataType::Int(8), {context_buffer_size}, tir::const_true(),
                           body);
      body =
          tir::AttrStmt(PrimExpr(), tvm::tir::attr::device_type, target_->kind->device_type, body);
      body = tir::AttrStmt(PrimExpr(), tvm::tir::attr::device_id, 0, body);
    }

    tir::PrimFunc replacement_func(func_signature, body, VoidType(), Map<tir::Var, tir::Buffer>(),
                                   DictAttrs(dict_attrs));
    ir_module_->Add(global_var, replacement_func);
  }

  void EmitConv2D(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* clip_call = nullptr;
    const CallNode* requantize_call = nullptr;
    const CallNode* bias_add_call = nullptr;
    const CallNode* conv2d_call = nullptr;
    const CallNode* final_call = expr.as<CallNode>();
    const OpNode* final_op = final_call->op.as<OpNode>();
    if (final_op->name == "clip") {
      clip_call = final_call;
      requantize_call = clip_call->args[0].as<CallNode>();
    } else {
      requantize_call = final_call;
    }
    const CallNode* requantize_input = requantize_call->args[0].as<CallNode>();
    const OpNode* requantize_input_op = requantize_input->op.as<OpNode>();
    if (requantize_input_op->name == "nn.bias_add") {
      bias_add_call = requantize_input;
      conv2d_call = bias_add_call->args[0].as<CallNode>();
    } else {
      conv2d_call = requantize_input;
    }

    // TIR variables are created in the order they appear in the Relay partitioned function
    // %1 = qnn.conv2d(%input, %weight_const_0, input_zero_point_scalar,
    //                 %cmsisnn_multiplier_const_1, %input_scale_scalar, %weight_scale_const_2)
    // %2 = nn.bias_add(%1, %bias_const_3, axis=3)
    // %3 = qnn.requantize(%2, %input_scale_const_4, %cmsisnn_shift_const_5,
    //                     %output_scale_scalar, %output_zero_point_scalar)
    // clip(%3, a_min=%min_scalar, a_max=%max_scalar)
    tir::Var input("input", DataType::Handle(8));
    tir::Var filter("filter", DataType::Handle(8));
    tir::Var multiplier("multiplier", DataType::Handle(32));
    tir::Var filter_scale("filter_scale", DataType::Handle(32));
    tir::Var bias("bias", DataType::Handle(32));
    tir::Var input_scale("input_scale", DataType::Handle(32));
    tir::Var shift("shift", DataType::Handle(32));
    tir::Var output("output", DataType::Handle(8));

    // Individual arguments to the structs arguments of the CMSIS-NN API are filled into call_extern
    // https://github.com/ARM-software/CMSIS_5/blob/def6f800f95661eb3451d317f7d0dde504f6020d/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c#L50

    // prepare cmsis_nn_conv_params
    const Conv2DAttrs* conv2d_attrs = conv2d_call->attrs.as<Conv2DAttrs>();
    int32_t input_offset = -GetScalarFromConstant<int32_t>(conv2d_call->args[2]);
    int32_t output_offset = GetScalarFromConstant<int32_t>(requantize_call->args[4]);
    int32_t stride_w = qnn::get_const_int(conv2d_attrs->strides[1]);
    int32_t stride_h = qnn::get_const_int(conv2d_attrs->strides[0]);
    int32_t padding_w = qnn::get_const_int(conv2d_attrs->padding[1]);
    int32_t padding_h = qnn::get_const_int(conv2d_attrs->padding[0]);
    int32_t dilation_w = qnn::get_const_int(conv2d_attrs->dilation[1]);
    int32_t dilation_h = qnn::get_const_int(conv2d_attrs->dilation[0]);
    int32_t out_channels = qnn::get_const_int(conv2d_attrs->channels);
    int32_t groups = conv2d_attrs->groups;
    std::string kernel_layout = conv2d_attrs->kernel_layout.c_str();
    int32_t clip_min, clip_max;
    if (clip_call) {
      const ClipAttrs* clip_attrs = clip_call->attrs.as<ClipAttrs>();
      clip_min = clip_attrs->a_min;
      clip_max = clip_attrs->a_max;
    } else {
      clip_min = -128;
      clip_max = 127;
    }

    tvm::Array<PrimExpr> scalar_args = {ToArg(input_offset), ToArg(output_offset), ToArg(stride_w),
                                        ToArg(stride_h),     ToArg(padding_w),     ToArg(padding_h),
                                        ToArg(dilation_w),   ToArg(dilation_h),    ToArg(clip_min),
                                        ToArg(clip_max)};

    // CMSIS-NN data structure "cmsis_nn_dims" for ifm expects input layout as NHWC
    // This is the same layout we expect in Relay
    Array<PrimExpr> input_shape = conv2d_call->args[0]->type_as<TensorTypeNode>()->shape;
    int32_t input_n = qnn::get_const_int(input_shape[0]);
    int32_t input_h = qnn::get_const_int(input_shape[1]);
    int32_t input_c = qnn::get_const_int(input_shape[3]);

    // CMSIS-NN data structure "cmsis_nn_dims" for weights expects following layouts
    // OHWI for Conv2D and IHWO for Depthwise convolutions
    Array<PrimExpr> filter_shape = conv2d_call->args[1]->type_as<TensorTypeNode>()->shape;

    Array<PrimExpr> bias_shape{1, 1, 1, out_channels};

    Array<PrimExpr> output_shape = conv2d_call->type_as<TensorTypeNode>()->shape;
    int32_t output_h = qnn::get_const_int(output_shape[1]);
    int32_t output_w = qnn::get_const_int(output_shape[2]);
    int32_t output_c = qnn::get_const_int(output_shape[3]);

    int32_t depth_multiplier = -1;
    int kernel_pos_o = kernel_layout.find("O");
    if (groups == qnn::get_const_int(input_shape[3]) &&
        groups == qnn::get_const_int(filter_shape[kernel_pos_o])) {
      int kernel_pos_i = kernel_layout.find("I");
      depth_multiplier = qnn::get_const_int(filter_shape[kernel_pos_i]);
    }
    scalar_args.push_back(ToArg(depth_multiplier));

    // original filter_layout for depthwise is HWOI
    std::string cmsisnn_api = "arm_convolve_wrapper_s8";
    bool is_depthwise = depth_multiplier != -1;
    if (is_depthwise) {
      cmsisnn_api = "arm_depthwise_conv_wrapper_s8";
      int filter_pos_h = kernel_layout.find("H");
      int filter_pos_w = kernel_layout.find("W");
      Array<PrimExpr> depthwise_filter_shape{1, filter_shape[filter_pos_h],
                                             filter_shape[filter_pos_w], out_channels};
      filter_shape = depthwise_filter_shape;
    }
    int32_t filter_h = qnn::get_const_int(filter_shape[1]);
    int32_t filter_w = qnn::get_const_int(filter_shape[2]);

    tvm::Array<PrimExpr> call_ext_args = {tir::StringImm(cmsisnn_api), input, filter, multiplier};
    if (bias_add_call) {
      call_ext_args.push_back(bias);
    }
    call_ext_args.push_back(shift);
    call_ext_args.push_back(output);

    std::string context_buffer_name = "NULL";
    CMSISNNFlags flags = GetCompilerFlags(transform::PassContext::Current());
    size_t context_buffer_size;
    if (is_depthwise) {
      context_buffer_size =
          DepthwiseConv2dBufferSize(flags, input_n, input_c, output_c, filter_w, filter_h);
    } else {
      context_buffer_size =
          Conv2dBufferSize(flags, padding_w, padding_h, input_n, input_h, input_c, output_h,
                           output_w, stride_w, stride_h, filter_w, filter_h);
    }

    if (context_buffer_size) {
      context_buffer_name = "context_buffer_" + std::to_string(context_buffer_id_++);
    }
    tvm::Array<PrimExpr> context_buffer_args = {tir::StringImm(context_buffer_name),
                                                ToArg(context_buffer_size)};

    scalar_args = tvm::runtime::Concat(context_buffer_args, scalar_args);
    scalar_args = tvm::runtime::Concat(scalar_args, input_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, filter_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, output_shape);
    call_ext_args = tvm::runtime::Concat(call_ext_args, scalar_args);

    Array<tir::Var> func_signature{input, filter, multiplier, filter_scale};
    if (bias_add_call) {
      func_signature.push_back(bias);
    }
    func_signature.push_back(input_scale);
    func_signature.push_back(shift);
    func_signature.push_back(output);

    CreatePrimFuncForExtern(global_var, func_signature, call_ext_args, context_buffer_name,
                            context_buffer_size);
  }

  void EmitFullyConnected(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* clip_call = nullptr;
    const CallNode* requantize_call = nullptr;
    const CallNode* bias_add_call = nullptr;
    const CallNode* fc_call = nullptr;
    const CallNode* final_call = expr.as<CallNode>();
    const OpNode* final_op = final_call->op.as<OpNode>();
    if (final_op->name == "clip") {
      clip_call = final_call;
      requantize_call = clip_call->args[0].as<CallNode>();
    } else {
      requantize_call = final_call;
    }
    const CallNode* requantize_input = requantize_call->args[0].as<CallNode>();
    const OpNode* requantize_input_op = requantize_input->op.as<OpNode>();
    if (requantize_input_op->name == "nn.bias_add") {
      bias_add_call = requantize_input;
      fc_call = bias_add_call->args[0].as<CallNode>();
    } else {
      fc_call = requantize_input;
    }

    // TIR variables are created in the order they appear in the Relay partitioned function
    // %1 = qnn.dense(%input, %weight_const_0, input_zero_point_scalar, kernel_zero_point_scalar,
    //                 %input_scale_scalar, %kernel_scale_scalar)
    // %2 = nn.bias_add(%1, %bias_const_1, axis=1)
    // %3 = qnn.requantize(%2, %req_input_scale_scalar, %req_input_zero_point_scalar,
    //                     %output_scale_scalar, %output_zero_point_scalar)
    // clip(%3, a_min=%min_scalar, a_max=%max_scalar)
    tir::Var input("input", DataType::Handle(8));
    tir::Var filter("filter", DataType::Handle(8));
    tir::Var bias("bias", DataType::Handle(32));
    tir::Var output("output", DataType::Handle(8));

    // Individual arguments to the structs arguments of the CMSIS-NN API are filled into call_extern
    // https://github.com/ARM-software/CMSIS_5/blob/def6f800f95661eb3451d317f7d0dde504f6020d/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c#L50

    // prepare cmsis_nn_fc_params
    const DenseAttrs* dense_attrs = fc_call->attrs.as<DenseAttrs>();
    int32_t input_offset = -GetScalarFromConstant<int32_t>(fc_call->args[2]);
    int32_t filter_offset = -GetScalarFromConstant<int32_t>(fc_call->args[3]);
    int32_t output_offset = GetScalarFromConstant<int32_t>(requantize_call->args[4]);
    float input_scale = GetScalarFromConstant<float>(requantize_call->args[1]);
    float output_scale = GetScalarFromConstant<float>(requantize_call->args[3]);
    int32_t out_channels = qnn::get_const_int(dense_attrs->units);
    int32_t clip_min, clip_max;
    if (clip_call) {
      const ClipAttrs* clip_attrs = clip_call->attrs.as<ClipAttrs>();
      clip_min = clip_attrs->a_min;
      clip_max = clip_attrs->a_max;
    } else {
      clip_min = -128;
      clip_max = 127;
    }

    double quantized_multiplier =
        static_cast<double>(input_scale) / static_cast<double>(output_scale);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(quantized_multiplier);
    int32_t multiplier = std::get<0>(mult_shift_pair);
    int32_t shift = std::get<1>(mult_shift_pair);

    tvm::Array<PrimExpr> scalar_args = {
        ToArg(input_offset), ToArg(filter_offset), ToArg(output_offset), ToArg(clip_min),
        ToArg(clip_max),     ToArg(multiplier),    ToArg(shift)};

    Array<PrimExpr> input_shape = fc_call->args[0]->type_as<TensorTypeNode>()->shape;
    int32_t batch_size = qnn::get_const_int(input_shape[0]);
    int32_t in_channels = qnn::get_const_int(input_shape[1]);
    Array<PrimExpr> cmsisnn_input_shape{input_shape[0], 1, 1, input_shape[1]};

    Array<PrimExpr> cmsisnn_filter_shape{in_channels, 1, 1, out_channels};

    Array<PrimExpr> bias_shape{1, 1, 1, out_channels};

    Array<PrimExpr> cmsisnn_output_shape{batch_size, 1, 1, out_channels};

    tvm::Array<PrimExpr> call_ext_args = {tir::StringImm("arm_fully_connected_s8"), input, filter};
    if (bias_add_call) {
      call_ext_args.push_back(bias);
    }
    call_ext_args.push_back(output);

    int context_buffer_size = 0;
    std::string context_buffer_name = "NULL";
    tvm::Array<PrimExpr> context_buffer_args = {tir::StringImm(context_buffer_name),
                                                ToArg(context_buffer_size)};

    scalar_args = tvm::runtime::Concat(context_buffer_args, scalar_args);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_input_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_filter_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, bias_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_output_shape);
    call_ext_args = tvm::runtime::Concat(call_ext_args, scalar_args);

    Array<tir::Var> func_signature{input, filter};
    if (bias_add_call) {
      func_signature.push_back(bias);
    }
    func_signature.push_back(output);
    CreatePrimFuncForExtern(global_var, func_signature, call_ext_args, context_buffer_name,
                            context_buffer_size);
  }

  void EmitPool2D(const GlobalVar& global_var, const Expr& expr, const String pool_name) {
    Call clip, pool;
    Call final_call = GetRef<Call>(expr.as<CallNode>());
    Op final_op = GetRef<Op>(final_call->op.as<OpNode>());
    if (final_op->name == "clip") {
      clip = final_call;
      Call clip_input = GetRef<Call>(clip->args[0].as<CallNode>());
      Op clip_input_op = GetRef<Op>(clip_input->op.as<OpNode>());
      if (clip_input_op->name == "cast") {
        pool = GetRef<Call>(clip_input->args[0].as<CallNode>());
      } else {  // max_pool2d
        pool = clip_input;
      }
    } else if (final_op->name == "cast") {
      pool = GetRef<Call>(final_call->args[0].as<CallNode>());
    } else {  // max_pool2d
      pool = final_call;
    }

    // prepare cmsis_nn_pool_params
    int32_t stride_h, stride_w, padding_h, padding_w, pool_size_h, pool_size_w;
    int32_t clip_min, clip_max;
    std::string cmsisnn_api;
    if (pool_name == "cmsis-nn.qnn_avg_pool2d") {
      cmsisnn_api = "arm_avgpool_s8";
      const AvgPool2DAttrs* attrs = pool->attrs.as<AvgPool2DAttrs>();
      stride_h = qnn::get_const_int(attrs->strides[0]);
      stride_w = qnn::get_const_int(attrs->strides[1]);
      padding_h = qnn::get_const_int(attrs->padding[0]);
      padding_w = qnn::get_const_int(attrs->padding[1]);
      pool_size_h = qnn::get_const_int(attrs->pool_size[0]);
      pool_size_w = qnn::get_const_int(attrs->pool_size[1]);
    } else {
      cmsisnn_api = "arm_max_pool_s8";
      const MaxPool2DAttrs* attrs = pool->attrs.as<MaxPool2DAttrs>();
      stride_h = qnn::get_const_int(attrs->strides[0]);
      stride_w = qnn::get_const_int(attrs->strides[1]);
      padding_h = qnn::get_const_int(attrs->padding[0]);
      padding_w = qnn::get_const_int(attrs->padding[1]);
      pool_size_h = qnn::get_const_int(attrs->pool_size[0]);
      pool_size_w = qnn::get_const_int(attrs->pool_size[1]);
    }
    if (clip.defined()) {
      const ClipAttrs* clip_attrs = clip->attrs.as<ClipAttrs>();
      clip_min = clip_attrs->a_min;
      clip_max = clip_attrs->a_max;
    } else {
      clip_min = -128;
      clip_max = 127;
    }

    tvm::Array<PrimExpr> scalar_args = {ToArg(stride_h),  ToArg(stride_w), ToArg(padding_h),
                                        ToArg(padding_w), ToArg(clip_min), ToArg(clip_max)};

    Array<PrimExpr> input_shape = pool->args[0]->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> cmsisnn_input_shape{1, input_shape[1], input_shape[2], input_shape[3]};

    Array<PrimExpr> cmsisnn_filter_shape{1, pool_size_h, pool_size_w, 1};

    Array<PrimExpr> output_shape = pool->type_as<TensorTypeNode>()->shape;
    Array<PrimExpr> cmsisnn_output_shape{1, output_shape[1], output_shape[2], output_shape[3]};

    tir::Var input("input", DataType::Handle(8));
    tir::Var output("output", DataType::Handle(8));
    tvm::Array<PrimExpr> call_ext_args = {tir::StringImm(cmsisnn_api), input, output};

    int context_buffer_size = 0;
    std::string context_buffer_name = "NULL";
    if (pool_name == "cmsisnn.qnn_avg_pool2d") {
      CMSISNNFlags flags = GetCompilerFlags(transform::PassContext::Current());
      int32_t input_c = qnn::get_const_int(input_shape[3]);
      context_buffer_size = AvgPoolBufferSize(flags, input_c);
      context_buffer_name = "context_buffer_" + std::to_string(context_buffer_id_++);
    }
    tvm::Array<PrimExpr> context_buffer_args = {tir::StringImm(context_buffer_name),
                                                ToArg(context_buffer_size)};

    scalar_args = tvm::runtime::Concat(context_buffer_args, scalar_args);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_input_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_filter_shape);
    scalar_args = tvm::runtime::Concat(scalar_args, cmsisnn_output_shape);
    call_ext_args = tvm::runtime::Concat(call_ext_args, scalar_args);

    Array<tir::Var> func_signature{input, output};

    CreatePrimFuncForExtern(global_var, func_signature, call_ext_args, context_buffer_name,
                            context_buffer_size);
  }

  void EmitSoftMax(const GlobalVar& global_var, const Expr& expr) {
    const CallNode* quantize_call = expr.as<CallNode>();
    const CallNode* softmax_call = quantize_call->args[0].as<CallNode>();
    const CallNode* dequant_call = softmax_call->args[0].as<CallNode>();
    const float quant_scale = GetScalarFromConstant<float>(dequant_call->args[1]);

    // assuming layout as NHWC
    auto shape = quantize_call->type_as<TensorTypeNode>()->shape;
    int trailing_dim = shape.size() - 1;
    int row_size = shape[trailing_dim].as<tir::IntImmNode>()->value;
    int num_rows = 1;
    for (int i = 0; i < trailing_dim; ++i) {
      num_rows *= shape[i].as<tir::IntImmNode>()->value;
    }

    // calculate multiplier and shift for CMSIS-NN softmax API
    // Note: TensorFlow Lite Micro assumptions
    // Output zero point and scale are fixed to -128 and 1 / 256
    // kScaledDiffIntegerBits, kInputBits, kBeta are described on the following github page
    // https://github.com/tensorflow/tflite-micro/blob/d97cd0908d8cf5021e9d86f05a49888bee28c2a4/tensorflow/lite/micro/kernels/softmax_common.cc#L47
    double beta_multiplier = (kBeta * quant_scale * (1 << (31 - kInputBits)));
    beta_multiplier = std::min<double>(beta_multiplier, (1ll << 31) - 1.0);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(beta_multiplier);
    int32_t mult = std::get<0>(mult_shift_pair);
    int32_t shift = std::get<1>(mult_shift_pair);
    int32_t diff_min = (1 << kScaledDiffIntegerBits) - 1;
    diff_min <<= (31 - kScaledDiffIntegerBits);
    diff_min >>= shift;
    diff_min *= -1;

    auto in_var = tir::Var("input", DataType::Handle(8));
    auto out_var = tir::Var("output", DataType::Handle(8));

    Array<tir::Var> func_signature{in_var, out_var};

    tvm::Array<PrimExpr> args = {
        tir::StringImm("arm_softmax_s8"),
        in_var,
        ToArg(num_rows),
        ToArg(row_size),
        ToArg(mult),
        ToArg(shift),
        ToArg(diff_min),
        out_var,
    };

    CreatePrimFuncForExtern(global_var, func_signature, args);
  }

  void EmitMul(const GlobalVar& global_var, const Expr& expr) {
    auto* mul_call = expr.as<CallNode>();

    const float input_0_scale = GetScalarFromConstant<float>(mul_call->args[2]);
    const int32_t input_0_zero_point = GetScalarFromConstant<int32_t>(mul_call->args[3]);
    const float input_1_scale = GetScalarFromConstant<float>(mul_call->args[4]);
    const int32_t input_1_zero_point = GetScalarFromConstant<int32_t>(mul_call->args[5]);
    const float output_scale = GetScalarFromConstant<float>(mul_call->args[6]);
    const int32_t output_zero_point = GetScalarFromConstant<int32_t>(mul_call->args[7]);

    double quantized_multiplier = static_cast<double>(input_0_scale) *
                                  static_cast<double>(input_1_scale) /
                                  static_cast<double>(output_scale);
    auto mult_shift_pair = tvm::relay::qnn::GetFixedPointMultiplierShift(quantized_multiplier);
    int32_t output_multiplier = std::get<0>(mult_shift_pair);
    int32_t output_shift = std::get<1>(mult_shift_pair);

    PrimExpr tensor_size = mul_call->type_as<TensorTypeNode>()->Size();

    tir::Var input_0("input_0", DataType::Handle(8));
    tir::Var input_1("input_1", DataType::Handle(8));
    tir::Var output("output", DataType::Handle(8));

    Array<tir::Var> func_signature{input_0, input_1, output};

    tvm::Array<PrimExpr> args = {
        tir::StringImm("arm_elementwise_mul_s8"),
        input_0,
        input_1,
        ToArg(-input_0_zero_point),
        ToArg(-input_1_zero_point),
        output,
        ToArg(output_zero_point),
        ToArg(output_multiplier),
        ToArg(output_shift),
        ToArg(std::numeric_limits<int8_t>::min()),
        ToArg(std::numeric_limits<int8_t>::max()),
        tensor_size,
    };

    CreatePrimFuncForExtern(global_var, func_signature, args);
  }

  void EmitAdd(const GlobalVar& global_var, const Expr& expr) {
    auto* add_call = expr.as<CallNode>();

    const float input_0_scale = GetScalarFromConstant<float>(add_call->args[2]);
    const int32_t input_0_zero_point = GetScalarFromConstant<int32_t>(add_call->args[3]);
    const float input_1_scale = GetScalarFromConstant<float>(add_call->args[4]);
    const int32_t input_1_zero_point = GetScalarFromConstant<int32_t>(add_call->args[5]);
    const float output_scale = GetScalarFromConstant<float>(add_call->args[6]);
    const int32_t output_zero_point = GetScalarFromConstant<int32_t>(add_call->args[7]);

    const int32_t left_shift = 20;
    const int32_t input_0_offset = -input_0_zero_point;
    const int32_t input_1_offset = -input_1_zero_point;

    const float max_input_scale = std::max(input_0_scale, input_1_scale);
    const double twice_max_input_scale = 2 * static_cast<double>(max_input_scale);
    const double scaled_input_0_scale = static_cast<double>(input_0_scale) / twice_max_input_scale;
    const double scaled_input_1_scale = static_cast<double>(input_1_scale) / twice_max_input_scale;
    const double scaled_output_scale =
        twice_max_input_scale / ((1 << left_shift) * static_cast<double>(output_scale));

    auto input_0_mult_shift_pair =
        tvm::relay::qnn::GetFixedPointMultiplierShift(scaled_input_0_scale);
    int32_t input_0_multiplier = std::get<0>(input_0_mult_shift_pair);
    int32_t input_0_shift = std::get<1>(input_0_mult_shift_pair);

    auto input_1_mult_shift_pair =
        tvm::relay::qnn::GetFixedPointMultiplierShift(scaled_input_1_scale);
    int32_t input_1_multiplier = std::get<0>(input_1_mult_shift_pair);
    int32_t input_1_shift = std::get<1>(input_1_mult_shift_pair);

    auto output_mult_shift_pair =
        tvm::relay::qnn::GetFixedPointMultiplierShift(scaled_output_scale);
    int32_t output_multiplier = std::get<0>(output_mult_shift_pair);
    int32_t output_shift = std::get<1>(output_mult_shift_pair);

    PrimExpr tensor_size = add_call->type_as<TensorTypeNode>()->Size();

    tir::Var input_0("input_0", DataType::Handle(8));
    tir::Var input_1("input_1", DataType::Handle(8));
    tir::Var output("output", DataType::Handle(8));

    Array<tir::Var> func_signature{input_0, input_1, output};

    tvm::Array<PrimExpr> args = {
        tir::StringImm("arm_elementwise_add_s8"),
        input_0,
        input_1,
        ToArg(input_0_offset),
        ToArg(input_0_multiplier),
        ToArg(input_0_shift),
        ToArg(input_1_offset),
        ToArg(input_1_multiplier),
        ToArg(input_1_shift),
        ToArg(left_shift),
        output,
        ToArg(output_zero_point),
        ToArg(output_multiplier),
        ToArg(output_shift),
        ToArg(std::numeric_limits<int8_t>::min()),
        ToArg(std::numeric_limits<int8_t>::max()),
        tensor_size,
    };

    CreatePrimFuncForExtern(global_var, func_signature, args);
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const CallNode* call = post.as<CallNode>()) {
      auto* func = call->op.as<FunctionNode>();
      if (func == nullptr) {
        return post;
      }

      auto codegen_name = func->GetAttr<String>(attr::kCompiler);
      if (codegen_name.defined() && codegen_name == "cmsis-nn") {
        const CallNode* inner_call = func->body.as<CallNode>();
        const FunctionNode* composite_func = inner_call->op.as<FunctionNode>();
        auto comp_name = composite_func->GetAttr<String>(attr::kComposite);
        auto func_name = func->GetAttr<String>(::tvm::attr::kGlobalSymbol);

        GlobalVar new_global_var(func_name.value());
        new_global_var->checked_type_ = composite_func->checked_type();

        if (comp_name == "cmsis-nn.qnn_softmax") {
          EmitSoftMax(new_global_var, composite_func->body);
        }
        if (comp_name == "cmsis-nn.qnn_mul") {
          EmitMul(new_global_var, composite_func->body);
        }
        if (comp_name == "cmsis-nn.qnn_add") {
          EmitAdd(new_global_var, composite_func->body);
        }
        if (comp_name == "cmsis-nn.qnn_conv2d") {
          EmitConv2D(new_global_var, composite_func->body);
        }
        if (comp_name == "cmsis-nn.qnn_fully_connected") {
          EmitFullyConnected(new_global_var, composite_func->body);
        }
        if (comp_name == "cmsis-nn.qnn_avg_pool2d" || comp_name == "cmsis-nn.qnn_max_pool2d") {
          EmitPool2D(new_global_var, composite_func->body, comp_name.value());
        }

        Array<Expr> args;
        for (const auto& arg : call->args) {
          args.push_back(VisitExpr(arg));
        }

        return Call(new_global_var, args, call->attrs, call->type_args, call->span);
      }
    }

    return post;
  }

 private:
  static constexpr int32_t kScaledDiffIntegerBits = 5;
  static constexpr int32_t kInputBits = 5;
  static constexpr double kBeta = 1.0;
  int32_t context_buffer_id_;
  IRModule ir_module_;
  Target target_;
};

tvm::transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule ir_module, transform::PassContext pass_context) {
        auto relay_to_tir = RelayToTIRVisitor(ir_module, Target("cmsis-nn"));
        return relay_to_tir.Mutate();
      };
  return tvm::transform::CreateModulePass(pass_func, 0, "RelayToTIR", {});
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
