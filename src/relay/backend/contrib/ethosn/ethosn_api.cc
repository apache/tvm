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
 * \file src/relay/backend/contrib/ethosn/ethosn_api.cc
 * \brief The Relay -> Arm(R) Ethos(TM)-N command stream compiler.
 */

#include "ethosn_api.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/analysis.h>

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ethosn_api_version.h"
#include "ethosn_support_library/Support.hpp"
#include "ethosn_support_library/SupportQueries.hpp"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

sl::TensorInfo EthosnAPI::DefaultInputTensor(const Expr& expr) {
  Call call = Downcast<Call>(expr);
  const auto* dtype = call->args[0]->checked_type().as<TensorTypeNode>();
  sl::DataType data_type;
  Tvm2Npu(dtype->dtype, &data_type);
  return sl::TensorInfo({}, data_type, sl::DataFormat::NHWC, {});
}

EthosnError EthosnAPI::QnnConv2d(const Expr& expr, ConvolutionParams* params) {
  Call requantize = Downcast<Call>(expr);
  Call bias_add = Downcast<Call>(requantize->args[0]);
  Call conv = Downcast<Call>(bias_add->args[0]);
  Call pad;
  if (conv->args[0]->IsInstance<CallNode>() &&
      Downcast<Call>(conv->args[0])->op == Op::Get("nn.pad"))
    pad = Downcast<Call>(conv->args[0]);
  const auto& conv_attr = conv->attrs.as<Conv2DAttrs>();
  params->is_depthwise = conv_attr->channels.defined() &&
                         tvm::tir::ExprDeepEqual()(conv_attr->channels, conv_attr->groups) &&
                         conv_attr->groups != 1;

  // Extract the quantization params from the arguments
  int input_zero_point;
  int kernel_zero_point;
  int output_zero_point;
  std::valarray<float> input_scale;
  std::valarray<float> kernel_scale;
  float output_scale;
  std::string s = conv_attr->kernel_layout;
  unsigned int qaxis = s.find("O");
  assert(conv->args[2].size() == 1);
  assert(conv->args[3] == 1);
  assert(requantize->args[4] == 1);
  assert(conv->args[4] == 4);
  assert(conv->args[1]->checked_type().shape[qaxis] == kernel_scale_axis.size());
  assert(requantize->args[3] == 1);

  EthosnError err = AsConstant(conv->args[2], &input_zero_point);
  err += AsConstant(conv->args[3], &kernel_zero_point);
  err += AsConstant(requantize->args[4], &output_zero_point);
  err += AsConstant(conv->args[4], &input_scale);
  err += AsConstant(conv->args[5], &kernel_scale);
  err += AsConstant(requantize->args[3], &output_scale);

  // Convert quantization params
  sl::QuantizationInfo input_q_info;
  sl::QuantizationInfo weights_q_info;
  sl::QuantizationInfo bias_q_info;
  sl::QuantizationInfo output_q_info;
  err += Tvm2Npu(input_zero_point, input_scale, qaxis, &input_q_info);
  err += Tvm2Npu(kernel_zero_point, kernel_scale, qaxis, &weights_q_info);
  std::valarray<float> bias = input_q_info.GetScales() * weights_q_info.GetScales();
  err += Tvm2Npu(0, bias, 3, &bias_q_info);
  err += Tvm2Npu(output_zero_point, output_scale, &output_q_info);

  // Convert convolution attributes
  sl::Padding padding;
  if (pad.defined()) {
    Tvm2Npu(conv_attr->padding, &padding);
    // Don't support both standalone operator padding and attribute defined padding
    if (padding != sl::Padding({0, 0, 0, 0})) {
      err += EthosnError(
          ErrStrm() << "both op and attr padding exist, must be either op/attr only or no padding");
    }
    err += Tvm2Npu(pad->attrs.as<PadAttrs>()->pad_width, &padding);
  } else {
    err += Tvm2Npu(conv_attr->padding, &padding);
  }
  sl::Stride stride;
  err += Tvm2Npu(conv_attr->strides, &stride);
  // Dilation is not supported
  std::array<uint32_t, 2> dilation = {1, 1};
  AsArray(conv_attr->dilation, &dilation);
  if (conv_attr->dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1) {
    err +=
        EthosnError(ErrStrm() << "dilation=" << conv_attr->dilation << ", dilation must = [1, 1]");
  }
  // Create convolution info
  params->conv_info = sl::ConvolutionInfo(padding, stride, output_q_info);

  // Create input info
  const TensorTypeNode* input_ttype;
  if (pad.defined()) {
    input_ttype = pad->args[0]->checked_type().as<TensorTypeNode>();
  } else {
    input_ttype = conv->args[0]->checked_type().as<TensorTypeNode>();
  }
  sl::TensorShape input_tensor_shape;
  sl::DataType input_data_type;
  err += Tvm2Npu(input_ttype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_ttype->dtype, &input_data_type);
  params->input_info =
      sl::TensorInfo(input_tensor_shape, input_data_type, sl::DataFormat::NHWC, input_q_info);

  // Create weights info
  const auto* weights_dtype = conv->args[1]->checked_type().as<TensorTypeNode>();
  sl::TensorShape weights_tensor_shape;
  sl::DataType weights_data_type;
  sl::DataFormat weights_data_format;
  // Ignore the error here because weights don't have a batch axis
  Tvm2Npu(weights_dtype->shape, &weights_tensor_shape);
  err += Tvm2Npu(weights_dtype->dtype, &weights_data_type);
  err += Tvm2Npu(params->is_depthwise ? "HWIM" : "HWIO", &weights_data_format);
  params->weights_info =
      sl::TensorInfo(weights_tensor_shape, weights_data_type, weights_data_format, weights_q_info);
  params->raw_weights = conv->args[1].as<ConstantNode>()->data->data;

  // Create bias info
  params->bias_info = sl::TensorInfo(
      {1, 1, 1, params->is_depthwise ? weights_tensor_shape[2] : weights_tensor_shape[3]},
      sl::DataType::INT32_QUANTIZED, sl::DataFormat::NHWC, bias_q_info);
  params->raw_bias = bias_add->args[1].as<ConstantNode>()->data->data;

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(requantize->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = output_q_info;
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::QnnFullyConnected(const Expr& expr, FullyConnectedParams* params) {
  Call requantize = Downcast<Call>(expr);
  Call bias_add = Downcast<Call>(requantize->args[0]);
  Call dense = Downcast<Call>(bias_add->args[0]);

  // Extract the quantization params from the arguments
  int input_zero_point;
  int kernel_zero_point;
  int output_zero_point;
  float input_scale;
  float kernel_scale;
  float output_scale;
  EthosnError err = AsConstant(dense->args[2], &input_zero_point);
  err += AsConstant(dense->args[3], &kernel_zero_point);
  err += AsConstant(requantize->args[4], &output_zero_point);
  err += AsConstant(dense->args[4], &input_scale);
  err += AsConstant(dense->args[5], &kernel_scale);
  err += AsConstant(requantize->args[3], &output_scale);

  // Convert quantization params
  sl::QuantizationInfo data_q_info;
  sl::QuantizationInfo weights_q_info;
  sl::QuantizationInfo bias_q_info;
  sl::QuantizationInfo output_q_info;
  err += Tvm2Npu(input_zero_point, input_scale, &data_q_info);
  err += Tvm2Npu(kernel_zero_point, kernel_scale, &weights_q_info);
  err += Tvm2Npu(0, data_q_info.GetScale() * weights_q_info.GetScale(), &bias_q_info);
  err += Tvm2Npu(output_zero_point, output_scale, &output_q_info);

  // Create fc info
  params->fc_info = sl::FullyConnectedInfo(output_q_info);

  // Create data info
  const TensorTypeNode* data_dtype = dense->args[0]->checked_type().as<TensorTypeNode>();
  sl::TensorShape data_tensor_shape;
  sl::DataType data_data_type;
  err += Tvm2Npu(data_dtype->shape, &data_tensor_shape);
  err += Tvm2Npu(data_dtype->dtype, &data_data_type);
  params->input_info = sl::TensorInfo({data_tensor_shape[0], 1, 1, data_tensor_shape[1]},
                                      data_data_type, sl::DataFormat::NHWC, data_q_info);

  // Create weights info
  const auto* weights_dtype = dense->args[1]->checked_type().as<TensorTypeNode>();
  sl::TensorShape weights_tensor_shape;
  sl::DataType weights_data_type;
  sl::DataFormat weights_data_format;
  // Ignore the error here because weights don't have a batch axis
  Tvm2Npu(weights_dtype->shape, &weights_tensor_shape);
  err += Tvm2Npu(weights_dtype->dtype, &weights_data_type);
  err += Tvm2Npu("HWIO", &weights_data_format);
  params->weights_info = sl::TensorInfo({1, 1, weights_tensor_shape[1], weights_tensor_shape[0]},
                                        weights_data_type, weights_data_format, weights_q_info);
  params->raw_weights = dense->args[1].as<ConstantNode>()->data->data;

  // Create bias info
  params->bias_info =
      sl::TensorInfo({1, 1, 1, weights_tensor_shape[0]}, sl::DataType::INT32_QUANTIZED,
                     sl::DataFormat::NHWC, bias_q_info);
  params->raw_bias = bias_add->args[1].as<ConstantNode>()->data->data;

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(requantize->checked_type(), &output_tensor_info);
  output_tensor_info.m_Dimensions = {data_tensor_shape[0], 1, 1, weights_tensor_shape[0]};
  output_tensor_info.m_QuantizationInfo = output_q_info;
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Pool2d(const Call& input, const Call& output, Array<IndexExpr> size,
                              Array<IndexExpr> strides, Array<IndexExpr> padding,
                              sl::PoolingType pooling_type, sl::PoolingInfo* pool_info,
                              sl::TensorInfo* input_info, sl::TensorInfo* output_info,
                              std::string layout) {
  uint32_t npu_sizex, npu_sizey;
  sl::Padding npu_padding;
  sl::Stride npu_stride;
  EthosnError err = Tvm2Npu(size, &npu_sizex, &npu_sizey);
  err += Tvm2Npu(padding, &npu_padding);
  err += Tvm2Npu(strides, &npu_stride);
  *pool_info = sl::PoolingInfo(npu_sizex, npu_sizey, npu_stride.m_X, npu_stride.m_Y, npu_padding,
                               pooling_type);

  // Create input info
  const auto* input_dtype = input->args[0]->checked_type().as<TensorTypeNode>();
  sl::TensorShape input_tensor_shape;
  sl::DataType input_data_type;
  sl::DataFormat input_data_format;
  err += Tvm2Npu(input_dtype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_dtype->dtype, &input_data_type);
  err += Tvm2Npu(layout, &input_data_format);
  if (input_data_format != sl::DataFormat::NHWC) {
    return EthosnError(ErrStrm() << "data format=" << layout << ", data format must = NHWC");
  }
  *input_info = sl::TensorInfo(input_tensor_shape, input_data_type, input_data_format,
                               input_info->m_QuantizationInfo);

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(output->checked_type(), &output_tensor_info);
  // output quantization is the same as the input
  output_tensor_info.m_QuantizationInfo = input_info->m_QuantizationInfo;
  *output_info = output_tensor_info;
  return err;
}

EthosnError EthosnAPI::MaxPool2D(const Expr& expr, MaxPool2DParams* params) {
  Call pool = Downcast<Call>(expr);
  const auto pool_attrs = pool->attrs.as<MaxPool2DAttrs>();
  return Pool2d(pool, pool, pool_attrs->pool_size, pool_attrs->strides, pool_attrs->padding,
                sl::PoolingType::MAX, &params->pool_info, &params->input_info, &params->output_info,
                pool_attrs->layout);
}

EthosnError EthosnAPI::AvgPool2D(const Expr& expr, AvgPool2DParams* params) {
  Call cast_0 = Downcast<Call>(expr);
  Call pool = Downcast<Call>(cast_0->args[0]);
  Call cast_1 = Downcast<Call>(pool->args[0]);
  const auto pool_attrs = pool->attrs.as<AvgPool2DAttrs>();
  return Pool2d(cast_1, cast_0, pool_attrs->pool_size, pool_attrs->strides, pool_attrs->padding,
                sl::PoolingType::AVG, &params->pool_info, &params->input_info, &params->output_info,
                pool_attrs->layout);
}

EthosnError EthosnAPI::Reshape(const Expr& expr, ReshapeParams* params) {
  // Create input info
  Call reshape = Downcast<Call>(expr);
  const auto* input_dtype = reshape->args[0]->checked_type().as<TensorTypeNode>();
  const auto& reshape_attrs = reshape->attrs.as<ReshapeAttrs>();

  if (reshape_attrs->newshape.size() > params->new_shape.size()) {
    return EthosnError(ErrStrm() << "reshape dimension=" << reshape_attrs->newshape.size()
                                 << ", reshape dimension must be <= " << params->new_shape.size());
  }

  sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
  sl::DataType input_data_type;
  EthosnError err = Tvm2Npu(input_dtype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_dtype->dtype, &input_data_type);
  int tensor_size = 1;
  for (const auto& dim : input_tensor_shape) {
    tensor_size *= dim;
  }

  int infer_index = -1;
  int reshaped_size = 1;
  Array<Integer> inferred_shape = {1, 1, 1, 1};
  for (size_t i = 0; i < reshape_attrs->newshape.size(); i++) {
    int value = reshape_attrs->newshape[i].as<IntImmNode>()->value;
    if (value < -1) {
      return EthosnError(ErrStrm()
                         << "reshape dimension=" << value << ", reshape dimension must be >= -1");
    }
    if (value == -1) {
      if (infer_index != -1) {
        return EthosnError("only one reshape dimension can be inferred");
      }
      infer_index = i;
    } else {
      inferred_shape.Set(i, value);
      reshaped_size *= value;
    }
  }

  if (infer_index != -1) {
    if (tensor_size % reshaped_size != 0) {
      return EthosnError(ErrStrm()
                         << "reshaped size=" << reshaped_size
                         << ", must be an integer factor of the input size " << tensor_size);
    }
    int value = tensor_size / reshaped_size;
    inferred_shape.Set(infer_index, Integer(value));
  }
  err += Tvm2Npu(inferred_shape, &params->new_shape);
  params->input_info =
      sl::TensorInfo(input_tensor_shape, input_data_type, params->input_info.m_DataFormat,
                     params->input_info.m_QuantizationInfo);

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(reshape->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = params->input_info.m_QuantizationInfo;
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Addition(const Expr& expr, AdditionParams* params) {
  Call call = Downcast<Call>(expr);
  // Extract the quantization params from the arguments
  float lhs_scale;
  int lhs_zero_point;
  float rhs_scale;
  int rhs_zero_point;
  float output_scale;
  int output_zero_point;
  EthosnError err = AsConstant(call->args[2], &lhs_scale);
  err += AsConstant(call->args[3], &lhs_zero_point);
  err += AsConstant(call->args[4], &rhs_scale);
  err += AsConstant(call->args[5], &rhs_zero_point);
  err += AsConstant(call->args[6], &output_scale);
  err += AsConstant(call->args[7], &output_zero_point);

  sl::QuantizationInfo lhs_q_info;
  sl::QuantizationInfo rhs_q_info;
  sl::QuantizationInfo output_q_info;
  err += Tvm2Npu(lhs_zero_point, lhs_scale, &lhs_q_info);
  err += Tvm2Npu(rhs_zero_point, rhs_scale, &rhs_q_info);
  err += Tvm2Npu(output_zero_point, output_scale, &output_q_info);
  params->output_quantization_info = output_q_info;

  // Create input info
  const auto* lhs_dtype = call->args[0]->checked_type().as<TensorTypeNode>();
  sl::TensorShape lhs_tensor_shape;
  sl::DataType lhs_data_type;
  err += Tvm2Npu(lhs_dtype->shape, &lhs_tensor_shape);
  err += Tvm2Npu(lhs_dtype->dtype, &lhs_data_type);
  params->lhs_info =
      sl::TensorInfo(lhs_tensor_shape, lhs_data_type, sl::DataFormat::NHWC, lhs_q_info);

  const auto* rhs_dtype = call->args[1]->checked_type().as<TensorTypeNode>();
  sl::TensorShape rhs_tensor_shape;
  sl::DataType rhs_data_type;
  err += Tvm2Npu(rhs_dtype->shape, &rhs_tensor_shape);
  err += Tvm2Npu(rhs_dtype->dtype, &rhs_data_type);
  params->rhs_info =
      sl::TensorInfo(rhs_tensor_shape, rhs_data_type, sl::DataFormat::NHWC, rhs_q_info);

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(call->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = output_q_info;
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Sigmoid(const Expr& expr, SigmoidParams* params) {
  Call quantize = Downcast<Call>(expr);
  Call sigmoid = Downcast<Call>(quantize->args[0]);
  Call dequantize = Downcast<Call>(sigmoid->args[0]);

  // Create input info
  const auto* input_dtype = dequantize->args[0]->checked_type().as<TensorTypeNode>();
  sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
  sl::DataType input_tensor_dtype;
  EthosnError err = Tvm2Npu(input_dtype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_dtype->dtype, &input_tensor_dtype);
  float input_sc;
  int input_zp;
  err += AsConstant(dequantize->args[2], &input_zp);
  err += AsConstant(dequantize->args[1], &input_sc);
  float output_sc;
  int output_zp;
  err += AsConstant(quantize->args[2], &output_zp);
  err += AsConstant(quantize->args[1], &output_sc);

  auto test_zp = input_dtype->dtype.is_int() ? -128 : 0;
  if (output_zp != test_zp || output_sc != 1.0f / 256.0f) {
    err += EthosnError(ErrStrm() << "output quantization params=(" << output_zp << ", " << output_sc
                                 << "), must = (" << test_zp << ", 1/256)");
  }

  params->input_info = sl::TensorInfo(input_tensor_shape, input_tensor_dtype, sl::DataFormat::NHWC,
                                      sl::QuantizationInfo(input_zp, input_sc));

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(quantize->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = sl::QuantizationInfo(output_zp, output_sc);
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Mean(const Expr& expr, MeanParams* params) {
  Call requantize = Downcast<Call>(expr);
  Call mean = Downcast<Call>(requantize->args[0]);
  Call cast_0 = Downcast<Call>(mean->args[0]);

  // Create input info
  const auto* input_ttype = cast_0->args[0]->checked_type().as<TensorTypeNode>();
  const auto* output_ttype = requantize->checked_type().as<TensorTypeNode>();
  sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
  sl::DataType input_tensor_dtype;
  EthosnError err = Tvm2Npu(input_ttype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_ttype->dtype, &input_tensor_dtype);
  sl::TensorShape output_tensor_shape = {1, 1, 1, 1};
  sl::DataType output_tensor_dtype;
  err += Tvm2Npu(output_ttype->shape, &output_tensor_shape);
  err += Tvm2Npu(output_ttype->dtype, &output_tensor_dtype);
  float input_sc;
  int input_zp;
  err += AsConstant(requantize->args[2], &input_zp);
  err += AsConstant(requantize->args[1], &input_sc);
  params->input_info = sl::TensorInfo(input_tensor_shape, input_tensor_dtype, sl::DataFormat::NHWC,
                                      sl::QuantizationInfo(input_zp, input_sc));

  float output_sc;
  int output_zp;
  err += AsConstant(requantize->args[3], &output_sc);
  err += AsConstant(requantize->args[4], &output_zp);
  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(requantize->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = sl::QuantizationInfo(output_zp, output_sc);
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Tanh(const Expr& expr, TanhParams* params) {
  Call quantize = Downcast<Call>(expr);
  Call tanh = Downcast<Call>(quantize->args[0]);
  Call dequantize = Downcast<Call>(tanh->args[0]);
  // Create input info
  const auto* input_dtype = quantize->checked_type().as<TensorTypeNode>();
  sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
  sl::DataType input_tensor_dtype;
  EthosnError err = Tvm2Npu(input_dtype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_dtype->dtype, &input_tensor_dtype);
  float input_sc;
  int input_zp;
  err += AsConstant(dequantize->args[2], &input_zp);
  err += AsConstant(dequantize->args[1], &input_sc);
  float output_sc;
  int output_zp;
  err += AsConstant(quantize->args[2], &output_zp);
  err += AsConstant(quantize->args[1], &output_sc);
  auto test_zp = input_dtype->dtype.is_uint() ? 128 : 0;
  if (output_zp != test_zp || output_sc != 0.0078125f) {
    err += EthosnError(ErrStrm() << "output quantization params=(" << output_zp << ", " << output_sc
                                 << "), must = (" << test_zp << ", 1/256)");
  }
  params->input_info = sl::TensorInfo(input_tensor_shape, input_tensor_dtype, sl::DataFormat::NHWC,
                                      sl::QuantizationInfo(input_zp, input_sc));

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(quantize->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = sl::QuantizationInfo(output_zp, output_sc);
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::LeakyReLU(const Expr& expr, LeakyReLUParams* params) {
  Call quantize = Downcast<Call>(expr);
  Call leaky_relu = Downcast<Call>(quantize->args[0]);
  Call dequantize = Downcast<Call>(leaky_relu->args[0]);

  const auto* input_dtype = quantize->checked_type().as<TensorTypeNode>();
  sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
  sl::DataType input_tensor_dtype;
  EthosnError err = Tvm2Npu(input_dtype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_dtype->dtype, &input_tensor_dtype);
  float input_sc;
  int input_zp;
  err += AsConstant(dequantize->args[2], &input_zp);
  err += AsConstant(dequantize->args[1], &input_sc);
  float output_sc;
  int output_zp;
  err += AsConstant(quantize->args[2], &output_zp);
  err += AsConstant(quantize->args[1], &output_sc);

  const auto* attrs = leaky_relu->attrs.as<LeakyReluAttrs>();
  double alpha = attrs->alpha;
  if (alpha >= 1.0f || alpha <= 0.0f) {
    err += EthosnError(
        ErrStrm() << "leaky relu alpha must be less than 1 and greater than 0, but was " << alpha);
    return err;
  }
  params->leaky_relu_info = sl::LeakyReluInfo(alpha, sl::QuantizationInfo(output_zp, output_sc));
  params->input_info = sl::TensorInfo(input_tensor_shape, input_tensor_dtype, sl::DataFormat::NHWC,
                                      sl::QuantizationInfo(input_zp, input_sc));

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(quantize->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = sl::QuantizationInfo(output_zp, output_sc);
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Concatenate(const Expr& expr, ConcatenateParams* params) {
  Call call = Downcast<Call>(expr);
  const auto& attrs = call->attrs.as<ConcatenateAttrs>();
  params->concat_info.m_Axis = attrs->axis;

  float output_sc;
  int output_zp;
  EthosnError err = AsConstant(call->args[3], &output_sc);
  err += AsConstant(call->args[4], &output_zp);
  params->concat_info.m_OutputQuantizationInfo = sl::QuantizationInfo(output_zp, output_sc);

  auto input_scales = call->args[1].as<TupleNode>()->fields;
  auto input_zero_points = call->args[2].as<TupleNode>()->fields;
  auto input_tensors = call->args[0]->checked_type().as<TupleTypeNode>()->fields;

  int index = 0;
  for (auto input_scale : input_scales) {
    auto input_dtype = input_tensors[index].as<TensorTypeNode>();
    auto input_zero_point = input_zero_points[index];
    float scale;
    int zp;
    err += AsConstant(input_scale, &scale);
    err += AsConstant(input_zero_point, &zp);
    sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
    sl::DataType input_data_type;
    err += Tvm2Npu(input_dtype->shape, &input_tensor_shape);
    err += Tvm2Npu(input_dtype->dtype, &input_data_type);
    params->input_infos.emplace_back(sl::TensorInfo(input_tensor_shape, input_data_type,
                                                    sl::DataFormat::NHWC,
                                                    sl::QuantizationInfo(zp, scale)));
    index++;
  }

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(call->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = sl::QuantizationInfo(output_zp, output_sc);
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Split(const Expr& expr, SplitParams* params) {
  Call call = Downcast<Call>(expr);
  const auto* input_tensor_type = call->args[0]->checked_type().as<TensorTypeNode>();
  const auto& attrs = call->attrs.as<SplitAttrs>();

  sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
  sl::DataType input_data_type;
  EthosnError err = Tvm2Npu(input_tensor_type->shape, &input_tensor_shape);
  err += Tvm2Npu(input_tensor_type->dtype, &input_data_type);
  params->input_info =
      sl::TensorInfo(input_tensor_shape, input_data_type, params->input_info.m_DataFormat,
                     params->input_info.m_QuantizationInfo);
  params->split_info.m_Axis = attrs->axis;
  if (attrs->indices_or_sections->IsInstance<IntImmNode>()) {
    auto sections = Downcast<IntImm>(attrs->indices_or_sections)->value;
    int size = input_tensor_shape[attrs->axis] / sections;
    for (int i = 0; i < sections; i++) {
      params->split_info.m_Sizes.push_back(size);
    }
  } else {
    auto indices = Downcast<tvm::Array<Integer>>(attrs->indices_or_sections);
    int last_index = 0;
    for (const auto& i : indices) {
      params->split_info.m_Sizes.push_back(i->value - last_index);
      last_index = i->value;
    }
    int axis_size = input_tensor_shape[attrs->axis];
    params->split_info.m_Sizes.push_back(axis_size - last_index);
  }

  Array<Type> output_tensors = call->checked_type().as<TupleTypeNode>()->fields;
  std::vector<sl::TensorInfo> output_infos = {};
  for (auto output_ttype : output_tensors) {
    sl::TensorInfo output_tensor_info;
    err += Tvm2Npu(output_ttype, &output_tensor_info);
    output_tensor_info.m_QuantizationInfo = params->input_info.m_QuantizationInfo;
    output_infos.push_back(output_tensor_info);
  }
  params->output_infos = output_infos;
  return err;
}

EthosnError EthosnAPI::DepthToSpace(const Expr& expr, DepthToSpaceParams* params) {
  Call call = Downcast<Call>(expr);
  const auto* input_dtype = call->args[0]->checked_type().as<TensorTypeNode>();
  const auto* attrs = call->attrs.as<SubPixelAttrs>();
  if (attrs->mode != "DCR") {
    return EthosnError(ErrStrm() << "mode=" << attrs->mode << ", mode must = DCR");
  }
  params->depth_info.m_BlockSize = attrs->block_size;

  sl::TensorShape input_tensor_shape;
  sl::DataType input_data_type;
  sl::DataFormat input_data_format;
  EthosnError err = Tvm2Npu(input_dtype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_dtype->dtype, &input_data_type);
  err += Tvm2Npu(attrs->layout, &input_data_format);
  params->input_info = sl::TensorInfo(input_tensor_shape, input_data_type, input_data_format,
                                      params->input_info.m_QuantizationInfo);

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(call->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = params->input_info.m_QuantizationInfo;
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Relu(const Expr& expr, ReluParams* params) {
  Call call = Downcast<Call>(expr);
  const auto* input_dtype = call->args[0]->checked_type().as<TensorTypeNode>();
  const auto* attrs = call->attrs.as<ClipAttrs>();
  params->relu_info.m_LowerBound = attrs->a_min;
  params->relu_info.m_UpperBound = attrs->a_max;

  sl::TensorShape input_tensor_shape = {1, 1, 1, 1};
  sl::DataType input_data_type;
  EthosnError err = Tvm2Npu(input_dtype->shape, &input_tensor_shape);
  err += Tvm2Npu(input_dtype->dtype, &input_data_type);
  params->input_info =
      sl::TensorInfo(input_tensor_shape, input_data_type, params->input_info.m_DataFormat,
                     params->input_info.m_QuantizationInfo);

  sl::TensorInfo output_tensor_info;
  err += Tvm2Npu(call->checked_type(), &output_tensor_info);
  output_tensor_info.m_QuantizationInfo = params->input_info.m_QuantizationInfo;
  params->output_info = output_tensor_info;

  return err;
}

EthosnError EthosnAPI::Tvm2Npu(const Array<IndexExpr>& padding, sl::Padding* npu_padding) {
  std::array<uint32_t, 4> dim;
  if (EthosnError err = AsArray<IndexExpr, uint32_t>(padding, &dim)) {
    return err;
  }
  switch (padding.size()) {
    case 1:
      *npu_padding = sl::Padding(dim[3], dim[3], dim[3], dim[3]);
      break;
    case 2:
      // Height, width -> top, bottom, left, right
      *npu_padding = sl::Padding(dim[3], dim[3], dim[2], dim[2]);
      break;
    case 4:
      // Top, left, bottom, right -> top, bottom, left, right
      *npu_padding = sl::Padding(dim[0], dim[2], dim[1], dim[3]);
      break;
    default:
      return EthosnError(ErrStrm() << "padding tuple size=" << padding.size()
                                   << ", padding tuple size must be {1, 2, 4}");
  }
  return EthosnError();
}

EthosnError EthosnAPI::Tvm2Npu(const Array<IndexExpr>& strides, sl::Stride* npu_stride) {
  if (strides.size() != 2) {
    return EthosnError(ErrStrm() << "stride size=" << strides.size() << ", stride size must = 2");
  }
  std::array<uint32_t, 2> dim;
  if (EthosnError err = AsArray<IndexExpr, uint32_t>(strides, &dim)) {
    return err;
  }
  *npu_stride = sl::Stride(dim[1], dim[0]);
  return EthosnError();
}

EthosnError EthosnAPI::Tvm2Npu(const Array<IndexExpr>& size, uint32_t* x, uint32_t* y) {
  if (size.size() != 2) {
    return EthosnError(ErrStrm() << "dimensions=" << size.size() << ", dimensions must = 2");
  }
  std::array<uint32_t, 2> dim;
  if (EthosnError err = AsArray<IndexExpr, uint32_t>(size, &dim)) {
    return err;
  }
  *x = dim[0];
  *y = dim[1];
  return EthosnError();
}

EthosnError EthosnAPI::Tvm2Npu(const std::string& dformat, sl::DataFormat* data_format) {
  *data_format = sl::DataFormat::NCHW;
  if (dformat == "NCHW") {
    return EthosnError();
  } else if (dformat == "NHWC") {
    *data_format = sl::DataFormat::NHWC;
    return EthosnError();
  } else if (dformat == "HWIO") {
    *data_format = sl::DataFormat::HWIO;
    return EthosnError();
  } else if (dformat == "HWIM") {
    *data_format = sl::DataFormat::HWIM;
    return EthosnError();
  }
  return EthosnError(ErrStrm() << "format=" << dformat
                               << ", format must be {NCHW, NHWC, HWIO, HWIM}");
}

EthosnError EthosnAPI::Tvm2Npu(const Array<IndexExpr>& shape, sl::TensorShape* npu_shape) {
  EthosnError err = AsArray<IndexExpr, uint32_t>(shape, npu_shape);
  if (npu_shape->front() != 1) {
    err += EthosnError(ErrStrm() << "batch size=" << npu_shape->front() << ", batch size must = 1");
  }
  return err;
}

EthosnError EthosnAPI::Tvm2Npu(const tvm::DataType& dtype, sl::DataType* data_type) {
  *data_type = sl::DataType::INT8_QUANTIZED;
  if (dtype.is_scalar() == 1) {
    if (dtype.is_uint() && dtype.bits() == 8) {
      *data_type = sl::DataType::UINT8_QUANTIZED;
      return EthosnError();
    } else if (dtype.is_int() && dtype.bits() == 8) {
      return EthosnError();
    } else if (dtype.is_int() && dtype.bits() == 32) {
      *data_type = sl::DataType::INT32_QUANTIZED;
      return EthosnError();
    }
  }
  return EthosnError(ErrStrm() << "dtype=\'" << dtype
                               << "\', dtype must be either uint8, int8 or int32");
}

EthosnError EthosnAPI::Tvm2Npu(const int32_t zero_point, const float scale,
                               sl::QuantizationInfo* npu_qinfo) {
  sl::QuantizationInfo q(zero_point, scale);
  *npu_qinfo = q;
  return EthosnError();
}

EthosnError EthosnAPI::Tvm2Npu(const int zero_point, std::valarray<float> scales, unsigned int axis,
                               sl::QuantizationInfo* npu_qinfo) {
  if (scales.size() == 1) {
    sl::QuantizationInfo q(zero_point, scales[0]);
    *npu_qinfo = q;
  } else {
    struct sl::QuantizationScales s(std::move(scales));
    sl::QuantizationInfo q(zero_point, s, axis);
    *npu_qinfo = q;
  }
  return EthosnError();
}

EthosnError EthosnAPI::Tvm2Npu(const Array<Integer>& shape, sl::TensorShape* npu_shape) {
  return AsArray<Integer, uint32_t>(shape, npu_shape);
}

EthosnError EthosnAPI::Tvm2Npu(const Array<Array<Integer>>& padding, sl::Padding* npu_padding) {
  if (padding.size() != 4) {
    return EthosnError(ErrStrm() << "padding tuple size=" << padding.size()
                                 << ", padding tuple size must = 4");
  }
  Array<IndexExpr> reduced_padding;
  reduced_padding.push_back(padding[1][0]);
  reduced_padding.push_back(padding[1][1]);
  reduced_padding.push_back(padding[2][0]);
  reduced_padding.push_back(padding[2][1]);
  std::array<uint32_t, 4> dim;
  if (EthosnError err = AsArray<IndexExpr, uint32_t>(reduced_padding, &dim)) {
    return err;
  }
  *npu_padding = sl::Padding(dim[0], dim[1], dim[2], dim[3]);
  return EthosnError();
}

EthosnError EthosnAPI::Tvm2Npu(const tvm::Type& type, sl::TensorInfo* npu_tinfo) {
  const TensorTypeNode* ttype = type.as<TensorTypeNode>();
  ICHECK(ttype) << "Expected TensorTypeNode but was " << ttype->GetTypeKey();

  sl::TensorShape shape = {1, 1, 1, 1};
  sl::DataType data_type;
  EthosnError err = Tvm2Npu(ttype->shape, &shape);
  err += Tvm2Npu(ttype->dtype, &data_type);
  *npu_tinfo = sl::TensorInfo(shape, data_type, sl::DataFormat::NHWC, {});
  return err;
}

// Convert an array of IntImmNodes into ValueT
// IndexT type of Array indexing variable
// ValueT type of resulting value
// N The size of the output array
template <typename IndexT, typename ValueT, size_t N>
EthosnError EthosnAPI::AsArray(const Array<IndexT>& arr, std::array<ValueT, N>* v) {
  if (arr.size() > N)
    return EthosnError(ErrStrm() << "dimensions=" << arr.size() << ", dimensions must be <= " << N);
  for (size_t i = 0; i < arr.size(); i++) {
    const PrimExpr& a = arr[i];
    const auto* intImm = a.as<IntImmNode>();
    if (intImm->value > std::numeric_limits<ValueT>::max()) {
      return EthosnError(ErrStrm() << "axis size=" << intImm->value << ", axis size must be <= "
                                   << std::numeric_limits<ValueT>::max());
    }
    (*v)[i] = static_cast<ValueT>(intImm->value);
  }
  return EthosnError();
}

// Get a std::valarray from a constant represented by a NDArray.
EthosnError EthosnAPI::AsConstant(const Expr& expr, std::valarray<float>* out) {
  if (!expr->IsInstance<ConstantNode>()) {
    return EthosnError("expected constant data");
  }
  const auto* data = expr.as<ConstantNode>();
  int64_t num_elems = 1;
  auto shape = data->data.Shape();
  for (size_t i = 0; i < shape.size(); i++) {
    num_elems *= shape[i];
  }
  out->resize(num_elems);
  for (int64_t i = 0; i < num_elems; i++) {
    (*out)[i] = static_cast<float*>(data->data->data)[i];
  }
  return EthosnError();
}

// Get a T from a constant represented by a NDArray.
template <typename T>
EthosnError EthosnAPI::AsConstant(const Expr& expr, T* out) {
  *out = {0};
  if (!expr->IsInstance<ConstantNode>()) {
    return EthosnError("expected constant data");
  }
  runtime::NDArray data = Downcast<Constant>(expr)->data;
  *out = *static_cast<T*>(data->data);
  return EthosnError();
}

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
