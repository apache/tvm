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
 * \file src/relay/qnn/op/convolution.cc
 * \brief Property def of qnn convolution operator.
 */
#include "../../op/nn/convolution.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_layout.h>

#include "../../transforms/pattern_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.conv2d

bool QnnConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  // Expected Types: data, weight, input_zero_point, weight_zero_point, input_scale, weight_scale,
  // out_type
  ICHECK_EQ(types.size(), 7);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr || weight == nullptr) return false;
  const auto* param = attrs.as<Conv2DAttrs>();
  ICHECK(param != nullptr) << "Conv2DAttrs cannot be nullptr.";
  ICHECK(data->dtype == DataType::Int(8) || data->dtype == DataType::UInt(8) ||
         data->dtype == DataType::Int(16))
      << "Expected qnn conv2d type(int8, uint8, int16) for input but was " << data->dtype;
  ICHECK(weight->dtype == DataType::Int(8) || weight->dtype == DataType::UInt(8) ||
         weight->dtype == DataType::Int(16))
      << "Expected qnn conv2d type(int8, uint8, int16) for weight but was " << weight->dtype;
  ICHECK(param->out_dtype == DataType::Int(16) || param->out_dtype == DataType::Int(32) ||
         param->out_dtype == DataType::Int(64))
      << "Expected qnn conv2d type(int16, int32, int64) for output but was " << param->out_dtype;
  ICHECK(param->out_dtype.bits() > 0) << "Output dtype bits should be greater than 0.";

  // Check the types of scale and zero points.
  for (size_t i = 2; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // input_zero_point
  ICHECK(IsScalarType(types[4], DataType::Float(32)));  // input_scale
  // Kernel scale can be a vector of length output_channels or a scalar.
  if (param->groups == 1) {
    size_t axis = param->kernel_layout.operator std::string().find('O');
    ICHECK(axis != std::string::npos) << "Kernel layout attribute is not defined";
    AssignType(types[5], DataType::Float(32), weight->shape[axis], reporter);  // weight_scale
  } else {
    size_t o_axis = param->kernel_layout.operator std::string().find('O');
    size_t i_axis = param->kernel_layout.operator std::string().find('I');
    size_t d_axis = param->data_layout.operator std::string().find('C');
    ICHECK(o_axis != std::string::npos || i_axis != std::string::npos)
        << "Kernel layout attribute is not defined";
    ICHECK(d_axis != std::string::npos) << "Data layout attribute is not defined";
    if (param->channels.defined() &&
        tvm::tir::ExprDeepEqual()(param->groups, data->shape[d_axis]) &&
        tvm::tir::ExprDeepEqual()(param->channels, weight->shape[i_axis])) {
      // This is depthwise convolution
      // Here, total number of output channels depend on depth multiplier.
      AssignType(types[5], DataType::Float(32), weight->shape[i_axis] * weight->shape[o_axis],
                 reporter);  // weight_scale
    } else {
      // This is grouped convolution
      AssignType(types[5], DataType::Float(32), param->channels,
                 reporter);  // weight_scale
    }
  }

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // Conv2D infer type function.
  Array<Type> tensor_types = {types[0], types[1], types[6]};
  return Conv2DRel(tensor_types, 3, attrs, reporter);
}

InferCorrectLayoutOutput QnnConvInferCorrectLayout(const Attrs& attrs,
                                                   const Array<Layout>& new_in_layouts,
                                                   const Array<Layout>& old_in_layouts,
                                                   const Array<tvm::relay::Type>& old_in_types) {
  // Use Relay Conv2D Infer correct layout.
  auto conv_new_layouts =
      ConvInferCorrectLayout<Conv2DAttrs>(attrs, new_in_layouts, old_in_layouts, old_in_types);

  // Fill the layouts of remaining input tensors - scales and zero points. The layouts of these
  // tensors can be treated as channel layout.
  Layout channel_layout = Layout("C");
  Array<Layout> input_layouts = {conv_new_layouts->input_layouts[0],
                                 conv_new_layouts->input_layouts[1],
                                 channel_layout,
                                 channel_layout,
                                 channel_layout,
                                 channel_layout};
  Array<Layout> output_layouts = conv_new_layouts->output_layouts;
  return InferCorrectLayoutOutput(input_layouts, output_layouts, attrs);
}

bool is_depthwise(const Conv2DAttrs* param) {
  return param->channels.defined() && tvm::tir::ExprDeepEqual()(param->channels, param->groups) &&
         param->groups != 1;
}

// Workload - batch_size, in_channels, out_channels, kernel_h, kernel_w, channel_multiplier
using WorkloadType = std::tuple<int, int, int, int, int, int>;

/*
 * \brief Get the conv parameters like batch_size, kernel_height etc.
 * \param ref_call The original callnode.
 * \param param The qnn conv2d attributes.
 * \return A tuple of workload.
 */
WorkloadType GetWorkload(const Array<tvm::relay::Type>& arg_types, const Conv2DAttrs* param) {
  // Get conv parameters.
  const auto in_shape = get_shape(arg_types[0]);
  int batch_size, in_channels;
  if (param->data_layout == "NCHW") {
    batch_size = get_const_int(in_shape[0]);
    in_channels = get_const_int(in_shape[1]);
  } else if (param->data_layout == "NHWC") {
    batch_size = get_const_int(in_shape[0]);
    in_channels = get_const_int(in_shape[3]);
  } else {
    LOG(FATAL) << "qnn.conv2d does not support " << param->data_layout << " layout";
  }

  const auto kernel_shape = get_shape(arg_types[1]);
  int out_channels, kernel_h, kernel_w;
  int channel_multiplier = -1;
  bool depthwise = is_depthwise(param);
  int index_k = 0;
  int index_h = 2;
  int index_w = 3;
  int index_c = 1;
  if (param->kernel_layout == "HWIO") {
    index_k = 3;
    index_h = 0;
    index_w = 1;
    index_c = 2;
  } else if (param->kernel_layout == "HWOI") {
    index_k = 2;
    index_h = 0;
    index_w = 1;
    index_c = 3;
  } else if (param->kernel_layout == "OHWI") {
    index_k = 0;
    index_h = 1;
    index_w = 2;
    index_c = 3;
  } else if (param->kernel_layout != "OIHW") {
    LOG(FATAL) << "qnn.conv2d does not support " << param->kernel_layout << " layout";
  }

  kernel_h = get_const_int(kernel_shape[index_h]);
  kernel_w = get_const_int(kernel_shape[index_w]);
  out_channels = get_const_int(kernel_shape[index_k]);

  if (depthwise) {
    channel_multiplier = get_const_int(kernel_shape[index_c]);
  }

  return std::make_tuple(batch_size, in_channels, out_channels, kernel_h, kernel_w,
                         channel_multiplier);
}

/*
 * \brief Fallback to simpler lowering for dilation (when non-zero kernel point) or grouped conv.
 * \param data The input expr.
 * \param weight The weight expr.
 * \param input_zero_point The input zero point expr.
 * \param kernel_zero_point The kernel zero point expr.
 * \param param The qnn conv2d attributes.
 * \return The fallback lowered sequence of Relay expr.
 * \note In case of dilation with non-zero kernel zero point, normal lowering would require a
 * dilated pool. Since, we don't have dilated pool, we fallback to a simpler sequence of Relay
 * operations. This will potentially lead to performance degradation as the convolution is called on
 * int32 tensors instead of int8 tensors.
 */
Expr Conv2DFallBack(const Expr& data, const Expr& weight, const Expr& input_zero_point,
                    const Expr& kernel_zero_point, const Conv2DAttrs* param) {
  // Upcast the parameters to be at least int32 to avoid overflow
  auto upcast_bits = param->out_dtype.bits() < 32 ? 32 : param->out_dtype.bits();

  auto zp_data = Cast(input_zero_point, DataType::Int(upcast_bits));
  auto zp_kernel = Cast(kernel_zero_point, DataType::Int(upcast_bits));

  auto shifted_data = Cast(data, DataType::Int(upcast_bits));
  auto zero_scalar = MakeConstantScalar(DataType::Int(upcast_bits), 0);
  if (!IsEqualScalar(input_zero_point, zero_scalar)) {
    shifted_data = Subtract(Cast(data, DataType::Int(upcast_bits)), zp_data);
  }

  auto shifted_kernel = Cast(weight, DataType::Int(upcast_bits));
  if (!IsEqualScalar(kernel_zero_point, zero_scalar)) {
    shifted_kernel = Subtract(Cast(weight, DataType::Int(upcast_bits)), zp_kernel);
  }

  return Conv2D(shifted_data, shifted_kernel, param->strides, param->padding, param->dilation,
                param->groups, param->channels, param->kernel_size, param->data_layout,
                param->kernel_layout, param->out_layout, param->out_dtype);
}

/*
 * \brief Pad the input data.
 * \param data The input expr.
 * \param input_zero_point The input zero point expr.
 * \return The padded input expr.
 * \note For quantized convolution, the input has to be padded with zero point
 *       instead of zero. This might lead to performance degradation as pad
 *       cannot be fused with conv in Relay. In case we see performance
 *       degradation, we can change the conv2D API to accept a pad_const value.
 */
Expr Conv2DPadInput(const Expr& data, const Expr& input_zero_point, const Conv2DAttrs* param) {
  // 1) Pad the input data
  auto padded_data = data;
  auto pad_top_value = get_const_int(param->padding[0]);
  auto pad_left_value = get_const_int(param->padding[1]);
  auto pad_bottom_value = get_const_int(param->padding[2]);
  auto pad_right_value = get_const_int(param->padding[3]);
  bool do_pad =
      pad_top_value != 0 || pad_left_value != 0 || pad_bottom_value != 0 || pad_right_value != 0;
  if (do_pad) {
    Array<IndexExpr> pad_n({0, 0});
    Array<IndexExpr> pad_c({0, 0});
    Array<IndexExpr> pad_h({param->padding[0], param->padding[2]});
    Array<IndexExpr> pad_w({param->padding[1], param->padding[3]});

    Array<Array<IndexExpr>> pad_width;
    if (param->data_layout == "NCHW") {
      pad_width = {pad_n, pad_c, pad_h, pad_w};
    } else if (param->data_layout == "NHWC") {
      pad_width = {pad_n, pad_h, pad_w, pad_c};
    } else {
      LOG(FATAL) << "qnn.conv2d does not support " << param->data_layout << " layout";
    }
    padded_data = Pad(data, pad_width, input_zero_point, "constant");
  }
  return padded_data;
}

/*
 * \brief Calculates the second term in the qnn.conv2d depthwise lowering sequence.
 * \param padded_data The padded data expr.
 * \param kernel_zero_point The kernel zero point expr.
 * \param param The qnn conv2d attributes.
 * \param kernel_h The height of kernel.
 * \param kernel_w The width of kernel.
 * \param channel_multiplier The channel/depth multiplier.
 * \return The sequence of Relay operators for term2.
 * \note The term2 looks like this
 *
 *       Sigma(r, s) zp_w * Qa(n, oc/cm, oh + r, ow + s)
 *
 *       Second term is not directly representable by one Relay operator.
 *       However, deeper analysis shows that we can reduce r,s using avg_pool2d,
 *       followed by repeat on the C axis by cm times.
 */
Expr DepthwiseConv2DSecondTerm(const Expr& padded_data, const Expr& kernel_zero_point,
                               const Conv2DAttrs* param, int kernel_h, int kernel_w,
                               int channel_multiplier) {
  auto casted_t2 = Cast(padded_data, DataType::Int(32));

  // We can reduce the H and W axis by using avg_pool2d. However, avg_pool2d averages the sum.
  // Since, this is integer division (floor), we can first multiply the data by the pool_size and
  // then perform avg_pool2d. Reversing this causes inaccuracy due to floor division. If the
  // pool_size and strides are 1x1, we don't need avg_pool2d.
  auto reduced_t2 = casted_t2;
  if (kernel_h * kernel_w != 1) {
    auto scaled_hw_t2 =
        Multiply(casted_t2, MakeConstantScalar(DataType::Int(32), kernel_h * kernel_w));
    Array<IndexExpr> padding({0, 0});
    reduced_t2 = AvgPool2D(scaled_hw_t2, param->kernel_size, param->strides, param->dilation,
                           padding, param->data_layout,
                           "",      // out_layout
                           false,   // ceil_mode
                           false);  // count_include_pad
  } else {
    int stride1 = get_const_int(param->strides[0]);
    int stride2 = get_const_int(param->strides[1]);
    if (stride1 * stride2 != 1) {
      Array<IndexExpr> padding({0, 0});
      reduced_t2 = AvgPool2D(reduced_t2, param->kernel_size, param->strides, param->dilation,
                             padding, param->data_layout,
                             "",      // out_layout
                             false,   // ceil_mode
                             false);  // count_include_pad
    }
  }

  auto multiplied_t2 = reduced_t2;
  auto one_scalar = MakeConstantScalar(DataType::Int(32), 1);
  if (!IsEqualScalar(kernel_zero_point, one_scalar)) {
    if (!IsConstScalar(kernel_zero_point)) {
      multiplied_t2 = Multiply(MakeRepeat(kernel_zero_point, channel_multiplier, 0), reduced_t2);
    } else {
      multiplied_t2 = Multiply(kernel_zero_point, reduced_t2);
    }
  }

  // Reduce the C dimension. Find the dimension.
  int axis_t2 = 0;
  if (param->data_layout == "NCHW") {
    axis_t2 = 1;
  } else if (param->data_layout == "NHWC") {
    axis_t2 = 3;
  } else {
    LOG(FATAL) << "qnn.conv2d does not support " << param->data_layout << " layout";
  }
  auto repeated_t2 = multiplied_t2;
  if (channel_multiplier != 1) {
    repeated_t2 = MakeRepeat(multiplied_t2, channel_multiplier, axis_t2);
  }
  return repeated_t2;
}

/*
 * \brief Calculates the third term in the qnn.conv2d depthwise lowering sequence.
 * \param weight The weight expr.
 * \param input_zero_point The input zero point expr.
 * \param param The qnn conv2d attributes.
 * \param out_channels The number of output channels.
 * \param channel_multiplier The channel/depth multiplier.
 * \return The sequence of Relay operatos for term3.
 * \note The term3 looks like this
 *
 *       Sigma(r, s) zp_a * Qw(oc/m, oc%m, r, s)
 *
 *       This can be achieved by calling reduce on r and s axis. The tensor can be then reshaped to
 *       (1, oc, 1, 1) as (oc/m, oc%m) are just contiguous memory locations.
 */
Expr DepthwiseConv2DThirdTerm(const Expr& weight, const Expr& input_zero_point,
                              const Conv2DAttrs* param, int out_channels, int channel_multiplier) {
  // Find which dimensions are R, S.
  Array<Integer> axes_t3;
  if (param->kernel_layout == "OIHW") {
    // For OIHW kernel layout, HW are reduce axis
    axes_t3 = {2, 3};
  } else if (param->kernel_layout == "HWIO") {
    axes_t3 = {0, 1};
  } else if (param->kernel_layout == "HWOI") {
    axes_t3 = {0, 1};
  } else {
    LOG(FATAL) << "qnn.conv2d does not support " << param->kernel_layout << " layout";
  }
  auto reduced_t3 = Sum(Cast(weight, DataType::Int(32)), axes_t3, false, false);

  // Find the newshape depending on NCHW/NHWC layout.
  Array<Integer> newshape;
  if (param->data_layout == "NCHW") {
    newshape = {1, out_channels * channel_multiplier, 1, 1};
  } else if (param->data_layout == "NHWC") {
    newshape = {1, 1, 1, out_channels * channel_multiplier};
  } else {
    LOG(FATAL) << "qnn.conv2d does not support " << param->data_layout << " layout";
  }
  auto reshaped_t3 = Reshape(reduced_t3, newshape);

  auto one_scalar = MakeConstantScalar(DataType::Int(32), 1);
  if (IsEqualScalar(input_zero_point, one_scalar)) {
    return reshaped_t3;
  }
  return Multiply(input_zero_point, reshaped_t3);
}

/*
 * \brief Calculates the fourth term in the qnn.conv2d depthwise lowering sequence.
 * \param input_zero_point_int The int value of input zero point.
 * \param kernel_zero_point_int The int value of kernel zero point.
 * \param kernel_h The height of kernel.
 * \param kernel_w The width of kernel.
 * \return The sequence of Relay operators for term4.
 * \note The term4 looks like this
 *
 *       Sigma(r, s) zp_a * zp_w
 */
Expr DepthwiseConv2DFourthTerm(int input_zero_point_int, int kernel_zero_point_int, int kernel_h,
                               int kernel_w) {
  int scalar_term4 = input_zero_point_int * kernel_zero_point_int * kernel_h * kernel_w;
  return MakeConstantScalar(DataType::Int(32), scalar_term4);
}

/*
 * \brief Calculates the fourth term in the qnn.conv2d depthwise lowering sequence
          for non-constant zero_points.
 * \param input_zero_point The Expr for the input zero point.
 * \param kernel_zero_point The Expr for the kernel zero point.
 * \param kernel_h The height of kernel.
 * \param kernel_w The width of kernel.
 * \return The sequence of Relay operators for term4.
 * \note The term4 looks like this
 *
 *       Sigma(r, s) zp_a * zp_w
 */
Expr DepthwiseConv2DFourthTerm(const Expr& input_zero_point, const Expr& kernel_zero_point,
                               int kernel_h, int kernel_w) {
  Expr scalar_term4 = MakeConstantScalar(DataType::Int(32), kernel_h * kernel_w);
  Expr variable_term4 = Multiply(input_zero_point, kernel_zero_point);
  return Multiply(scalar_term4, variable_term4);
}

/*
 * \brief Calculates the first term in the qnn.conv2d lowering sequence.
 * \param data The input expr.
 * \param weight The weight expr.
 * \param param The qnn conv2d attributes.
 * \return The sequence of Relay operators for term1.
 * \note The term1 is
 *       Sigma(c,r,s) QW(k, c, r, s) * QA(n, c, h + r, w + s)
 *       This is just conv2d on int tensors.
 */
Expr Conv2DFirstTerm(const Expr& padded_data, const Expr& weight, const Conv2DAttrs* param) {
  // Lowering for Term 1
  Array<IndexExpr> padding({0, 0, 0, 0});
  return Conv2D(padded_data, weight, param->strides, padding, param->dilation, param->groups,
                param->channels, param->kernel_size, param->data_layout, param->kernel_layout,
                param->out_layout, param->out_dtype);
}

/*
 * \brief Calculates the second term in the qnn.conv2d lowering sequence.
 * \param padded_data The padded data expr.
 * \param kernel_zero_point The kernel zero point expr.
 * \param param The qnn conv2d attributes.
 * \param kernel_h The height of kernel.
 * \param kernel_w The width of kernel.
 * \return The sequence of Relay operators for term2.
 * \note The term2 looks like this
 *
 *       Sigma(c,r,s) zp_w * QA(n, c, h + r, w + s)
 *
 *       Second term is not directly representable by one Relay operator.
 *       However, deeper analysis shows that we can reduce r,s using avg_pool2d,
 *       followed by a reduce on the C axis. Using avg_pool2d also gives an
 *       opportunity to reuse alter_op_layout infrastructure.
 */
Expr Conv2DSecondTerm(const Expr& padded_data, const Expr& kernel_zero_point,
                      const Conv2DAttrs* param, int kernel_h, int kernel_w, int out_channels) {
  auto casted_t2 = Cast(padded_data, DataType::Int(32));

  // We can reduce the H and W axis by using avg_pool2d. However, avg_pool2d averages the sum.
  // Since, this is integer division (floor), we can first multiply the data by the pool_size and
  // then perform avg_pool2d. Reversing this causes inaccuracy due to floor division.
  Array<IndexExpr> padding({0, 0});

  // Reduce the C dimension. Find the dimension.
  Array<Integer> axes_t2;
  if (param->data_layout == "NCHW") {
    axes_t2 = {1};
  } else if (param->data_layout == "NHWC") {
    axes_t2 = {3};
  } else {
    LOG(FATAL) << "qnn.conv2d does not support " << param->data_layout << " layout";
  }
  // Keep dims true to retain 4D tensor
  auto reduced_c_t2 = Sum(casted_t2, axes_t2, true, false);

  // If the pool_size and strides are 1x1, we don't need avg_pool2d.
  auto reduced_t2 = reduced_c_t2;
  if (kernel_h * kernel_w != 1) {
    reduced_c_t2 =
        Multiply(reduced_c_t2, MakeConstantScalar(DataType::Int(32), kernel_h * kernel_w));
    reduced_t2 = AvgPool2D(reduced_c_t2, param->kernel_size, param->strides, param->dilation,
                           padding, param->data_layout,
                           "",      // out_layout
                           false,   // ceil_mode
                           false);  // count_include_pad
  } else {
    int stride1 = get_const_int(param->strides[0]);
    int stride2 = get_const_int(param->strides[1]);
    if (stride1 * stride2 != 1) {
      reduced_t2 = AvgPool2D(reduced_c_t2, param->kernel_size, param->strides, param->dilation,
                             padding, param->data_layout,
                             "",      // out_layout
                             false,   // ceil_mode
                             false);  // count_include_pad
    }
  }

  auto multiplied_t2 = reduced_t2;
  auto one_scalar = MakeConstantScalar(DataType::Int(32), 1);
  if (!IsEqualScalar(kernel_zero_point, one_scalar)) {
    if (!IsConstScalar(kernel_zero_point)) {
      Layout layout(param->data_layout);
      int channel_axis = layout.IndexOf(LayoutAxis::Get('C'));
      reduced_t2 = MakeRepeat(reduced_t2, out_channels, channel_axis);
    }
    multiplied_t2 = Multiply(kernel_zero_point, reduced_t2);
  }
  return multiplied_t2;
}

/*
 * \brief Calculates the third term in the qnn.conv2d lowering sequence.
 * \param weight The weight expr.
 * \param input_zero_point The input zero point expr.
 * \param param The qnn conv2d attributes.
 * \param out_channels The number of output channels.
 * \return The sequence of Relay operators for term3.
 * \note The term3 looks like this
 *
 *       Sigma(c,r,s) zp_a * QW(k, c, r, s)
 *
 *       This can be achieved by calling reduce on c, r and s axis, resulting in
 *       a 1D tensor. The tensor is then reshaped to conform to NHWC/NCHW
 *       format.
 */
Expr Conv2DThirdTerm(const Expr& weight, const Expr& input_zero_point, const Conv2DAttrs* param,
                     int out_channels) {
  // Find which dimensions are C, R, S.
  Array<Integer> axes_t3;
  if (param->kernel_layout == "OIHW") {
    // For OIHW kernel layout, IHW are reduce axis
    axes_t3 = {1, 2, 3};
  } else if (param->kernel_layout == "HWIO") {
    axes_t3 = {0, 1, 2};
  } else if (param->kernel_layout == "HWOI") {
    axes_t3 = {0, 1, 3};
  } else if (param->kernel_layout == "OHWI") {
    axes_t3 = {1, 2, 3};
  } else {
    LOG(FATAL) << "qnn.conv2d does not support " << param->kernel_layout << " layout";
  }
  auto reduced_t3 = Sum(Cast(weight, DataType::Int(32)), axes_t3, false, false);

  // Find the newshape depending on NCHW/NHWC layout.
  Array<Integer> newshape;
  if (param->data_layout == "NCHW") {
    newshape = {1, out_channels, 1, 1};
  } else if (param->data_layout == "NHWC") {
    newshape = {1, 1, 1, out_channels};
  } else {
    LOG(FATAL) << "qnn.conv2d does not support " << param->data_layout << " layout";
  }
  auto reshaped_t3 = Reshape(reduced_t3, newshape);

  auto one_scalar = MakeConstantScalar(DataType::Int(32), 1);
  if (IsEqualScalar(input_zero_point, one_scalar)) {
    return reshaped_t3;
  }
  return Multiply(input_zero_point, reshaped_t3);
}

/*
 * \brief Calculates the fourth term in the qnn.conv2d lowering sequence.
 * \param input_zero_point_int The int value of input zero point.
 * \param kernel_zero_point_int The int value of kernel zero point.
 * \param in_channels The number of input channels.
 * \param kernel_h The height of kernel.
 * \param kernel_w The width of kernel.
 * \param param The qnn conv2d attributes.
 * \return The sequence of Relay operators for term4.
 * \note The term4 looks like this
 *
 *       Sigma(c,r,s) zp_a * zp_w
 *
 */
Expr Conv2DFourthTerm(int input_zero_point_int, int kernel_zero_point_int, int in_channels,
                      int kernel_h, int kernel_w, const Conv2DAttrs* param) {
  auto upcast_bits = param->out_dtype.bits() < 32 ? 32 : param->out_dtype.bits();
  int scalar_term4 =
      input_zero_point_int * kernel_zero_point_int * in_channels * kernel_h * kernel_w;
  return MakeConstantScalar(DataType::Int(upcast_bits), scalar_term4);
}

/*
 * \brief Calculates the fourth term in the qnn.conv2d lowering sequence
          for non-constant zero_points.
 * \param input_zero_point The Expr for the input zero point.
 * \param kernel_zero_point The Expr for the kernel zero point.
 * \param in_channels The number of input channels.
 * \param kernel_h The height of kernel.
 * \param kernel_w The width of kernel.
 * \param param The qnn conv2d attributes.
 * \return The sequence of Relay operators for term4.
 * \note The term4 looks like this
 *
 *       Sigma(c,r,s) zp_a * zp_w
 *
 */
Expr Conv2DFourthTerm(const Expr& input_zero_point, const Expr& kernel_zero_point, int in_channels,
                      int kernel_h, int kernel_w, const Conv2DAttrs* param) {
  auto upcast_bits = param->out_dtype.bits() < 32 ? 32 : param->out_dtype.bits();
  Expr scalar_term4 =
      MakeConstantScalar(DataType::Int(upcast_bits), in_channels * kernel_h * kernel_w);
  Expr variable_term4 = Multiply(input_zero_point, kernel_zero_point);
  return Multiply(scalar_term4, variable_term4);
}

/*
 * \brief Combines different terms of qnn conv2d lowering.
 * \param term1 The term1 of qnn conv2d lowering.
 * \param term2 The term2 of qnn conv2d lowering.
 * \param term3 The term3 of qnn conv2d lowering.
 * \param term4 The term4 of qnn conv2d lowering.
 * \param input_zero_point_int The int value of input zero point.
 * \param kernel_zero_point_int The int value of kernel zero point.
 * \param param The qnn conv2d attributes.
 * \return The combined sequence of relay operations.
 * \note The combined operation looks like this
 *
 *       Sigma(c,r,s) QW(k, c, r, s) * QA(n, c, h + r, w + s)  // Term1
 *     - Sigma(c,r,s) zp_w * QA(n, c, h + r, w + s)            // Term2
 *     - Sigma(c,r,s) zp_a * QW(k, c, r, s)                    // Term3
 *     + Sigma(c,r,s) zp_a * zp_w                              // Term4
 *
 */
Expr Conv2DCombineTerms(const Expr& term1, const Expr& term2, const Expr& term3, const Expr& term4,
                        int input_zero_point_int, int kernel_zero_point_int) {
  if (input_zero_point_int == 0 && kernel_zero_point_int == 0) {
    // term 2, 3 and 4 become zero.
    return term1;
  } else if (input_zero_point_int == 0 && kernel_zero_point_int != 0) {
    // term 3 and term 4 become zero.
    return Subtract(term1, term2);
  } else if (input_zero_point_int != 0 && kernel_zero_point_int == 0) {
    // term 2 and term 4 become zero.
    return Subtract(term1, term3);
  } else {
    auto data_term = Subtract(term1, term2);
    // Putting constant terms together, so that constant folding can fold it.
    auto const_term = Subtract(term4, term3);
    return Add(data_term, const_term);
  }
}

/*
 * \brief Forward rewrite the qnn conv2d op.
 * \param attrs The QNN conv2d attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for qnn cov2d op.
 * \node Lowering of the qnn.conv2d operator
 *       A quantized tensor is represented in following manner
 *          A = scale_a x (QA - zp_A)
 *       where QA is quantized tensor, scale_a and zp_A are quantization
 *       params.
 *
 *       Quantized convolution will convolve two quantized tensors and returns a
 *       quantized tensor of default dtype of int32, with scale equaling to the
 *       product of scales of input tensors, and a zero point of zero.
 *
 *       For symmetric quantization, the zp_* for all tensors is 0. So, the
 *       lowering of qnn.conv2d is
 *
 *          QA(n, ic, oh + r, ow + s) (conv) QW(oc, ic, r, s)
 *
 *       For asymmetric computation, we can perform similar unrolling. We can
 *       find more details at
 *       https://discuss.tvm.ai/t/tf-lite-quantized-conv2d-operator-conversion/2651/8?u=janimesh
 *       The computation gets unrolled into following 4 terms
 *
 *            Sigma(c,r,s) QW(k, c, r, s) * QA(n, c, h + r, w + s)  // Term1
 *          - Sigma(c,r,s) zp_w * QA(n, c, h + r, w + s)            // Term2
 *          - Sigma(c,r,s) zp_a * QW(k, c, r, s)                    // Term3
 *          + Sigma(c,r,s) zp_a * zp_w                              // Term4
 *
 *       Term3 and Term4 can be computed at compile time.
 *
 *       Key points to notice:
 *         1) Padding is done explicitly because the input has to be padded with
 *         zero point. This might leave some performance opportunity at the
 *         table. Can be avoided by modifying conv2d API to accept the
 *         pad_const_value.
 *         2) Second term is not directly representable by one Relay operator.
 *         However, deeper analysis shows that we can reduce r,s using
 *         avg_pool2d, followed by a reduce on the C axis. Using avg_pool2d also
 *         gives an opportunity to reuse alter_op_layout infrastructure.
 *         3) For dilated conv, in current lowering, we need dilated pool. So as
 *         a workaround, we fall back to simpler lowering using int32 conv if
 *         the conv is dilated. We fallback also in case of grouped conv.
 *
 *       For depthwise, we can similarly unroll the computation. The initial compute is as follows
 *       where cm = channel_multiplier
 *
 *       Qc(n, oc, oh, ow) = Sigma(r, s) (Qw(oc/m, oc%/m, r, s) - zp_w)
 *                                     * (Qa(n, oc/cm, oh + r, ow + s) - zp_a)
 *
 *       This can be written as
 *
 *            Sigma(r, s) Qw(oc/m, oc%/m, r, s) * Qa(n, oc/cm, oh + r, ow + s)
 *          - Sigma(r, s) zp_w * Qa(n, oc/cm, oh + r, ow + s)
 *          - Sigma(r, s) zp_a * Qw(oc/m, oc%m, r, s)
 *          - Sigma(r, s) zp_a * zp_w
 *
 *       The whole process can be broken down into following steps
 *       * Assertion checks for existing support, fallback if necessary
 *       * Pad the input.
 *       * Get Term1.
 *       * Get Term2.
 *       * Get Term3.
 *       * Get Term4.
 *       * Combine the terms.
 */
Expr QnnConv2DCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                           const Array<tvm::relay::Type>& arg_types) {
  ICHECK_EQ(new_args.size(), 6);
  Expr data = new_args[0];
  Expr weight = new_args[1];
  Expr input_zero_point = new_args[2];
  Expr kernel_zero_point = new_args[3];
  const auto* param = attrs.as<Conv2DAttrs>();
  ICHECK(param != nullptr);
  // Assertion checks for existing support.
  ICHECK(param->data_layout == "NCHW" || param->data_layout == "NHWC")
      << "qnn.conv2d supports only NCHW/NHWC input data layout.";
  ICHECK(param->kernel_layout == "OIHW" || param->kernel_layout == "HWIO" ||
         param->kernel_layout == "HWOI" || param->kernel_layout == "OHWI")
      << "qnn.conv2d supports only OIHW/HWIO/HWOI/OHWI kernel data layout.";
  ICHECK(param->kernel_size.defined()) << "qnn.conv2d requires kernel size to be specified.";

  auto [batch_size, in_channels, out_channels, kernel_h, kernel_w, channel_multiplier] =
      GetWorkload(arg_types, param);
  (void)batch_size;  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767

  // zero points are allowed to be non-scalar. Let's check if that's the case.
  bool dynamic_zp = false;
  // Use -1 zero point as a default for dynamic.
  int input_zero_point_int = -1;
  int kernel_zero_point_int = -1;

  // Input zero point can either be a constant or a scalar expression.
  if (IsConstScalar(input_zero_point) && (IsConstScalar(kernel_zero_point))) {
    // Extract the integer zero points.
    input_zero_point_int = GetScalarFromConstant<int>(input_zero_point);
    kernel_zero_point_int = GetScalarFromConstant<int>(kernel_zero_point);
  } else {
    // Make kernel_zero_point expression a 1-D tensor for consistent shape.
    kernel_zero_point = Reshape(kernel_zero_point, {
                                                       -1,
                                                   });
    dynamic_zp = true;
  }

  // Fallback to int32 conv if there is dilation with non-zero kernel point or grouped conv2d
  // For dilated conv, if the kernel zero point is non-zero, the pooling operator also has to
  // traverse the elements in dilated manner. Currently, we do not have strided pool. So, in case of
  // dilated conv with non-zero kernel point, we fall back to simpler but slow lowering.

  ICHECK_EQ(param->dilation.size(), 2) << "qnn.conv2d only supports 2D dilation";
  auto dilation_h = get_const_int(param->dilation[0]);
  auto dilation_w = get_const_int(param->dilation[1]);
  // Check if qnn supports the conv2d parameters. If not, fallback to regular conv2d.
  bool supported_dilation = (kernel_zero_point_int == 0) || (dilation_h == 1 && dilation_w == 1);
  bool supported_groups = (param->groups == 1 || is_depthwise(param));
  bool conv2d_params_supported = supported_dilation && supported_groups;

  // If we need to fall back to default conv2d, kernel zp may need to be broadcast to kernel_layout.
  // Otherwise, we broadcast it to data_layout for qnn lowering.
  if (dynamic_zp) {
    if (!conv2d_params_supported) {
      Layout kernel_layout(param->kernel_layout);
      int kernel_axis = kernel_layout.IndexOf(LayoutAxis::Get("O"));
      kernel_zero_point = ExpandBiasToMatchAxis(kernel_zero_point, 4, {kernel_axis});
    } else {
      Layout data_layout(param->data_layout);
      int channel_axis = data_layout.IndexOf(LayoutAxis::Get("C"));
      kernel_zero_point = ExpandBiasToMatchAxis(kernel_zero_point, 4, {channel_axis});
    }
  }

  if (!conv2d_params_supported) {
    return Conv2DFallBack(data, weight, input_zero_point, kernel_zero_point, param);
  } else if (is_depthwise(param)) {
    ICHECK_NE(channel_multiplier, -1);
    auto padded_data = Conv2DPadInput(data, input_zero_point, param);
    auto term1 = Conv2DFirstTerm(padded_data, weight, param);
    auto term2 = DepthwiseConv2DSecondTerm(padded_data, kernel_zero_point, param, kernel_h,
                                           kernel_w, channel_multiplier);
    auto term3 =
        DepthwiseConv2DThirdTerm(weight, input_zero_point, param, out_channels, channel_multiplier);
    Expr term4;
    if (dynamic_zp) {
      term4 = DepthwiseConv2DFourthTerm(input_zero_point, kernel_zero_point, kernel_h, kernel_w);
    } else {
      term4 = DepthwiseConv2DFourthTerm(input_zero_point_int, kernel_zero_point_int, kernel_h,
                                        kernel_w);
    }
    return Conv2DCombineTerms(term1, term2, term3, term4, input_zero_point_int,
                              kernel_zero_point_int);
  }

  auto padded_data = Conv2DPadInput(data, input_zero_point, param);
  auto term1 = Conv2DFirstTerm(padded_data, weight, param);
  auto term2 =
      Conv2DSecondTerm(padded_data, kernel_zero_point, param, kernel_h, kernel_w, out_channels);
  auto term3 = Conv2DThirdTerm(weight, input_zero_point, param, out_channels);
  Expr term4;
  if (dynamic_zp) {
    term4 = Conv2DFourthTerm(input_zero_point, kernel_zero_point, in_channels, kernel_h, kernel_w,
                             param);
  } else {
    term4 = Conv2DFourthTerm(input_zero_point_int, kernel_zero_point_int, in_channels, kernel_h,
                             kernel_w, param);
  }
  return Conv2DCombineTerms(term1, term2, term3, term4, input_zero_point_int,
                            kernel_zero_point_int);
}

// Positional relay function to create quantized conv2d operator
// used by frontend FFI.
Expr MakeQnnConv2D(Expr data, Expr weight, Expr input_zero_point, Expr kernel_zero_point,
                   Expr input_scale, Expr kernel_scale, Array<IndexExpr> strides,
                   Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
                   IndexExpr channels, Array<IndexExpr> kernel_size, String data_layout,
                   String kernel_layout, String out_layout, DataType out_dtype) {
  auto attrs = make_object<Conv2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("qnn.conv2d");
  return Call(op, {data, weight, input_zero_point, kernel_zero_point, input_scale, kernel_scale},
              Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.conv2d")
    .describe(R"code(2D quantized convolution layer.
This operator convolves quantized weight with quantized data. The scale of the
output quantized tensor is the product of the weight_scale and input_scale of
the input quantized tensors. The zero point of the output quantized tensor is
0. By default, the dtype of output is int32. Please also refer to Requantize
operator to understand how to scale back the int32 output to (u)int8 or (u)int16.
- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DAttrs>()
    .set_num_inputs(6)
    .add_argument("data", "Tensor", "The quantized input data tensor.")
    .add_argument("weight", "Tensor", "The quantized weight tensor.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("weight_scale", "Tensor", "The quantization scale of the weight tensor.")
    .add_argument("weight_zero_point", "Tensor",
                  "The quantization zero_point of the weight tensor.")
    .set_support_level(11)
    .add_type_rel("QnnConv2D", QnnConv2DRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnConv2DCanonicalize)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", QnnConvInferCorrectLayout)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.conv2d").set_body_typed(MakeQnnConv2D);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
