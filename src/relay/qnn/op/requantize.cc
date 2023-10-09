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
 * \file src/relay/qnn/op/requantize.cc
 * \brief QNN requantize operator.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../op/op_common.h"
#include "../../transforms/infer_layout_utils.h"
#include "../../transforms/pattern_utils.h"
#include "../utils.h"
#include "./requantize_config.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(RequantizeAttrs);

InferCorrectLayoutOutput RequantizeInferCorrectLayout(const Attrs& attrs,
                                                      const Array<Layout>& new_in_layouts,
                                                      const Array<Layout>& old_in_layouts,
                                                      const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<RequantizeAttrs>();
  ICHECK(attrs_ptr);
  ObjectPtr<RequantizeAttrs> param = make_object<RequantizeAttrs>(*attrs_ptr);

  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }

  Array<Layout> input_layouts, output_layouts;
  if (new_in_layouts.defined()) {
    // Adapt to new layout. The axis has to change.
    // Record original reduce axis. Convert to the modified layout axis.
    ICHECK_EQ(new_in_layouts.size(), 5);
    ICHECK_EQ(old_in_layouts.size(), 5);

    // 1) Get the axis.
    int axis = param->axis;
    axis = (axis == -1) ? old_in_shapes[0].size() - 1 : axis;

    // 2) Collect the original axis
    std::string old_dim = old_in_layouts[0][axis].name();

    // 3) Collect the new axes by walking new_layout.
    tvm::Integer new_axis;
    std::string new_layout_string = "";
    int axis_index = 0;
    for (auto iter_var : new_in_layouts[0]->axes) {
      const auto& layout_axis = LayoutAxis::Get(iter_var);
      const std::string& layout_dim = layout_axis.name();
      if (old_dim == layout_dim) {
        new_axis = tvm::Integer(axis_index);
      }

      if (layout_axis.IsPrimal()) {
        new_layout_string += layout_dim;
        axis_index++;
      } else {
        // Propogate layout if input_zero_point and input_scale are scalar values.
        ICHECK_GE(old_in_types.size(), 3);
        if (IsScalarType(old_in_types[1]) && IsScalarType(old_in_types[2])) {
          new_layout_string += std::to_string(new_in_layouts[0].FactorOf(layout_axis)) + layout_dim;
          axis_index++;
        }
      }
    }

    // 4) Set the new axis and layout.
    Layout new_layout = Layout(new_layout_string);

    // Fill the layouts of remaining input tensors - scales and zero points. The layouts of these
    // tensors can be treated as channel layout.
    Layout channel_layout = Layout("C");
    input_layouts = {new_layout, channel_layout, channel_layout, channel_layout, channel_layout};
    output_layouts = {new_layout};
    param->axis = new_axis.IntValue();
  } else if (old_in_layouts.defined()) {
    // If the new layout is undefined, set the old layout as the inferred layout.
    ICHECK_EQ(old_in_layouts.size(), 5);

    Layout old_layout = old_in_layouts[0];

    // Fill the layouts of remaining input tensors - scales and zero points. The layouts of these
    // tensors can be treated as channel layout.
    Layout channel_layout = Layout("C");
    input_layouts = {old_layout, channel_layout, channel_layout, channel_layout, channel_layout};
    output_layouts = {old_layout};
  } else {
    // Set the layouts to undef.
    Layout undef = Layout::Undef();
    input_layouts = Array<Layout>(5, undef);
    output_layouts = {undef};
  }

  return InferCorrectLayoutOutput(input_layouts, output_layouts, Attrs(param));
}

bool has_current_target_sse41_support() {
  auto target_has_feature_fn_ptr = tvm::runtime::Registry::Get("target.target_has_feature");
  ICHECK(target_has_feature_fn_ptr) << "Function target.target_has_feature not found";
  return (*target_has_feature_fn_ptr)("sse4.1", Target::Current(true));
}

/*
 * \brief TONEAREST is the standard rounding where the value is rounded away
 *         from zero at midpoints (for example, -1.5 rounds to -2).
 * \param input_tensor The input tensor to rounding op.
 * \return The sequence of existing Relay ops.
 */
template <int Bits>
Expr Tonearest(const Expr& input_tensor) {
  if (has_current_target_sse41_support()) return Round(input_tensor);

  auto half = MakeConstantScalar(DataType::Float(Bits), 0.5f);
  auto zero = MakeConstantScalar(DataType::Float(Bits), 0.f);
  auto pos_one = MakeConstantScalar(DataType::Float(Bits), +1.f);
  auto neg_one = MakeConstantScalar(DataType::Float(Bits), -1.f);
  auto multiplier = Where(Less(input_tensor, zero), neg_one, pos_one);
  auto half_multiplied = Multiply(half, multiplier);
  auto input_tensor_biased = Add(input_tensor, half_multiplied);
  auto input_tensor_biased_multiplied = Multiply(input_tensor_biased, multiplier);
  auto input_tensor_biased_multiplied_int =
      Cast(input_tensor_biased_multiplied, DataType::Int(Bits));
  auto input_tensor_biased_multiplied_float =
      Cast(input_tensor_biased_multiplied_int, DataType::Float(Bits));
  auto input_tensor_rounded = Multiply(input_tensor_biased_multiplied_float, multiplier);
  return Where(IsFinite(input_tensor), input_tensor_rounded, input_tensor);
}

/*
 * \brief UPWARD is the standard rounding except at midpoints where the value
 *        is rounded to positive infinity (for example, -1.5 rounds to -1).
 * \param input_tensor The input tensor to rounding op.
 * \return The sequence of existing Relay ops.
 */
template <int Bits>
Expr Upward(const Expr& input_tensor) {
  auto half = MakeConstantScalar(DataType::Float(Bits), 0.5f);
  auto input_tensor_biased = Add(input_tensor, half);
  if (has_current_target_sse41_support()) return Floor(input_tensor_biased);

  auto zero = MakeConstantScalar(DataType::Float(Bits), 0.f);
  auto one = MakeConstantScalar(DataType::Float(Bits), +1.f);
  auto input_tensor_biased_int = Cast(input_tensor_biased, DataType::Int(Bits));
  auto input_tensor_biased_float = Cast(input_tensor_biased_int, DataType::Float(Bits));
  auto is_subtraction_not_necessary =
      LogicalOr(Equal(input_tensor_biased, input_tensor_biased_float),
                GreaterEqual(input_tensor_biased, zero));
  auto input_tensor_rounded = Where(is_subtraction_not_necessary, input_tensor_biased_float,
                                    Subtract(input_tensor_biased_float, one));
  return Where(IsFinite(input_tensor), input_tensor_rounded, input_tensor);
}

// Lowering of qnn.requantize op

/*
 * \brief Lower requantize to a sequence of ops.
 * \param input_tensor The input tensor to requantize op.
 * \param param The requantize op attrs.
 * \param input_shape The input tensor shape of the requantize op.
 * \return The sequence of existing Relay ops.
 * \note RequantizationInt using only integer computation. Here, the computation is
 *       converted to a fixed point computation by computing output multiplier
 *       and shift. This is useful, if the target device does not support/have
 *       very expensive floating point computations.
 *
 *       The whole computation this can be broken down into following steps
 *       1) Calculate the integer multiplier and integer shift.
 *       2) Subtract the input integer zero point.
 *       3) Perform fixed point multiplication.
 *       4) Add the output zero point.
 *       5) Cast to the out_dtype.
 */
Expr RequantizeLowerInt(const Expr& input_tensor, const Expr& input_scale,
                        const Expr& input_zero_point, const Expr& output_scale,
                        const Expr& output_zero_point, const RequantizeAttrs* param,
                        const Array<IndexExpr>& input_shape, const DataType& out_dtype) {
  auto tensor = Cast(input_tensor, DataType::Int(32));
  auto zero_scalar = MakeConstantScalar(DataType::Int(32), 0);
  if (!IsEqualScalar(input_zero_point, zero_scalar)) {
    // Broadcast input zero point if needed.
    int rank = static_cast<int>(input_shape.size());
    int axis = (param->axis < 0) ? ((rank > 0) ? rank + param->axis : 0) : param->axis;
    Expr input_zero_broadcast = ExpandBiasToMatchAxis(Reshape(input_zero_point,
                                                              {
                                                                  -1,
                                                              }),
                                                      rank, {axis});
    tensor = Subtract(tensor, Cast(input_zero_broadcast, DataType::Int(32)));
  }

  // 2) If the input and output scales are same, we can skip the fixed point multiplication. Check
  // if the input scale is per-tensor or per-channel. If it is per-tensor, there is single scale for
  // the whole tensor. For per-channel (aka per-axis), there is a vector of scales for the input
  // tensor. Depending on the quantization type, the fixed point multiplication routing is called.
  const bool is_upward_rounding = (param->rounding == "UPWARD");
  auto scaled_int32_t = tensor;
  float output_scale_float = GetScalarFromConstant<float>(output_scale);
  if (IsConstScalar(input_scale)) {
    // This is per-tensor quantization. Single scale.
    float input_scale_float = GetScalarFromConstant<float>(input_scale);
    double double_multiplier =
        static_cast<double>(input_scale_float) / static_cast<double>(output_scale_float);
    // Skip if input and output scales are same.
    if (!IsEqualScalar(input_scale, output_scale)) {
      auto [fixed_point_multiplier, shift] = GetFixedPointMultiplierShift(double_multiplier);

      // When using upward rounding (i.e., x.5 rounded to x+1), leverage
      // the FixedPointMultiply operator
      scaled_int32_t =
          (is_upward_rounding
               ? FixedPointMultiply(scaled_int32_t, fixed_point_multiplier, shift)
               : FixedPointMultiplyToNearest(scaled_int32_t, double_multiplier, input_shape));
    }

  } else {
    // This is per-channel (per=axis) quantization.
    std::vector<double> double_multipliers;
    auto input_axis_scales = GetFloatVectorFromConstant(input_scale);
    for (auto input_axis_scale : input_axis_scales) {
      double multiplier =
          static_cast<double>(input_axis_scale) / static_cast<double>(output_scale_float);
      double_multipliers.push_back(multiplier);
    }
    int axis = param->axis;
    axis = (axis == -1) ? input_shape.size() - 1 : axis;

    // When using "upward" rounding, leverage the FixedPointMultiplyPerAxis operator,
    // for "tonearest" rounding - lower to multiply, add, shift operators sequence.
    scaled_int32_t = is_upward_rounding
                         ? FixedPointMultiplyPerChannel(scaled_int32_t, double_multipliers, axis)
                         : FixedPointMultiplyPerChannelToNearest(scaled_int32_t, double_multipliers,
                                                                 input_shape, axis);
  }

  // 3) Add the output zero point.
  auto shifted_int32_t = scaled_int32_t;
  if (!IsEqualScalar(output_zero_point, zero_scalar)) {
    shifted_int32_t = Add(Cast(output_zero_point, DataType::Int(32)), scaled_int32_t);
  }

  // 4) Clip to the out_dtype min/max. Skip clipping if out_dtype is Int32. The fixed point
  // multiplication keeps the value in int32 range.
  if (out_dtype == DataType::Int(32)) {
    return shifted_int32_t;
  }

  auto q_min = GetQmin(out_dtype);
  auto q_max = GetQmax(out_dtype);
  auto clipped_t = Clip(shifted_int32_t, q_min, q_max);
  return Cast(clipped_t, out_dtype);
}

// Lowering of qnn.requantize op

/*
 * \brief Lower requantize to a sequence of ops.
 * \param input_tensor The input tensor to requantize op.
 * \param param The requantize op attrs.
 * \param input_shape The input tensor shape of the requantize op.
 * \return The sequence of existing Relay ops.
 * \note RequantizationFP using floating computation. All multiplication/sub/sum
 *       occurs in floating point data type and only at the end is converted to
 *       int32 data type and clamped for output data type.
 *
 *       The whole computation this can be broken down into following steps
 *       1) Subtract the input zero point.
 *       2) Perform multiplication.
 *       3) Add the output zero point.
 *       4) Cast to the out_dtype.
 */
template <int Bits>
Expr RequantizeLowerFP(const Expr& input_tensor, const Expr& input_scale,
                       const Expr& input_zero_point, const Expr& output_scale,
                       const Expr& output_zero_point, const RequantizeAttrs* param,
                       const Array<IndexExpr>& input_shape, const DataType& out_dtype) {
  auto tensor = Cast(input_tensor, DataType::Float(Bits));
  auto zero_scalar = MakeConstantScalar(DataType::Int(32), 0);
  if (!IsEqualScalar(input_zero_point, zero_scalar)) {
    // Broadcast input zero point if needed.
    int rank = static_cast<int>(input_shape.size());
    int axis = (param->axis < 0) ? ((rank > 0) ? rank + param->axis : 0) : param->axis;
    Expr input_zero_broadcast = ExpandBiasToMatchAxis(Reshape(input_zero_point,
                                                              {
                                                                  -1,
                                                              }),
                                                      rank, {axis});
    tensor = Subtract(tensor, Cast(input_zero_broadcast, DataType::Float(Bits)));
  }

  // 2) If the input and output scales are same, we can skip the multiplication. Check
  // if the input scale is per-tensor or per-channel. If it is per-tensor, there is single scale for
  // the whole tensor. For per-channel (aka per-axis), there is a vector of scales for the input
  // tensor. Depending on the quantization type, the fixed point multiplication routing is called.
  auto scaled_fp_t = tensor;
  double output_scale_float = GetScalarFromConstant<float>(output_scale);
  if (IsConstScalar(input_scale)) {
    // This is per-tensor quantization. Single scale.
    double input_scale_float = GetScalarFromConstant<float>(input_scale);
    double double_multiplier = static_cast<double>(input_scale_float) / output_scale_float;
    // Skip if input and output scales are same.
    if (!IsEqualScalar(input_scale, output_scale)) {
      double multiplier = double_multiplier;
      auto m_scalar = MakeConstantScalar(DataType::Float(Bits), multiplier);
      scaled_fp_t = Multiply(m_scalar, scaled_fp_t);
    }

  } else {
    // This is per-channel (per=axis) quantization.
    std::vector<double> double_multipliers;
    auto input_axis_scales = GetFloatVectorFromConstant(input_scale);
    double output_scale_float = GetScalarFromConstant<float>(output_scale);
    for (auto input_axis_scale : input_axis_scales) {
      double multiplier = static_cast<double>(input_axis_scale) / output_scale_float;
      double_multipliers.push_back(multiplier);
    }
    int axis = param->axis;
    axis = (axis == -1) ? input_shape.size() - 1 : axis;

    auto fixed_pt_multiplier_expr = MakeConstantTensor(
        DataType::Float(Bits), {(int64_t)double_multipliers.size()}, double_multipliers);
    size_t n_dim = input_shape.size();
    auto exp_fixed_pt_multiplier_expr =
        ExpandBiasToMatchAxis(fixed_pt_multiplier_expr, n_dim, {axis});

    scaled_fp_t = Multiply(scaled_fp_t, exp_fixed_pt_multiplier_expr);
  }

  // 3) Add the output zero point.
  auto shifted_fp_t = scaled_fp_t;
  if (!IsEqualScalar(output_zero_point, zero_scalar)) {
    shifted_fp_t = Add(shifted_fp_t, Cast(output_zero_point, DataType::Float(Bits)));
  }

  if (param->rounding == "UPWARD") {
    shifted_fp_t = Upward<Bits>(shifted_fp_t);
  } else /*if (param->rounding == "TONEAREST")*/ {
    shifted_fp_t = Tonearest<Bits>(shifted_fp_t);
  }

  shifted_fp_t = Cast(shifted_fp_t, DataType::Int(32));
  // 4) Clip to the out_dtype min/max. Skip clipping if out_dtype is Int32. The fixed point
  // multiplication keeps the value in int32 range.
  if (out_dtype == DataType::Int(32)) {
    return shifted_fp_t;
  }

  auto q_min = GetQmin(out_dtype);
  auto q_max = GetQmax(out_dtype);
  auto clipped_t = Clip(shifted_fp_t, q_min, q_max);
  return Cast(clipped_t, out_dtype);
}

// Lowering of qnn.requantize op
/*
 * \brief Lower requantize to a sequence of ops.
 * \param input_tensor The input tensor to requantize op.
 * \param param The requantize op attrs.
 * \param input_shape The input tensor shape of the requantize op.
 * \return The sequence of existing Relay ops.
 */
Expr RequantizeLower(const Expr& input_tensor, const Expr& input_scale,
                     const Expr& input_zero_point, const Expr& output_scale,
                     const Expr& output_zero_point, const RequantizeAttrs* param,
                     const Array<IndexExpr>& input_shape, const DataType& out_dtype) {
  // Check output scale validity.
  ICHECK_NE(GetScalarFromConstant<float>(output_scale), 0.0)
      << "QNN requantize output scale can not be equal to 0.0";
  // Check rounding validity.
  ICHECK(param->rounding == "UPWARD" || param->rounding == "TONEAREST")
      << "QNN requantize supports two rounding modes - UPWARD and "
      << "TONEAREST";
  // Check compute_dtype validity.
  ICHECK(param->compute_dtype == "int64" || param->compute_dtype == "float32" ||
         param->compute_dtype == "float64")
      << "QNN requantize supports three compute_dtype variants - \"int64\", \"float32\" and "
         "\"float64\"";
  if (param->compute_dtype == "float32") {
    return RequantizeLowerFP<32>(input_tensor, input_scale, input_zero_point, output_scale,
                                 output_zero_point, param, input_shape, out_dtype);
  } else if (param->compute_dtype == "float64") {
    return RequantizeLowerFP<64>(input_tensor, input_scale, input_zero_point, output_scale,
                                 output_zero_point, param, input_shape, out_dtype);
  } else /*if (param->compute_dtype == "int64") */ {
    return RequantizeLowerInt(input_tensor, input_scale, input_zero_point, output_scale,
                              output_zero_point, param, input_shape, out_dtype);
  }
}

/*
 * \brief Forward rewrite the requantize op.
 * \param ref_call The original call that will be lowered.
 * \param new_args The new mutated args to the call node.
 * \param ctx The node context.
 * \return The sequence of Relay ops for requantize op.
 * \note Lowering of the requantize operation. The requantize operator converts
 *       one quantized tensor to another quantized tensor. For the output
 *       tensor, we are provided with output scale and zero point. The
 *       computation looks like this
 *
 * Q_output = zp_output +  (scale_input)/(scale_ouptut) * (Q_input - zp_input)
 */
Expr RequantizeQnnCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                               const Array<tvm::relay::Type>& types) {
  ICHECK_EQ(new_args.size(), 5);
  auto& quantized_data = new_args[0];
  auto& input_scale = new_args[1];
  auto& input_zero_point = new_args[2];
  auto& output_scale = new_args[3];
  auto& output_zero_point = new_args[4];
  const auto* param = attrs.as<RequantizeAttrs>();
  const RequantizeConfig& cfg = RequantizeConfig::Current();

  ICHECK(param != nullptr);

  const_cast<RequantizeAttrs*>(param)->rounding =
      SelectRequntizeParameter(param->rounding, cfg->get_rounding(), cfg->is_default, "rounding");
  const_cast<RequantizeAttrs*>(param)->compute_dtype = SelectRequntizeParameter(
      param->compute_dtype, cfg->get_compute_dtype(), cfg->is_default, "compute_dtype");

  // Find input shape.
  ICHECK_EQ(types.size(), 6);
  auto in_type = types[0];
  auto in_tensor_type = in_type.as<TensorTypeNode>();
  ICHECK(in_tensor_type != nullptr) << "Type information missing."
                                    << " Please run infer_type pass.";
  Array<IndexExpr> input_shape = in_tensor_type->shape;

  // Find the output dtype.
  auto out_type = types[5];
  auto out_tensor_type = out_type.as<TensorTypeNode>();
  ICHECK(out_tensor_type != nullptr) << "Type information missing."
                                     << " Please run infer_type pass.";
  auto out_dtype = out_tensor_type->dtype;
  return RequantizeLower(quantized_data, input_scale, input_zero_point, output_scale,
                         output_zero_point, param, input_shape, out_dtype);
}

/*
 * \brief Infer shape function of Requantize op.
 * \param types The types of input args.
 * \param num_inputs The number of inputs.
 * \param attrs The op attributes.
 * \param reporter The type reporter that sets the dtype and shapes.
 * \return True if the infer shape succeeded.
 */
bool RequantizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  // Expected Types: data, input_scale, input_zero_point, output_scale, output_zero_point, output
  ICHECK_EQ(types.size(), 6);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) {
    return false;
  }

  // Check the scale and zero point types
  for (size_t i = 3; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  const auto in_dtype = data->dtype;
  ICHECK(in_dtype == DataType::Int(8) || in_dtype == DataType::UInt(8) ||
         in_dtype == DataType::Int(16) || in_dtype == DataType::Int(32) ||
         in_dtype == DataType::Int(64))
      << "Input type should be one of [int8, uint8, int16, int32, int64] but was " << in_dtype;

  const RequantizeAttrs* requantize_attrs = attrs.as<RequantizeAttrs>();
  int axis = requantize_attrs->axis;
  auto rank = static_cast<int>(data->shape.size());
  axis = (axis < 0) ? ((rank > 0) ? data->shape.size() + axis : 0) : axis;
  ICHECK_LT(axis, rank > 0 ? rank : 1) << "axis " << requantize_attrs->axis << " is out of range";
  ICHECK_GE(axis, 0) << "axis " << requantize_attrs->axis << " is out of range";

  PrimExpr axis_shape;
  if (rank > 0) {
    axis_shape = data->shape[axis];
  } else {
    axis_shape = Integer(1);
  }
  // Check and assign types for scale and zero points.
  AssignType(types[1], DataType::Float(32), axis_shape, reporter);  // input_scale
  AssignType(types[2], DataType::Int(32), axis_shape, reporter);    // input_zero_pt
  // For now, requantize output tensor is limited to full tensor uniform quantization.
  ICHECK(IsScalarType(types[3], DataType::Float(32)));  // output_scale
  ICHECK(IsScalarType(types[4], DataType::Int(32)));    // output_zero_point

  const Array<tvm::PrimExpr> oshape = data->shape;
  // assign output type
  auto out_dtype = requantize_attrs->out_dtype;
  ICHECK(out_dtype == DataType::Int(8) || out_dtype == DataType::UInt(8) ||
         out_dtype == DataType::Int(16) || out_dtype == DataType::Int(32))
      << "Output type should be one of [int8, uint8, int16, int32] but was " << out_dtype;
  reporter->Assign(types[5], TensorType(oshape, out_dtype));
  return true;
}

// Positional relay function to create qnn requantize operator
// used by frontend FFI.
Expr MakeRequantize(Expr data, Expr input_scale, Expr input_zero_point, Expr output_scale,
                    Expr output_zero_point, int axis, String rounding, String compute_dtype,
                    DataType out_dtype) {
  auto attrs = make_object<RequantizeAttrs>();
  attrs->axis = axis;
  attrs->rounding = std::move(rounding);
  attrs->out_dtype = std::move(out_dtype);
  attrs->compute_dtype = std::move(compute_dtype);
  static const Op& op = Op::Get("qnn.requantize");
  return Call(op, {data, input_scale, input_zero_point, output_scale, output_zero_point},
              Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.requantize")
    .describe(R"code(Requantize operator.
The requantize operator converts one quantized tensor to another quantized
tensor. For the output tensor, we are provided with output scale and zero
point. The computation looks like this

Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<RequantizeAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "The quantized input tensor.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("output_zero_point", "Tensor",
                  "The quantization zero_point of the output tensor.")
    .set_support_level(11)
    .add_type_rel("Requantize", RequantizeRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", RequantizeQnnCanonicalize)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", RequantizeInferCorrectLayout);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.requantize").set_body_typed(MakeRequantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
