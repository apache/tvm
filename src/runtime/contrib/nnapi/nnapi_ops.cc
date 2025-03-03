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

#ifdef TVM_GRAPH_EXECUTOR_NNAPI
#include "nnapi_ops.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nnapi_builder.h"

namespace tvm {
namespace runtime {
namespace contrib {

NNAPIOpConverterParams::NNAPIOpConverterParams(const JSONGraphNode& node) : node(node) {}

NNAPIOpConverter::NNAPIOpConverter(std::string op_name) : op_name_(op_name) {}

void ElwBinaryOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                                   const std::vector<NNAPIOperand>& inputs,
                                   std::vector<NNAPIOperand>& outputs) const {
  // A map from op names to NNAPI OperationCode and whether it requires a FuseCode.
  static const std::unordered_map<std::string, std::tuple<ANeuralNetworksOperationType, bool>>
      op_map = {
          {"add", {ANEURALNETWORKS_ADD, true}},
          {"mul", {ANEURALNETWORKS_MUL, true}},
          {"div", {ANEURALNETWORKS_DIV, true}},
          {"sub", {ANEURALNETWORKS_SUB, true}},
          {"pow", {ANEURALNETWORKS_POW, false}},
          {"equal", {ANEURALNETWORKS_EQUAL, false}},
          {"greater", {ANEURALNETWORKS_GREATER, false}},
          {"greater_equal", {ANEURALNETWORKS_GREATER_EQUAL, false}},
          {"less", {ANEURALNETWORKS_LESS, false}},
          {"less_equal", {ANEURALNETWORKS_LESS_EQUAL, false}},
          {"not_equal", {ANEURALNETWORKS_NOT_EQUAL, false}},
          {"maximum", {ANEURALNETWORKS_MAXIMUM, false}},
          {"minimum", {ANEURALNETWORKS_MINIMUM, false}},
      };

  auto it = op_map.find(op_name_);
  ICHECK(it != op_map.end()) << "Unsupported binary operation type " << op_name_;
  const ANeuralNetworksOperationType operation_type = std::get<0>(it->second);
  const bool requires_fuse_code = std::get<1>(it->second);

  ICHECK_EQ(inputs.size(), 2) << "Expected binary operation to have 2 inputs but got "
                              << inputs.size();

  auto input_indices = ExtractOperandIndices(inputs);
  const auto output_indices = ExtractOperandIndices(outputs);

  if (requires_fuse_code) {
    // Create an extra input at index 2 for the fuse code.
    const int32_t fused_none = ANEURALNETWORKS_FUSED_NONE;
    const NNAPIOperand fuse_code_operand = builder.CreateScalarOperandWithValue(
        ANEURALNETWORKS_INT32, &fused_none, sizeof(fused_none));
    input_indices.push_back(fuse_code_operand.GetOperandIndex());
  }

  builder.AddOperation(operation_type, input_indices, output_indices);
}

void UnaryOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                               const std::vector<NNAPIOperand>& inputs,
                               std::vector<NNAPIOperand>& outputs) const {
  static const std::unordered_map<std::string, ANeuralNetworksOperationType> op_map = {
      // clang-format off
      {"floor", ANEURALNETWORKS_FLOOR},
      {"logistic", ANEURALNETWORKS_LOGISTIC},
      {"relu", ANEURALNETWORKS_RELU},
      {"tanh", ANEURALNETWORKS_TANH},
      {"abs", ANEURALNETWORKS_ABS},
      {"exp", ANEURALNETWORKS_EXP},
      {"log", ANEURALNETWORKS_LOG},
      {"neg", ANEURALNETWORKS_NEG},
      {"sqrt", ANEURALNETWORKS_SQRT},
      {"rsqrt", ANEURALNETWORKS_RSQRT},
      // clang-format on
  };
  auto it = op_map.find(op_name_);
  ICHECK(it != op_map.end()) << "Unsupported unary operation type " << op_name_;
  const ANeuralNetworksOperationType operation_type = it->second;

  const auto input_indices = ExtractOperandIndices(inputs);
  const auto output_indices = ExtractOperandIndices(outputs);
  builder.AddOperation(operation_type, input_indices, output_indices);
}

void SoftmaxOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                                 const std::vector<NNAPIOperand>& inputs,
                                 std::vector<NNAPIOperand>& outputs) const {
  ICHECK_EQ(inputs.size(), 1) << "Unsupported number of inputs for NNAPI softmax operation: "
                              << inputs.size();

  auto input_indices = ExtractOperandIndices(inputs);
  const auto output_indices = ExtractOperandIndices(outputs);

  // Add the scalar input for beta value at index 1.
  const auto& input = inputs[0];
  // TODO(PLLab): Conditionally use float16 beta for float16 input.
  ICHECK_EQ(input.GetTensorType(), ANEURALNETWORKS_TENSOR_FLOAT32)
      << "NNAPI runtime does not support non-float32 inputs for softmax yet";
  const float beta = 1.0f;
  const NNAPIOperand beta_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_FLOAT32, &beta, sizeof beta);
  input_indices.push_back(beta_operand.GetOperandIndex());

  builder.AddOperation(ANEURALNETWORKS_SOFTMAX, input_indices, output_indices);
}

// Insert a reshape operation that reshapes `operand` to `dimensions` and return the reshaped
// operand.
NNAPIOperand ReshapeOperand(NNAPIModelBuilder& builder, const NNAPIOperand& operand,  // NOLINT(*)
                            std::vector<int64_t> dimensions) {
  // ANEURALNETWORKS_RESHAPE requires the dimensions to be specified in a int32 tensor.
  const std::vector<int32_t> dimensions_int32(dimensions.begin(), dimensions.end());
  const std::vector<int64_t> dim_of_dims{static_cast<int64_t>(dimensions_int32.size())};

  const NNAPIOperand reshape_shape_operand =
      builder.CreateOperandWithValue(ANEURALNETWORKS_TENSOR_INT32, dim_of_dims, 0.0f, 0,
                                     reinterpret_cast<const void*>(dimensions_int32.data()),
                                     dimensions_int32.size() * sizeof(*dimensions_int32.data()));
  const NNAPIOperand reshaped_operand = builder.CreateOperand(
      operand.GetTensorType(), dimensions, operand.GetScale(), operand.GetZeroPoint());

  builder.AddOperation(
      ANEURALNETWORKS_RESHAPE,
      std::vector<uint32_t>{operand.GetOperandIndex(), reshape_shape_operand.GetOperandIndex()},
      std::vector<uint32_t>{reshaped_operand.GetOperandIndex()});
  return reshaped_operand;
}

NNAPIOperand TransposeOperand(NNAPIModelBuilder& builder, const NNAPIOperand& operand,  // NOLINT(*)
                              std::vector<int64_t> dimensions) {
  const std::vector<int32_t> dimensions_int32(dimensions.begin(), dimensions.end());
  const std::vector<int64_t> dim_of_axes{static_cast<int64_t>(dimensions_int32.size())};
  std::vector<int64_t> result_dimension;
  for (size_t i = 0; i < dimensions.size(); i++) {
    result_dimension.push_back(operand.GetDimensions()[dimensions_int32[i]]);
  }

  const NNAPIOperand transpose_shape_operand =
      builder.CreateOperandWithValue(ANEURALNETWORKS_TENSOR_INT32, dim_of_axes, 0.0f, 0,
                                     reinterpret_cast<const void*>(dimensions_int32.data()),
                                     dimensions_int32.size() * sizeof(*dimensions_int32.data()));
  const NNAPIOperand transposed_operand = builder.CreateOperand(
      operand.GetTensorType(), result_dimension, operand.GetScale(), operand.GetZeroPoint());

  builder.AddOperation(
      ANEURALNETWORKS_TRANSPOSE,
      std::vector<uint32_t>{operand.GetOperandIndex(), transpose_shape_operand.GetOperandIndex()},
      std::vector<uint32_t>{transposed_operand.GetOperandIndex()});

  return transposed_operand;
}

void MatmulOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                                const std::vector<NNAPIOperand>& inputs,
                                std::vector<NNAPIOperand>& outputs) const {
  ICHECK_EQ(inputs.size(), 2);

  auto input_indices = ExtractOperandIndices(inputs);
  const auto output_indices = ExtractOperandIndices(outputs);

  const size_t input0_ndim = inputs[0].GetDimensions().size();
  const size_t input1_ndim = inputs[1].GetDimensions().size();
  if (input0_ndim != input1_ndim) {
    if (input0_ndim > input1_ndim) {
      // Check that the extra leading dimensions on input 0 are all ones.
      const size_t diff = input0_ndim - input1_ndim;
      for (size_t i = 0; i < diff; ++i) {
        ICHECK_EQ(inputs[0].GetDimensions()[i], 1);
      }

      // Expand input 1's dimensions.
      std::vector<int64_t> reshaped_dimensions(diff, 1);
      reshaped_dimensions.insert(reshaped_dimensions.end(), inputs[1].GetDimensions().begin(),
                                 inputs[1].GetDimensions().end());
      const auto reshaped_operand = ReshapeOperand(builder, inputs[1], reshaped_dimensions);
      input_indices[1] = reshaped_operand.GetOperandIndex();
    } else {
      // input0_ndim < input1_ndim
      // Check that the extra leading dimensions on input 1 are all ones.
      const size_t diff = input1_ndim - input0_ndim;
      for (size_t i = 0; i < diff; ++i) {
        ICHECK_EQ(inputs[1].GetDimensions()[i], 1);
      }

      // Expand input 0's dimensions.
      std::vector<int64_t> reshaped_dimensions(diff, 1);
      reshaped_dimensions.insert(reshaped_dimensions.end(), inputs[0].GetDimensions().begin(),
                                 inputs[0].GetDimensions().end());
      const auto reshaped_operand = ReshapeOperand(builder, inputs[0], reshaped_dimensions);
      input_indices[0] = reshaped_operand.GetOperandIndex();
    }
  }

  {
    const unsigned char adj_x = 0;
    const NNAPIOperand adj_x_operand =
        builder.CreateScalarOperandWithValue(ANEURALNETWORKS_BOOL, &adj_x, sizeof(adj_x));
    input_indices.push_back(adj_x_operand.GetOperandIndex());
  }

  {
    const unsigned char adj_y = 0;
    const NNAPIOperand adj_y_operand =
        builder.CreateScalarOperandWithValue(ANEURALNETWORKS_BOOL, &adj_y, sizeof(adj_y));
    input_indices.push_back(adj_y_operand.GetOperandIndex());
  }

  builder.AddOperation(ANEURALNETWORKS_BATCH_MATMUL, input_indices, output_indices);
}

void TransposeOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                                   const std::vector<NNAPIOperand>& inputs,
                                   std::vector<NNAPIOperand>& outputs) const {
  ICHECK_EQ(inputs.size(), 1);

  auto input_indices = ExtractOperandIndices(inputs);
  auto output_indices = ExtractOperandIndices(outputs);

  std::vector<int32_t> axes;
  if (node.HasAttr("axes")) {
    const auto axes_attr = node.GetAttr<std::vector<std::string>>("axes");
    for (auto str_axis : axes_attr) {
      axes.push_back(std::stoi(str_axis));
    }
  } else {
    for (size_t i = 0; i < inputs[0].GetDimensions().size(); ++i) {
      axes.push_back(i);
    }
    std::reverse(axes.begin(), axes.end());
  }

  const std::vector<int64_t> dim_of_axes{static_cast<int64_t>(axes.size())};
  const NNAPIOperand perm_operand = builder.CreateOperandWithValue(
      ANEURALNETWORKS_TENSOR_INT32, dim_of_axes, 0.0f, 0,
      reinterpret_cast<const void*>(axes.data()), axes.size() * sizeof(*axes.data()));
  input_indices.push_back(perm_operand.GetOperandIndex());

  builder.AddOperation(ANEURALNETWORKS_TRANSPOSE, input_indices, output_indices);
}

void CastOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                              const std::vector<NNAPIOperand>& inputs,
                              std::vector<NNAPIOperand>& outputs) const {
  auto input_indices = ExtractOperandIndices(inputs);
  auto output_indices = ExtractOperandIndices(outputs);

  // Extract the dtype attribute and check that the output operand type matches the dtype specified.
  const auto dtype_attr = node.GetAttr<std::vector<std::string>>("astype_dtype");
  ICHECK(dtype_attr.size() == 1);
  const auto dtype_str = dtype_attr[0];
  const DLDataType dtype = runtime::String2DLDataType(dtype_str);
  ICHECK(outputs.size() == 1);
  const auto output_tensor_type = outputs[0].GetTensorType();
  ICHECK(TensorTypeFromDLDataType(dtype) == output_tensor_type)
      << "Expect a cast to dtype " << dtype_str << " but got output operand of type "
      << output_tensor_type;

  builder.AddOperation(ANEURALNETWORKS_CAST, input_indices, output_indices);
}

template <int TensorType, typename DataType>
NNAPIOperand CreateConv2DBiasOperand(NNAPIModelBuilder& builder,  // NOLINT(*)
                                     int64_t output_depth) {
  std::vector<DataType> bias(output_depth, 0.0f);

  const std::vector<int64_t> dim_of_bias{static_cast<int64_t>(bias.size())};
  const NNAPIOperand bias_operand = builder.CreateOperandWithValue(
      TensorType, dim_of_bias, 0.0f, 0, reinterpret_cast<const void*>(bias.data()),
      bias.size() * sizeof(*bias.data()));
  return bias_operand;
}

void Conv2dOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                                const std::vector<NNAPIOperand>& inputs,
                                std::vector<NNAPIOperand>& outputs) const {
  auto input_indices = ExtractOperandIndices(inputs);
  auto output_indices = ExtractOperandIndices(outputs);

  ICHECK(inputs.size() >= 2);
  const auto input_tensor_type = inputs[0].GetTensorType();
  const auto filter_tensor_type = inputs[1].GetTensorType();
  ICHECK(input_tensor_type == filter_tensor_type);
  ICHECK(input_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT32 ||
         input_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT16);
  ICHECK(filter_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT32 ||
         filter_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT16);

  // transpose kernel
  std::vector<int64_t> transposed_dimensions{0, 2, 3, 1};
  const auto transposed_operand = TransposeOperand(builder, inputs[1], transposed_dimensions);

  input_indices[1] = transposed_operand.GetOperandIndex();

  // bias operand
  if (input_indices.size() == 2) {
    const int output_depth = inputs[1].GetDimensions()[0];
    if (input_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT32) {
      const NNAPIOperand bias_operand =
          CreateConv2DBiasOperand<ANEURALNETWORKS_TENSOR_FLOAT32, float>(builder, output_depth);
      input_indices.push_back(bias_operand.GetOperandIndex());
    } else if (input_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT16) {
      const NNAPIOperand bias_operand =
          CreateConv2DBiasOperand<ANEURALNETWORKS_TENSOR_FLOAT16, uint16_t>(builder, output_depth);
      input_indices.push_back(bias_operand.GetOperandIndex());
    }
  } else {
    int64_t bias_dim;
    for (int i = 0; i < inputs[2].GetDimensions().size(); i++) {
      if (inputs[2].GetDimensions()[i] != 1) {
        bias_dim = inputs[2].GetDimensions()[i];
      }
    }
    std::vector<int64_t> bias_dimension = {bias_dim};
    NNAPIOperand bias_operand = ReshapeOperand(builder, inputs[2], bias_dimension);
    input_indices[2] = bias_operand.GetOperandIndex();
  }
  // padding operand
  std::vector<int32_t> padding;
  const auto padding_attr = node.GetAttr<std::vector<std::string>>("padding");

  for (auto str_pad : padding_attr) {
    padding.push_back(std::stoi(str_pad));
  }

  ICHECK(padding.size() == 4) << "NNAPI runtime currently only supports 4-way padding for Conv2D";
  const NNAPIOperand padding_left_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &padding[1], sizeof(padding[1]));
  input_indices.push_back(padding_left_operand.GetOperandIndex());

  const NNAPIOperand padding_right_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &padding[3], sizeof(padding[3]));
  input_indices.push_back(padding_right_operand.GetOperandIndex());

  const NNAPIOperand padding_top_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &padding[0], sizeof(padding[0]));
  input_indices.push_back(padding_top_operand.GetOperandIndex());

  const NNAPIOperand padding_bottom_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &padding[2], sizeof(padding[2]));
  input_indices.push_back(padding_bottom_operand.GetOperandIndex());

  // stride operand
  std::vector<int32_t> stride;
  const auto stride_attr = node.GetAttr<std::vector<std::string>>("strides");
  for (auto str_stride : stride_attr) {
    stride.push_back(std::stoi(str_stride));
  }

  ICHECK(stride.size() == 2);
  const NNAPIOperand stride_width_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &stride[0], sizeof(stride[0]));
  input_indices.push_back(stride_width_operand.GetOperandIndex());

  const NNAPIOperand stride_height_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &stride[1], sizeof(stride[1]));
  input_indices.push_back(stride_height_operand.GetOperandIndex());

  // group
  int32_t group;
  const auto group_attr = node.GetAttr<std::vector<std::string>>("group");
  for (auto str_group : group_attr) {
    group = std::stoi(str_group);
  }

  if (group > 1) {
    const NNAPIOperand group_operand =
        builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &group, sizeof(group));
    input_indices.push_back(group_operand.GetOperandIndex());
  }

  // fuse code
  const int32_t fused_none = ANEURALNETWORKS_FUSED_NONE;
  const NNAPIOperand fuse_code_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &fused_none, sizeof(fused_none));
  input_indices.push_back(fuse_code_operand.GetOperandIndex());

  // layout
  // Use NCHW layout for input 0 and output 0.
  const bool layout = true;
  const NNAPIOperand layout_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_BOOL, &layout, sizeof(layout));
  input_indices.push_back(layout_operand.GetOperandIndex());

  if (group > 1) {
    builder.AddOperation(ANEURALNETWORKS_GROUPED_CONV_2D, input_indices, output_indices);
  } else {
    builder.AddOperation(ANEURALNETWORKS_CONV_2D, input_indices, output_indices);
  }
}

void MaxPool2dOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                                   const std::vector<NNAPIOperand>& inputs,
                                   std::vector<NNAPIOperand>& outputs) const {
  auto input_indices = ExtractOperandIndices(inputs);
  auto output_indices = ExtractOperandIndices(outputs);

  // padding operand
  std::vector<int32_t> padding;
  const auto padding_attr = node.GetAttr<std::vector<std::string>>("padding");

  for (auto str_pad : padding_attr) {
    padding.push_back(std::stoi(str_pad));
  }

  const NNAPIOperand padding_left_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &padding[1], sizeof(padding[1]));
  input_indices.push_back(padding_left_operand.GetOperandIndex());

  const NNAPIOperand padding_right_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &padding[3], sizeof(padding[3]));
  input_indices.push_back(padding_right_operand.GetOperandIndex());

  const NNAPIOperand padding_top_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &padding[0], sizeof(padding[0]));
  input_indices.push_back(padding_top_operand.GetOperandIndex());

  const NNAPIOperand padding_bottom_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &padding[2], sizeof(padding[2]));
  input_indices.push_back(padding_bottom_operand.GetOperandIndex());

  // stride operand
  std::vector<int32_t> stride;
  const auto stride_attr = node.GetAttr<std::vector<std::string>>("strides");
  for (auto str_stride : stride_attr) {
    stride.push_back(std::stoi(str_stride));
  }

  const NNAPIOperand stride_width_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &stride[0], sizeof(stride[0]));
  input_indices.push_back(stride_width_operand.GetOperandIndex());

  const NNAPIOperand stride_height_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &stride[1], sizeof(stride[1]));
  input_indices.push_back(stride_height_operand.GetOperandIndex());

  // filter operand
  std::vector<int32_t> pool_size;
  const auto pool_size_attr = node.GetAttr<std::vector<std::string>>("pool_size");
  for (auto size : pool_size_attr) {
    pool_size.push_back(std::stoi(size));
  }

  const NNAPIOperand pool_size_width_operand = builder.CreateScalarOperandWithValue(
      ANEURALNETWORKS_INT32, &pool_size[0], sizeof(pool_size[0]));
  input_indices.push_back(pool_size_width_operand.GetOperandIndex());

  const NNAPIOperand pool_size_height_operand = builder.CreateScalarOperandWithValue(
      ANEURALNETWORKS_INT32, &pool_size[1], sizeof(pool_size[1]));
  input_indices.push_back(pool_size_height_operand.GetOperandIndex());

  // fuse code
  const int32_t fused_none = ANEURALNETWORKS_FUSED_NONE;
  const NNAPIOperand fuse_code_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &fused_none, sizeof(fused_none));
  input_indices.push_back(fuse_code_operand.GetOperandIndex());

  // layout
  const bool layout = true;
  const NNAPIOperand layout_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_BOOL, &layout, sizeof(layout));
  input_indices.push_back(layout_operand.GetOperandIndex());

  builder.AddOperation(ANEURALNETWORKS_MAX_POOL_2D, input_indices, output_indices);
}

void DenseOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                               const std::vector<NNAPIOperand>& inputs,
                               std::vector<NNAPIOperand>& outputs) const {
  auto input_indices = ExtractOperandIndices(inputs);
  auto output_indices = ExtractOperandIndices(outputs);
  const auto input_tensor_type = inputs[0].GetTensorType();
  const auto filter_tensor_type = inputs[1].GetTensorType();
  ICHECK(input_tensor_type == filter_tensor_type);
  ICHECK(input_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT32 ||
         input_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT16);
  ICHECK(filter_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT32 ||
         filter_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT16);

  if (input_indices.size() == 2) {
    const int output_depth = inputs[1].GetDimensions()[0];
    if (input_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT32) {
      const NNAPIOperand bias_operand =
          CreateConv2DBiasOperand<ANEURALNETWORKS_TENSOR_FLOAT32, float>(builder, output_depth);
      input_indices.push_back(bias_operand.GetOperandIndex());
    } else if (input_tensor_type == ANEURALNETWORKS_TENSOR_FLOAT16) {
      const NNAPIOperand bias_operand =
          CreateConv2DBiasOperand<ANEURALNETWORKS_TENSOR_FLOAT16, uint16_t>(builder, output_depth);
      input_indices.push_back(bias_operand.GetOperandIndex());
    }
  }

  // fuse code
  const int32_t fused_none = ANEURALNETWORKS_FUSED_NONE;
  const NNAPIOperand fuse_code_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &fused_none, sizeof(fused_none));
  input_indices.push_back(fuse_code_operand.GetOperandIndex());

  builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indices, output_indices);
}

void MeanOpConverter::Convert(NNAPIModelBuilder& builder, const JSONGraphNode& node,
                              const std::vector<NNAPIOperand>& inputs,
                              std::vector<NNAPIOperand>& outputs) const {
  auto input_indices = ExtractOperandIndices(inputs);
  auto output_indices = ExtractOperandIndices(outputs);

  // Extract the axis attribute and create an operand for it.
  const auto axis_attr = node.GetAttr<std::vector<std::string>>("axis");
  std::vector<int32_t> axis;
  for (auto dim : axis_attr) {
    axis.push_back(std::stoi(dim));
  }
  const std::vector<int64_t> dim_of_axis{static_cast<int64_t>(axis.size())};

  const NNAPIOperand axis_operand = builder.CreateOperandWithValue(
      ANEURALNETWORKS_TENSOR_INT32, dim_of_axis, 0.0f, 0,
      reinterpret_cast<const void*>(axis.data()), axis.size() * sizeof(*axis.data()));
  input_indices.push_back(axis_operand.GetOperandIndex());

  // Extract the keepdims attribute and create an operand for it.
  const auto keepdims_attr = node.GetAttr<std::vector<std::string>>("keepdims");
  ICHECK(keepdims_attr.size() == 1);
  const int32_t keepdims = keepdims_attr[0] == "1";

  const NNAPIOperand keepdims_operand =
      builder.CreateScalarOperandWithValue(ANEURALNETWORKS_INT32, &keepdims, sizeof keepdims);
  input_indices.push_back(keepdims_operand.GetOperandIndex());

  builder.AddOperation(ANEURALNETWORKS_MEAN, input_indices, output_indices);
}

const std::unordered_map<std::string, std::unique_ptr<NNAPIOpConverter>>& GetOpConverters() {
  static const std::unordered_map<std::string, std::unique_ptr<NNAPIOpConverter>> map = []() {
    std::unordered_map<std::string, std::unique_ptr<NNAPIOpConverter>> map;
    map.emplace("nnapi.add", std::make_unique<ElwBinaryOpConverter>("add"));
    map.emplace("nnapi.mul", std::make_unique<ElwBinaryOpConverter>("mul"));
    map.emplace("nnapi.div", std::make_unique<ElwBinaryOpConverter>("div"));
    map.emplace("nnapi.sub", std::make_unique<ElwBinaryOpConverter>("sub"));
    map.emplace("nnapi.pow", std::make_unique<ElwBinaryOpConverter>("pow"));
    map.emplace("nnapi.equal", std::make_unique<ElwBinaryOpConverter>("equal"));
    map.emplace("nnapi.greater", std::make_unique<ElwBinaryOpConverter>("greater"));
    map.emplace("nnapi.greater_equal", std::make_unique<ElwBinaryOpConverter>("greater_equal"));
    map.emplace("nnapi.less", std::make_unique<ElwBinaryOpConverter>("less"));
    map.emplace("nnapi.less_equal", std::make_unique<ElwBinaryOpConverter>("less_equal"));
    map.emplace("nnapi.not_equal", std::make_unique<ElwBinaryOpConverter>("not_equal"));
    map.emplace("nnapi.maximum", std::make_unique<ElwBinaryOpConverter>("maximum"));
    map.emplace("nnapi.minimum", std::make_unique<ElwBinaryOpConverter>("minimum"));
    map.emplace("nnapi.floor", std::make_unique<UnaryOpConverter>("floor"));
    map.emplace("nnapi.logistic", std::make_unique<UnaryOpConverter>("logistic"));
    map.emplace("nnapi.relu", std::make_unique<UnaryOpConverter>("relu"));
    map.emplace("nnapi.tanh", std::make_unique<UnaryOpConverter>("tanh"));
    map.emplace("nnapi.abs", std::make_unique<UnaryOpConverter>("abs"));
    map.emplace("nnapi.exp", std::make_unique<UnaryOpConverter>("exp"));
    map.emplace("nnapi.log", std::make_unique<UnaryOpConverter>("log"));
    map.emplace("nnapi.neg", std::make_unique<UnaryOpConverter>("neg"));
    map.emplace("nnapi.sqrt", std::make_unique<UnaryOpConverter>("sqrt"));
    map.emplace("nnapi.rsqrt", std::make_unique<UnaryOpConverter>("rsqrt"));
    map.emplace("nnapi.softmax", std::make_unique<SoftmaxOpConverter>());
    map.emplace("nnapi.batch_matmul", std::make_unique<MatmulOpConverter>());
    map.emplace("nnapi.transpose", std::make_unique<TransposeOpConverter>());
    map.emplace("nnapi.cast", std::make_unique<CastOpConverter>("cast"));
    map.emplace("nnapi.mean", std::make_unique<MeanOpConverter>("mean"));
    map.emplace("nnapi.conv2d", std::make_unique<Conv2dOpConverter>());
    map.emplace("nnapi.fully_connected", std::make_unique<DenseOpConverter>());
    map.emplace("nnapi.max_pool_2d", std::make_unique<MaxPool2dOpConverter>());
    return map;
  }();
  return map;
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_GRAPH_EXECUTOR_NNAPI
