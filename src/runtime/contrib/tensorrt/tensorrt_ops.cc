/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tensorrt/tensorrt_ops.cc
 * \brief Converters from Relay ops into TensorRT layers. Converters should
 * inherit from TensorRTOpConverter and implement the Convert() method.
 */

#include "tensorrt_ops.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "NvInfer.h"
#include "tensorrt_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

TensorRTOpConverter::TensorRTOpConverter(const std::vector<TensorRTInputType>& input_types,
                                         bool variable_input_count)
    : input_types(input_types), variable_input_count(variable_input_count) {}

nvinfer1::ITensor* TensorRTOpConverter::Reshape(TensorRTOpConverterParams* params,
                                                nvinfer1::ITensor* input,
                                                const std::vector<int>& new_shape) const {
  auto layer = params->network->addShuffle(*input);
  ICHECK(layer != nullptr);
  layer->setReshapeDimensions(VectorToTrtDims(new_shape));
  return layer->getOutput(0);
}

nvinfer1::ITensor* TensorRTOpConverter::Transpose(TensorRTOpConverterParams* params,
                                                  nvinfer1::ITensor* input,
                                                  const std::vector<int>& order) const {
  auto layer = params->network->addShuffle(*input);
  ICHECK(layer != nullptr);
  nvinfer1::Permutation perm;
  if (TRT_HAS_IMPLICIT_BATCH(params)) {
    // Batch dimension cannot be modified.
    ICHECK_EQ(input->getDimensions().nbDims, order.size() - 1);
    ICHECK_EQ(order[0], 0);
    for (size_t i = 0; i < order.size(); ++i) {
      perm.order[i] = order[i + 1] - 1;
    }
  } else {
    ICHECK_EQ(input->getDimensions().nbDims, order.size());
    for (size_t i = 0; i < order.size(); ++i) {
      perm.order[i] = order[i];
    }
  }
  layer->setFirstTranspose(perm);
  return layer->getOutput(0);
}

int TensorRTOpConverter::ConvertAxis(TensorRTOpConverterParams* params, int axis,
                                     int input_rank) const {
  // Add 1 for missing batch dim.
  if (TRT_HAS_IMPLICIT_BATCH(params)) {
    input_rank += 1;
  }
  ICHECK(axis >= -input_rank && axis < input_rank);
  if (axis < 0) axis += input_rank;
  if (TRT_HAS_IMPLICIT_BATCH(params)) {
    // Can't modify batch dimenson.
    ICHECK_NE(axis, 0);
    // Subtract 1 for implicit batch dim.
    axis -= 1;
  }
  return axis;
}

nvinfer1::ITensor* TensorRTOpConverter::CreateScalar(
    TensorRTOpConverterParams* params, float value, const nvinfer1::Dims& broadcast_to_dims) const {
  nvinfer1::Dims dims;
  dims.nbDims = broadcast_to_dims.nbDims;
  std::fill_n(dims.d, dims.nbDims, 1);
  float* values = new float[1];
  values[0] = value;
  nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, static_cast<void*>(values), 1};
  params->trt_weights->push_back(weights);
  return params->network->addConstant(dims, weights)->getOutput(0);
}

void TensorRTOpConverter::GetPadding(const std::vector<std::string>& padding,
                                     bool* use_asymmetric_padding, nvinfer1::DimsHW* prepadding,
                                     nvinfer1::DimsHW* postpadding) const {
  ICHECK(padding.size() == 1 || padding.size() == 2 || padding.size() == 4);
  if (padding.size() == 4) {
    // four int : padding width in the order of (top, left, bottom, right).
    *prepadding = nvinfer1::DimsHW(std::stoi(padding[0]), std::stoi(padding[1]));
    *postpadding = nvinfer1::DimsHW(std::stoi(padding[2]), std::stoi(padding[3]));
    *use_asymmetric_padding = true;
  } else if (padding.size() == 2) {
    // two int : bottom, right will use same padding as top, left
    *prepadding = nvinfer1::DimsHW(std::stoi(padding[0]), std::stoi(padding[1]));
    *postpadding = *prepadding;
    *use_asymmetric_padding = false;
  } else {
    // one int : same padding used on all sides
    *prepadding = nvinfer1::DimsHW(std::stoi(padding[0]), std::stoi(padding[0]));
    *postpadding = *prepadding;
    *use_asymmetric_padding = false;
  }
}

void TensorRTOpConverter::GetPadding3D(const std::vector<std::string>& padding,
                                       bool* use_asymmetric_padding, nvinfer1::Dims* prepadding,
                                       nvinfer1::Dims* postpadding) const {
  ICHECK(padding.size() == 1 || padding.size() == 3 || padding.size() == 6);
  if (padding.size() == 6) {
    // six int : padding width in the order of (front, top, left, back, bottom, right)
    *prepadding =
        nvinfer1::Dims3(std::stoi(padding[0]), std::stoi(padding[1]), std::stoi(padding[2]));
    *postpadding =
        nvinfer1::Dims3(std::stoi(padding[3]), std::stoi(padding[4]), std::stoi(padding[5]));
    *use_asymmetric_padding = true;
  } else if (padding.size() == 3) {
    // three int : back, bottom, right will use same padding as front, top, left
    *prepadding =
        nvinfer1::Dims3(std::stoi(padding[0]), std::stoi(padding[1]), std::stoi(padding[2]));
    *postpadding = *prepadding;
    *use_asymmetric_padding = false;
  } else {
    // one int : same padding used on all sides
    *prepadding =
        nvinfer1::Dims3(std::stoi(padding[0]), std::stoi(padding[0]), std::stoi(padding[0]));
    *postpadding = *prepadding;
    *use_asymmetric_padding = false;
  }
}

class ActivationOpConverter : public TensorRTOpConverter {
 public:
  ActivationOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    static const std::unordered_map<std::string, nvinfer1::ActivationType> op_map = {
      {"nn.relu", nvinfer1::ActivationType::kRELU},
      {"sigmoid", nvinfer1::ActivationType::kSIGMOID},
      {"tanh", nvinfer1::ActivationType::kTANH},
#if TRT_VERSION_GE(5, 1, 5)
      {"clip", nvinfer1::ActivationType::kCLIP},
      {"nn.leaky_relu", nvinfer1::ActivationType::kLEAKY_RELU},
#endif
    };
    auto it = op_map.find(params->op_name);
    ICHECK(it != op_map.end()) << "Unsupported activation type " << params->op_name;
    nvinfer1::IActivationLayer* act_layer =
        params->network->addActivation(*params->inputs.at(0).tensor, it->second);
#if TRT_VERSION_GE(5, 1, 5)
    if (params->op_name == "clip") {
      float a_min = std::stof(params->node.GetAttr<std::vector<std::string>>("a_min")[0]);
      float a_max = std::stof(params->node.GetAttr<std::vector<std::string>>("a_max")[0]);
      act_layer->setAlpha(a_min);
      act_layer->setBeta(a_max);
    } else if (params->op_name == "nn.leaky_relu") {
      float alpha = std::stof(params->node.GetAttr<std::vector<std::string>>("alpha")[0]);
      act_layer->setAlpha(alpha);
    }
#endif
    ICHECK(act_layer != nullptr);
    params->outputs.push_back(act_layer->getOutput(0));
  }
};

class ElementWiseBinaryOpConverter : public TensorRTOpConverter {
 public:
  ElementWiseBinaryOpConverter() : TensorRTOpConverter({kTensor, kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> op_map = {
        {"add", nvinfer1::ElementWiseOperation::kSUM},
        {"subtract", nvinfer1::ElementWiseOperation::kSUB},
        {"multiply", nvinfer1::ElementWiseOperation::kPROD},
        {"divide", nvinfer1::ElementWiseOperation::kDIV},
        {"power", nvinfer1::ElementWiseOperation::kPOW},
        {"maximum", nvinfer1::ElementWiseOperation::kMAX},
        {"minimum", nvinfer1::ElementWiseOperation::kMIN}};
    auto it = op_map.find(params->op_name);
    ICHECK(it != op_map.end()) << "Unsupported elementwise type " << params->op_name;
    // Broadcast
    auto input0 = params->inputs.at(0).tensor;
    auto input0_dims = TrtDimsToVector(input0->getDimensions());
    auto input1 = params->inputs.at(1).tensor;
    auto input1_dims = TrtDimsToVector(input1->getDimensions());
    const bool need_broadcast = input0_dims.size() != input1_dims.size();
    if (need_broadcast) {
      if (input0_dims.size() < input1_dims.size()) {
        std::vector<int> new_shape(input0_dims);
        while (new_shape.size() < input1_dims.size()) new_shape.insert(new_shape.begin(), 1);
        input0 = Reshape(params, input0, new_shape);
      } else if (input1_dims.size() < input0_dims.size()) {
        std::vector<int> new_shape(input1_dims);
        while (new_shape.size() < input0_dims.size()) new_shape.insert(new_shape.begin(), 1);
        input1 = Reshape(params, input1, new_shape);
      }
    }

    nvinfer1::IElementWiseLayer* elemwise_layer =
        params->network->addElementWise(*input0, *input1, it->second);
    ICHECK(elemwise_layer != nullptr);
    params->outputs.push_back(elemwise_layer->getOutput(0));
  }
};

class Conv1DOpConverter : public TensorRTOpConverter {
 public:
  Conv1DOpConverter() : TensorRTOpConverter({kTensor, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params->inputs.at(1).weight_shape;
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCW");
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIW");
    auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
    auto str_dilation = params->node.GetAttr<std::vector<std::string>>("dilation");
    auto str_padding = params->node.GetAttr<std::vector<std::string>>("padding");
    int groups = std::stoi(params->node.GetAttr<std::vector<std::string>>("groups")[0]);
    int channels = weight_shape[0];
    if (params->node.HasAttr("channels") &&
        !params->node.GetAttr<std::vector<std::string>>("channels")[0].empty()) {
      channels = std::stoi(params->node.GetAttr<std::vector<std::string>>("channels")[0]);
    }

    auto shuffle_layer = params->network->addShuffle(*input_tensor);
    std::vector<int> new_shape = {input_dims[0], input_dims[1], 1};
    shuffle_layer->setReshapeDimensions(VectorToTrtDims(new_shape));
    input_tensor = shuffle_layer->getOutput(0);

    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], 1);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto conv_layer = params->network->addConvolution(*input_tensor, channels, kernel_size,
                                                      params->inputs.at(1).weight, bias);
    ICHECK(conv_layer != nullptr);
    conv_layer->setPadding(nvinfer1::DimsHW(std::stoi(str_padding[0]), 0));
    ICHECK_EQ(str_strides.size(), 1);
    const auto strides = nvinfer1::DimsHW(std::stoi(str_strides[0]), 1);
    conv_layer->setStride(strides);
    ICHECK_EQ(str_dilation.size(), 1);
    const auto dilation = nvinfer1::DimsHW(std::stoi(str_dilation[0]), 1);
    conv_layer->setDilation(dilation);
    conv_layer->setNbGroups(groups);
    input_tensor = conv_layer->getOutput(0);

    auto conv_output_dims = TrtDimsToVector(input_tensor->getDimensions());
    std::vector<int> back_shape = {0, 0};
    auto shuffle_back_layer = params->network->addShuffle(*input_tensor);
    shuffle_back_layer->setReshapeDimensions(VectorToTrtDims(back_shape));
    params->outputs.push_back(shuffle_back_layer->getOutput(0));
  }
};

class Conv2DOpConverter : public TensorRTOpConverter {
 public:
  Conv2DOpConverter() : TensorRTOpConverter({kTensor, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params->inputs.at(1).weight_shape;
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
    ICHECK(params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "" ||
           params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "NCHW");
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
    auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
    auto str_dilation = params->node.GetAttr<std::vector<std::string>>("dilation");
    auto str_padding = params->node.GetAttr<std::vector<std::string>>("padding");
    int groups = std::stoi(params->node.GetAttr<std::vector<std::string>>("groups")[0]);
    int channels = weight_shape[0];
    if (params->node.HasAttr("channels") &&
        !params->node.GetAttr<std::vector<std::string>>("channels")[0].empty()) {
      channels = std::stoi(params->node.GetAttr<std::vector<std::string>>("channels")[0]);
    }
    // TRT conv2d op doesn't support asymmetric padding before 5.1, so we
    // workaround by adding a padding layer before the pooling op.
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding(str_padding, &use_asymmetric_padding, &prepadding, &postpadding);
#if !TRT_VERSION_GE(5, 1, 5)
    if (use_asymmetric_padding) {
      auto pad_layer = params->network->addPadding(*input_tensor, prepadding, postpadding);
      ICHECK(pad_layer != nullptr);
      input_tensor = pad_layer->getOutput(0);
      // No need for conv op to do any padding.
      use_asymmetric_padding = false;
      prepadding = nvinfer1::DimsHW(0, 0);
    }
#endif

    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto conv_layer = params->network->addConvolution(*input_tensor, channels, kernel_size,
                                                      params->inputs.at(1).weight, bias);
    ICHECK(conv_layer != nullptr);
    if (use_asymmetric_padding) {
#if TRT_VERSION_GE(5, 1, 5)
      conv_layer->setPrePadding(prepadding);
      conv_layer->setPostPadding(postpadding);
#endif
    } else {
      conv_layer->setPadding(prepadding);
    }
    ICHECK_EQ(str_strides.size(), 2);
    const auto strides = nvinfer1::DimsHW(std::stoi(str_strides[0]), std::stoi(str_strides[1]));
    conv_layer->setStride(strides);
    ICHECK_EQ(str_dilation.size(), 2);
    const auto dilation = nvinfer1::DimsHW(std::stoi(str_dilation[0]), std::stoi(str_dilation[1]));
    conv_layer->setDilation(dilation);
    conv_layer->setNbGroups(groups);
    params->outputs.push_back(conv_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class Conv3DOpConverter : public TensorRTOpConverter {
 public:
  Conv3DOpConverter() : TensorRTOpConverter({kTensor, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params->inputs.at(1).weight_shape;
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCDHW");
    ICHECK(params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "" ||
           params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "NCDHW");
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIDHW");
    auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
    auto str_dilation = params->node.GetAttr<std::vector<std::string>>("dilation");
    auto str_padding = params->node.GetAttr<std::vector<std::string>>("padding");
    int groups = std::stoi(params->node.GetAttr<std::vector<std::string>>("groups")[0]);

    nvinfer1::Dims prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding3D(str_padding, &use_asymmetric_padding, &prepadding, &postpadding);

    const int num_outputs =
        std::stoi(params->node.GetAttr<std::vector<std::string>>("channels")[0]);
    const auto kernel_size = nvinfer1::Dims3(weight_shape[2], weight_shape[3], weight_shape[4]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto conv_layer = params->network->addConvolutionNd(*input_tensor, num_outputs, kernel_size,
                                                        params->inputs.at(1).weight, bias);
    ICHECK(conv_layer != nullptr);
    if (use_asymmetric_padding) {
      conv_layer->setPrePadding(prepadding);
      conv_layer->setPostPadding(postpadding);
    } else {
      conv_layer->setPaddingNd(prepadding);
    }
    ICHECK_EQ(str_strides.size(), 3);
    const auto strides = nvinfer1::Dims3(std::stoi(str_strides[0]), std::stoi(str_strides[1]),
                                         std::stoi(str_strides[2]));
    conv_layer->setStrideNd(strides);
    ICHECK_EQ(str_dilation.size(), 3);
    const auto dilation = nvinfer1::Dims3(std::stoi(str_dilation[0]), std::stoi(str_dilation[1]),
                                          std::stoi(str_dilation[2]));
    conv_layer->setDilationNd(dilation);
    conv_layer->setNbGroups(groups);
    params->outputs.push_back(conv_layer->getOutput(0));
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

class DenseOpConverter : public TensorRTOpConverter {
 public:
  DenseOpConverter() : TensorRTOpConverter({kTensor, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    ICHECK(input_dims.size() > 0 && input_dims.size() <= 3);
    const size_t required_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    const bool need_reshape_on_input = input_dims.size() != required_rank;
    if (need_reshape_on_input) {
      // Add dims of size 1 until rank is required_rank.
      std::vector<int> new_shape(input_dims);
      while (new_shape.size() < required_rank) new_shape.insert(new_shape.end(), 1);
      input_tensor = Reshape(params, input_tensor, new_shape);
    }
    // Weights are in KC format.
    ICHECK_EQ(params->inputs.at(1).weight_shape.size(), 2);
    const int num_units = params->inputs.at(1).weight_shape[0];
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IFullyConnectedLayer* fc_layer = params->network->addFullyConnected(
        *input_tensor, num_units, params->inputs.at(1).weight, bias);
    ICHECK(fc_layer != nullptr);
    auto output_tensor = fc_layer->getOutput(0);
    if (need_reshape_on_input) {
      // Remove added dims.
      input_dims[input_dims.size() - 1] = num_units;
      output_tensor = Reshape(params, output_tensor, input_dims);
    }
    params->outputs.push_back(output_tensor);
  }
};

class BatchNormOpConverter : public TensorRTOpConverter {
 public:
  BatchNormOpConverter() : TensorRTOpConverter({kTensor, kWeight, kWeight, kWeight, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto gamma = params->inputs.at(1).weight;
    auto beta = params->inputs.at(2).weight;
    auto mean = params->inputs.at(3).weight;
    auto var = params->inputs.at(4).weight;
    ICHECK_EQ(gamma.count, beta.count);
    ICHECK_EQ(gamma.count, mean.count);
    ICHECK_EQ(gamma.count, var.count);
    const float epsilon = std::stof(params->node.GetAttr<std::vector<std::string>>("epsilon")[0]);
    const int axis = std::stoi(params->node.GetAttr<std::vector<std::string>>("axis")[0]);
    const bool scale = std::stoi(params->node.GetAttr<std::vector<std::string>>("scale")[0]);
    const bool center = std::stoi(params->node.GetAttr<std::vector<std::string>>("center")[0]);
    auto input_dims = TrtDimsToVector(input->getDimensions());
    const size_t min_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    const size_t max_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 4 : 5;
    ICHECK_LE(input_dims.size(), max_rank);
    const bool need_reshape = input_dims.size() < min_rank;
    const bool need_transpose = axis != 1;

    // Reshape if needed
    if (need_reshape) {
      // Add dims of size 1 until rank is required_rank.
      std::vector<int> new_shape(input_dims);
      while (new_shape.size() < min_rank) new_shape.insert(new_shape.end(), 1);
      input = Reshape(params, input, new_shape);
    }

    // Transpose if needed.
    const int input_rank_with_batch =
        input->getDimensions().nbDims + (TRT_HAS_IMPLICIT_BATCH(params) ? 1 : 0);
    ICHECK(input_rank_with_batch == 4 || input_rank_with_batch == 5);
    std::vector<int> transpose_order(input_rank_with_batch);
    if (need_transpose) {
      // Move axis dim to first dim after batch.
      for (int i = 0; i < input_rank_with_batch; ++i) {
        transpose_order[i] = i;
      }
      transpose_order[1] = axis;
      transpose_order[axis] = 1;
      input = Transpose(params, input, transpose_order);
    }

    void* weight_scale_ptr = new float[gamma.count];
    nvinfer1::Weights weight_scale{nvinfer1::DataType::kFLOAT, weight_scale_ptr, gamma.count};
    params->trt_weights->push_back(weight_scale);
    void* weight_shift_ptr = new float[gamma.count];
    nvinfer1::Weights weight_shift{nvinfer1::DataType::kFLOAT, weight_shift_ptr, gamma.count};
    params->trt_weights->push_back(weight_shift);
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    // fill in the content of weights for the Scale layer
    const float* gamma_ptr = reinterpret_cast<const float*>(gamma.values);
    const float* beta_ptr = reinterpret_cast<const float*>(beta.values);
    const float* mean_ptr = reinterpret_cast<const float*>(mean.values);
    const float* var_ptr = reinterpret_cast<const float*>(var.values);
    float* scale_ptr = reinterpret_cast<float*>(weight_scale_ptr);
    float* shift_ptr = reinterpret_cast<float*>(weight_shift_ptr);
    for (int i = 0; i < gamma.count; ++i) {
      scale_ptr[i] = 1.0 / std::sqrt(var_ptr[i] + epsilon);
      if (scale) {
        scale_ptr[i] *= gamma_ptr[i];
      }
      shift_ptr[i] = -mean_ptr[i] * scale_ptr[i];
      if (center) {
        shift_ptr[i] += beta_ptr[i];
      }
    }

#if TRT_VERSION_GE(6, 0, 1)
    const int channel_dim = TRT_HAS_IMPLICIT_BATCH(params) ? 0 : 1;
    nvinfer1::IScaleLayer* scale_layer = params->network->addScaleNd(
        *input, nvinfer1::ScaleMode::kCHANNEL, weight_shift, weight_scale, power, channel_dim);
#else
    ICHECK_EQ(input->getDimensions().nbDims, 3);
    nvinfer1::IScaleLayer* scale_layer = params->network->addScale(
        *input, nvinfer1::ScaleMode::kCHANNEL, weight_shift, weight_scale, power);
#endif
    ICHECK(scale_layer != nullptr);
    auto output = scale_layer->getOutput(0);
    if (need_transpose) {
      output = Transpose(params, output, transpose_order);
    }
    if (need_reshape) {
      output = Reshape(params, output, input_dims);
    }
    params->outputs.push_back(output);
  }
};

class LayerNormOpConverter : public TensorRTOpConverter {
 public:
  LayerNormOpConverter() : TensorRTOpConverter({kTensor, kWeight, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto gamma_input = params->inputs.at(1).weight;
    auto beta_input = params->inputs.at(2).weight;
    ICHECK_EQ(gamma_input.count, beta_input.count);

    const float epsilon = std::stof(params->node.GetAttr<std::vector<std::string>>("epsilon")[0]);
    const bool scale = std::stoi(params->node.GetAttr<std::vector<std::string>>("scale")[0]);
    const bool center = std::stoi(params->node.GetAttr<std::vector<std::string>>("center")[0]);
    const int input_rank = input->getDimensions().nbDims;
    const int original_axis = std::stoi(params->node.GetAttr<std::vector<std::string>>("axis")[0]);
    const int axis = ConvertAxis(params, original_axis, input_rank);

    std::vector<int> weight_shape(input_rank, 1);
    weight_shape[axis] = gamma_input.count;
    auto gamma =
        params->network->addConstant(VectorToTrtDims(weight_shape), gamma_input)->getOutput(0);
    auto beta =
        params->network->addConstant(VectorToTrtDims(weight_shape), beta_input)->getOutput(0);

    // Compute mean
    auto mean_layer = params->network->addReduce(*input, nvinfer1::ReduceOperation::kAVG, 1 << axis,
                                                 /*keepdims=*/true);
    ICHECK(mean_layer != nullptr);
    auto mean = mean_layer->getOutput(0);
    // Compute variance
    auto diff_layer =
        params->network->addElementWise(*input, *mean, nvinfer1::ElementWiseOperation::kSUB);
    ICHECK(diff_layer != nullptr);
    auto square_layer =
        params->network->addElementWise(*diff_layer->getOutput(0), *diff_layer->getOutput(0),
                                        nvinfer1::ElementWiseOperation::kPROD);
    ICHECK(square_layer != nullptr);
    auto var_layer = params->network->addReduce(
        *square_layer->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1 << axis, /*keepdims=*/true);
    ICHECK(var_layer != nullptr);
    auto var = var_layer->getOutput(0);
    // sqrt(var + epsilon)
    auto epsilon_tensor = CreateScalar(params, epsilon, var->getDimensions());
    auto denom_add_layer = params->network->addElementWise(*var, *epsilon_tensor,
                                                           nvinfer1::ElementWiseOperation::kSUM);
    ICHECK(denom_add_layer != nullptr);
    auto denom_layer =
        params->network->addUnary(*denom_add_layer->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    ICHECK(denom_layer != nullptr);
    // (input - mean) / sqrt(var + epsilon)
    auto output_layer =
        params->network->addElementWise(*diff_layer->getOutput(0), *denom_layer->getOutput(0),
                                        nvinfer1::ElementWiseOperation::kDIV);
    ICHECK(output_layer != nullptr);
    auto output = output_layer->getOutput(0);

    if (scale) {
      auto scale_layer =
          params->network->addElementWise(*output, *gamma, nvinfer1::ElementWiseOperation::kPROD);
      ICHECK(scale_layer != nullptr);
      output = scale_layer->getOutput(0);
    }
    if (center) {
      auto center_layer =
          params->network->addElementWise(*output, *beta, nvinfer1::ElementWiseOperation::kSUM);
      ICHECK(center_layer != nullptr);
      output = center_layer->getOutput(0);
    }
    params->outputs.push_back(output);
  }
};

class BatchFlattenOpConverter : public TensorRTOpConverter {
 public:
  BatchFlattenOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    std::vector<int> new_shape{-1};
    if (!TRT_HAS_IMPLICIT_BATCH(params)) {
      new_shape.insert(new_shape.begin(), params->inputs.at(0).tensor->getDimensions().d[0]);
    }
    params->outputs.push_back(Reshape(params, params->inputs.at(0).tensor, new_shape));
  }
};

class SoftmaxOpConverter : public TensorRTOpConverter {
 public:
  SoftmaxOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    const int input_rank = input->getDimensions().nbDims;
    const int original_axis = std::stoi(params->node.GetAttr<std::vector<std::string>>("axis")[0]);
    const int axis = ConvertAxis(params, original_axis, input_rank);
    nvinfer1::ISoftMaxLayer* softmax_layer = params->network->addSoftMax(*input);
    softmax_layer->setAxes(1 << axis);
    ICHECK(softmax_layer != nullptr);
    params->outputs.push_back(softmax_layer->getOutput(0));
  }
};

class PoolingOpConverter : public TensorRTOpConverter {
 public:
  PoolingOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params->op_name);
    ICHECK(it != op_map.end()) << "Unsupported pooling type " << params->op_name << " in TensorRT";
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("layout")[0], "NCHW");
    auto str_pool_size = params->node.GetAttr<std::vector<std::string>>("pool_size");
    auto str_padding = params->node.GetAttr<std::vector<std::string>>("padding");
    auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding(str_padding, &use_asymmetric_padding, &prepadding, &postpadding);
    bool ceil_mode = std::stoi(params->node.GetAttr<std::vector<std::string>>("ceil_mode")[0]);

// TRT pooling op doesn't support asymmetric padding before 5.1, so we
// workaround by adding a padding layer before the pooling op.
#if !TRT_VERSION_GE(5, 1, 5)
    if (use_asymmetric_padding) {
      auto pad_layer = params->network->addPadding(*input, prepadding, postpadding);
      ICHECK(pad_layer != nullptr);
      input = pad_layer->getOutput(0);
      // No need for pooling op to do any padding.
      use_asymmetric_padding = false;
      prepadding = nvinfer1::DimsHW(0, 0);
    }
#endif

    nvinfer1::DimsHW window_size =
        nvinfer1::DimsHW(std::stoi(str_pool_size[0]), std::stoi(str_pool_size[1]));
    auto pool_layer = params->network->addPooling(*input, it->second, window_size);
    ICHECK(pool_layer != nullptr);
    nvinfer1::DimsHW strides =
        nvinfer1::DimsHW(std::stoi(str_strides[0]), std::stoi(str_strides[1]));
    pool_layer->setStride(strides);
    if (use_asymmetric_padding) {
#if TRT_VERSION_GE(5, 1, 5)
      pool_layer->setPrePadding(prepadding);
      pool_layer->setPostPadding(postpadding);
#endif
    } else {
      pool_layer->setPadding(prepadding);
    }
    if (params->op_name == "nn.avg_pool2d") {
      bool count_include_pad =
          std::stoi(params->node.GetAttr<std::vector<std::string>>("count_include_pad")[0]);
      // count_include_pad=True is useless if there is no padding. TRT doesn't
      // like count_include_pad in combination with strides even when there is
      // no padding or assymetric padding even, so turn off inclusive to avoid
      // error message. Note: Padding will always be symmetric with
      // count_include_pad since partitioner will prevent unsupported case.
      if (prepadding.h() == 0 && prepadding.w() == 0) {
        count_include_pad = false;
      }
      pool_layer->setAverageCountExcludesPadding(!count_include_pad);
    }
#if TRT_VERSION_GE(5, 1, 5)
    if (ceil_mode) {
      pool_layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP);
    }
#else
    ICHECK(!ceil_mode);
#endif
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class Pooling3DOpConverter : public TensorRTOpConverter {
 public:
  Pooling3DOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.max_pool3d", nvinfer1::PoolingType::kMAX},
        {"nn.avg_pool3d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params->op_name);
    ICHECK(it != op_map.end()) << "Unsupported pooling type " << params->op_name << " in TensorRT";
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("layout")[0], "NCDHW");
    auto str_pool_size = params->node.GetAttr<std::vector<std::string>>("pool_size");
    auto str_padding = params->node.GetAttr<std::vector<std::string>>("padding");
    auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding3D(str_padding, &use_asymmetric_padding, &prepadding, &postpadding);
    bool ceil_mode = std::stoi(params->node.GetAttr<std::vector<std::string>>("ceil_mode")[0]);
    nvinfer1::Dims window_size = nvinfer1::Dims3(
        std::stoi(str_pool_size[0]), std::stoi(str_pool_size[1]), std::stoi(str_pool_size[2]));
    auto pool_layer = params->network->addPoolingNd(*input, it->second, window_size);
    ICHECK(pool_layer != nullptr);
    nvinfer1::Dims strides = nvinfer1::Dims3(std::stoi(str_strides[0]), std::stoi(str_strides[1]),
                                             std::stoi(str_strides[2]));
    pool_layer->setStrideNd(strides);
    if (use_asymmetric_padding) {
      pool_layer->setPrePadding(prepadding);
      pool_layer->setPostPadding(postpadding);
    } else {
      pool_layer->setPaddingNd(prepadding);
    }
    if (params->op_name == "nn.avg_pool3d") {
      bool count_include_pad =
          std::stoi(params->node.GetAttr<std::vector<std::string>>("count_include_pad")[0]);
      pool_layer->setAverageCountExcludesPadding(!count_include_pad);
    }
    if (ceil_mode) {
      pool_layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP);
    }
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

class GlobalPoolingOpConverter : public TensorRTOpConverter {
 public:
  GlobalPoolingOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.global_max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.global_avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params->op_name);
    ICHECK(it != op_map.end()) << "Unsupported pooling type " << params->op_name << " in TensorRT";
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("layout")[0], "NCHW");
    const int h = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[1] : input_dims[2];
    const int w = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[2] : input_dims[3];
    auto pool_layer =
        params->network->addPooling(*input_tensor, it->second, nvinfer1::DimsHW(h, w));
    ICHECK(pool_layer != nullptr);
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

class ExpandDimsOpConverter : public TensorRTOpConverter {
 public:
  ExpandDimsOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    const int original_axis = std::stoi(params->node.GetAttr<std::vector<std::string>>("axis")[0]);
    const int num_newaxis =
        std::stoi(params->node.GetAttr<std::vector<std::string>>("num_newaxis")[0]);
    const int axis = ConvertAxis(params, original_axis, input_dims.size() + 1);
    for (int i = 0; i < num_newaxis; ++i) {
      input_dims.insert(input_dims.begin() + axis, 1);
    }
    params->outputs.push_back(Reshape(params, params->inputs.at(0).tensor, input_dims));
  }
};

class SqueezeOpConverter : public TensorRTOpConverter {
 public:
  SqueezeOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto str_axis = params->node.GetAttr<std::vector<std::string>>("axis");
    for (size_t i = 0; i < str_axis.size(); ++i) {
      const int axis = ConvertAxis(params, std::stoi(str_axis[i]), input_dims.size());
      input_dims[axis] = 0;
    }
    input_dims.erase(std::remove(input_dims.begin(), input_dims.end(), 0), input_dims.end());
    params->outputs.push_back(Reshape(params, params->inputs.at(0).tensor, input_dims));
  }
};

class UnaryOpConverter : public TensorRTOpConverter {
 public:
  UnaryOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    // The following ops are supported by TRT but don't exist in relay yet:
    // recip, tan, sinh, cosh, asin, acos, asinh, acosh, atanh
    static const std::unordered_map<std::string, nvinfer1::UnaryOperation> op_map = {
      {"exp", nvinfer1::UnaryOperation::kEXP},
      {"log", nvinfer1::UnaryOperation::kLOG},
      {"sqrt", nvinfer1::UnaryOperation::kSQRT},
      {"abs", nvinfer1::UnaryOperation::kABS},
      {"negative", nvinfer1::UnaryOperation::kNEG},
#if TRT_VERSION_GE(5, 1, 5)
      {"sin", nvinfer1::UnaryOperation::kSIN},
      {"cos", nvinfer1::UnaryOperation::kCOS},
      {"atan", nvinfer1::UnaryOperation::kATAN},
      {"ceil", nvinfer1::UnaryOperation::kCEIL},
      {"floor", nvinfer1::UnaryOperation::kFLOOR},
#endif
#if TRT_VERSION_GE(7, 0, 0)
      {"erf", nvinfer1::UnaryOperation::kERF},
#endif
    };
    auto it = op_map.find(params->op_name);
    ICHECK(it != op_map.end()) << "Unsupported unary type " << params->op_name;
    nvinfer1::IUnaryLayer* unary_layer =
        params->network->addUnary(*params->inputs.at(0).tensor, it->second);
    ICHECK(unary_layer != nullptr);
    params->outputs.push_back(unary_layer->getOutput(0));
  }
};

class ConcatOpConverter : public TensorRTOpConverter {
 public:
  ConcatOpConverter() : TensorRTOpConverter({}, /*variable_input_count=*/true) {}

  void Convert(TensorRTOpConverterParams* params) const {
    const int num_inputs = params->inputs.size();
    ICHECK_GT(num_inputs, 0);
    const int input_rank = params->inputs[0].tensor->getDimensions().nbDims;
    std::vector<nvinfer1::ITensor*> input_tensors;
    for (auto input : params->inputs) {
      ICHECK(input.type == kTensor);
      ICHECK_EQ(input_rank, input.tensor->getDimensions().nbDims);
      input_tensors.push_back(input.tensor);
    }

    const int original_axis = std::stoi(params->node.GetAttr<std::vector<std::string>>("axis")[0]);
    const int axis = ConvertAxis(params, original_axis, input_rank);

    nvinfer1::IConcatenationLayer* concat_layer =
        params->network->addConcatenation(input_tensors.data(), input_tensors.size());
    ICHECK(concat_layer != nullptr);
    concat_layer->setAxis(axis);
    params->outputs.push_back(concat_layer->getOutput(0));
  }
};

class SplitOpConverter : public TensorRTOpConverter {
 public:
  SplitOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    const int original_axis = std::stoi(params->node.GetAttr<std::vector<std::string>>("axis")[0]);
    const int axis = ConvertAxis(params, original_axis, input_dims.size());
    auto indices_or_sections =
        params->node.GetAttr<std::vector<std::string>>("indices_or_sections");
    auto mode = params->node.GetAttr<std::vector<std::string>>("mode")[0];

    std::vector<int> split_starts;
    std::vector<int> split_sizes;
    if (mode == "sections") {
      int sections = std::stoi(indices_or_sections[0]);
      int size = input_dims[axis] / sections;
      for (int i = 0; i < sections; i++) {
        split_starts.push_back(i * size);
        split_sizes.push_back(size);
      }
    } else {
      int last_index = 0;
      for (size_t i = 0; i < indices_or_sections.size(); ++i) {
        int index = std::stoi(indices_or_sections[i]);
        split_starts.push_back(last_index);
        split_sizes.push_back(index - last_index);
        last_index = index;
      }
      split_starts.push_back(last_index);
      split_sizes.push_back(input_dims[axis] - last_index);
    }

    std::vector<int> start(input_dims.size(), 0);
    std::vector<int> size(input_dims.begin(), input_dims.end());
    std::vector<int> strides(input_dims.size(), 1);
    for (size_t i = 0; i < split_sizes.size(); ++i) {
      start[axis] = split_starts[i];
      size[axis] = split_sizes[i];
      auto slice_layer = params->network->addSlice(*input, VectorToTrtDims(start),
                                                   VectorToTrtDims(size), VectorToTrtDims(strides));
      params->outputs.push_back(slice_layer->getOutput(0));
    }
  }
};

class BiasAddOpConverter : public TensorRTOpConverter {
 public:
  BiasAddOpConverter() : TensorRTOpConverter({kTensor, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    const size_t required_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    ICHECK(input_dims.size() > 0 && input_dims.size() <= required_rank);
    const bool need_reshape_on_input = input_dims.size() != required_rank;
    if (need_reshape_on_input) {
      // Add dims of size 1 until rank is required_rank.
      std::vector<int> new_shape(input_dims);
      while (new_shape.size() < required_rank) new_shape.insert(new_shape.end(), 1);
      input_tensor = Reshape(params, input_tensor, new_shape);
    }

    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IScaleLayer* scale_layer = params->network->addScale(
        *input_tensor, nvinfer1::ScaleMode::kCHANNEL, params->inputs.at(1).weight, shift, power);
    ICHECK(scale_layer != nullptr);
    auto output_tensor = scale_layer->getOutput(0);
    if (need_reshape_on_input) {
      // Remove added dims.
      output_tensor = Reshape(params, output_tensor, input_dims);
    }
    params->outputs.push_back(output_tensor);
  }
};

class Conv2DTransposeOpConverter : public TensorRTOpConverter {
 public:
  Conv2DTransposeOpConverter() : TensorRTOpConverter({kTensor, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto weight_shape = params->inputs.at(1).weight_shape;
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
    ICHECK(params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "" ||
           params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "NCHW");
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
    auto str_dilation = params->node.GetAttr<std::vector<std::string>>("dilation");
    ICHECK(std::stoi(str_dilation[0]) == 1 && std::stoi(str_dilation[1]) == 1);
    auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
    auto str_padding = params->node.GetAttr<std::vector<std::string>>("padding");
    auto str_output_padding = params->node.GetAttr<std::vector<std::string>>("output_padding");
    int groups = std::stoi(params->node.GetAttr<std::vector<std::string>>("groups")[0]);

    // TRT deconv op doesn't support asymmetric padding before 5.1, so we
    // workaround by adding a padding layer before the pooling op.
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding(str_padding, &use_asymmetric_padding, &prepadding, &postpadding);
#if !TRT_VERSION_GE(5, 1, 5)
    if (use_asymmetric_padding) {
      auto pad_layer = params->network->addPadding(*input_tensor, prepadding, postpadding);
      ICHECK(pad_layer != nullptr);
      input_tensor = pad_layer->getOutput(0);
      // No need for conv op to do any padding.
      use_asymmetric_padding = false;
      prepadding = nvinfer1::DimsHW(0, 0);
    }
#endif

    const int num_outputs =
        std::stoi(params->node.GetAttr<std::vector<std::string>>("channels")[0]);
    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto deconv_layer = params->network->addDeconvolution(*input_tensor, num_outputs, kernel_size,
                                                          params->inputs.at(1).weight, bias);
    ICHECK(deconv_layer != nullptr);
    if (use_asymmetric_padding) {
#if TRT_VERSION_GE(5, 1, 5)
      deconv_layer->setPrePadding(prepadding);
      deconv_layer->setPostPadding(postpadding);
#endif
    } else {
      deconv_layer->setPadding(prepadding);
    }
    const auto strides = nvinfer1::DimsHW(std::stoi(str_strides[0]), std::stoi(str_strides[1]));
    deconv_layer->setStride(strides);
    deconv_layer->setNbGroups(groups);
    nvinfer1::ITensor* output = deconv_layer->getOutput(0);
    // Output padding.
    if (str_output_padding.size()) {
      GetPadding(str_output_padding, &use_asymmetric_padding, &prepadding, &postpadding);
      if (prepadding.h() != 0 || prepadding.w() != 0 || postpadding.h() != 0 ||
          postpadding.w() != 0) {
        // Output padding for Conv2D transpose is always asymmetric and applied to post only.
        prepadding = nvinfer1::DimsHW(0, 0);
        auto pad_layer = params->network->addPadding(*output, prepadding, postpadding);
        output = pad_layer->getOutput(0);
      }
    }
    params->outputs.push_back(output);
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class Conv3DTransposeOpConverter : public TensorRTOpConverter {
 public:
  Conv3DTransposeOpConverter() : TensorRTOpConverter({kTensor, kWeight}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto weight_shape = params->inputs.at(1).weight_shape;
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCDHW");
    ICHECK(params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "" ||
           params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "NCDHW");
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIDHW");
    auto str_dilation = params->node.GetAttr<std::vector<std::string>>("dilation");
    ICHECK_EQ(str_dilation.size(), 3);
    ICHECK(std::stoi(str_dilation[0]) == 1 && std::stoi(str_dilation[1]) == 1 &&
           std::stoi(str_dilation[2]) == 1);
    auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
    auto str_padding = params->node.GetAttr<std::vector<std::string>>("padding");
    auto str_output_padding = params->node.GetAttr<std::vector<std::string>>("output_padding");
    int groups = std::stoi(params->node.GetAttr<std::vector<std::string>>("groups")[0]);
    nvinfer1::Dims prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding3D(str_padding, &use_asymmetric_padding, &prepadding, &postpadding);

    const int num_outputs =
        std::stoi(params->node.GetAttr<std::vector<std::string>>("channels")[0]);
    const auto kernel_size = nvinfer1::Dims3(weight_shape[2], weight_shape[3], weight_shape[4]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto deconv_layer = params->network->addDeconvolutionNd(*input_tensor, num_outputs, kernel_size,
                                                            params->inputs.at(1).weight, bias);
    ICHECK(deconv_layer != nullptr);
    if (use_asymmetric_padding) {
      deconv_layer->setPrePadding(prepadding);
      deconv_layer->setPostPadding(postpadding);
    } else {
      deconv_layer->setPaddingNd(prepadding);
    }
    ICHECK_EQ(str_strides.size(), 3);
    const auto strides = nvinfer1::Dims3(std::stoi(str_strides[0]), std::stoi(str_strides[1]),
                                         std::stoi(str_strides[2]));
    deconv_layer->setStrideNd(strides);
    deconv_layer->setNbGroups(groups);
    nvinfer1::ITensor* output = deconv_layer->getOutput(0);
    // Output padding.
    if (str_output_padding.size()) {
      GetPadding3D(str_output_padding, &use_asymmetric_padding, &prepadding, &postpadding);
      // Are any post-padding values non-zero?
      ICHECK(!std::any_of(postpadding.d, postpadding.d + postpadding.nbDims, [](int x) {
        return x != 0;
      })) << "TRT does not support padding on 3 dimensions.";
    }
    params->outputs.push_back(output);
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

class TransposeOpConverter : public TensorRTOpConverter {
 public:
  TransposeOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto str_axes = params->node.GetAttr<std::vector<std::string>>("axes");
    std::vector<int> order;
    for (size_t i = 0; i < str_axes.size(); ++i) {
      order.push_back(std::stoi(str_axes[i]));
    }
    params->outputs.push_back(Transpose(params, input, order));
  }
};

class LayoutTransformOpConverter : public TensorRTOpConverter {
 public:
  LayoutTransformOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto src = params->node.GetAttr<std::vector<std::string>>("src_layout")[0];
    auto dst = params->node.GetAttr<std::vector<std::string>>("dst_layout")[0];
    std::vector<int> order;
    if (src == "NCHW" && dst == "NHWC") {
      order = {0, 2, 3, 1};
    } else if (src == "NHWC" && dst == "NCHW") {
      order = {0, 3, 1, 2};
    } else if (src == "NDHWC" && dst == "NCDHW") {
      order = {0, 4, 1, 2, 3};
    } else if (src == "NCDHW" && dst == "NDHWC") {
      order = {0, 2, 3, 4, 1};
    }
    params->outputs.push_back(Transpose(params, input, order));
  }
};

class ReshapeOpConverter : public TensorRTOpConverter {
 public:
  ReshapeOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto str_newshape = params->node.GetAttr<std::vector<std::string>>("newshape");
    std::vector<int> new_shape;
    const int start_index = TRT_HAS_IMPLICIT_BATCH(params) ? 1 : 0;
    for (size_t i = start_index; i < str_newshape.size(); ++i) {
      const int value = std::stoi(str_newshape[i]);
      ICHECK_GE(value, -1);
      new_shape.push_back(value);
    }
    params->outputs.push_back(Reshape(params, input, new_shape));
  }
};

class PadOpConverter : public TensorRTOpConverter {
 public:
  PadOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto str_paddding = params->node.GetAttr<std::vector<std::string>>("padding");
    nvinfer1::DimsHW prepadding =
        nvinfer1::DimsHW(std::stoi(str_paddding[0]), std::stoi(str_paddding[1]));
    nvinfer1::DimsHW postpadding =
        nvinfer1::DimsHW(std::stoi(str_paddding[2]), std::stoi(str_paddding[3]));
    auto pad_layer = params->network->addPadding(*input, prepadding, postpadding);
    params->outputs.push_back(pad_layer->getOutput(0));
  }
};

class ReduceOpConverter : public TensorRTOpConverter {
 public:
  ReduceOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    static const std::unordered_map<std::string, nvinfer1::ReduceOperation> op_map = {
        {"sum", nvinfer1::ReduceOperation::kSUM},
        {"prod", nvinfer1::ReduceOperation::kPROD},
        {"max", nvinfer1::ReduceOperation::kMAX},
        {"min", nvinfer1::ReduceOperation::kMIN},
        {"mean", nvinfer1::ReduceOperation::kAVG}};
    auto it = op_map.find(params->op_name);
    ICHECK(it != op_map.end()) << "Unsupported reduce type " << params->op_name;

    auto input = params->inputs.at(0).tensor;
    ICHECK_EQ(std::stoi(params->node.GetAttr<std::vector<std::string>>("exclude")[0]), false);
    bool keepdims = std::stoi(params->node.GetAttr<std::vector<std::string>>("keepdims")[0]);
    auto str_axis = params->node.GetAttr<std::vector<std::string>>("axis");
    // TODO(trevmorr): Support reduce to scalar.
    ICHECK_GT(str_axis.size(), 0);
    uint32_t reduce_axes = 0;

    if (str_axis.size() == 1 && str_axis[0].length() == 0) {
      // Reduce to scalar
      for (int i = 0; i < input->getDimensions().nbDims; ++i) {
        reduce_axes |= 1 << i;
      }
    } else {
      for (size_t i = 0; i < str_axis.size(); ++i) {
        const int axis = ConvertAxis(params, std::stoi(str_axis[i]), input->getDimensions().nbDims);
        reduce_axes |= 1 << axis;
      }
    }
    auto reduce_layer = params->network->addReduce(*input, it->second, reduce_axes, keepdims);
    params->outputs.push_back(reduce_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(5, 1, 5)
class StridedSliceOpConverter : public TensorRTOpConverter {
 public:
  StridedSliceOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    auto str_start = params->node.GetAttr<std::vector<std::string>>("start");
    auto str_size = params->node.GetAttr<std::vector<std::string>>("size");
    auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
    std::vector<int> start, size, strides;
    std::transform(str_start.begin(), str_start.end(), std::back_inserter(start),
                   [](const std::string& s) { return std::stoi(s); });
    std::transform(str_size.begin(), str_size.end(), std::back_inserter(size),
                   [](const std::string& s) { return std::stoi(s); });
    std::transform(str_strides.begin(), str_strides.end(), std::back_inserter(strides),
                   [](const std::string& s) { return std::stoi(s); });
    if (TRT_HAS_IMPLICIT_BATCH(params)) {
      start.erase(start.begin());
      size.erase(size.begin());
      strides.erase(strides.begin());
    }
    auto slice_layer = params->network->addSlice(*input, VectorToTrtDims(start),
                                                 VectorToTrtDims(size), VectorToTrtDims(strides));
    params->outputs.push_back(slice_layer->getOutput(0));
  }
};
#endif

class AdaptivePoolingOpConverter : public TensorRTOpConverter {
 public:
  AdaptivePoolingOpConverter() : TensorRTOpConverter({kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.adaptive_max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.adaptive_avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params->op_name);
    ICHECK(it != op_map.end()) << "Unsupported pooling type " << params->op_name << " in TensorRT";
    ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("layout")[0], "NCHW");

    // This is an approximation of adaptive pooling. Results will not be
    // mathematically exact except when output_size is (1, 1).
    // Annotation rules will only allow output size of (1, 1).
    auto output_size = nvinfer1::DimsHW(1, 1);
    const int h = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[1] : input_dims[2];
    const int w = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[2] : input_dims[3];
    const auto stride = nvinfer1::DimsHW(h / output_size.h(), w / output_size.w());
    const auto window_size = nvinfer1::DimsHW(h - (output_size.h() - 1) * stride.h(),
                                              w - (output_size.w() - 1) * stride.w());
    auto pool_layer = params->network->addPooling(*input_tensor, it->second, window_size);
    ICHECK(pool_layer != nullptr);
    pool_layer->setStride(stride);
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

class BatchMatmulOpConverter : public TensorRTOpConverter {
 public:
  BatchMatmulOpConverter() : TensorRTOpConverter({kTensor, kTensor}) {}

  void Convert(TensorRTOpConverterParams* params) const {
    auto transa = std::stoi(params->node.GetAttr<std::vector<std::string>>("transpose_a")[0]);
    auto transb = std::stoi(params->node.GetAttr<std::vector<std::string>>("transpose_b")[0]);
    nvinfer1::MatrixOperation trt_transa =
        transa ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
    nvinfer1::MatrixOperation trt_transb =
        transb ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
    nvinfer1::IMatrixMultiplyLayer* matmul_layer = params->network->addMatrixMultiply(
        *params->inputs.at(0).tensor, trt_transa, *params->inputs.at(1).tensor, trt_transb);
    ICHECK(matmul_layer != nullptr);
    params->outputs.push_back(matmul_layer->getOutput(0));
  }
};

const std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<TensorRTOpConverter>>>
GetOpConverters() {
  static auto map =
      std::make_shared<std::unordered_map<std::string, std::shared_ptr<TensorRTOpConverter>>>();
  if (!map->empty()) return map;
  map->emplace("nn.relu", std::make_shared<ActivationOpConverter>());
  map->emplace("sigmoid", std::make_shared<ActivationOpConverter>());
  map->emplace("tanh", std::make_shared<ActivationOpConverter>());
  map->emplace("nn.batch_norm", std::make_shared<BatchNormOpConverter>());
  map->emplace("nn.layer_norm", std::make_shared<LayerNormOpConverter>());
  map->emplace("nn.softmax", std::make_shared<SoftmaxOpConverter>());
  map->emplace("nn.conv1d", std::make_shared<Conv1DOpConverter>());
  map->emplace("nn.conv2d", std::make_shared<Conv2DOpConverter>());
  map->emplace("nn.dense", std::make_shared<DenseOpConverter>());
  map->emplace("nn.bias_add", std::make_shared<BiasAddOpConverter>());
  map->emplace("add", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("subtract", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("multiply", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("divide", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("power", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("maximum", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("minimum", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("nn.max_pool2d", std::make_shared<PoolingOpConverter>());
  map->emplace("nn.avg_pool2d", std::make_shared<PoolingOpConverter>());
  map->emplace("nn.global_max_pool2d", std::make_shared<GlobalPoolingOpConverter>());
  map->emplace("nn.global_avg_pool2d", std::make_shared<GlobalPoolingOpConverter>());
  map->emplace("exp", std::make_shared<UnaryOpConverter>());
  map->emplace("log", std::make_shared<UnaryOpConverter>());
  map->emplace("sqrt", std::make_shared<UnaryOpConverter>());
  map->emplace("abs", std::make_shared<UnaryOpConverter>());
  map->emplace("negative", std::make_shared<UnaryOpConverter>());
  map->emplace("nn.batch_flatten", std::make_shared<BatchFlattenOpConverter>());
  map->emplace("expand_dims", std::make_shared<ExpandDimsOpConverter>());
  map->emplace("squeeze", std::make_shared<SqueezeOpConverter>());
  map->emplace("concatenate", std::make_shared<ConcatOpConverter>());
  map->emplace("split", std::make_shared<SplitOpConverter>());
  map->emplace("nn.conv2d_transpose", std::make_shared<Conv2DTransposeOpConverter>());
  map->emplace("transpose", std::make_shared<TransposeOpConverter>());
  map->emplace("layout_transform", std::make_shared<LayoutTransformOpConverter>());
  map->emplace("reshape", std::make_shared<ReshapeOpConverter>());
  map->emplace("nn.pad", std::make_shared<PadOpConverter>());
  map->emplace("sum", std::make_shared<ReduceOpConverter>());
  map->emplace("prod", std::make_shared<ReduceOpConverter>());
  map->emplace("max", std::make_shared<ReduceOpConverter>());
  map->emplace("min", std::make_shared<ReduceOpConverter>());
  map->emplace("mean", std::make_shared<ReduceOpConverter>());
  map->emplace("nn.adaptive_max_pool2d", std::make_shared<AdaptivePoolingOpConverter>());
  map->emplace("nn.adaptive_avg_pool2d", std::make_shared<AdaptivePoolingOpConverter>());
  map->emplace("nn.batch_matmul", std::make_shared<BatchMatmulOpConverter>());
#if TRT_VERSION_GE(5, 1, 5)
  map->emplace("clip", std::make_shared<ActivationOpConverter>());
  map->emplace("nn.leaky_relu", std::make_shared<ActivationOpConverter>());
  map->emplace("sin", std::make_shared<UnaryOpConverter>());
  map->emplace("cos", std::make_shared<UnaryOpConverter>());
  map->emplace("atan", std::make_shared<UnaryOpConverter>());
  map->emplace("ceil", std::make_shared<UnaryOpConverter>());
  map->emplace("floor", std::make_shared<UnaryOpConverter>());
  map->emplace("strided_slice", std::make_shared<StridedSliceOpConverter>());
#endif  // TRT_VERSION_GE(5, 1, 5)
#if TRT_VERSION_GE(6, 0, 1)
  map->emplace("nn.conv3d", std::make_shared<Conv3DOpConverter>());
  map->emplace("nn.max_pool3d", std::make_shared<Pooling3DOpConverter>());
  map->emplace("nn.avg_pool3d", std::make_shared<Pooling3DOpConverter>());
  map->emplace("nn.conv3d_transpose", std::make_shared<Conv3DTransposeOpConverter>());
#endif  // TRT_VERSION_GE(6, 0, 1)
#if TRT_VERSION_GE(7, 0, 0)
  map->emplace("erf", std::make_shared<UnaryOpConverter>());
#endif  // TRT_VERSION_GE(7, 0, 0)
  return map;
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
