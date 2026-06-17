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
 * \brief Converters from ops into TensorRT layers. Converters should
 * inherit from TensorRTOpConverter and implement the Convert() method.
 */

#include "tensorrt_ops.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../json/json_node.h"
#include "NvInfer.h"
#include "tensorrt_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

TensorRTOpConverter::TensorRTOpConverter(std::string op_name,
                                         const std::vector<TensorRTInputType>& input_types,
                                         bool variable_input_count)
    : op_name(std::move(op_name)),
      input_types(input_types),
      variable_input_count(variable_input_count) {}

nvinfer1::ITensor* TensorRTOpConverter::Reshape(TensorRTOpConverterParams* params,
                                                nvinfer1::ITensor* input,
                                                const std::vector<int>& new_shape) const {
  auto layer = params->network->addShuffle(*input);
  TVM_FFI_ICHECK(layer != nullptr);
  layer->setReshapeDimensions(VectorToTrtDims(new_shape));
  layer->setOutputType(0, input->getType());
  return layer->getOutput(0);
}

nvinfer1::ITensor* TensorRTOpConverter::Transpose(TensorRTOpConverterParams* params,
                                                  nvinfer1::ITensor* input,
                                                  const std::vector<int>& order) const {
  auto layer = params->network->addShuffle(*input);
  TVM_FFI_ICHECK(layer != nullptr);
  nvinfer1::Permutation perm;
  if (TRT_HAS_IMPLICIT_BATCH(params)) {
    // Batch dimension cannot be modified.
    TVM_FFI_ICHECK_EQ(input->getDimensions().nbDims, order.size() - 1);
    TVM_FFI_ICHECK_EQ(order[0], 0);
    for (size_t i = 0; i + 1 < order.size(); ++i) {
      perm.order[i] = order[i + 1] - 1;
    }
  } else {
    TVM_FFI_ICHECK_EQ(input->getDimensions().nbDims, order.size());
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
  TVM_FFI_ICHECK(axis >= -input_rank && axis < input_rank);
  if (axis < 0) axis += input_rank;
  if (TRT_HAS_IMPLICIT_BATCH(params)) {
    // Can't modify batch dimenson.
    TVM_FFI_ICHECK_NE(axis, 0);
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
  const nvinfer1::DataType weight_type = params->inputs.at(1).weight.type;
  nvinfer1::Weights weights{weight_type, static_cast<void*>(values), 1};
  params->trt_weights->push_back(weights);
  return params->network->addConstant(dims, weights)->getOutput(0);
}

void TensorRTOpConverter::GetPadding(const ffi::Array<int64_t>& padding,
                                     bool* use_asymmetric_padding, nvinfer1::DimsHW* prepadding,
                                     nvinfer1::DimsHW* postpadding) const {
  TVM_FFI_ICHECK(padding.size() == 1 || padding.size() == 2 || padding.size() == 4);
  if (padding.size() == 4) {
    // four int : padding width in the order of (top, left, bottom, right).
    *prepadding = nvinfer1::DimsHW(static_cast<int>(padding[0]), static_cast<int>(padding[1]));
    *postpadding = nvinfer1::DimsHW(static_cast<int>(padding[2]), static_cast<int>(padding[3]));
    *use_asymmetric_padding = true;
  } else if (padding.size() == 2) {
    // two int : bottom, right will use same padding as top, left
    *prepadding = nvinfer1::DimsHW(static_cast<int>(padding[0]), static_cast<int>(padding[1]));
    *postpadding = *prepadding;
    *use_asymmetric_padding = false;
  } else {
    // one int : same padding used on all sides
    *prepadding = nvinfer1::DimsHW(static_cast<int>(padding[0]), static_cast<int>(padding[0]));
    *postpadding = *prepadding;
    *use_asymmetric_padding = false;
  }
}

void TensorRTOpConverter::GetPadding3D(const ffi::Array<int64_t>& padding,
                                       bool* use_asymmetric_padding, nvinfer1::Dims* prepadding,
                                       nvinfer1::Dims* postpadding) const {
  TVM_FFI_ICHECK(padding.size() == 1 || padding.size() == 3 || padding.size() == 6);
  if (padding.size() == 6) {
    // six int : padding width in the order of (front, top, left, back, bottom, right)
    *prepadding = nvinfer1::Dims3(static_cast<int>(padding[0]), static_cast<int>(padding[1]),
                                  static_cast<int>(padding[2]));
    *postpadding = nvinfer1::Dims3(static_cast<int>(padding[3]), static_cast<int>(padding[4]),
                                   static_cast<int>(padding[5]));
    *use_asymmetric_padding = true;
  } else if (padding.size() == 3) {
    // three int : back, bottom, right will use same padding as front, top, left
    *prepadding = nvinfer1::Dims3(static_cast<int>(padding[0]), static_cast<int>(padding[1]),
                                  static_cast<int>(padding[2]));
    *postpadding = *prepadding;
    *use_asymmetric_padding = false;
  } else {
    // one int : same padding used on all sides
    *prepadding = nvinfer1::Dims3(static_cast<int>(padding[0]), static_cast<int>(padding[0]),
                                  static_cast<int>(padding[0]));
    *postpadding = *prepadding;
    *use_asymmetric_padding = false;
  }
}

class ActivationOpConverter : public TensorRTOpConverter {
 public:
  explicit ActivationOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~ActivationOpConverter() = default;

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
    auto it = op_map.find(op_name);
    TVM_FFI_ICHECK(it != op_map.end()) << "Unsupported activation type " << op_name;
    nvinfer1::IActivationLayer* act_layer =
        params->network->addActivation(*params->inputs.at(0).tensor, it->second);
#if TRT_VERSION_GE(5, 1, 5)
    if (op_name == "clip") {
      // Relax clip min/max are PrimValue args (serialized as arg_min/arg_max), not Relay attrs.
      float a_min = static_cast<float>(params->node.GetAttr<double>("arg_min"));
      float a_max = static_cast<float>(params->node.GetAttr<double>("arg_max"));
      act_layer->setAlpha(a_min);
      act_layer->setBeta(a_max);
    } else if (op_name == "nn.leaky_relu") {
      float alpha = static_cast<float>(params->node.GetAttr<double>("alpha"));
      act_layer->setAlpha(alpha);
    }
#endif
    TVM_FFI_ICHECK(act_layer != nullptr);
    params->outputs.push_back(act_layer->getOutput(0));
  }
};

class ElementWiseBinaryOpConverter : public TensorRTOpConverter {
 public:
  explicit ElementWiseBinaryOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kTensor}) {}
  ~ElementWiseBinaryOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> op_map = {
        {"add", nvinfer1::ElementWiseOperation::kSUM},
        {"subtract", nvinfer1::ElementWiseOperation::kSUB},
        {"multiply", nvinfer1::ElementWiseOperation::kPROD},
        {"divide", nvinfer1::ElementWiseOperation::kDIV},
        {"power", nvinfer1::ElementWiseOperation::kPOW},
        {"maximum", nvinfer1::ElementWiseOperation::kMAX},
        {"minimum", nvinfer1::ElementWiseOperation::kMIN}};
    auto it = op_map.find(op_name);
    TVM_FFI_ICHECK(it != op_map.end()) << "Unsupported elementwise type " << op_name;
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
    TVM_FFI_ICHECK(elemwise_layer != nullptr);
    params->outputs.push_back(elemwise_layer->getOutput(0));
  }
};

class Conv1DOpConverter : public TensorRTOpConverter {
 public:
  explicit Conv1DOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight}) {}
  ~Conv1DOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params->inputs.at(1).weight_shape;
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("data_layout"), "NCW");
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("kernel_layout"), "OIW");
    auto strides = params->node.GetAttr<ffi::Array<int64_t>>("strides");
    auto dilation = params->node.GetAttr<ffi::Array<int64_t>>("dilation");
    auto padding = params->node.GetAttr<ffi::Array<int64_t>>("padding");
    int groups = static_cast<int>(params->node.GetAttr<int64_t>("groups"));
    // Relax conv attrs carry no "channels" field (unlike Relay); the number of output channels is
    // the first dimension of the OIHW/OIW kernel.
    int channels = weight_shape[0];

    auto shuffle_layer = params->network->addShuffle(*input_tensor);
    // Emulate a 1D convolution with a 2D convolution by appending a trailing unit spatial
    // dimension (NCW -> NCW1). In explicit-batch mode (TensorRT 10) input_dims already includes the
    // batch dimension, so derive the reshape from the full input rank instead of hard-coding it.
    std::vector<int> new_shape(input_dims);
    new_shape.push_back(1);
    shuffle_layer->setReshapeDimensions(VectorToTrtDims(new_shape));
    input_tensor = shuffle_layer->getOutput(0);

    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], 1);
    const nvinfer1::DataType weight_type = params->inputs.at(1).weight.type;

    nvinfer1::Weights bias{weight_type, nullptr, 0};

    auto conv_layer = params->network->addConvolutionNd(*input_tensor, channels, kernel_size,
                                                        params->inputs.at(1).weight, bias);
    TVM_FFI_ICHECK(conv_layer != nullptr);
    conv_layer->setPaddingNd(nvinfer1::DimsHW(static_cast<int>(padding[0]), 0));
    TVM_FFI_ICHECK_EQ(strides.size(), 1);
    const auto trt_strides = nvinfer1::DimsHW(static_cast<int>(strides[0]), 1);
    conv_layer->setStrideNd(trt_strides);
    TVM_FFI_ICHECK_EQ(dilation.size(), 1);
    const auto trt_dilation = nvinfer1::DimsHW(static_cast<int>(dilation[0]), 1);
    conv_layer->setDilationNd(trt_dilation);
    conv_layer->setNbGroups(groups);
    input_tensor = conv_layer->getOutput(0);

    // Drop the trailing unit dimension (NOW1 -> NOW); 0 copies the corresponding input dimension,
    // so the number of leading dims to keep matches the original input rank.
    std::vector<int> back_shape(input_dims.size(), 0);
    auto shuffle_back_layer = params->network->addShuffle(*input_tensor);
    shuffle_back_layer->setReshapeDimensions(VectorToTrtDims(back_shape));
    params->outputs.push_back(shuffle_back_layer->getOutput(0));
  }
};

class Conv2DOpConverter : public TensorRTOpConverter {
 public:
  explicit Conv2DOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight}) {}
  ~Conv2DOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params->inputs.at(1).weight_shape;
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("data_layout"), "NCHW");
    TVM_FFI_ICHECK(params->node.GetAttr<ffi::String>("out_layout") == "" ||
                   params->node.GetAttr<ffi::String>("out_layout") == "NCHW");
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("kernel_layout"), "OIHW");
    auto strides = params->node.GetAttr<ffi::Array<int64_t>>("strides");
    auto dilation = params->node.GetAttr<ffi::Array<int64_t>>("dilation");
    auto padding = params->node.GetAttr<ffi::Array<int64_t>>("padding");
    int groups = static_cast<int>(params->node.GetAttr<int64_t>("groups"));
    // Relax conv attrs carry no "channels" field (unlike Relay); the number of output channels is
    // the first dimension of the OIHW/OIW kernel.
    int channels = weight_shape[0];
    // TRT conv2d op doesn't support asymmetric padding before 5.1, so we
    // workaround by adding a padding layer before the pooling op.
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding(padding, &use_asymmetric_padding, &prepadding, &postpadding);

    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
    const nvinfer1::DataType weight_type = params->inputs.at(1).weight.type;
    nvinfer1::Weights bias{weight_type, nullptr, 0};
    auto conv_layer = params->network->addConvolutionNd(*input_tensor, channels, kernel_size,
                                                        params->inputs.at(1).weight, bias);
    TVM_FFI_ICHECK(conv_layer != nullptr);
    conv_layer->setName(params->LayerName().c_str());
    if (use_asymmetric_padding) {
      conv_layer->setPrePadding(prepadding);
      conv_layer->setPostPadding(postpadding);
    } else {
      conv_layer->setPaddingNd(prepadding);
    }
    TVM_FFI_ICHECK_EQ(strides.size(), 2);
    const auto trt_strides =
        nvinfer1::DimsHW(static_cast<int>(strides[0]), static_cast<int>(strides[1]));
    conv_layer->setStrideNd(trt_strides);
    TVM_FFI_ICHECK_EQ(dilation.size(), 2);
    const auto trt_dilation =
        nvinfer1::DimsHW(static_cast<int>(dilation[0]), static_cast<int>(dilation[1]));
    conv_layer->setDilationNd(trt_dilation);
    conv_layer->setNbGroups(groups);
    params->outputs.push_back(conv_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class Conv3DOpConverter : public TensorRTOpConverter {
 public:
  explicit Conv3DOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight}) {}
  ~Conv3DOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params->inputs.at(1).weight_shape;
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("data_layout"), "NCDHW");
    TVM_FFI_ICHECK(params->node.GetAttr<ffi::String>("out_layout") == "" ||
                   params->node.GetAttr<ffi::String>("out_layout") == "NCDHW");
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("kernel_layout"), "OIDHW");
    auto strides = params->node.GetAttr<ffi::Array<int64_t>>("strides");
    auto dilation = params->node.GetAttr<ffi::Array<int64_t>>("dilation");
    auto padding = params->node.GetAttr<ffi::Array<int64_t>>("padding");
    int groups = static_cast<int>(params->node.GetAttr<int64_t>("groups"));

    nvinfer1::Dims prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding3D(padding, &use_asymmetric_padding, &prepadding, &postpadding);

    // Relax conv3d has no "channels" attr; output channels = weight_shape[0] (OIDHW kernel).
    const int num_outputs = static_cast<int>(weight_shape[0]);
    const auto kernel_size = nvinfer1::Dims3(weight_shape[2], weight_shape[3], weight_shape[4]);
    const nvinfer1::DataType weight_type = params->inputs.at(1).weight.type;
    nvinfer1::Weights bias{weight_type, nullptr, 0};
    auto conv_layer = params->network->addConvolutionNd(*input_tensor, num_outputs, kernel_size,
                                                        params->inputs.at(1).weight, bias);
    TVM_FFI_ICHECK(conv_layer != nullptr);
    if (use_asymmetric_padding) {
      conv_layer->setPrePadding(prepadding);
      conv_layer->setPostPadding(postpadding);
    } else {
      conv_layer->setPaddingNd(prepadding);
    }
    TVM_FFI_ICHECK_EQ(strides.size(), 3);
    const auto trt_strides = nvinfer1::Dims3(
        static_cast<int>(strides[0]), static_cast<int>(strides[1]), static_cast<int>(strides[2]));
    conv_layer->setStrideNd(trt_strides);
    TVM_FFI_ICHECK_EQ(dilation.size(), 3);
    const auto trt_dilation =
        nvinfer1::Dims3(static_cast<int>(dilation[0]), static_cast<int>(dilation[1]),
                        static_cast<int>(dilation[2]));
    conv_layer->setDilationNd(trt_dilation);
    conv_layer->setNbGroups(groups);
    params->outputs.push_back(conv_layer->getOutput(0));
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

class DenseOpConverter : public TensorRTOpConverter {
 public:
  explicit DenseOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight}) {}
  ~DenseOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    // Weights are in KC (out_units x in_features) format.
    TVM_FFI_ICHECK_EQ(params->inputs.at(1).weight_shape.size(), 2);
    // addMatrixMultiply requires the input to have at least 2 dimensions (rows x K); the old
    // FullyConnected path padded the rank, so guard explicitly now that it is gone.
    TVM_FFI_ICHECK_GE(input_tensor->getDimensions().nbDims, 2)
        << "TensorRT dense expects an input of rank >= 2 (got "
        << input_tensor->getDimensions().nbDims << ")";
    // TensorRT 10 removed IFullyConnectedLayer/addFullyConnected. Implement dense as a matrix
    // multiply: out[.., O] = in[.., K] * weightᵀ, with weight a constant of shape [O, K].
    // IMatrixMultiplyLayer contracts the last dim of `input` (K) with the last dim of the
    // transposed weight (also K) and broadcasts the remaining leading dimensions, which matches
    // nn.dense semantics for any input rank >= 2 without the rank-padding reshape FC required.
    auto* weight_tensor = params->network
                              ->addConstant(VectorToTrtDims(params->inputs.at(1).weight_shape),
                                            params->inputs.at(1).weight)
                              ->getOutput(0);
    auto* matmul_layer =
        params->network->addMatrixMultiply(*input_tensor, nvinfer1::MatrixOperation::kNONE,
                                           *weight_tensor, nvinfer1::MatrixOperation::kTRANSPOSE);
    TVM_FFI_ICHECK(matmul_layer != nullptr);
    params->outputs.push_back(matmul_layer->getOutput(0));
  }
};

class BatchNormOpConverter : public TensorRTOpConverter {
 public:
  explicit BatchNormOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight, kWeight, kWeight, kWeight}) {}
  ~BatchNormOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto gamma = params->inputs.at(1).weight;
    auto beta = params->inputs.at(2).weight;
    auto mean = params->inputs.at(3).weight;
    auto var = params->inputs.at(4).weight;
    TVM_FFI_ICHECK_EQ(gamma.count, beta.count);
    TVM_FFI_ICHECK_EQ(gamma.count, mean.count);
    TVM_FFI_ICHECK_EQ(gamma.count, var.count);
    const float epsilon = static_cast<float>(params->node.GetAttr<double>("epsilon"));
    const int axis = static_cast<int>(params->node.GetAttr<int64_t>("axis"));
    const bool scale = static_cast<int>(params->node.GetAttr<int64_t>("scale"));
    const bool center = static_cast<int>(params->node.GetAttr<int64_t>("center"));
    auto input_dims = TrtDimsToVector(input->getDimensions());
    const size_t min_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    const size_t max_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 4 : 5;
    TVM_FFI_ICHECK_LE(input_dims.size(), max_rank);
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
    TVM_FFI_ICHECK(input_rank_with_batch == 4 || input_rank_with_batch == 5);
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
    const nvinfer1::DataType weight_type_scale = params->inputs.at(1).weight.type;
    nvinfer1::Weights weight_scale{weight_type_scale, weight_scale_ptr, gamma.count};
    params->trt_weights->push_back(weight_scale);
    void* weight_shift_ptr = new float[gamma.count];
    const nvinfer1::DataType weight_type_shift = params->inputs.at(2).weight.type;
    nvinfer1::Weights weight_shift{weight_type_shift, weight_shift_ptr, gamma.count};
    params->trt_weights->push_back(weight_shift);
    const nvinfer1::DataType weight_type_power = params->inputs.at(3).weight.type;
    nvinfer1::Weights power{weight_type_power, nullptr, 0};

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
    TVM_FFI_ICHECK_EQ(input->getDimensions().nbDims, 3);
    nvinfer1::IScaleLayer* scale_layer = params->network->addScale(
        *input, nvinfer1::ScaleMode::kCHANNEL, weight_shift, weight_scale, power);
#endif
    TVM_FFI_ICHECK(scale_layer != nullptr);
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
  explicit LayerNormOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight, kWeight}) {}
  ~LayerNormOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto gamma_input = params->inputs.at(1).weight;
    auto beta_input = params->inputs.at(2).weight;
    TVM_FFI_ICHECK_EQ(gamma_input.count, beta_input.count);

    const float epsilon = static_cast<float>(params->node.GetAttr<double>("epsilon"));
    const bool scale = static_cast<int>(params->node.GetAttr<int64_t>("scale"));
    const bool center = static_cast<int>(params->node.GetAttr<int64_t>("center"));
    const int input_rank = input->getDimensions().nbDims;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    // Relax layer_norm normalizes over an `axes` list (Relay used a single `axis`).
    auto axes_attr = params->node.GetAttr<ffi::Array<int64_t>>("axes");
    uint32_t reduce_axes = 0;
    std::vector<int> weight_shape(input_rank, 1);
    int64_t normalized_count = 1;
    for (size_t i = 0; i < axes_attr.size(); ++i) {
      const int axis = ConvertAxis(params, static_cast<int>(axes_attr[i]), input_rank);
      reduce_axes |= 1 << axis;
      weight_shape[axis] = input_dims[axis];
      normalized_count *= input_dims[axis];
    }
    TVM_FFI_ICHECK_EQ(normalized_count, gamma_input.count)
        << "TensorRT layer_norm expects gamma/beta to cover exactly the normalized axes";
    auto gamma =
        params->network->addConstant(VectorToTrtDims(weight_shape), gamma_input)->getOutput(0);
    auto beta =
        params->network->addConstant(VectorToTrtDims(weight_shape), beta_input)->getOutput(0);

    // Compute mean
    auto mean_layer = params->network->addReduce(*input, nvinfer1::ReduceOperation::kAVG,
                                                 reduce_axes, /*keepdims=*/true);
    TVM_FFI_ICHECK(mean_layer != nullptr);
    auto mean = mean_layer->getOutput(0);
    // Compute variance
    auto diff_layer =
        params->network->addElementWise(*input, *mean, nvinfer1::ElementWiseOperation::kSUB);
    TVM_FFI_ICHECK(diff_layer != nullptr);
    auto square_layer =
        params->network->addElementWise(*diff_layer->getOutput(0), *diff_layer->getOutput(0),
                                        nvinfer1::ElementWiseOperation::kPROD);
    TVM_FFI_ICHECK(square_layer != nullptr);
    auto var_layer = params->network->addReduce(*square_layer->getOutput(0),
                                                nvinfer1::ReduceOperation::kAVG, reduce_axes,
                                                /*keepdims=*/true);
    TVM_FFI_ICHECK(var_layer != nullptr);
    auto var = var_layer->getOutput(0);
    // sqrt(var + epsilon)
    auto epsilon_tensor = CreateScalar(params, epsilon, var->getDimensions());
    auto denom_add_layer = params->network->addElementWise(*var, *epsilon_tensor,
                                                           nvinfer1::ElementWiseOperation::kSUM);
    TVM_FFI_ICHECK(denom_add_layer != nullptr);
    auto denom_layer =
        params->network->addUnary(*denom_add_layer->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    TVM_FFI_ICHECK(denom_layer != nullptr);
    // (input - mean) / sqrt(var + epsilon)
    auto output_layer =
        params->network->addElementWise(*diff_layer->getOutput(0), *denom_layer->getOutput(0),
                                        nvinfer1::ElementWiseOperation::kDIV);
    TVM_FFI_ICHECK(output_layer != nullptr);
    auto output = output_layer->getOutput(0);

    if (scale) {
      auto scale_layer =
          params->network->addElementWise(*output, *gamma, nvinfer1::ElementWiseOperation::kPROD);
      TVM_FFI_ICHECK(scale_layer != nullptr);
      output = scale_layer->getOutput(0);
    }
    if (center) {
      auto center_layer =
          params->network->addElementWise(*output, *beta, nvinfer1::ElementWiseOperation::kSUM);
      TVM_FFI_ICHECK(center_layer != nullptr);
      output = center_layer->getOutput(0);
    }
    params->outputs.push_back(output);
  }
};

class BatchFlattenOpConverter : public TensorRTOpConverter {
 public:
  explicit BatchFlattenOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~BatchFlattenOpConverter() = default;

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
  explicit SoftmaxOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~SoftmaxOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    const int input_rank = input->getDimensions().nbDims;
    const int original_axis = static_cast<int>(params->node.GetAttr<int64_t>("axis"));
    const int axis = ConvertAxis(params, original_axis, input_rank);
    nvinfer1::ISoftMaxLayer* softmax_layer = params->network->addSoftMax(*input);
    softmax_layer->setAxes(1 << axis);
    TVM_FFI_ICHECK(softmax_layer != nullptr);
    params->outputs.push_back(softmax_layer->getOutput(0));
  }
};

class PoolingOpConverter : public TensorRTOpConverter {
 public:
  explicit PoolingOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~PoolingOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(op_name);
    TVM_FFI_ICHECK(it != op_map.end()) << "Unsupported pooling type " << op_name << " in TensorRT";
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("layout"), "NCHW");
    auto pool_size = params->node.GetAttr<ffi::Array<int64_t>>("pool_size");
    auto padding = params->node.GetAttr<ffi::Array<int64_t>>("padding");
    auto strides = params->node.GetAttr<ffi::Array<int64_t>>("strides");
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding(padding, &use_asymmetric_padding, &prepadding, &postpadding);
    bool ceil_mode = static_cast<int>(params->node.GetAttr<int64_t>("ceil_mode"));

    nvinfer1::DimsHW window_size =
        nvinfer1::DimsHW(static_cast<int>(pool_size[0]), static_cast<int>(pool_size[1]));
    auto pool_layer = params->network->addPoolingNd(*input, it->second, window_size);
    TVM_FFI_ICHECK(pool_layer != nullptr);
    nvinfer1::DimsHW trt_strides =
        nvinfer1::DimsHW(static_cast<int>(strides[0]), static_cast<int>(strides[1]));
    pool_layer->setStrideNd(trt_strides);
    if (use_asymmetric_padding) {
      pool_layer->setPrePadding(prepadding);
      pool_layer->setPostPadding(postpadding);
    } else {
      pool_layer->setPaddingNd(prepadding);
    }
    if (op_name == "nn.avg_pool2d") {
      bool count_include_pad = static_cast<int>(params->node.GetAttr<int64_t>("count_include_pad"));
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
    TVM_FFI_ICHECK(!ceil_mode);
#endif
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class Pooling3DOpConverter : public TensorRTOpConverter {
 public:
  explicit Pooling3DOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~Pooling3DOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.max_pool3d", nvinfer1::PoolingType::kMAX},
        {"nn.avg_pool3d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(op_name);
    TVM_FFI_ICHECK(it != op_map.end()) << "Unsupported pooling type " << op_name << " in TensorRT";
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("layout"), "NCDHW");
    auto pool_size = params->node.GetAttr<ffi::Array<int64_t>>("pool_size");
    auto padding = params->node.GetAttr<ffi::Array<int64_t>>("padding");
    auto strides = params->node.GetAttr<ffi::Array<int64_t>>("strides");
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding3D(padding, &use_asymmetric_padding, &prepadding, &postpadding);
    bool ceil_mode = static_cast<int>(params->node.GetAttr<int64_t>("ceil_mode"));
    nvinfer1::Dims window_size =
        nvinfer1::Dims3(static_cast<int>(pool_size[0]), static_cast<int>(pool_size[1]),
                        static_cast<int>(pool_size[2]));
    auto pool_layer = params->network->addPoolingNd(*input, it->second, window_size);
    TVM_FFI_ICHECK(pool_layer != nullptr);
    nvinfer1::Dims trt_strides = nvinfer1::Dims3(
        static_cast<int>(strides[0]), static_cast<int>(strides[1]), static_cast<int>(strides[2]));
    pool_layer->setStrideNd(trt_strides);
    if (use_asymmetric_padding) {
      pool_layer->setPrePadding(prepadding);
      pool_layer->setPostPadding(postpadding);
    } else {
      pool_layer->setPaddingNd(prepadding);
    }
    if (op_name == "nn.avg_pool3d") {
      bool count_include_pad = static_cast<int>(params->node.GetAttr<int64_t>("count_include_pad"));
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
  explicit GlobalPoolingOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~GlobalPoolingOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.global_max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.global_avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(op_name);
    TVM_FFI_ICHECK(it != op_map.end()) << "Unsupported pooling type " << op_name << " in TensorRT";
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("layout"), "NCHW");
    const int h = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[1] : input_dims[2];
    const int w = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[2] : input_dims[3];
    auto pool_layer =
        params->network->addPoolingNd(*input_tensor, it->second, nvinfer1::DimsHW(h, w));
    TVM_FFI_ICHECK(pool_layer != nullptr);
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

class ExpandDimsOpConverter : public TensorRTOpConverter {
 public:
  explicit ExpandDimsOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~ExpandDimsOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    // Relax expand_dims carries an `axis` list (not Relay's `axis` + `num_newaxis`).
    auto axes = params->node.GetAttr<ffi::Array<int64_t>>("axis");
    const int output_ndim = static_cast<int>(input_dims.size() + axes.size());
    std::vector<int> new_axes;
    for (size_t i = 0; i < axes.size(); ++i) {
      new_axes.push_back(ConvertAxis(params, static_cast<int>(axes[i]), output_ndim));
    }
    std::sort(new_axes.begin(), new_axes.end());
    for (int axis : new_axes) {
      input_dims.insert(input_dims.begin() + axis, 1);
    }
    params->outputs.push_back(Reshape(params, params->inputs.at(0).tensor, input_dims));
  }
};

class SqueezeOpConverter : public TensorRTOpConverter {
 public:
  explicit SqueezeOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~SqueezeOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto axes = params->node.GetAttr<ffi::Array<int64_t>>("axis");
    for (size_t i = 0; i < axes.size(); ++i) {
      const int axis = ConvertAxis(params, static_cast<int>(axes[i]), input_dims.size());
      input_dims[axis] = 0;
    }
    input_dims.erase(std::remove(input_dims.begin(), input_dims.end(), 0), input_dims.end());
    params->outputs.push_back(Reshape(params, params->inputs.at(0).tensor, input_dims));
  }
};

class UnaryOpConverter : public TensorRTOpConverter {
 public:
  explicit UnaryOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~UnaryOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    // The following ops are supported by TRT but don't exist in TVM yet:
    // recip, tan, sinh, cosh, asin, acos, asinh, acosh, atanh
    static const std::unordered_map<std::string, nvinfer1::UnaryOperation> op_map = {
        {"exp", nvinfer1::UnaryOperation::kEXP},      {"log", nvinfer1::UnaryOperation::kLOG},
        {"sqrt", nvinfer1::UnaryOperation::kSQRT},    {"abs", nvinfer1::UnaryOperation::kABS},
        {"negative", nvinfer1::UnaryOperation::kNEG},
#if TRT_VERSION_GE(5, 1, 5)
        {"sin", nvinfer1::UnaryOperation::kSIN},      {"cos", nvinfer1::UnaryOperation::kCOS},
        {"atan", nvinfer1::UnaryOperation::kATAN},    {"ceil", nvinfer1::UnaryOperation::kCEIL},
        {"floor", nvinfer1::UnaryOperation::kFLOOR},
#endif
#if TRT_VERSION_GE(7, 0, 0)
        {"erf", nvinfer1::UnaryOperation::kERF},
#endif
    };
    auto it = op_map.find(op_name);
    TVM_FFI_ICHECK(it != op_map.end()) << "Unsupported unary type " << op_name;
    nvinfer1::IUnaryLayer* unary_layer =
        params->network->addUnary(*params->inputs.at(0).tensor, it->second);
    TVM_FFI_ICHECK(unary_layer != nullptr);
    params->outputs.push_back(unary_layer->getOutput(0));
  }
};

class ConcatOpConverter : public TensorRTOpConverter {
 public:
  explicit ConcatOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {}, /*variable_input_count=*/true) {}
  ~ConcatOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    const int num_inputs = params->inputs.size();
    TVM_FFI_ICHECK_GT(num_inputs, 0);
    const int input_rank = params->inputs[0].tensor->getDimensions().nbDims;
    std::vector<nvinfer1::ITensor*> input_tensors;
    for (auto input : params->inputs) {
      TVM_FFI_ICHECK_EQ(input.type, kTensor);
      TVM_FFI_ICHECK_EQ(input_rank, input.tensor->getDimensions().nbDims);
      input_tensors.push_back(input.tensor);
    }

    const int original_axis = static_cast<int>(params->node.GetAttr<int64_t>("axis"));
    const int axis = ConvertAxis(params, original_axis, input_rank);

    nvinfer1::IConcatenationLayer* concat_layer =
        params->network->addConcatenation(input_tensors.data(), input_tensors.size());
    TVM_FFI_ICHECK(concat_layer != nullptr);
    concat_layer->setAxis(axis);
    params->outputs.push_back(concat_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(5, 1, 5)
class SplitOpConverter : public TensorRTOpConverter {
 public:
  explicit SplitOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~SplitOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    const int original_axis = static_cast<int>(params->node.GetAttr<int64_t>("axis"));
    const int axis = ConvertAxis(params, original_axis, input_dims.size());
    // No Relay "mode": derive each output's extent along `axis` from the per-output shapes.
    auto output_shapes = params->node.GetAttr<ffi::Array<ffi::Array<int64_t>>>("shape");

    std::vector<int> start(input_dims.size(), 0);
    std::vector<int> size(input_dims.begin(), input_dims.end());
    std::vector<int> strides(input_dims.size(), 1);
    int offset = 0;
    for (size_t i = 0; i < output_shapes.size(); ++i) {
      start[axis] = offset;
      size[axis] = static_cast<int>(output_shapes[i][axis]);
      auto slice_layer = params->network->addSlice(*input, VectorToTrtDims(start),
                                                   VectorToTrtDims(size), VectorToTrtDims(strides));
      params->outputs.push_back(slice_layer->getOutput(0));
      offset += size[axis];
    }
  }
};
#endif

class BiasAddOpConverter : public TensorRTOpConverter {
 public:
  explicit BiasAddOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight}) {}
  ~BiasAddOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    size_t required_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    const size_t input_nbDims = input_tensor->getDimensions().nbDims;
    int axis = static_cast<int>(params->node.GetAttr<int64_t>("axis"));
    if (axis == -1) {
      // Make sure there are 2 dimensions after channel dimension,
      if (input_nbDims + 2 > required_rank) required_rank = input_nbDims + 2;
      axis = input_nbDims - 1;
    } else if (TRT_HAS_IMPLICIT_BATCH(params)) {
      axis -= 1;
    }
    TVM_FFI_ICHECK(input_dims.size() > 0 && input_dims.size() <= required_rank);
    const bool need_reshape_on_input = input_dims.size() != required_rank;
    if (need_reshape_on_input) {
      // Add dims of size 1 until rank is required_rank.
      std::vector<int> new_shape(input_dims);
      while (new_shape.size() < required_rank) new_shape.insert(new_shape.end(), 1);
      input_tensor = Reshape(params, input_tensor, new_shape);
    }

    const nvinfer1::DataType weight_type = params->inputs.at(1).weight.type;

    nvinfer1::Weights scale{weight_type, nullptr, 0};
    nvinfer1::Weights power{weight_type, nullptr, 0};
    nvinfer1::IScaleLayer* scale_layer =
        params->network->addScaleNd(*input_tensor, nvinfer1::ScaleMode::kCHANNEL,
                                    params->inputs.at(1).weight, scale, power, axis);
    TVM_FFI_ICHECK(scale_layer != nullptr);
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
  explicit Conv2DTransposeOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight}) {}
  ~Conv2DTransposeOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto weight_shape = params->inputs.at(1).weight_shape;
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("data_layout"), "NCHW");
    TVM_FFI_ICHECK(params->node.GetAttr<ffi::String>("out_layout") == "" ||
                   params->node.GetAttr<ffi::String>("out_layout") == "NCHW");
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("kernel_layout"), "IOHW");
    auto dilation = params->node.GetAttr<ffi::Array<int64_t>>("dilation");
    TVM_FFI_ICHECK(static_cast<int>(dilation[0]) == 1 && static_cast<int>(dilation[1]) == 1);
    auto strides = params->node.GetAttr<ffi::Array<int64_t>>("strides");
    auto padding = params->node.GetAttr<ffi::Array<int64_t>>("padding");
    auto output_padding = params->node.GetAttr<ffi::Array<int64_t>>("output_padding");
    int groups = static_cast<int>(params->node.GetAttr<int64_t>("groups"));

    // TRT deconv op doesn't support asymmetric padding before 5.1, so we
    // workaround by adding a padding layer before the pooling op.
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding(padding, &use_asymmetric_padding, &prepadding, &postpadding);

    // Relax conv2d_transpose uses an IOHW kernel ([in, out, h, w]) by default, which is also the
    // layout TensorRT's deconvolution expects, so the weight is passed through unchanged and the
    // output channel count is the second kernel dimension.
    const int num_outputs = static_cast<int>(weight_shape[1]);
    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
    const nvinfer1::DataType weight_type = params->inputs.at(1).weight.type;
    nvinfer1::Weights bias{weight_type, nullptr, 0};
    auto deconv_layer = params->network->addDeconvolutionNd(*input_tensor, num_outputs, kernel_size,
                                                            params->inputs.at(1).weight, bias);
    TVM_FFI_ICHECK(deconv_layer != nullptr);
    if (use_asymmetric_padding) {
      deconv_layer->setPrePadding(prepadding);
      deconv_layer->setPostPadding(postpadding);
    } else {
      deconv_layer->setPaddingNd(prepadding);
    }
    const auto trt_strides =
        nvinfer1::DimsHW(static_cast<int>(strides[0]), static_cast<int>(strides[1]));
    deconv_layer->setStrideNd(trt_strides);
    deconv_layer->setNbGroups(groups);
    nvinfer1::ITensor* output = deconv_layer->getOutput(0);
    // Output padding.
    if (output_padding.size()) {
      GetPadding(output_padding, &use_asymmetric_padding, &prepadding, &postpadding);
      if (prepadding.h() != 0 || prepadding.w() != 0 || postpadding.h() != 0 ||
          postpadding.w() != 0) {
        // Output padding for Conv2D transpose is always asymmetric and applied to post only.
        prepadding = nvinfer1::DimsHW(0, 0);
        auto pad_layer = params->network->addPaddingNd(*output, prepadding, postpadding);
        output = pad_layer->getOutput(0);
      }
    }
    params->outputs.push_back(output);
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class Conv3DTransposeOpConverter : public TensorRTOpConverter {
 public:
  explicit Conv3DTransposeOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kWeight}) {}
  ~Conv3DTransposeOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto weight_shape = params->inputs.at(1).weight_shape;
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("data_layout"), "NCDHW");
    TVM_FFI_ICHECK(params->node.GetAttr<ffi::String>("out_layout") == "" ||
                   params->node.GetAttr<ffi::String>("out_layout") == "NCDHW");
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("kernel_layout"), "IODHW");
    auto dilation = params->node.GetAttr<ffi::Array<int64_t>>("dilation");
    TVM_FFI_ICHECK_EQ(dilation.size(), 3);
    TVM_FFI_ICHECK(static_cast<int>(dilation[0]) == 1 && static_cast<int>(dilation[1]) == 1 &&
                   static_cast<int>(dilation[2]) == 1);
    auto strides = params->node.GetAttr<ffi::Array<int64_t>>("strides");
    auto padding = params->node.GetAttr<ffi::Array<int64_t>>("padding");
    auto output_padding = params->node.GetAttr<ffi::Array<int64_t>>("output_padding");
    int groups = static_cast<int>(params->node.GetAttr<int64_t>("groups"));
    nvinfer1::Dims prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding3D(padding, &use_asymmetric_padding, &prepadding, &postpadding);

    // Relax conv3d_transpose uses an IODHW kernel ([in, out, d, h, w]) by default, matching the
    // layout TensorRT's deconvolution expects, so the weight passes through unchanged and the
    // output channel count is the second kernel dimension.
    const int num_outputs = static_cast<int>(weight_shape[1]);
    const auto kernel_size = nvinfer1::Dims3(weight_shape[2], weight_shape[3], weight_shape[4]);
    const nvinfer1::DataType weight_type = params->inputs.at(1).weight.type;
    nvinfer1::Weights bias{weight_type, nullptr, 0};
    auto deconv_layer = params->network->addDeconvolutionNd(*input_tensor, num_outputs, kernel_size,
                                                            params->inputs.at(1).weight, bias);
    TVM_FFI_ICHECK(deconv_layer != nullptr);
    if (use_asymmetric_padding) {
      deconv_layer->setPrePadding(prepadding);
      deconv_layer->setPostPadding(postpadding);
    } else {
      deconv_layer->setPaddingNd(prepadding);
    }
    TVM_FFI_ICHECK_EQ(strides.size(), 3);
    const auto trt_strides = nvinfer1::Dims3(
        static_cast<int>(strides[0]), static_cast<int>(strides[1]), static_cast<int>(strides[2]));
    deconv_layer->setStrideNd(trt_strides);
    deconv_layer->setNbGroups(groups);
    nvinfer1::ITensor* output = deconv_layer->getOutput(0);
    // Output padding.
    if (output_padding.size()) {
      GetPadding3D(output_padding, &use_asymmetric_padding, &prepadding, &postpadding);
      // Are any post-padding values non-zero?
      TVM_FFI_ICHECK(!std::any_of(postpadding.d, postpadding.d + postpadding.nbDims, [](int x) {
        return x != 0;
      })) << "TRT does not support padding on 3 dimensions.";
    }
    params->outputs.push_back(output);
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

class TransposeOpConverter : public TensorRTOpConverter {
 public:
  explicit TransposeOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~TransposeOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto axes = params->node.GetAttr<ffi::Array<int64_t>>("axes");
    std::vector<int> order;
    for (size_t i = 0; i < axes.size(); ++i) {
      order.push_back(static_cast<int>(axes[i]));
    }
    params->outputs.push_back(Transpose(params, input, order));
  }
};

class LayoutTransformOpConverter : public TensorRTOpConverter {
 public:
  explicit LayoutTransformOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~LayoutTransformOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    // The codegen emits a pure-permutation IndexMap as "arg_axes"; a missing key => unsupported
    // map.
    TVM_FFI_ICHECK(params->node.HasAttr("arg_axes"))
        << "TensorRT layout_transform supports only pure-permutation index maps";
    auto axes = params->node.GetAttr<ffi::Array<int64_t>>("arg_axes");
    std::vector<int> order;
    for (size_t i = 0; i < axes.size(); ++i) {
      order.push_back(static_cast<int>(axes[i]));
    }
    params->outputs.push_back(Transpose(params, input, order));
  }
};

class ReshapeOpConverter : public TensorRTOpConverter {
 public:
  explicit ReshapeOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~ReshapeOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    // Relax reshape's shape is a Shape arg (serialized as arg_shape); a missing key => non-static.
    TVM_FFI_ICHECK(params->node.HasAttr("arg_shape"))
        << "TensorRT reshape supports only a fully static target shape";
    auto newshape = params->node.GetAttr<ffi::Array<int64_t>>("arg_shape");
    std::vector<int> new_shape;
    int start_index = TRT_HAS_IMPLICIT_BATCH(params) ? 1 : 0;
    if (static_cast<int>(newshape[0]) == -1) start_index = 0;
    for (size_t i = start_index; i < newshape.size(); ++i) {
      const int value = static_cast<int>(newshape[i]);
      TVM_FFI_ICHECK_GE(value, -1);
      new_shape.push_back(value);
    }
    params->outputs.push_back(Reshape(params, input, new_shape));
  }
};

class PadOpConverter : public TensorRTOpConverter {
 public:
  explicit PadOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kIgnored}) {}
  ~PadOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto padding_arr = params->node.GetAttr<ffi::Array<int64_t>>("padding");
    nvinfer1::DimsHW prepadding =
        nvinfer1::DimsHW(static_cast<int>(padding_arr[0]), static_cast<int>(padding_arr[1]));
    nvinfer1::DimsHW postpadding =
        nvinfer1::DimsHW(static_cast<int>(padding_arr[2]), static_cast<int>(padding_arr[3]));
    auto pad_layer = params->network->addPaddingNd(*input, prepadding, postpadding);
    params->outputs.push_back(pad_layer->getOutput(0));
  }
};

class ReduceOpConverter : public TensorRTOpConverter {
 public:
  explicit ReduceOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~ReduceOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    static const std::unordered_map<std::string, nvinfer1::ReduceOperation> op_map = {
        {"sum", nvinfer1::ReduceOperation::kSUM},
        {"prod", nvinfer1::ReduceOperation::kPROD},
        {"max", nvinfer1::ReduceOperation::kMAX},
        {"min", nvinfer1::ReduceOperation::kMIN},
        {"mean", nvinfer1::ReduceOperation::kAVG}};
    auto it = op_map.find(op_name);
    TVM_FFI_ICHECK(it != op_map.end()) << "Unsupported reduce type " << op_name;

    auto input = params->inputs.at(0).tensor;
    // No Relay "exclude"; axis is materialized to a concrete list by the codegen (None -> all
    // axes).
    bool keepdims = static_cast<int>(params->node.GetAttr<int64_t>("keepdims"));
    const int input_rank = input->getDimensions().nbDims;
    auto axes = params->node.GetAttr<ffi::Array<int64_t>>("axis");
    uint32_t reduce_axes = 0;
    for (size_t i = 0; i < axes.size(); ++i) {
      reduce_axes |= 1 << ConvertAxis(params, static_cast<int>(axes[i]), input_rank);
    }
    auto reduce_layer = params->network->addReduce(*input, it->second, reduce_axes, keepdims);
    params->outputs.push_back(reduce_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(5, 1, 5)
class StridedSliceOpConverter : public TensorRTOpConverter {
 public:
  explicit StridedSliceOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~StridedSliceOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    const int rank = static_cast<int>(input_dims.size());
    // axes/begin/end/strides are tuple args (serialized by the codegen); only listed axes are
    // sliced.
    auto axes = params->node.GetAttr<ffi::Array<int64_t>>("arg_axes");
    auto begin = params->node.GetAttr<ffi::Array<int64_t>>("arg_begin");
    auto end = params->node.GetAttr<ffi::Array<int64_t>>("arg_end");
    std::vector<int64_t> stride_values;
    if (params->node.HasAttr("arg_strides")) {
      auto attr_strides = params->node.GetAttr<ffi::Array<int64_t>>("arg_strides");
      stride_values.assign(attr_strides.begin(), attr_strides.end());
    }

    std::vector<int> start(rank, 0);
    std::vector<int> size(input_dims.begin(), input_dims.end());
    std::vector<int> strides(rank, 1);
    for (size_t i = 0; i < axes.size(); ++i) {
      const int axis = ConvertAxis(params, static_cast<int>(axes[i]), rank);
      const int dim = input_dims[axis];
      const int stride = stride_values.empty() ? 1 : static_cast<int>(stride_values[i]);
      TVM_FFI_ICHECK_GT(stride, 0) << "TensorRT strided_slice supports only positive strides";
      int b = static_cast<int>(begin[i]);
      int e = static_cast<int>(end[i]);
      if (b < 0) b += dim;
      if (e < 0) e += dim;
      b = std::max(0, std::min(b, dim));
      e = std::max(0, std::min(e, dim));
      start[axis] = b;
      strides[axis] = stride;
      size[axis] = e > b ? (e - b + stride - 1) / stride : 0;
    }
    auto slice_layer = params->network->addSlice(*input, VectorToTrtDims(start),
                                                 VectorToTrtDims(size), VectorToTrtDims(strides));
    params->outputs.push_back(slice_layer->getOutput(0));
  }
};
#endif

class AdaptivePoolingOpConverter : public TensorRTOpConverter {
 public:
  explicit AdaptivePoolingOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor}) {}
  ~AdaptivePoolingOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.adaptive_max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.adaptive_avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(op_name);
    TVM_FFI_ICHECK(it != op_map.end()) << "Unsupported pooling type " << op_name << " in TensorRT";
    TVM_FFI_ICHECK_EQ(params->node.GetAttr<ffi::String>("layout"), "NCHW");

    // This is an approximation of adaptive pooling. Results will not be
    // mathematically exact except when output_size is (1, 1).
    // Annotation rules will only allow output size of (1, 1).
    auto output_size = nvinfer1::DimsHW(1, 1);
    const int h = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[1] : input_dims[2];
    const int w = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[2] : input_dims[3];
    const auto stride = nvinfer1::DimsHW(h / output_size.h(), w / output_size.w());
    const auto window_size = nvinfer1::DimsHW(h - (output_size.h() - 1) * stride.h(),
                                              w - (output_size.w() - 1) * stride.w());
    auto pool_layer = params->network->addPoolingNd(*input_tensor, it->second, window_size);
    TVM_FFI_ICHECK(pool_layer != nullptr);
    pool_layer->setStrideNd(stride);
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

class BatchMatmulOpConverter : public TensorRTOpConverter {
 public:
  explicit BatchMatmulOpConverter(std::string op_name)
      : TensorRTOpConverter(std::move(op_name), {kTensor, kTensor}) {}
  ~BatchMatmulOpConverter() = default;

  void Convert(TensorRTOpConverterParams* params) const {
    // Relax matmul has no transpose flags; multiply both operands as-is.
    nvinfer1::IMatrixMultiplyLayer* matmul_layer = params->network->addMatrixMultiply(
        *params->inputs.at(0).tensor, nvinfer1::MatrixOperation::kNONE,
        *params->inputs.at(1).tensor, nvinfer1::MatrixOperation::kNONE);
    TVM_FFI_ICHECK(matmul_layer != nullptr);
    params->outputs.push_back(matmul_layer->getOutput(0));
  }
};

const std::unordered_map<std::string, std::unique_ptr<TensorRTOpConverter>>& GetOpConverters() {
  static const std::unordered_map<std::string, std::unique_ptr<TensorRTOpConverter>>* map = []() {
    std::vector<std::unique_ptr<TensorRTOpConverter>> all_converters;
    all_converters.emplace_back(std::make_unique<ActivationOpConverter>("nn.relu"));
    all_converters.emplace_back(std::make_unique<ActivationOpConverter>("sigmoid"));
    all_converters.emplace_back(std::make_unique<ActivationOpConverter>("tanh"));
    all_converters.emplace_back(std::make_unique<BatchNormOpConverter>("nn.batch_norm"));
    all_converters.emplace_back(std::make_unique<LayerNormOpConverter>("nn.layer_norm"));
    all_converters.emplace_back(std::make_unique<SoftmaxOpConverter>("nn.softmax"));
    all_converters.emplace_back(std::make_unique<Conv1DOpConverter>("nn.conv1d"));
    all_converters.emplace_back(std::make_unique<Conv2DOpConverter>("nn.conv2d"));
    all_converters.emplace_back(std::make_unique<DenseOpConverter>("nn.dense"));
    all_converters.emplace_back(std::make_unique<BatchMatmulOpConverter>("nn.batch_matmul"));
    all_converters.emplace_back(std::make_unique<BiasAddOpConverter>("nn.bias_add"));
    all_converters.emplace_back(std::make_unique<ElementWiseBinaryOpConverter>("add"));
    all_converters.emplace_back(std::make_unique<ElementWiseBinaryOpConverter>("subtract"));
    all_converters.emplace_back(std::make_unique<ElementWiseBinaryOpConverter>("multiply"));
    all_converters.emplace_back(std::make_unique<ElementWiseBinaryOpConverter>("divide"));
    all_converters.emplace_back(std::make_unique<ElementWiseBinaryOpConverter>("power"));
    all_converters.emplace_back(std::make_unique<ElementWiseBinaryOpConverter>("maximum"));
    all_converters.emplace_back(std::make_unique<ElementWiseBinaryOpConverter>("minimum"));
    all_converters.emplace_back(std::make_unique<PoolingOpConverter>("nn.max_pool2d"));
    all_converters.emplace_back(std::make_unique<PoolingOpConverter>("nn.avg_pool2d"));
    all_converters.emplace_back(std::make_unique<GlobalPoolingOpConverter>("nn.global_max_pool2d"));
    all_converters.emplace_back(std::make_unique<GlobalPoolingOpConverter>("nn.global_avg_pool2d"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("exp"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("log"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("sqrt"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("abs"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("negative"));
    all_converters.emplace_back(std::make_unique<BatchFlattenOpConverter>("nn.batch_flatten"));
    all_converters.emplace_back(std::make_unique<ExpandDimsOpConverter>("expand_dims"));
    all_converters.emplace_back(std::make_unique<SqueezeOpConverter>("squeeze"));
    all_converters.emplace_back(std::make_unique<ConcatOpConverter>("concatenate"));
    all_converters.emplace_back(
        std::make_unique<Conv2DTransposeOpConverter>("nn.conv2d_transpose"));
    all_converters.emplace_back(std::make_unique<TransposeOpConverter>("transpose"));
    all_converters.emplace_back(std::make_unique<LayoutTransformOpConverter>("layout_transform"));
    all_converters.emplace_back(std::make_unique<ReshapeOpConverter>("reshape"));
    all_converters.emplace_back(std::make_unique<PadOpConverter>("nn.pad"));
    all_converters.emplace_back(std::make_unique<ReduceOpConverter>("sum"));
    all_converters.emplace_back(std::make_unique<ReduceOpConverter>("prod"));
    all_converters.emplace_back(std::make_unique<ReduceOpConverter>("max"));
    all_converters.emplace_back(std::make_unique<ReduceOpConverter>("min"));
    all_converters.emplace_back(std::make_unique<ReduceOpConverter>("mean"));
    all_converters.emplace_back(
        std::make_unique<AdaptivePoolingOpConverter>("nn.adaptive_max_pool2d"));
    all_converters.emplace_back(
        std::make_unique<AdaptivePoolingOpConverter>("nn.adaptive_avg_pool2d"));
    all_converters.emplace_back(std::make_unique<BatchMatmulOpConverter>("nn.batch_matmul"));
#if TRT_VERSION_GE(5, 1, 5)
    all_converters.emplace_back(std::make_unique<ActivationOpConverter>("clip"));
    all_converters.emplace_back(std::make_unique<ActivationOpConverter>("nn.leaky_relu"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("sin"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("cos"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("atan"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("ceil"));
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("floor"));
    all_converters.emplace_back(std::make_unique<SplitOpConverter>("split"));
    all_converters.emplace_back(std::make_unique<StridedSliceOpConverter>("strided_slice"));
#endif  // TRT_VERSION_GE(5, 1, 5)
#if TRT_VERSION_GE(6, 0, 1)
    all_converters.emplace_back(std::make_unique<Conv3DOpConverter>("nn.conv3d"));
    all_converters.emplace_back(std::make_unique<Pooling3DOpConverter>("nn.max_pool3d"));
    all_converters.emplace_back(std::make_unique<Pooling3DOpConverter>("nn.avg_pool3d"));
    all_converters.emplace_back(
        std::make_unique<Conv3DTransposeOpConverter>("nn.conv3d_transpose"));
#endif  // TRT_VERSION_GE(6, 0, 1)
#if TRT_VERSION_GE(7, 0, 0)
    all_converters.emplace_back(std::make_unique<UnaryOpConverter>("erf"));
#endif  // TRT_VERSION_GE(7, 0, 0)
    auto* map = new std::unordered_map<std::string, std::unique_ptr<TensorRTOpConverter>>();
    for (auto& converter : all_converters) {
      map->emplace("tensorrt." + converter->op_name, std::move(converter));
    }
    return map;
  }();
  return *map;
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
