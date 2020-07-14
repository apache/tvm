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
 * \file runtime/contrib/tensorrt/tensorrt_ops.h
 * \brief Converters from Relay ops into TensorRT layers. Converters should
 * inherit from TrtOpConverter and implement the Convert() method.
 */

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_OPS_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_OPS_H_

#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/attrs/vision.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "utils.h"

#if TRT_VERSION_GE(6, 0, 1)
#define TRT_HAS_IMPLICIT_BATCH(params) (params->network->hasImplicitBatchDimension())
#else
#define TRT_HAS_IMPLICIT_BATCH(params) (true)
#endif

namespace tvm {
namespace relay {
namespace contrib {

/*! \brief Parameters to convert an Op from relay to TensorRT. */
struct AddTrtLayerParams {
  /*! \brief The corresponding relay Call node. */
  const CallNode* call;
  /*! \brief The TRT network that the new layer should be added to. */
  nvinfer1::INetworkDefinition* network;
  /*! \brief The type of op. */
  std::string op_name;
  /*! \brief Inputs to the op. */
  std::vector<TrtOpInput> inputs;
  /*! \brief Outputs of the op should be populated here during Convert(). */
  std::vector<nvinfer1::ITensor*> outputs;
  /*! \brief Any newly allocated weights should be stored here also. */
  std::vector<nvinfer1::Weights>* trt_weights;

  AddTrtLayerParams(nvinfer1::INetworkDefinition* network, const CallNode* call,
                    std::vector<nvinfer1::Weights>* trt_weights)
      : network(network), call(call), trt_weights(trt_weights) {
    if (auto* op = call->op.as<OpNode>()) {
      op_name = op->name;
    } else if (call->op->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(call->op);
      const auto name_node = func->GetAttr<String>(attr::kComposite);
      if (!name_node.defined() || name_node.value() == "") {
        LOG(FATAL) << "Only composite functions can be converted.";
      }
      op_name = name_node.value();
    } else {
      LOG(FATAL) << "Call must be Op or Function.";
    }
  }
};

/*! \brief Base class for an op converter from Relay to TRT. */
class TrtOpConverter {
 public:
  /*! \brief Used to specify whether each input is tensor or weight. */
  const std::vector<TrtInputType> input_types;
  /*! \brief If set to true, any number of tensor inputs can be used for the op.
   */
  const bool variable_input_count;

  /*!
   * \brief Converter subclasses should call this constructor to set
   * input_types or variable_input_count.
   * \param input_types For each input to the op, there should be a
   * corresponding entry in input_types to determine whether that input should
   * be a tensor or a weight. TrtBuilder will prepare inputs in
   * AddTrtLayerParams according to this.
   * \param variable_input_count If the op can have multiple inputs, set this to
   * true. input_types vector will be ignored and any number of input tensors
   * can be used for this op. All inputs will be tensors and not weights.
   */
  explicit TrtOpConverter(const std::vector<TrtInputType>& input_types,
                          bool variable_input_count = false)
      : input_types(input_types), variable_input_count(variable_input_count) {}

  /*!
   * \brief Convert to TRT. Implementation should use inputs and attributes
   * from the CallNode to add the corresponding TRT layers to network. Outputs
   * should be pushed to outputs vector.
   * \param params Parameters for this op.
   */
  virtual void Convert(AddTrtLayerParams* params) const = 0;

  /*!
   * \brief Helper function to reshape a tensor.
   * \param params Parameters for this op.
   * \param input Tensor to reshape.
   * \param new_shape New shape, does not include batch dim.
   * \return Reshaped tensor
   */
  nvinfer1::ITensor* Reshape(AddTrtLayerParams* params, nvinfer1::ITensor* input,
                             const std::vector<int>& new_shape) const {
    auto layer = params->network->addShuffle(*input);
    CHECK(layer != nullptr);
    layer->setReshapeDimensions(VectorToTrtDims(new_shape));
    return layer->getOutput(0);
  }

  /*!
   * \brief Helper function to transpose a tensor.
   * \param params Parameters for this op.
   * \param input Tensor to transpose.
   * \param order New order of axes, does include batch dim.
   * \return Transposed tensor
   */
  nvinfer1::ITensor* Transpose(AddTrtLayerParams* params, nvinfer1::ITensor* input,
                               const std::vector<int>& order) const {
    auto layer = params->network->addShuffle(*input);
    CHECK(layer != nullptr);
    nvinfer1::Permutation perm;
    if (TRT_HAS_IMPLICIT_BATCH(params)) {
      // Batch dimension cannot be modified.
      CHECK_EQ(input->getDimensions().nbDims, order.size() - 1);
      CHECK_EQ(order[0], 0);
      for (int i = 0; i < order.size(); ++i) {
        perm.order[i] = order[i + 1] - 1;
      }
    } else {
      CHECK_EQ(input->getDimensions().nbDims, order.size());
      for (int i = 0; i < order.size(); ++i) {
        perm.order[i] = order[i];
      }
    }
    layer->setFirstTranspose(perm);
    return layer->getOutput(0);
  }

  /*!
   * \brief Helper function to convert an axis to TRT format.
   * \param axis Axis from TVM.
   * \param input_rank Rank of input, does not include batch dim.
   * \return Axis in TRT format.
   */
  int ConvertAxis(AddTrtLayerParams* params, int axis, int input_rank) const {
    // Add 1 for missing batch dim.
    if (TRT_HAS_IMPLICIT_BATCH(params)) {
      input_rank += 1;
    }
    CHECK(axis >= -input_rank && axis < input_rank);
    if (axis < 0) axis += input_rank;
    if (TRT_HAS_IMPLICIT_BATCH(params)) {
      // Can't modify batch dimenson.
      CHECK_NE(axis, 0);
      // Subtract 1 for implicit batch dim.
      axis -= 1;
    }
    return axis;
  }

  /*!
   * \brief Create constant that is broadcastable.
   * \param params Parameters for this op.
   * \param value Value of scalar.
   * \param broadcast_to_dims Dims that scalar should be broadcastable against.
   * \return Constant tensor.
   */
  nvinfer1::ITensor* CreateScalar(AddTrtLayerParams* params, float value,
                                  const nvinfer1::Dims& broadcast_to_dims) const {
    nvinfer1::Dims dims;
    dims.nbDims = broadcast_to_dims.nbDims;
    std::fill_n(dims.d, dims.nbDims, 1);
    float* values = new float[1];
    values[0] = value;
    nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, static_cast<void*>(values), 1};
    params->trt_weights->push_back(weights);
    return params->network->addConstant(dims, weights)->getOutput(0);
  }

  /*!
   * \brief Get pre/post padding values from padding attributes array.
   * \param padding Padding from relay op attributes.
   * \param padding_is_asymmetric True if both pre and post are needed for asymmetric padding.
   * \param prepadding Prepadding value or symmetric padding values if !padding_is_asymmetric.
   * \param postpadding Postpadding value if padding_is_asymmetric.
   */
  void GetPadding(const Array<IndexExpr>& padding, bool* use_asymmetric_padding,
                  nvinfer1::DimsHW* prepadding, nvinfer1::DimsHW* postpadding) const {
    CHECK(padding.size() == 1 || padding.size() == 2 || padding.size() == 4);
    if (padding.size() == 4) {
      // four int : padding width in the order of (top, left, bottom, right).
      *prepadding =
          nvinfer1::DimsHW(padding[0].as<IntImmNode>()->value, padding[1].as<IntImmNode>()->value);
      *postpadding =
          nvinfer1::DimsHW(padding[2].as<IntImmNode>()->value, padding[3].as<IntImmNode>()->value);
      *use_asymmetric_padding = true;
    } else if (padding.size() == 2) {
      // two int : bottom, right will use same padding as top, left
      *prepadding =
          nvinfer1::DimsHW(padding[0].as<IntImmNode>()->value, padding[1].as<IntImmNode>()->value);
      *postpadding = *prepadding;
      *use_asymmetric_padding = false;
    } else {
      // one int : same padding used on all sides
      *prepadding =
          nvinfer1::DimsHW(padding[0].as<IntImmNode>()->value, padding[0].as<IntImmNode>()->value);
      *postpadding = *prepadding;
      *use_asymmetric_padding = false;
    }
  }

  /*!
   * \brief Get pre/post padding values from padding attributes array for volumetric ops.
   * \param padding Padding from relay op attributes.
   * \param padding_is_asymmetric True if both pre and post are needed for asymmetric padding.
   * \param prepadding Prepadding value or symmetric padding values if !padding_is_asymmetric.
   * \param postpadding Postpadding value if padding_is_asymmetric.
   */
  void GetPadding3D(const Array<IndexExpr>& padding, bool* use_asymmetric_padding,
                    nvinfer1::Dims* prepadding, nvinfer1::Dims* postpadding) const {
    CHECK(padding.size() == 1 || padding.size() == 3 || padding.size() == 6);
    if (padding.size() == 6) {
      // six int : padding width in the order of (front, top, left, back, bottom, right)
      *prepadding =
          nvinfer1::Dims3(padding[0].as<IntImmNode>()->value, padding[1].as<IntImmNode>()->value,
                          padding[2].as<IntImmNode>()->value);
      *postpadding =
          nvinfer1::Dims3(padding[3].as<IntImmNode>()->value, padding[4].as<IntImmNode>()->value,
                          padding[5].as<IntImmNode>()->value);
      *use_asymmetric_padding = true;
    } else if (padding.size() == 3) {
      // three int : back, bottom, right will use same padding as front, top, left
      *prepadding =
          nvinfer1::Dims3(padding[0].as<IntImmNode>()->value, padding[1].as<IntImmNode>()->value,
                          padding[2].as<IntImmNode>()->value);
      *postpadding = *prepadding;
      *use_asymmetric_padding = false;
    } else {
      // one int : same padding used on all sides
      *prepadding =
          nvinfer1::Dims3(padding[0].as<IntImmNode>()->value, padding[0].as<IntImmNode>()->value,
                          padding[0].as<IntImmNode>()->value);
      *postpadding = *prepadding;
      *use_asymmetric_padding = false;
    }
  }
};

class ActivationOpConverter : public TrtOpConverter {
 public:
  ActivationOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    CHECK_EQ(params->inputs.size(), 1) << "Activation op expects 1 input.";
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
    CHECK(it != op_map.end()) << "Unsupported activation type " << params->op_name;
    nvinfer1::IActivationLayer* act_layer =
        params->network->addActivation(*params->inputs.at(0).tensor, it->second);
#if TRT_VERSION_GE(5, 1, 5)
    if (params->op_name == "clip") {
      const auto* clip_attr = params->call->attrs.as<ClipAttrs>();
      act_layer->setAlpha(clip_attr->a_min);
      act_layer->setBeta(clip_attr->a_max);
    } else if (params->op_name == "nn.leaky_relu") {
      const auto* leaky_relu_attr = params->call->attrs.as<LeakyReluAttrs>();
      act_layer->setAlpha(leaky_relu_attr->alpha);
    }
#endif
    CHECK(act_layer != nullptr);
    params->outputs.push_back(act_layer->getOutput(0));
  }
};

class ClipLegacyOpConverter : public TrtOpConverter {
 public:
  ClipLegacyOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    const auto* attrs = params->call->attrs.as<ClipAttrs>();
    CHECK_EQ(params->inputs.size(), 1) << "Activation op expects 1 input.";
    auto input = params->inputs.at(0).tensor;
    // relu(x)
    nvinfer1::ITensor* output = nullptr;
    if (attrs->a_min == 0.0f) {
      // Use relu instead of max(x, 0) because relu can be fused.
      nvinfer1::IActivationLayer* relu_layer =
          params->network->addActivation(*input, nvinfer1::ActivationType::kRELU);
      CHECK(relu_layer != nullptr);
      output = relu_layer->getOutput(0);
    } else {
      // max(x, a_min)
      nvinfer1::ITensor* a_min = CreateScalar(params, attrs->a_min, input->getDimensions());
      nvinfer1::IElementWiseLayer* max_layer =
          params->network->addElementWise(*input, *a_min, nvinfer1::ElementWiseOperation::kMAX);
      CHECK(max_layer != nullptr);
      output = max_layer->getOutput(0);
    }
    // min(relu(x), a_max)
    nvinfer1::ITensor* a_max = CreateScalar(params, attrs->a_max, input->getDimensions());
    nvinfer1::IElementWiseLayer* min_layer =
        params->network->addElementWise(*output, *a_max, nvinfer1::ElementWiseOperation::kMIN);
    params->outputs.push_back(min_layer->getOutput(0));
  }
};

class ElementWiseBinaryOpConverter : public TrtOpConverter {
 public:
  ElementWiseBinaryOpConverter() : TrtOpConverter({kTensor, kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> op_map = {
        {"add", nvinfer1::ElementWiseOperation::kSUM},
        {"subtract", nvinfer1::ElementWiseOperation::kSUB},
        {"multiply", nvinfer1::ElementWiseOperation::kPROD},
        {"divide", nvinfer1::ElementWiseOperation::kDIV},
        {"power", nvinfer1::ElementWiseOperation::kPOW},
        {"maximum", nvinfer1::ElementWiseOperation::kMAX},
        {"minimum", nvinfer1::ElementWiseOperation::kMIN}};
    auto it = op_map.find(params->op_name);
    CHECK(it != op_map.end()) << "Unsupported elementwise type " << params->op_name;
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
    CHECK(elemwise_layer != nullptr);
    params->outputs.push_back(elemwise_layer->getOutput(0));
  }
};

class Conv2DOpConverter : public TrtOpConverter {
 public:
  Conv2DOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params->inputs.at(1).weight_shape;
    const auto* conv2d_attr = params->call->attrs.as<Conv2DAttrs>();
    CHECK_EQ(conv2d_attr->data_layout, "NCHW");
    CHECK(conv2d_attr->out_layout == "" || conv2d_attr->out_layout == "NCHW");
    CHECK_EQ(conv2d_attr->kernel_layout, "OIHW");

    // TRT conv2d op doesn't support asymmetric padding before 5.1, so we
    // workaround by adding a padding layer before the pooling op.
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding(conv2d_attr->padding, &use_asymmetric_padding, &prepadding, &postpadding);
#if !TRT_VERSION_GE(5, 1, 5)
    if (use_asymmetric_padding) {
      auto pad_layer = params->network->addPadding(*input_tensor, prepadding, postpadding);
      CHECK(pad_layer != nullptr);
      input_tensor = pad_layer->getOutput(0);
      // No need for conv op to do any padding.
      use_asymmetric_padding = false;
      prepadding = nvinfer1::DimsHW(0, 0);
    }
#endif

    // Could use conv2d_attr->channels.as<IntImmNode>()->value
    const int num_outputs = weight_shape[0];
    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto conv_layer = params->network->addConvolution(*input_tensor, num_outputs, kernel_size,
                                                      params->inputs.at(1).weight, bias);
    CHECK(conv_layer != nullptr);
    if (use_asymmetric_padding) {
#if TRT_VERSION_GE(5, 1, 5)
      conv_layer->setPrePadding(prepadding);
      conv_layer->setPostPadding(postpadding);
#endif
    } else {
      conv_layer->setPadding(prepadding);
    }
    CHECK_EQ(conv2d_attr->strides.size(), 2);
    const auto strides = nvinfer1::DimsHW(conv2d_attr->strides[0].as<IntImmNode>()->value,
                                          conv2d_attr->strides[1].as<IntImmNode>()->value);
    conv_layer->setStride(strides);
    CHECK_EQ(conv2d_attr->dilation.size(), 2);
    const auto dilation = nvinfer1::DimsHW(conv2d_attr->dilation[0].as<IntImmNode>()->value,
                                           conv2d_attr->dilation[1].as<IntImmNode>()->value);
    conv_layer->setDilation(dilation);
    conv_layer->setNbGroups(conv2d_attr->groups);
    params->outputs.push_back(conv_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class Conv3DOpConverter : public TrtOpConverter {
 public:
  Conv3DOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    auto weight_shape = params->inputs.at(1).weight_shape;
    const auto* attrs = params->call->attrs.as<Conv3DAttrs>();
    CHECK_EQ(attrs->data_layout, "NCDHW");
    CHECK(attrs->out_layout == "" || attrs->out_layout == "NCDHW");
    CHECK_EQ(attrs->kernel_layout, "OIDHW");

    nvinfer1::Dims prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding3D(attrs->padding, &use_asymmetric_padding, &prepadding, &postpadding);

    // Could use attrs->channels.as<IntImmNode>()->value
    const int num_outputs = weight_shape[0];
    const auto kernel_size = nvinfer1::Dims3(weight_shape[2], weight_shape[3], weight_shape[4]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto conv_layer = params->network->addConvolutionNd(*input_tensor, num_outputs, kernel_size,
                                                        params->inputs.at(1).weight, bias);
    CHECK(conv_layer != nullptr);
    if (use_asymmetric_padding) {
      conv_layer->setPrePadding(prepadding);
      conv_layer->setPostPadding(postpadding);
    } else {
      conv_layer->setPaddingNd(prepadding);
    }
    CHECK_EQ(attrs->strides.size(), 3);
    const auto strides = nvinfer1::Dims3(attrs->strides[0].as<IntImmNode>()->value,
                                         attrs->strides[1].as<IntImmNode>()->value,
                                         attrs->strides[2].as<IntImmNode>()->value);
    conv_layer->setStrideNd(strides);
    CHECK_EQ(attrs->dilation.size(), 3);
    const auto dilation = nvinfer1::Dims3(attrs->dilation[0].as<IntImmNode>()->value,
                                          attrs->dilation[1].as<IntImmNode>()->value,
                                          attrs->dilation[2].as<IntImmNode>()->value);
    conv_layer->setDilationNd(dilation);
    conv_layer->setNbGroups(attrs->groups);
    params->outputs.push_back(conv_layer->getOutput(0));
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

// Using FullyConnected
class DenseOpConverter : public TrtOpConverter {
 public:
  DenseOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    CHECK(input_dims.size() > 0 && input_dims.size() <= 3);
    const int required_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    const bool need_reshape_on_input = input_dims.size() != required_rank;
    if (need_reshape_on_input) {
      // Add dims of size 1 until rank is required_rank.
      std::vector<int> new_shape(input_dims);
      while (new_shape.size() < required_rank) new_shape.insert(new_shape.end(), 1);
      input_tensor = Reshape(params, input_tensor, new_shape);
    }
    // Weights are in KC format.
    CHECK_EQ(params->inputs.at(1).weight_shape.size(), 2);
    const int num_units = params->inputs.at(1).weight_shape[0];
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IFullyConnectedLayer* fc_layer = params->network->addFullyConnected(
        *input_tensor, num_units, params->inputs.at(1).weight, bias);
    CHECK(fc_layer != nullptr);
    auto output_tensor = fc_layer->getOutput(0);
    if (need_reshape_on_input) {
      // Remove added dims.
      input_dims[input_dims.size() - 1] = num_units;
      output_tensor = Reshape(params, output_tensor, input_dims);
    }
    params->outputs.push_back(output_tensor);
  }
};

class BatchNormOpConverter : public TrtOpConverter {
 public:
  BatchNormOpConverter() : TrtOpConverter({kTensor, kWeight, kWeight, kWeight, kWeight}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto gamma = params->inputs.at(1).weight;
    auto beta = params->inputs.at(2).weight;
    auto mean = params->inputs.at(3).weight;
    auto var = params->inputs.at(4).weight;
    const auto* bn_attr = params->call->attrs.as<BatchNormAttrs>();
    CHECK_EQ(gamma.count, beta.count);
    CHECK_EQ(gamma.count, mean.count);
    CHECK_EQ(gamma.count, var.count);
    CHECK(bn_attr->axis == 1 || bn_attr->axis == 3);
    const bool need_transpose = bn_attr->axis == 3;

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
      scale_ptr[i] = 1.0 / std::sqrt(var_ptr[i] + bn_attr->epsilon);
      if (bn_attr->scale) {
        scale_ptr[i] *= gamma_ptr[i];
      }
      shift_ptr[i] = -mean_ptr[i] * scale_ptr[i];
      if (bn_attr->center) {
        shift_ptr[i] += beta_ptr[i];
      }
    }
    if (need_transpose) {
      input = Transpose(params, input, {0, 3, 1, 2});
    }
    nvinfer1::IScaleLayer* scale_layer = params->network->addScale(
        *input, nvinfer1::ScaleMode::kCHANNEL, weight_shift, weight_scale, power);
    CHECK(scale_layer != nullptr);
    auto output = scale_layer->getOutput(0);
    if (need_transpose) {
      output = Transpose(params, output, {0, 2, 3, 1});
    }
    params->outputs.push_back(output);
  }
};

class BatchFlattenOpConverter : public TrtOpConverter {
 public:
  BatchFlattenOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    std::vector<int> new_shape{-1};
    if (!TRT_HAS_IMPLICIT_BATCH(params)) {
      new_shape.insert(new_shape.begin(), params->inputs.at(0).tensor->getDimensions().d[0]);
    }
    params->outputs.push_back(Reshape(params, params->inputs.at(0).tensor, new_shape));
  }
};

class SoftmaxOpConverter : public TrtOpConverter {
 public:
  SoftmaxOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    const int input_rank = input->getDimensions().nbDims;
    const auto* softmax_attr = params->call->attrs.as<SoftmaxAttrs>();
    const int axis = ConvertAxis(params, softmax_attr->axis, input_rank);
    nvinfer1::ISoftMaxLayer* softmax_layer = params->network->addSoftMax(*input);
    softmax_layer->setAxes(1 << axis);
    CHECK(softmax_layer != nullptr);
    params->outputs.push_back(softmax_layer->getOutput(0));
  }
};

class PoolingOpConverter : public TrtOpConverter {
 public:
  PoolingOpConverter() : TrtOpConverter({kTensor}) {}

  // Get attributes from MaxPool2DAttrs or AvgPool2DAttrs. If
  // use_assymetric_padding is false, symmetric padding values will be returned
  // in prepadding only.
  template <class PoolAttrs>
  void GetPoolAttrs(const PoolAttrs* attrs, nvinfer1::DimsHW* prepadding,
                    nvinfer1::DimsHW* postpadding, nvinfer1::DimsHW* window_size,
                    nvinfer1::DimsHW* strides, bool* ceil_mode,
                    bool* use_asymmetric_padding) const {
    CHECK_EQ(attrs->layout, "NCHW");
    GetPadding(attrs->padding, use_asymmetric_padding, prepadding, postpadding);
    *window_size = nvinfer1::DimsHW(attrs->pool_size[0].template as<IntImmNode>()->value,
                                    attrs->pool_size[1].template as<IntImmNode>()->value);
    *strides = nvinfer1::DimsHW(attrs->strides[0].template as<IntImmNode>()->value,
                                attrs->strides[1].template as<IntImmNode>()->value);
    *ceil_mode = attrs->ceil_mode;
  }

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params->op_name);
    CHECK(it != op_map.end()) << "Unsupported pooling type " << params->op_name << " in TensorRT";

    nvinfer1::DimsHW prepadding, postpadding, window_size, strides;
    bool use_asymmetric_padding = false, ceil_mode = false, count_include_pad = true;
    if (params->op_name == "nn.max_pool2d") {
      const auto* attrs = params->call->attrs.as<MaxPool2DAttrs>();
      GetPoolAttrs<MaxPool2DAttrs>(attrs, &prepadding, &postpadding, &window_size, &strides,
                                   &ceil_mode, &use_asymmetric_padding);
    } else if (params->op_name == "nn.avg_pool2d") {
      const auto* attrs = params->call->attrs.as<AvgPool2DAttrs>();
      count_include_pad = attrs->count_include_pad;
      GetPoolAttrs<AvgPool2DAttrs>(attrs, &prepadding, &postpadding, &window_size, &strides,
                                   &ceil_mode, &use_asymmetric_padding);
    }

// TRT pooling op doesn't support asymmetric padding before 5.1, so we
// workaround by adding a padding layer before the pooling op.
#if !TRT_VERSION_GE(5, 1, 5)
    if (use_asymmetric_padding) {
      auto pad_layer = params->network->addPadding(*input, prepadding, postpadding);
      CHECK(pad_layer != nullptr);
      input = pad_layer->getOutput(0);
      // No need for pooling op to do any padding.
      use_asymmetric_padding = false;
      prepadding = nvinfer1::DimsHW(0, 0);
    }
#endif

    auto pool_layer = params->network->addPooling(*input, it->second, window_size);
    CHECK(pool_layer != nullptr);
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
    CHECK(!ceil_mode);
#endif
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class Pooling3DOpConverter : public TrtOpConverter {
 public:
  Pooling3DOpConverter() : TrtOpConverter({kTensor}) {}

  // Get attributes from MaxPool2DAttrs or AvgPool2DAttrs. If
  // use_assymetric_padding is false, symmetric padding values will be returned
  // in prepadding only.
  template <class PoolAttrs>
  void GetPoolAttrs(const PoolAttrs* attrs, nvinfer1::Dims* prepadding, nvinfer1::Dims* postpadding,
                    nvinfer1::Dims* window_size, nvinfer1::Dims* strides, bool* ceil_mode,
                    bool* use_asymmetric_padding) const {
    CHECK_EQ(attrs->layout, "NCDHW");
    GetPadding3D(attrs->padding, use_asymmetric_padding, prepadding, postpadding);
    *window_size = nvinfer1::Dims3(attrs->pool_size[0].template as<IntImmNode>()->value,
                                   attrs->pool_size[1].template as<IntImmNode>()->value,
                                   attrs->pool_size[2].template as<IntImmNode>()->value);
    *strides = nvinfer1::Dims3(attrs->strides[0].template as<IntImmNode>()->value,
                               attrs->strides[1].template as<IntImmNode>()->value,
                               attrs->strides[2].template as<IntImmNode>()->value);
    *ceil_mode = attrs->ceil_mode;
  }

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.max_pool3d", nvinfer1::PoolingType::kMAX},
        {"nn.avg_pool3d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params->op_name);
    CHECK(it != op_map.end()) << "Unsupported pooling type " << params->op_name << " in TensorRT";

    nvinfer1::Dims prepadding, postpadding, window_size, strides;
    bool use_asymmetric_padding = false, ceil_mode = false, count_include_pad = true;
    if (params->op_name == "nn.max_pool3d") {
      const auto* attrs = params->call->attrs.as<MaxPool3DAttrs>();
      GetPoolAttrs<MaxPool3DAttrs>(attrs, &prepadding, &postpadding, &window_size, &strides,
                                   &ceil_mode, &use_asymmetric_padding);
    } else if (params->op_name == "nn.avg_pool3d") {
      const auto* attrs = params->call->attrs.as<AvgPool3DAttrs>();
      count_include_pad = attrs->count_include_pad;
      GetPoolAttrs<AvgPool3DAttrs>(attrs, &prepadding, &postpadding, &window_size, &strides,
                                   &ceil_mode, &use_asymmetric_padding);
    }
    auto pool_layer = params->network->addPoolingNd(*input, it->second, window_size);
    CHECK(pool_layer != nullptr);
    pool_layer->setStrideNd(strides);
    if (use_asymmetric_padding) {
      pool_layer->setPrePadding(prepadding);
      pool_layer->setPostPadding(postpadding);
    } else {
      pool_layer->setPaddingNd(prepadding);
    }
    if (params->op_name == "nn.avg_pool3d") {
      // count_include_pad=True is useless if there is no padding. TRT doesn't
      // like count_include_pad in combination with strides even when there is
      // no padding or assymetric padding even, so turn off inclusive to avoid
      // error message. Note: Padding will always be symmetric with
      // count_include_pad since partitioner will prevent unsupported case.
      pool_layer->setAverageCountExcludesPadding(!count_include_pad);
    }
    if (ceil_mode) {
      pool_layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP);
    }
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

class GlobalPoolingOpConverter : public TrtOpConverter {
 public:
  GlobalPoolingOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.global_max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.global_avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params->op_name);
    CHECK(it != op_map.end()) << "Unsupported pooling type " << params->op_name << " in TensorRT";
    const auto* pool_attr = params->call->attrs.as<GlobalPool2DAttrs>();
    CHECK_EQ(pool_attr->layout, "NCHW");
    const int h = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[1] : input_dims[2];
    const int w = TRT_HAS_IMPLICIT_BATCH(params) ? input_dims[2] : input_dims[3];
    auto pool_layer =
        params->network->addPooling(*input_tensor, it->second, nvinfer1::DimsHW(h, w));
    CHECK(pool_layer != nullptr);
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

class ExpandDimsOpConverter : public TrtOpConverter {
 public:
  ExpandDimsOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    const auto* attrs = params->call->attrs.as<ExpandDimsAttrs>();
    const int axis = ConvertAxis(params, attrs->axis, input_dims.size() + 1);
    for (int i = 0; i < attrs->num_newaxis; ++i) {
      input_dims.insert(input_dims.begin() + axis, 1);
    }
    params->outputs.push_back(Reshape(params, params->inputs.at(0).tensor, input_dims));
  }
};

class SqueezeOpConverter : public TrtOpConverter {
 public:
  SqueezeOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    const auto* attrs = params->call->attrs.as<SqueezeAttrs>();
    // TODO(tmorris): if axis not defined, squeeze all dimensions with size 1.
    CHECK(attrs->axis.defined());
    for (size_t i = 0; i < attrs->axis.size(); ++i) {
      const int axis =
          ConvertAxis(params, attrs->axis[i].as<IntImmNode>()->value, input_dims.size());
      input_dims[axis] = 0;
    }
    input_dims.erase(std::remove(input_dims.begin(), input_dims.end(), 0), input_dims.end());
    params->outputs.push_back(Reshape(params, params->inputs.at(0).tensor, input_dims));
  }
};

class UnaryOpConverter : public TrtOpConverter {
 public:
  UnaryOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
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
    };
    auto it = op_map.find(params->op_name);
    CHECK(it != op_map.end()) << "Unsupported unary type " << params->op_name;
    nvinfer1::IUnaryLayer* unary_layer =
        params->network->addUnary(*params->inputs.at(0).tensor, it->second);
    CHECK(unary_layer != nullptr);
    params->outputs.push_back(unary_layer->getOutput(0));
  }
};

class ConcatOpConverter : public TrtOpConverter {
 public:
  ConcatOpConverter() : TrtOpConverter({}, /*variable_input_count=*/true) {}

  void Convert(AddTrtLayerParams* params) const {
    const int num_inputs = params->inputs.size();
    CHECK_GT(num_inputs, 0);
    const int input_rank = params->inputs[0].tensor->getDimensions().nbDims;
    std::vector<nvinfer1::ITensor*> input_tensors;
    for (auto input : params->inputs) {
      CHECK(input.type == kTensor);
      CHECK_EQ(input_rank, input.tensor->getDimensions().nbDims);
      input_tensors.push_back(input.tensor);
    }

    const auto* concat_attr = params->call->attrs.as<ConcatenateAttrs>();
    const int axis = ConvertAxis(params, concat_attr->axis, input_rank);

    nvinfer1::IConcatenationLayer* concat_layer =
        params->network->addConcatenation(input_tensors.data(), input_tensors.size());
    CHECK(concat_layer != nullptr);
    concat_layer->setAxis(axis);
    params->outputs.push_back(concat_layer->getOutput(0));
  }
};

class BiasAddOpConverter : public TrtOpConverter {
 public:
  BiasAddOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    const int required_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    CHECK(input_dims.size() > 0 && input_dims.size() <= required_rank);
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
    CHECK(scale_layer != nullptr);
    auto output_tensor = scale_layer->getOutput(0);
    if (need_reshape_on_input) {
      // Remove added dims.
      output_tensor = Reshape(params, output_tensor, input_dims);
    }
    params->outputs.push_back(output_tensor);
  }
};

class Conv2DTransposeOpConverter : public TrtOpConverter {
 public:
  Conv2DTransposeOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto weight_shape = params->inputs.at(1).weight_shape;
    const auto* conv2d_attr = params->call->attrs.as<Conv2DTransposeAttrs>();
    CHECK_EQ(conv2d_attr->data_layout, "NCHW");
    CHECK(conv2d_attr->out_layout == "" || conv2d_attr->out_layout == "NCHW");
    CHECK_EQ(conv2d_attr->kernel_layout, "OIHW");
    CHECK(conv2d_attr->dilation[0].as<IntImmNode>()->value == 1 &&
          conv2d_attr->dilation[1].as<IntImmNode>()->value == 1);

    // TRT deconv op doesn't support asymmetric padding before 5.1, so we
    // workaround by adding a padding layer before the pooling op.
    nvinfer1::DimsHW prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding(conv2d_attr->padding, &use_asymmetric_padding, &prepadding, &postpadding);
#if !TRT_VERSION_GE(5, 1, 5)
    if (use_asymmetric_padding) {
      auto pad_layer = params->network->addPadding(*input_tensor, prepadding, postpadding);
      CHECK(pad_layer != nullptr);
      input_tensor = pad_layer->getOutput(0);
      // No need for conv op to do any padding.
      use_asymmetric_padding = false;
      prepadding = nvinfer1::DimsHW(0, 0);
    }
#endif

    // Could use conv2d_attr->channels.as<IntImmNode>()->value
    const int num_outputs = weight_shape[1];
    const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto deconv_layer = params->network->addDeconvolution(*input_tensor, num_outputs, kernel_size,
                                                          params->inputs.at(1).weight, bias);
    CHECK(deconv_layer != nullptr);
    if (use_asymmetric_padding) {
#if TRT_VERSION_GE(5, 1, 5)
      deconv_layer->setPrePadding(prepadding);
      deconv_layer->setPostPadding(postpadding);
#endif
    } else {
      deconv_layer->setPadding(prepadding);
    }
    const auto strides = nvinfer1::DimsHW(conv2d_attr->strides[0].as<IntImmNode>()->value,
                                          conv2d_attr->strides[1].as<IntImmNode>()->value);
    deconv_layer->setStride(strides);
    deconv_layer->setNbGroups(conv2d_attr->groups);
    nvinfer1::ITensor* output = deconv_layer->getOutput(0);
    // Output padding.
    if (conv2d_attr->output_padding.size()) {
      GetPadding(conv2d_attr->output_padding, &use_asymmetric_padding, &prepadding, &postpadding);
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
class Conv3DTransposeOpConverter : public TrtOpConverter {
 public:
  Conv3DTransposeOpConverter() : TrtOpConverter({kTensor, kWeight}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto weight_shape = params->inputs.at(1).weight_shape;
    const auto* attrs = params->call->attrs.as<Conv3DTransposeAttrs>();
    CHECK_EQ(attrs->data_layout, "NCDHW");
    CHECK(attrs->out_layout == "" || attrs->out_layout == "NCDHW");
    CHECK_EQ(attrs->kernel_layout, "OIDHW");
    CHECK(attrs->dilation[0].as<IntImmNode>()->value == 1 &&
          attrs->dilation[1].as<IntImmNode>()->value == 1 &&
          attrs->dilation[2].as<IntImmNode>()->value == 1);

    nvinfer1::Dims prepadding, postpadding;
    bool use_asymmetric_padding;
    GetPadding3D(attrs->padding, &use_asymmetric_padding, &prepadding, &postpadding);

    // Could use attrs->channels.as<IntImmNode>()->value
    const int num_outputs = weight_shape[1];
    const auto kernel_size = nvinfer1::Dims3(weight_shape[2], weight_shape[3], weight_shape[4]);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto deconv_layer = params->network->addDeconvolutionNd(*input_tensor, num_outputs, kernel_size,
                                                            params->inputs.at(1).weight, bias);
    CHECK(deconv_layer != nullptr);
    if (use_asymmetric_padding) {
      deconv_layer->setPrePadding(prepadding);
      deconv_layer->setPostPadding(postpadding);
    } else {
      deconv_layer->setPaddingNd(prepadding);
    }
    const auto strides = nvinfer1::Dims3(attrs->strides[0].as<IntImmNode>()->value,
                                         attrs->strides[1].as<IntImmNode>()->value,
                                         attrs->strides[2].as<IntImmNode>()->value);
    deconv_layer->setStrideNd(strides);
    deconv_layer->setNbGroups(attrs->groups);
    nvinfer1::ITensor* output = deconv_layer->getOutput(0);
    // Output padding.
    if (attrs->output_padding.size()) {
      GetPadding3D(attrs->output_padding, &use_asymmetric_padding, &prepadding, &postpadding);
      // Are any post-padding values non-zero?
      CHECK(!std::any_of(postpadding.d, postpadding.d + postpadding.nbDims, [](int x) {
        return x != 0;
      })) << "TRT does not support padding on 3 dimensions.";
    }
    params->outputs.push_back(output);
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

class TransposeOpConverter : public TrtOpConverter {
 public:
  TransposeOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    const auto* attrs = params->call->attrs.as<TransposeAttrs>();
    std::vector<int> order;
    for (size_t i = 0; i < attrs->axes.size(); ++i) {
      order.push_back(attrs->axes[i].as<IntImmNode>()->value);
    }
    params->outputs.push_back(Transpose(params, input, order));
  }
};

class ReshapeOpConverter : public TrtOpConverter {
 public:
  ReshapeOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    const auto* attrs = params->call->attrs.as<ReshapeAttrs>();
    CHECK_EQ(attrs->reverse, false);
    std::vector<int> new_shape;
    const int start_index = TRT_HAS_IMPLICIT_BATCH(params) ? 1 : 0;
    for (size_t i = start_index; i < attrs->newshape.size(); ++i) {
      CHECK(attrs->newshape[i].defined());
      const int value = attrs->newshape[i]->value;
      CHECK_GE(value, -1);
      new_shape.push_back(value);
    }
    params->outputs.push_back(Reshape(params, input, new_shape));
  }
};

class PadOpConverter : public TrtOpConverter {
 public:
  PadOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    const auto* attrs = params->call->attrs.as<PadAttrs>();
    const int input_rank_with_batch =
        input->getDimensions().nbDims + (TRT_HAS_IMPLICIT_BATCH(params) ? 1 : 0);
    CHECK_EQ(input_rank_with_batch, attrs->pad_width.size());
    CHECK(!TRT_HAS_IMPLICIT_BATCH(params) || (attrs->pad_width[0][0].as<IntImmNode>()->value == 0 &&
                                              attrs->pad_width[0][1].as<IntImmNode>()->value == 0))
        << "Cannot pad on batch dimension.";

    nvinfer1::DimsHW prepadding, postpadding;
    // Check if we need to transpose from NHWC -> NCHW.
    const bool need_transpose = attrs->pad_width[1][0].as<IntImmNode>()->value != 0 ||
                                attrs->pad_width[1][1].as<IntImmNode>()->value != 0;
    if (need_transpose) {
      input = Transpose(params, input, {0, 3, 1, 2});
      prepadding = nvinfer1::DimsHW(attrs->pad_width[1][0].as<IntImmNode>()->value,
                                    attrs->pad_width[2][0].as<IntImmNode>()->value);
      postpadding = nvinfer1::DimsHW(attrs->pad_width[1][1].as<IntImmNode>()->value,
                                     attrs->pad_width[2][1].as<IntImmNode>()->value);
    } else {
      prepadding = nvinfer1::DimsHW(attrs->pad_width[2][0].as<IntImmNode>()->value,
                                    attrs->pad_width[3][0].as<IntImmNode>()->value);
      postpadding = nvinfer1::DimsHW(attrs->pad_width[2][1].as<IntImmNode>()->value,
                                     attrs->pad_width[3][1].as<IntImmNode>()->value);
    }
    auto pad_layer = params->network->addPadding(*input, prepadding, postpadding);
    CHECK(pad_layer != nullptr);
    auto output = pad_layer->getOutput(0);
    if (need_transpose) {
      // NCHW -> NHWC
      output = Transpose(params, output, {0, 2, 3, 1});
    }
    params->outputs.push_back(output);
  }
};

class ReduceOpConverter : public TrtOpConverter {
 public:
  ReduceOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    static const std::unordered_map<std::string, nvinfer1::ReduceOperation> op_map = {
        {"sum", nvinfer1::ReduceOperation::kSUM},
        {"prod", nvinfer1::ReduceOperation::kPROD},
        {"max", nvinfer1::ReduceOperation::kMAX},
        {"min", nvinfer1::ReduceOperation::kMIN},
        {"mean", nvinfer1::ReduceOperation::kAVG}};
    auto it = op_map.find(params->op_name);
    CHECK(it != op_map.end()) << "Unsupported reduce type " << params->op_name;

    auto input = params->inputs.at(0).tensor;
    const auto* attrs = params->call->attrs.as<ReduceAttrs>();
    CHECK(attrs->exclude == false);
    // TODO(trevmorr): Support reduce to scalar.
    CHECK(attrs->axis.defined() && attrs->axis.size() > 0);
    uint32_t reduce_axes = 0;
    for (size_t i = 0; i < attrs->axis.size(); ++i) {
      const int axis = ConvertAxis(params, attrs->axis[i].as<IntImmNode>()->value,
                                   input->getDimensions().nbDims);
      reduce_axes |= 1 << axis;
    }
    auto reduce_layer =
        params->network->addReduce(*input, it->second, reduce_axes, attrs->keepdims);
    params->outputs.push_back(reduce_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(5, 1, 5)
class StridedSliceOpConverter : public TrtOpConverter {
 public:
  StridedSliceOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    const auto* attrs = params->call->attrs.as<StridedSliceAttrs>();
    // Dynamic shapes not supported.
    CHECK(attrs->begin && attrs->end && attrs->strides);
    const int input_rank_with_batch =
        input->getDimensions().nbDims + (TRT_HAS_IMPLICIT_BATCH(params) ? 1 : 0);
    CHECK_EQ(input_rank_with_batch, attrs->begin.value().size());
    CHECK_EQ(input_rank_with_batch, attrs->end.value().size());
    const bool default_strides =
        !attrs->strides.value().defined() || attrs->strides.value().size() == 0;
    if (TRT_HAS_IMPLICIT_BATCH(params)) {
      CHECK(default_strides || !attrs->strides.value()[0].defined() ||
            attrs->strides.value()[0].as<IntImmNode>()->value == 1);
    }

    auto process_slice_index = [](Integer x, int default_value, int dim_value) {
      if (!x.defined()) return default_value;
      int value = x.as<IntImmNode>()->value;
      if (value < 0) value += dim_value;
      return value;
    };

    const int start_index = TRT_HAS_IMPLICIT_BATCH(params) ? 1 : 0;
    std::vector<int> start, size, strides;
    for (size_t i = start_index; i < attrs->begin.value().size(); ++i) {
      const int begin_value =
          process_slice_index(attrs->begin.value()[i], 0, input_dims[i - start_index]);
      const int end_value = process_slice_index(attrs->end.value()[i], input_dims[i - start_index],
                                                input_dims[i - start_index]);
      const int stride_value = (default_strides || i >= attrs->strides.value().size() ||
                                !attrs->strides.value()[i].defined())
                                   ? 1
                                   : attrs->strides.value()[i].as<IntImmNode>()->value;
      CHECK_GT(stride_value, 0);
      const int size_value = (end_value - begin_value + stride_value - 1) / stride_value;
      CHECK_GE(begin_value, 0);
      CHECK_GT(size_value, 0);
      start.push_back(begin_value);
      size.push_back(size_value);
      strides.push_back(stride_value);
    }

    auto slice_layer = params->network->addSlice(*input, VectorToTrtDims(start),
                                                 VectorToTrtDims(size), VectorToTrtDims(strides));
    params->outputs.push_back(slice_layer->getOutput(0));
  }
};
#endif

class AdaptivePoolingOpConverter : public TrtOpConverter {
 public:
  AdaptivePoolingOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input_tensor = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
    static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map = {
        {"nn.adaptive_max_pool2d", nvinfer1::PoolingType::kMAX},
        {"nn.adaptive_avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
    auto it = op_map.find(params->op_name);
    CHECK(it != op_map.end()) << "Unsupported pooling type " << params->op_name << " in TensorRT";
    const auto* attrs = params->call->attrs.as<AdaptivePool2DAttrs>();
    CHECK_EQ(attrs->layout, "NCHW");

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
    CHECK(pool_layer != nullptr);
    pool_layer->setStride(stride);
    params->outputs.push_back(pool_layer->getOutput(0));
  }
};

#if TRT_VERSION_GE(6, 0, 1)
class ResizeOpConverter : public TrtOpConverter {
 public:
  ResizeOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    const auto* attrs = params->call->attrs.as<ResizeAttrs>();
    static const std::unordered_map<std::string, nvinfer1::ResizeMode> op_map = {
        {"nearest_neighbor", nvinfer1::ResizeMode::kNEAREST},
        {"bilinear", nvinfer1::ResizeMode::kLINEAR}};
    auto it = op_map.find(attrs->method);
    CHECK(it != op_map.end()) << "Unsupported resize type " << attrs->method;
    CHECK_EQ(attrs->size.size(), 2);
    auto output_dims = TrtDimsToVector(input->getDimensions());
    const int required_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    CHECK_EQ(output_dims.size(), required_rank);
    CHECK(attrs->layout == "NCHW" || attrs->layout == "NHWC");
    int h_index = attrs->layout == "NCHW" ? 2 : 1;
    int w_index = attrs->layout == "NCHW" ? 3 : 2;
    if (TRT_HAS_IMPLICIT_BATCH(params)) {
      h_index -= 1;
      w_index -= 1;
    }
    output_dims[h_index] = attrs->size[0].as<IntImmNode>()->value;
    output_dims[w_index] = attrs->size[1].as<IntImmNode>()->value;

    nvinfer1::IResizeLayer* resize_layer = params->network->addResize(*input);
    CHECK(resize_layer != nullptr);
    resize_layer->setResizeMode(it->second);
    resize_layer->setOutputDimensions(VectorToTrtDims(output_dims));
    resize_layer->setAlignCorners(attrs->coordinate_transformation_mode == "align_corners");
    params->outputs.push_back(resize_layer->getOutput(0));
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

#if TRT_VERSION_GE(5, 1, 5)
class SplitOpConverter : public TrtOpConverter {
 public:
  SplitOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    const auto* attrs = params->call->attrs.as<SplitAttrs>();
    const int input_rank = input->getDimensions().nbDims;
    const int axis = ConvertAxis(params, attrs->axis, input_dims.size());
    const int sections = attrs->indices_or_sections.as<IntImmNode>()->value;

    std::vector<int> start(input_dims.size(), 0);
    std::vector<int> size(input_dims.begin(), input_dims.end());
    size[axis] = input_dims[axis] / sections;
    std::vector<int> strides(input_dims.size(), 1);
    for (int i = 0; i < sections; ++i) {
      start[axis] = i * size[axis];
      auto slice_layer = params->network->addSlice(*input, VectorToTrtDims(start),
                                                   VectorToTrtDims(size), VectorToTrtDims(strides));

      params->outputs.push_back(slice_layer->getOutput(0));
    }
  }
};
#endif  // TRT_VERSION_GE(5, 1, 5)

#if TRT_VERSION_GE(5, 1, 5)
// TODO(trevmorr): Not needed due to SimplifySliceLike which converts slice_like
// to strided_slice. slice_like has a false dependency on the second input
// tensor since only the shape is needed. This confuses TRT.
class SliceLikeOpConverter : public TrtOpConverter {
 public:
  SliceLikeOpConverter() : TrtOpConverter({kTensor, kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    auto input_2 = params->inputs.at(1).tensor;
    auto input_dims = TrtDimsToVector(input->getDimensions());
    auto new_dims = TrtDimsToVector(input_2->getDimensions());
    const auto* attrs = params->call->attrs.as<SliceLikeAttrs>();
    if (attrs->axes.defined()) {
      for (int i = 0; i < attrs->axes.size(); i++) {
        const int axis =
            ConvertAxis(params, attrs->axes[i].as<IntImmNode>()->value, input_dims.size());
        input_dims[axis] = new_dims[axis];
      }
    } else {
      // Use all dims when axes is not defined.
      CHECK_EQ(input_dims.size(), new_dims.size());
      input_dims = new_dims;
    }

    // slice_like always begins at 0.
    std::vector<int> start(input_dims.size(), 0);
    std::vector<int> strides(input_dims.size(), 1);
    auto slice_layer = params->network->addSlice(
        *input, VectorToTrtDims(start), VectorToTrtDims(input_dims), VectorToTrtDims(strides));

    params->outputs.push_back(slice_layer->getOutput(0));
  }
};
#endif  // TRT_VERSION_GE(5, 1, 5)

#if TRT_VERSION_GE(6, 0, 1)
class UpsamplingOpConverter : public TrtOpConverter {
 public:
  UpsamplingOpConverter() : TrtOpConverter({kTensor}) {}

  void Convert(AddTrtLayerParams* params) const {
    auto input = params->inputs.at(0).tensor;
    const auto* attrs = params->call->attrs.as<UpSamplingAttrs>();
    static const std::unordered_map<std::string, nvinfer1::ResizeMode> op_map = {
        {"nearest_neighbor", nvinfer1::ResizeMode::kNEAREST},
        {"bilinear", nvinfer1::ResizeMode::kLINEAR}};
    auto it = op_map.find(attrs->method);
    CHECK(it != op_map.end()) << "Unsupported resize type " << attrs->method;
    auto output_dims = TrtDimsToVector(input->getDimensions());
    const int required_rank = TRT_HAS_IMPLICIT_BATCH(params) ? 3 : 4;
    CHECK_EQ(output_dims.size(), required_rank);
    CHECK(attrs->layout == "NCHW" || attrs->layout == "NHWC");
    int h_index = attrs->layout == "NCHW" ? 2 : 1;
    int w_index = attrs->layout == "NCHW" ? 3 : 2;
    if (TRT_HAS_IMPLICIT_BATCH(params)) {
      h_index -= 1;
      w_index -= 1;
    }
    output_dims[h_index] *= attrs->scale_h;
    output_dims[w_index] *= attrs->scale_w;

    nvinfer1::IResizeLayer* resize_layer = params->network->addResize(*input);
    CHECK(resize_layer != nullptr);
    resize_layer->setResizeMode(it->second);
    resize_layer->setOutputDimensions(VectorToTrtDims(output_dims));
    resize_layer->setAlignCorners(attrs->align_corners);
    params->outputs.push_back(resize_layer->getOutput(0));
  }
};
#endif  // TRT_VERSION_GE(6, 0, 1)

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_OPS_H_
