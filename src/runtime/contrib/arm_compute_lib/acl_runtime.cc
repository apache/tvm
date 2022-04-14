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
 * \file src/runtime/contrib/arm_compute_lib/acl_runtime.cc
 * \brief A simple JSON runtime for Arm Compute Library.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_EXECUTOR_ARM_COMPUTE_LIB
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticAddition.h>
#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>
#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include <arm_compute/runtime/NEON/functions/NEPoolingLayer.h>
#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>

#include "acl_allocator.h"
#include "acl_utils.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

class ACLRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The ACL runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit ACLRuntime(const std::string& symbol_name, const std::string& graph_json,
                      const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const override { return "arm_compute_lib"; }

  /*!
   * \brief Initialize runtime. Create ACL layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);
    BuildEngine();
  }

#ifdef TVM_GRAPH_EXECUTOR_ARM_COMPUTE_LIB
  /*!
   * \brief Unpack inputs and outputs and run inference on a given layer.
   *
   * \param args Access inputs and outputs.
   * \param function The layer to execute inference on.
   * \return Status of inference.
   */
  void Run() override {
    for (size_t nid_idx = 0; nid_idx < input_nodes_.size(); ++nid_idx) {
      auto nid = input_nodes_[nid_idx];
      if (nodes_[nid].GetOpType() == "input") {
        for (uint32_t eid_idx = 0; eid_idx < nodes_[nid].GetNumOutput(); eid_idx++) {
          uint32_t eid = EntryID(nid, eid_idx);
          void* data = data_entry_[eid]->data;
          auto key = std::pair<uint32_t, uint32_t>(nid, eid_idx);
          if (layer_.json_inputid_to_layer_inputid.count(key) > 0) {
            CheckACLError(
                layer_.inputs[layer_.json_inputid_to_layer_inputid[key]].allocator()->import_memory(
                    data));
          } else {
            CheckACLError(layer_.inputs[nid_idx].allocator()->import_memory(data));
          }
        }
      }
    }

    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      void* data = data_entry_[eid]->data;
      CheckACLError(layer_.outputs[i].allocator()->import_memory(data));
    }

    this->layer_.function->run();
  }

 private:
  /*!
   * \brief Build ACL layer from JSON representation and cache.
   *
   * \note For the time being only one layer or operator is supported
   * per engine.
   */
  void BuildEngine() {
    std::shared_ptr<arm_compute::MemoryManagerOnDemand> mm = MakeACLMemoryManager();
    int num_pools = 0;
    bool found_kernel_node = false;
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (found_kernel_node) {
        LOG(FATAL)
            << "Arm Compute Library runtime module only supports one kernel node per function.";
      }
      if (node.GetOpType() == "kernel") {
        found_kernel_node = true;
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name || "qnn.conv2d" == op_name) {
          CreateConvolution2DLayer(&layer_, node, mm);
          num_pools++;
        } else if ("nn.depthwise_conv2d" == op_name || "qnn.depthwise_conv2d" == op_name) {
          CreateDepthwiseConvolution2DLayer(&layer_, node, mm);
          num_pools++;
        } else if ("nn.dense" == op_name || "qnn.dense" == op_name) {
          CreateFullyConnectedLayer(&layer_, node, mm);
          num_pools++;
        } else if ("nn.max_pool2d" == op_name || "nn.avg_pool2d" == op_name ||
                   "nn.l2_pool2d" == op_name) {
          CreatePoolingLayer(&layer_, node);
        } else if ("nn.global_max_pool2d" == op_name || "nn.global_avg_pool2d" == op_name) {
          CreateGlobalPoolingLayer(&layer_, node);
        } else if ("reshape" == op_name) {
          CreateReshapeLayer(&layer_, node);
        } else if ("maximum" == op_name) {
          CreateMaximumLayer(&layer_, node);
        } else if ("add" == op_name || "qnn.add" == op_name) {
          CreateAddLayer(&layer_, node);
        } else if ("concatenate" == op_name) {
          CreateConcatenateLayer(&layer_, node);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
    this->layer_.function->prepare();
    if (num_pools > 0) mm->populate(this->allocator_, num_pools);
  }

  /*!
   * \brief ACL objects we cache in order to avoid needing to construct
   * a new layer each time.
   */
  struct CachedLayer {
    std::shared_ptr<arm_compute::IFunction> function;
    std::vector<arm_compute::Tensor> inputs;
    std::vector<arm_compute::Tensor> outputs;
    // maps the input index of JSON node to the index of the ACL layer's inputs
    // this is optional (i.e.only when an operator uses the eid index)
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> json_inputid_to_layer_inputid;
  };

  /*!
   * \brief Create an ACL tensor given the JSON representation. If scale
   * and offset are given, then create a quantized ACL tensor.
   *
   * \param tensor The tensor to represent.
   * \param scale (optional) The scale of the tensor as an input.
   * \param offset (optional) The offset of the tensor as an input.
   * \param apply_dim_correction (Optional) Flag to state whether apply dimension correction after
   * setting one dimension. E.g. when permuting NCHW -> NHWC, 1x1x2 would become 2x1x1, but
   * _num_dimensions should be 3 rather than 1.
   * \param increase_dim_unit (Optional) Set to true if new unit dimensions increase the number of
   * dimensions of the shape.
   * \return ACL Tensor.
   */
  arm_compute::Tensor MakeACLTensorFromJSONEntry(const JSONGraphNodeEntry& tensor,
                                                 JSONGraphNodeEntry* scale = nullptr,
                                                 JSONGraphNodeEntry* offset = nullptr,
                                                 bool apply_dim_correction = true,
                                                 bool increase_dim_unit = true) {
    JSONGraphNode node = nodes_[tensor.id_];
    void* node_data = nullptr;
    if (node.GetOpType() == "const") {
      node_data = data_entry_[EntryID(tensor)]->data;
    }
    return MakeACLTensorFromJSONNode(node, scale, offset, node_data, apply_dim_correction,
                                     increase_dim_unit, tensor.index_);
  }

  /*!
   * \brief Create an ACL tensor given the JSON representation. If scale
   * and offset are given, then create a quantized ACL tensor.
   *
   * \param node The tensor to represent.
   * \param scale (optional) The scale of the tensor as an input.
   * \param offset (optional) The offset of the tensor as an input.
   * \param data (optional) Constant data of input node.
   * \param apply_dim_correction (Optional) Flag to state whether apply dimension correction after
   * setting one dimension. E.g. when permuting NCHW -> NHWC, 1x1x2 would become 2x1x1, but
   * _num_dimensions should be 3 rather than 1.
   * \param increase_dim_unit (Optional) Set to true if new unit dimensions increase the number of
   * dimensions of the shape.
   * \param entry_index The entry index.
   * \return ACL Tensor.
   */
  arm_compute::Tensor MakeACLTensorFromJSONNode(
      const JSONGraphNode& node, JSONGraphNodeEntry* scale = nullptr,
      JSONGraphNodeEntry* offset = nullptr, void* data = nullptr, bool apply_dim_correction = true,
      bool increase_dim_unit = true, uint32_t entry_index = 0) {
    const DLTensor* scale_data = nullptr;
    const DLTensor* offset_data = nullptr;
    if (scale && offset) {
      scale_data = data_entry_[EntryID(*scale)];
      offset_data = data_entry_[EntryID(*offset)];
    }
    return MakeACLTensor(node, data, scale_data, offset_data, apply_dim_correction,
                         increase_dim_unit, entry_index);
  }

  /*!
   * \brief Create a 2D convolution layer.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   * \param mm The ACL conv2d layer can request auxiliary memory from TVM.
   */
  void CreateConvolution2DLayer(CachedLayer* layer, const JSONGraphNode& node,
                                const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& mm) {
    std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> dilation = node.GetAttr<std::vector<std::string>>("dilation");
    arm_compute::PadStrideInfo pad_stride_info = MakeACLPadStride(padding, strides);

    int groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    ICHECK(groups == 1) << "Arm Compute Library NEON convolution only supports group size of 1.";

    arm_compute::ActivationLayerInfo act_info;
    if (node.HasAttr("activation_type")) {
      std::string activation_type = node.GetAttr<std::vector<std::string>>("activation_type")[0];
      act_info = MakeACLActivationInfo(activation_type);
    }

    arm_compute::Size2D dilation_2d(std::stoi(dilation[0]), std::stoi(dilation[1]));

    // Collect inputs and outputs, handling both nn.conv2d and qnn.conv2d cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs = inputs.size();
    bool has_bias;
    if (node.GetOpName() == "qnn.conv2d") {
      ICHECK(num_inputs >= 8U && num_inputs <= 9U)
          << "Quantized convolution requires 9 inputs with a bias, 8 inputs without.";
      has_bias = num_inputs == 9;
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[0], &inputs[4], &inputs[2]));
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[1], &inputs[5], &inputs[3]));
      if (has_bias) {
        layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[6]));
      }
      layer->outputs.push_back(
          MakeACLTensorFromJSONNode(node, &inputs[6 + has_bias], &inputs[7 + has_bias]));
    } else {
      ICHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Convolution requires 3 inputs with a bias, 2 inputs without.";
      has_bias = num_inputs == 3;
      for (const auto& i : inputs) {
        layer->inputs.push_back(MakeACLTensorFromJSONEntry(i));
      }
      layer->outputs.push_back(MakeACLTensorFromJSONNode(node));
    }

    auto function = std::make_shared<arm_compute::NEConvolutionLayer>(mm);
    function->configure(&layer->inputs[0], &layer->inputs[1],
                        has_bias ? &layer->inputs[2] : nullptr, &layer->outputs[0], pad_stride_info,
                        arm_compute::WeightsInfo(), dilation_2d, act_info);
    layer->function = function;
  }

  /*!
   * \brief Create a 2D depthwise convolution layer.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   * \param mm The ACL conv2d layer can request auxiliary memory from TVM.
   */
  void CreateDepthwiseConvolution2DLayer(
      CachedLayer* layer, const JSONGraphNode& node,
      const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& mm) {
    std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> dilation = node.GetAttr<std::vector<std::string>>("dilation");
    arm_compute::PadStrideInfo pad_stride_info = MakeACLPadStride(padding, strides);

    arm_compute::ActivationLayerInfo act_info;
    if (node.HasAttr("activation_type")) {
      std::string activation_type = node.GetAttr<std::vector<std::string>>("activation_type")[0];
      act_info = MakeACLActivationInfo(activation_type);
    }

    arm_compute::Size2D dilation_2d(std::stoi(dilation[0]), std::stoi(dilation[1]));

    // Collect inputs and outputs, handling both nn.conv2d and qnn.conv2d cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs = inputs.size();
    bool has_bias;
    if (node.GetOpName() == "qnn.depthwise_conv2d") {
      ICHECK(num_inputs >= 8U && num_inputs <= 9U)
          << "Quantized convolution requires 9 inputs with a bias, 8 inputs without.";
      has_bias = num_inputs == 9;
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[0], &inputs[4], &inputs[2]));
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[1], &inputs[5], &inputs[3]));
      if (has_bias) {
        layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[6]));
      }
      layer->outputs.push_back(
          MakeACLTensorFromJSONNode(node, &inputs[6 + has_bias], &inputs[7 + has_bias]));
    } else {
      ICHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Convolution requires 3 inputs with a bias, 2 inputs without.";
      has_bias = num_inputs == 3;
      for (const auto& i : inputs) {
        layer->inputs.push_back(MakeACLTensorFromJSONEntry(i));
      }
      layer->outputs.push_back(MakeACLTensorFromJSONNode(node));
    }

    // Depth multiplier is the final dimension in acl weights tensor (IWH*M*)
    int depth_multiplier = layer->inputs[1].info()->tensor_shape()[3];

    auto function = std::make_shared<arm_compute::NEDepthwiseConvolutionLayer>(mm);
    function->configure(&layer->inputs[0], &layer->inputs[1],
                        has_bias ? &layer->inputs[2] : nullptr, &layer->outputs[0], pad_stride_info,
                        depth_multiplier, act_info, dilation_2d);
    layer->function = function;
  }

  /*!
   * \brief Create a fully connected (dense) layer.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   * \param mm The ACL fully connected layer can request auxiliary memory from TVM.
   */
  void CreateFullyConnectedLayer(CachedLayer* layer, const JSONGraphNode& node,
                                 const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& mm) {
    arm_compute::FullyConnectedLayerInfo fc_info;
    fc_info.set_weights_trained_layout(arm_compute::DataLayout::NHWC);

    // Collect inputs and outputs, handling both nn.dense and qnn.dense cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs = inputs.size();
    bool has_bias;
    if (node.GetOpName() == "qnn.dense") {
      ICHECK(num_inputs >= 8U && num_inputs <= 9U)
          << "Quantized fully connected (dense) layer requires 9 inputs with a bias, 8 inputs "
             "without.";
      has_bias = num_inputs == 9;
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[0], &inputs[4], &inputs[2]));
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[1], &inputs[5], &inputs[3]));
      if (has_bias) {
        layer->inputs.push_back(MakeACLTensorFromJSONEntry(inputs[6]));
      }
      layer->outputs.push_back(
          MakeACLTensorFromJSONNode(node, &inputs[6 + has_bias], &inputs[7 + has_bias]));
    } else {
      ICHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Fully connected (dense) layer requires 3 inputs with a bias, 2 inputs without.";
      has_bias = num_inputs == 3;
      for (const auto& i : inputs) {
        layer->inputs.push_back(MakeACLTensorFromJSONEntry(i));
      }
      layer->outputs.push_back(MakeACLTensorFromJSONNode(node));
    }

    auto function = std::make_shared<arm_compute::NEFullyConnectedLayer>(mm);
    function->configure(&layer->inputs[0], &layer->inputs[1],
                        has_bias ? &layer->inputs[2] : nullptr, &layer->outputs[0], fc_info);
    layer->function = function;
  }

  /*!
   * \brief Create a pooling layer.
   *
   * \note Currently max_pool2d, avg_pool2d and L2 pooling are supported.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   */
  void CreatePoolingLayer(CachedLayer* layer, const JSONGraphNode& node) {
    std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> dilation = node.GetAttr<std::vector<std::string>>("dilation");
    bool ceil_mode = std::stoi(node.GetAttr<std::vector<std::string>>("ceil_mode")[0]);
    arm_compute::PadStrideInfo pad_stride_info = MakeACLPadStride(padding, strides, ceil_mode);
    auto attr_pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
    int pool_size_h = std::stoi(attr_pool_size[0]);
    int pool_size_w = std::stoi(attr_pool_size[1]);

    // Only applies to average pool and l2 pool.
    // ACL exclude pad option is inverse to Relays include pad option.
    bool exclude_pad = false;
    if (node.HasAttr("count_include_pad")) {
      int count_include_pad =
          std::stoi(node.GetAttr<std::vector<std::string>>("count_include_pad")[0]);
      exclude_pad = !count_include_pad;
    }

    arm_compute::PoolingType pool_type;
    if (node.GetOpName() == "nn.max_pool2d") {
      pool_type = arm_compute::PoolingType::MAX;
    } else if (node.GetOpName() == "nn.avg_pool2d") {
      pool_type = arm_compute::PoolingType::AVG;
    } else if (node.GetOpName() == "nn.l2_pool2d") {
      pool_type = arm_compute::PoolingType::L2;
    } else {
      LOG(FATAL) << "Pooling type not supported";
    }

    ICHECK(dilation.size() == 2 && dilation[0] == "1" && dilation[1] == "1")
        << "Dilation other than (1, 1) not supported";
    arm_compute::PoolingLayerInfo pool_info =
        arm_compute::PoolingLayerInfo(pool_type, arm_compute::Size2D(pool_size_h, pool_size_w),
                                      arm_compute::DataLayout::NHWC, pad_stride_info, exclude_pad);

    layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[0]));
    layer->outputs.push_back(MakeACLTensorFromJSONNode(node));

    auto function = std::make_shared<arm_compute::NEPoolingLayer>();
    function->configure(&layer->inputs[0], &layer->outputs[0], pool_info);
    layer->function = function;
  }

  /*!
   * \brief Create a global pooling layer.
   *
   * \note Currently global_max_pool2d and global_avg_pool2d are supported.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   */
  void CreateGlobalPoolingLayer(CachedLayer* layer, const JSONGraphNode& node) {
    arm_compute::PoolingType pool_type;
    if (node.GetOpName() == "nn.global_max_pool2d") {
      pool_type = arm_compute::PoolingType::MAX;
    } else if (node.GetOpName() == "nn.global_avg_pool2d") {
      pool_type = arm_compute::PoolingType::AVG;
    } else {
      LOG(FATAL) << "Pooling type not supported";
    }

    arm_compute::PoolingLayerInfo pool_info =
        arm_compute::PoolingLayerInfo(pool_type, arm_compute::DataLayout::NHWC);

    layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[0]));
    layer->outputs.push_back(MakeACLTensorFromJSONNode(node));

    auto function = std::make_shared<arm_compute::NEPoolingLayer>();
    function->configure(&layer->inputs[0], &layer->outputs[0], pool_info);
    layer->function = function;
  }

  /*!
   * \brief Create a reshape layer.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   */
  void CreateReshapeLayer(CachedLayer* layer, const JSONGraphNode& node) {
    layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[0]));
    layer->outputs.push_back(MakeACLTensorFromJSONNode(node));
    auto function = std::make_shared<arm_compute::NEReshapeLayer>();
    function->configure(&layer->inputs[0], &layer->outputs[0]);
    layer->function = function;
  }

  /*!
   * \brief Create a maximum layer.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   */
  void CreateMaximumLayer(CachedLayer* layer, const JSONGraphNode& node) {
    layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[0]));
    layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[1]));
    layer->outputs.push_back(MakeACLTensorFromJSONNode(node));
    auto function = std::make_shared<arm_compute::NEElementwiseMax>();
    function->configure(&layer->inputs[0], &layer->inputs[1], &layer->outputs[0]);
    layer->function = function;
  }
  /*!
   * \brief Creates an add/qnn.add layer
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node  The JSON representation of the operator.
   */
  void CreateAddLayer(CachedLayer* layer, const JSONGraphNode& node) {
    auto op_name = node.GetOpName();
    if ("add" == op_name) {
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[0]));
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[1]));
      layer->outputs.push_back(MakeACLTensorFromJSONNode(node));
    } else if ("qnn.add" == op_name) {
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[0], &node.GetInputs()[2],
                                                         &node.GetInputs()[3]));
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(node.GetInputs()[1], &node.GetInputs()[4],
                                                         &node.GetInputs()[5]));
      layer->outputs.push_back(
          MakeACLTensorFromJSONNode(node, &node.GetInputs()[6], &node.GetInputs()[7]));
    } else {
      LOG(FATAL) << "Unsupported form of add op: " + op_name;
    }

    auto f = std::make_shared<arm_compute::NEArithmeticAddition>();

    // SATURATE is used as add_QASYMM8_QASYMM8_QASYMM8 always saturates result
    f->configure(&layer->inputs[0], &layer->inputs[1], &layer->outputs[0],
                 arm_compute::ConvertPolicy::SATURATE);
    layer->function = f;
  }

  /*!
   * \brief Create a Concatenate layer.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.c
   * \param node The JSON representation of the operator.
   */
  void CreateConcatenateLayer(CachedLayer* layer, const JSONGraphNode& node) {
    std::vector<std::string> axis = node.GetAttr<std::vector<std::string>>("axis");
    std::vector<const arm_compute::ITensor*> inputs;
    for (auto input : node.GetInputs()) {
      layer->inputs.push_back(MakeACLTensorFromJSONEntry(input, nullptr, nullptr, false));
      layer->json_inputid_to_layer_inputid[std::pair<uint32_t, uint32_t>(input.id_, input.index_)] =
          layer->inputs.size() - 1;
    }
    for (size_t i = 0; i < layer->inputs.size(); i++) {
      inputs.push_back(&layer->inputs[i]);
    }
    layer->outputs.push_back(MakeACLTensorFromJSONNode(node));
    int dimNum = layer->inputs[0].info()->num_dimensions();
    auto function = std::make_shared<arm_compute::NEConcatenateLayer>();
    // the shape of input tensor will be reversed after passing to ACL
    // for example a tensor with shape [1, 2, 3, 4] will be changed to
    // [4, 3, 2, 1] at ACL side. So the axis here should be preprocessed.
    auto a = std::stoi(axis[0]);
    function->configure(inputs, &layer->outputs[0], a < 0 ? -a - 1 : dimNum - a - 1);
    layer->function = function;
  }

  /*! \brief Allow ACL functions to request auxiliary memory from TVM. */
  ACLAllocator allocator_;
  /*!
   * \brief The network layers represented by acl functions.
   * \note Currently only supports a single layer.
   */
  CachedLayer layer_;
#else
  void Run() override {
    LOG(FATAL) << "Cannot call run on Arm Compute Library module without runtime enabled. "
               << "Please build with USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR.";
  }

  void BuildEngine() {
    LOG(WARNING) << "Arm Compute Library engine is not initialized. "
                 << "Please build with USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR.";
  }
#endif
};
runtime::Module ACLRuntimeCreate(const String& symbol_name, const String& graph_json,
                                 const Array<String>& const_names) {
  auto n = make_object<ACLRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.arm_compute_lib_runtime_create").set_body_typed(ACLRuntimeCreate);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_arm_compute_lib")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<ACLRuntime>);
}  //  namespace contrib
}  //  namespace runtime
}  //  namespace tvm
