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

#include "../../file_util.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_GRAPH_RUNTIME_ARM_COMPUTE_LIB
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
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
    CHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);
    BuildEngine();
  }

#ifdef TVM_GRAPH_RUNTIME_ARM_COMPUTE_LIB
  /*!
   * \brief Unpack inputs and outputs and run inference on a given layer.
   *
   * \param args Access inputs and outputs.
   * \param function The layer to execute inference on.
   * \return Status of inference.
   */
  void Run() override {
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      uint32_t eid = EntryID(nid, 0);
      if (nodes_[nid].GetOpType() == "input") {
        void* data = data_entry_[eid]->data;
        CheckACLError(layer_.inputs[i].allocator()->import_memory(data));
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
    std::shared_ptr<arm_compute::MemoryManagerOnDemand> mm = MakeMemoryManager();
    int num_pools = 0;

    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      uint32_t nid = input_nodes_[i];
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "input") {
        layer_.inputs.push_back(MakeTensor(node));
      } else if (node.GetOpType() == "const") {
        uint32_t eid = EntryID(nid, 0);
        void* data = data_entry_[eid]->data;
        layer_.const_inputs.push_back(MakeTensor(node, data));
      }
    }

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
        if ("nn.conv2d" == op_name) {
          CreateConvolution2DLayer(&layer_, node, mm);
          num_pools++;
        } else if ("nn.max_pool2d" == op_name) {
          CreatePoolingLayer(&layer_, node);
        } else if ("reshape" == op_name) {
          CreateReshapeLayer(&layer_, node);
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
    std::vector<arm_compute::Tensor> const_inputs;
    std::vector<arm_compute::Tensor> outputs;
  };

  /*!
   * \brief Create a 2D convolution layer.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   * \param mm The ACL conv2d layer can request auxiliary memory from TVM.
   */
  static void CreateConvolution2DLayer(
      CachedLayer* layer, const JSONGraphNode& node,
      const std::shared_ptr<arm_compute::MemoryManagerOnDemand>& mm) {
    std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> dilation = node.GetAttr<std::vector<std::string>>("dilation");
    arm_compute::PadStrideInfo pad_stride_info = ToACLPadStride(padding, strides);

    int groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    CHECK(groups == 1) << "Arm Compute Library NEON convolution only supports group size of 1.";

    arm_compute::ActivationLayerInfo act_info;
    if (node.HasAttr("activation_type")) {
      std::string activation_type = node.GetAttr<std::vector<std::string>>("activation_type")[0];
      if (activation_type == "relu") {
        act_info = arm_compute::ActivationLayerInfo(
            arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
      } else {
        LOG(FATAL) << "Unsupported activation function";
      }
    }

    arm_compute::Size2D dilation_2d(std::stoi(dilation[0]), std::stoi(dilation[1]));

    layer->outputs.push_back(MakeOutputTensor(node.GetOpShape()[0]));

    auto function = std::make_shared<arm_compute::NEConvolutionLayer>(mm);
    function->configure(&layer->inputs[0], &layer->const_inputs[0],
                        layer->const_inputs.size() > 1 ? &layer->const_inputs[1] : nullptr,
                        &layer->outputs[0], pad_stride_info, arm_compute::WeightsInfo(),
                        dilation_2d, act_info);
    layer->function = function;
  }

  /*!
   * \brief Create a pooling layer.
   *
   * \note Currently only maxpool is supported.
   *
   * \param layer The ACL layer to build. Containing inputs, outputs and the ACL function.
   * \param node The JSON representation of the operator.
   */
  static void CreatePoolingLayer(CachedLayer* layer, const JSONGraphNode& node) {
    std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    arm_compute::PadStrideInfo pad_stride_info = ToACLPadStride(padding, strides);

    auto attr_pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
    int pool_size_h = std::stoi(attr_pool_size[0]);
    int pool_size_w = std::stoi(attr_pool_size[1]);

    arm_compute::PoolingType pool_type;
    if (node.GetOpName() == "nn.max_pool2d") {
      pool_type = arm_compute::PoolingType::MAX;
    } else {
      LOG(FATAL) << "Pooling type not supported";
    }

    arm_compute::PoolingLayerInfo pool_info =
        arm_compute::PoolingLayerInfo(pool_type, arm_compute::Size2D(pool_size_h, pool_size_w),
                                      arm_compute::DataLayout::NHWC, pad_stride_info);

    layer->outputs.push_back(MakeOutputTensor(node.GetOpShape()[0]));

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
  static void CreateReshapeLayer(CachedLayer* layer, const JSONGraphNode& node) {
    layer->outputs.push_back(MakeOutputTensor(node.GetOpShape()[0]));
    auto function = std::make_shared<arm_compute::NEReshapeLayer>();
    function->configure(&layer->inputs[0], &layer->outputs[0]);
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
               << "Please build with USE_ARM_COMPUTE_LIB_GRAPH_RUNTIME.";
  }

  void BuildEngine() {
    LOG(WARNING) << "Arm Compute Library engine is not initialized. "
                 << "Please build with USE_ARM_COMPUTE_LIB_GRAPH_RUNTIME.";
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

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
