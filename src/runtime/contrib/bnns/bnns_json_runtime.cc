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

/**
 * \file
 * \brief Simple JSON runtime for Apple BNNS primitives
 */

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "bnns_wrp.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace ::tvm::runtime;
using namespace ::tvm::runtime::json;
using namespace ::tvm::runtime::contrib::BNNS;

struct ThreadingConfig {
  /**
   * Internal parallelism level ov BNNS primitive specified via BNNSFilterParameters
   * struct. BNNS doesn't provide real control of internal threading, so it may be
   * ignored by BNNS implementation.
   *
   * Valid values:
   *   0  use default num of threads suggested by BNNS implementation
   *  >0  suggests to use this num of internal BNNS threads
   */
  size_t internalConcurrency = 0;

  /**
   * TVM level parallelism for BNNS runtime.
   * BNNS runtime will split primitive into set of independent sub primitives which
   * can be executed in parallel. As a rule the splitting are performed through output
   * channels, so the effective shape of executed primitive is changed.
   *
   * Valid values:
   *   0  do not use graph level treading
   *  >0  split into this num of primitives
   */
  size_t externalConcurrency = 0;
};

/**
 * Depends on platform hardware the optimal ThreadingConfig may differ.
 * This function contains a priori knowledge about some Apple platforms
 * and their specific.
 *
 * @return default ThreadingConfig suggested for this platform
 */
ThreadingConfig getDefaultThreadingConfig() {
  // TODO(apeskov): have to implement CPU/iOS version check.
  //  meanwhile will use {0, 2} stub to utilize big cores of A13/A14 CPU.
  return {0, 2};
}

/**
 * Main entry point to BNNS runtime
 */
class BNNSJSONRuntime : public JSONRuntimeBase {
 public:
  BNNSJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const override { return "bnns_json"; }

  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    SetupConstants(consts);
    BindInputsAndOutputs();
    AllocateIntermediateTensors();
    BuildEngine();
  }

  void Run() override {
    // Wrap external handler into BNNS tensor representation
    auto bind_ext_hdl_to_tensor = [this](uint32_t eid) {
      const auto& ext_dlt = *data_entry_[eid];
      auto& bnns_tensor = tensors_eid_[eid];
      bnns_tensor->set_data_hdl(ext_dlt.data);
    };

    // Bind all input/output external data object into internal abstractions
    for (const auto& eid : input_var_eid_) bind_ext_hdl_to_tensor(eid);
    for (const auto& out_entity : outputs_) bind_ext_hdl_to_tensor(EntryID(out_entity));

    // Invoke primitives in topological order
    for (const auto& prim : primitives_) prim->execute();
  }

 private:
  /** Make corresponding input/output tensor stubs */
  void BindInputsAndOutputs() {
    tensors_eid_.resize(data_entry_.size());
    auto createTensor = [&](JSONGraphNodeEntry entry) {
      auto node = nodes_[entry.id_];
      auto dlshape = node.GetOpShape()[entry.index_];
      auto dltype = node.GetOpDataType()[entry.index_];
      void* data = nullptr;
      if (data_entry_[entry.id_] != nullptr) data = data_entry_[entry.id_]->data;
      tensors_eid_[entry.id_] = std::make_shared<BNNS::Tensor>(
          BNNS::Shape{dlshape.begin(), dlshape.end()}, convertToBNNS(dltype), data);
    };

    for (auto& id : input_nodes_) {
      auto eid = JSONGraphNodeEntry(id, 0);
      createTensor(eid);
    }

    for (auto entry : outputs_) {
      createTensor(entry);
    }
  }

  /** Allocate intermediate tensors */
  void AllocateIntermediateTensors() {
    for (int i = 0; i < nodes_.size(); ++i) {
      auto eid = JSONGraphNodeEntry(i, 0);
      if (tensors_eid_[eid.id_] != nullptr) continue;
      auto node = nodes_[i];
      auto dlshape = node.GetOpShape()[0];
      auto dltype = node.GetOpDataType()[0];
      tensors_eid_[eid.id_] = std::make_shared<BNNS::Tensor>(
          BNNS::Shape{dlshape.begin(), dlshape.end()}, convertToBNNS(dltype), nullptr);
      tensors_eid_[eid.id_]->allocate_memory();
    }
  }

  // Build up the engine based on the input graph.
  void BuildEngine() {
    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("bnns.conv2d_relu" == op_name) {
          Conv2d(nid, false, "relu");
        } else if ("bnns.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, "relu");
        } else if ("bnns.conv2d_sigmoid" == op_name) {
          Conv2d(nid, false, "sigmoid");
        } else if ("bnns.conv2d_bias_sigmoid" == op_name) {
          Conv2d(nid, true, "sigmoid");
        } else if ("bnns.conv2d_bias" == op_name) {
          Conv2d(nid, true);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("bnns.dense_bias" == op_name) {
          Dense(nid, true);
        } else if ("bnns.dense_bias_gelu" == op_name) {
          Dense(nid, true, true);
        } else if ("nn.batch_matmul" == op_name) {
          MatMul(nid);
        } else if ("nn.instance_norm" == op_name) {
          InstanceNormalization(nid);
        } else if ("nn.max_pool2d" == op_name) {
          Pooling(nid, false);
        } else if ("nn.avg_pool2d" == op_name) {
          Pooling(nid, true);
        } else if ("nn.global_max_pool2d" == op_name) {
          Pooling(nid, false, true);
        } else if ("nn.global_avg_pool2d" == op_name) {
          Pooling(nid, true, true);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Get BNNS tensor.
  std::shared_ptr<BNNS::Tensor> GetBNNSTensor(const JSONGraphNodeEntry& entry) {
    auto eid = EntryID(entry);
    ICHECK(eid < tensors_eid_.size());
    return tensors_eid_[eid];
  }

  void Conv2d(const size_t& nid, const bool has_bias = false,
              const std::string activation_type = "none") {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_entry = node.GetInputs()[0];
    auto wgh_entry = node.GetInputs()[1];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);

    auto dl_input_shape = nodes_[src_entry.id_].GetOpShape()[src_entry.index_];
    auto dl_weight_shape = nodes_[wgh_entry.id_].GetOpShape()[wgh_entry.index_];
    BNNS::Shape input_shape{dl_input_shape.begin(), dl_input_shape.end()};
    BNNS::Shape weight_shape{dl_weight_shape.begin(), dl_weight_shape.end()};
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilation = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    BNNS::Dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    BNNS::Dim PH_L = std::stoi(str_padding[0]),  // height padding: left
        PH_R = std::stoi(str_padding[2]),        // height padding: right
        PW_L = std::stoi(str_padding[1]),        // width padding: left
        PW_R = std::stoi(str_padding[3]),        // width padding: right
        SH = std::stoi(str_strides[0]),          // height-wise stride
        SW = std::stoi(str_strides[1]),          // weight-wise stride
        DH = std::stoi(str_dilation[0]),         // height kernel dilation
        DW = std::stoi(str_dilation[1]);         // width kernel dilation

    // Memory descriptions.
    const auto& src_t = GetBNNSTensor(src_entry);
    const auto& wgh_t = GetBNNSTensor(wgh_entry);
    const auto& dst_t = GetBNNSTensor(dst_entry);

    auto src_view = TView::as_is(src_t).extract_outer_dim().with_layout(BNNSDataLayoutImageCHW);
    auto wgh_view = TView::as_is(wgh_t).with_layout(BNNSDataLayoutConvolutionWeightsOIHW);
    auto dst_view = TView::as_is(dst_t).extract_outer_dim().with_layout(BNNSDataLayoutImageCHW);
    TView bias_view;

    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];

      auto bias_t = GetBNNSTensor(bias_entry);
      bias_view = TView::as_is(bias_t).squeeze().with_layout(BNNSDataLayoutVector);
    }

    BNNSActivation activation = {BNNSActivationFunctionIdentity};
    if (activation_type == "relu")
      activation = {BNNSActivationFunctionRectifiedLinear};
    else if (activation_type == "sigmoid")
      activation = {BNNSActivationFunctionSigmoid};

    BNNSLayerParametersConvolution conv_param = {
        src_view.get_bnns_view(),
        wgh_view.get_bnns_view(),
        dst_view.get_bnns_view(),
        bias_view.get_bnns_view(),
        activation,
        SW,                      /* x_stride */
        SH,                      /* y_stride */
        DW,                      /* x_dilation_stride */
        DH,                      /* y_dilation_stride */
        0,                       /* x_padding, explicit pads will be used */
        0,                       /* y_padding, explicit pads will be used */
        groups,                  /* groups */
        {PW_L, PW_R, PH_L, PH_R} /* explicit pad values */
    };

    size_t num_sub_prim = default_thread_config.externalConcurrency;
    std::vector<BNNSLayerParametersConvolution> params;
    std::tie(params, src_view, dst_view) =
        split_to_n(num_sub_prim, conv_param, src_view, wgh_view, bias_view, dst_view);

    std::vector<BNNSFilter> filters(params.size(), nullptr);
    for (int i = 0; i < params.size(); i++) {
      auto common_filter_param = getCommonFilterParams();
      filters[i] = BNNSFilterCreateLayerConvolution(&params[i], &common_filter_param);
      ICHECK(filters[i]) << "BNNS primitive was not created. Unsupported attributes configuration";
    }

    primitives_.emplace_back(std::make_shared<BNNS::Primitive>(filters, src_view, dst_view));
  }

  void Dense(const size_t& nid, const bool has_bias = false, const bool has_gelu = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);

    // Memory descriptions.
    auto src_t = GetBNNSTensor(src_entry);
    auto wgh_t = GetBNNSTensor(weight_entry);
    auto dst_t = GetBNNSTensor(dst_entry);

    auto src_view = TView::as_is(src_t).extract_outer_dim().with_layout(BNNSDataLayoutVector);
    auto wgh_view = TView::as_is(wgh_t).with_layout(BNNSDataLayoutRowMajorMatrix);
    auto dst_view = TView::as_is(dst_t).extract_outer_dim().with_layout(BNNSDataLayoutVector);

    TView bias_view;
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      auto bias_md = GetBNNSTensor(bias_entry);
      bias_view = TView::as_is(bias_md).with_layout(BNNSDataLayoutVector);
    }

    BNNSActivation activation = {BNNSActivationFunctionIdentity};
    if (has_gelu) {
      activation = {BNNSActivationFunctionGELUApproximation};
      activation.alpha = std::sqrt(2.0 / M_PI);
      activation.beta = 0.044715;
    }

    BNNSLayerParametersFullyConnected layerParameters = {
        src_view.get_bnns_view(),
        wgh_view.get_bnns_view(),
        dst_view.get_bnns_view(),
        bias_view.get_bnns_view(),
        activation,
    };

    auto common_filter_param = getCommonFilterParams();
    auto filter = BNNSFilterCreateLayerFullyConnected(&layerParameters, &common_filter_param);
    ICHECK(filter) << "BNNS primitive was not created. Unsupported attributes configuration";
    std::vector<BNNSFilter> filters = {filter};
    primitives_.emplace_back(std::make_shared<BNNS::Primitive>(filters, src_view, dst_view));
  }

  void MatMul(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto a_entry = node.GetInputs()[0];
    auto b_entry = node.GetInputs()[1];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);
    bool a_is_weighted = data_entry_[EntryID(a_entry)] != nullptr;
    bool b_is_weighted = data_entry_[EntryID(b_entry)] != nullptr;

    // Memory descriptions.
    auto a_t = GetBNNSTensor(a_entry);
    auto b_t = GetBNNSTensor(b_entry);
    auto dst_t = GetBNNSTensor(dst_entry);

    auto a_view = TView::as_is(a_t);
    auto b_view = TView::as_is(b_t);
    auto dst_view = TView::as_is(dst_t);

    BNNSLayerParametersBroadcastMatMul layerParameters = {1,      // alpha
                                                          0,      // beta
                                                          false,  // transA
                                                          true,   // transB
                                                          false,  // quadratic
                                                          a_is_weighted,
                                                          b_is_weighted,
                                                          a_view.get_bnns_view(),
                                                          b_view.get_bnns_view(),
                                                          dst_view.get_bnns_view()};

    // BNNS limitation: MatMul use reverse dims values. However strides are calculated correctly
    //    based on BNNSNDArrayDescriptor::layout value.
    std::reverse(layerParameters.iA_desc.size, layerParameters.iA_desc.size + 3);
    std::reverse(layerParameters.iB_desc.size, layerParameters.iB_desc.size + 3);
    std::reverse(layerParameters.o_desc.size, layerParameters.o_desc.size + 3);

    auto common_filter_param = getCommonFilterParams();
    auto filter = BNNSFilterCreateLayerBroadcastMatMul(&layerParameters, &common_filter_param);
    ICHECK(filter) << "BNNS primitive was not created. Unsupported attributes configuration";

    std::vector<BNNSFilter> filters{filter};
    if (a_is_weighted || b_is_weighted) {
      auto src_view = a_is_weighted ? b_view : a_view;
      primitives_.emplace_back(std::make_shared<BNNS::Primitive>(filters, src_view, dst_view));
    } else {
      primitives_.emplace_back(
          std::make_shared<BNNS::TwoInputPrimitive>(filters, a_view, b_view, dst_view));
    }
  }

  void InstanceNormalization(const size_t& nid) {
    auto node = nodes_[nid];
    size_t axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);
    bool center = std::stoi(node.GetAttr<std::vector<std::string>>("center")[0]);
    bool scale = std::stoi(node.GetAttr<std::vector<std::string>>("scale")[0]);

    // Setup attributes.
    auto src_entry = node.GetInputs()[0];
    auto scale_entry = node.GetInputs()[1];
    auto bias_entry = node.GetInputs()[2];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);

    // Memory descriptions.
    auto src_t = GetBNNSTensor(src_entry);
    auto scale_t = GetBNNSTensor(scale_entry);
    auto bias_t = GetBNNSTensor(bias_entry);
    auto dst_t = GetBNNSTensor(dst_entry);

    auto src_view = TView::as_is(src_t);
    auto dst_view = TView::as_is(dst_t);
    size_t src_rank = Tensor::getRank(src_view.get_bnns_view());
    size_t dst_rank = Tensor::getRank(dst_view.get_bnns_view());
    ICHECK_EQ(src_rank, dst_rank);
    ICHECK_LE(src_rank, 4);
    if (src_rank < 4) {
      src_view = src_view.unsqueeze(4);
      dst_view = dst_view.unsqueeze(4);
    }
    src_view = src_view.extract_outer_dim().with_layout(BNNSDataLayoutImageCHW);
    dst_view = dst_view.extract_outer_dim().with_layout(BNNSDataLayoutImageCHW);
    auto scale_view = TView::as_is(scale_t).with_layout(BNNSDataLayoutVector);
    auto bias_view = TView::as_is(bias_t).with_layout(BNNSDataLayoutVector);
    BNNSActivation activation = {BNNSActivationFunctionIdentity};

    auto b_desc = bias_view.get_bnns_view();
    if (!center) b_desc = {};
    auto s_desc = scale_view.get_bnns_view();
    if (!scale) s_desc = {};

    // NOTE: Axis option is ignored in BNNS. The result doesn't depends on value of axis.
    BNNSLayerParametersNormalization layerParameters = {src_view.get_bnns_view(),  // i_desc
                                                        dst_view.get_bnns_view(),  // o_desc
                                                        b_desc,                    // beta_desc
                                                        s_desc,                    // gamma_desc
                                                        {},          // moving_mean_desc
                                                        {},          // moving_variance_desc
                                                        1.f,         // momentum
                                                        epsilon,     // epsilon
                                                        activation,  // activation
                                                        1,           // num_groups
                                                        axis};       // normalization_axis

    BNNSFilterType filter_type = BNNSInstanceNorm;
    auto common_filter_param = getCommonFilterParams();
    auto filter =
        BNNSFilterCreateLayerNormalization(filter_type, &layerParameters, &common_filter_param);
    ICHECK(filter) << "BNNS primitive was not created. Unsupported attributes configuration";

    std::vector<BNNSFilter> filters{filter};
    primitives_.emplace_back(std::make_shared<BNNS::NormPrimitive>(filters, src_view, dst_view));
  }

  void Pooling(const size_t& nid, bool avg_pooling, bool global = false) {
    auto node = nodes_[nid];

    auto src_entry = node.GetInputs()[0];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);

    // Memory descriptions.
    auto src_t = GetBNNSTensor(src_entry);
    auto dst_t = GetBNNSTensor(dst_entry);

    auto src_view = TView::as_is(src_t);
    auto dst_view = TView::as_is(dst_t);
    size_t src_rank = Tensor::getRank(src_view.get_bnns_view());
    size_t dst_rank = Tensor::getRank(dst_view.get_bnns_view());
    ICHECK_EQ(src_rank, dst_rank);
    ICHECK_LE(src_rank, 4);
    if (src_rank < 4) {
      src_view = src_view.unsqueeze(4);
      dst_view = dst_view.unsqueeze(4);
    }
    src_view = src_view.extract_outer_dim().with_layout(BNNSDataLayoutImageCHW);
    dst_view = dst_view.extract_outer_dim().with_layout(BNNSDataLayoutImageCHW);
    BNNSActivation activation = {BNNSActivationFunctionIdentity};
    BNNSPoolingFunction pf = {BNNSPoolingFunctionMax};
    if (avg_pooling) pf = {BNNSPoolingFunctionAverageCountExcludePadding};

    // Setup attributes.
    size_t k_height = 0;
    size_t k_width = 0;
    size_t y_padding = 0;
    size_t x_padding = 0;
    size_t y_stride = 1;
    size_t x_stride = 1;
    if (!global) {
      std::vector<std::string> pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
      std::vector<std::string> padding = node.GetAttr<std::vector<std::string>>("padding");
      std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
      k_height = std::stoi(pool_size[0]);
      k_width = std::stoi(pool_size[1]);
      y_padding = std::stoi(padding[0]);
      x_padding = std::stoi(padding[1]);
      y_stride = std::stoi(strides[0]);
      x_stride = std::stoi(strides[1]);
    } else {
      auto sv = src_view.get_bnns_view();
      k_height = sv.size[1];
      k_width = sv.size[0];
    }

    BNNSLayerParametersPooling layerParameters = {src_view.get_bnns_view(),  // i_desc
                                                  dst_view.get_bnns_view(),  // o_desc
                                                  {},                        // bias
                                                  activation,                // activation
                                                  pf,                        // pooling_function
                                                  k_width,                   // k_width
                                                  k_height,                  // k_height
                                                  x_stride,                  // x_stride
                                                  y_stride,                  // y_stride
                                                  0,                         // x_dilation_stride
                                                  0,                         // y_dilation_stride
                                                  x_padding,                 // x_padding
                                                  y_padding,                 // y_padding
                                                  {}};  // pad left, right, up, down padding

    auto common_filter_param = getCommonFilterParams();
    auto filter = BNNSFilterCreateLayerPooling(&layerParameters, &common_filter_param);
    ICHECK(filter) << "BNNS primitive was not created. Unsupported attributes configuration";

    std::vector<BNNSFilter> filters{filter};
    primitives_.emplace_back(std::make_shared<BNNS::PoolingPrimitive>(filters, src_view, dst_view));
  }

  BNNS::Dtype convertToBNNS(const DLDataType& dl_dtype) {
    if (dl_dtype.code == DLDataTypeCode::kDLFloat) {
      if (dl_dtype.bits == 32) return BNNSDataTypeFloat32;
      if (dl_dtype.bits == 16) return BNNSDataTypeFloat16;
    }
    if (dl_dtype.code == DLDataTypeCode::kDLInt) {
      if (dl_dtype.bits == 32) return BNNSDataTypeInt32;
      if (dl_dtype.bits == 16) return BNNSDataTypeInt16;
      if (dl_dtype.bits == 8) return BNNSDataTypeInt8;
    }
    if (dl_dtype.code == DLDataTypeCode::kDLUInt) {
      if (dl_dtype.bits == 32) return BNNSDataTypeUInt32;
      if (dl_dtype.bits == 16) return BNNSDataTypeUInt16;
      if (dl_dtype.bits == 8) return BNNSDataTypeUInt8;
    }
    LOG(FATAL) << "Unsupported data type for BNNS runtime";
  }

  BNNSFilterParameters getCommonFilterParams() {
    // NOTE: To force weights tensor copy on stage of filter create
    //       just change : BNNSFlagsUseClientPtr -> 0
    return {BNNSFlagsUseClientPtr, default_thread_config.internalConcurrency};
  }

  /** Default threading config. Should be used if there are
   *  no other threading specificator. */
  const ThreadingConfig default_thread_config = getDefaultThreadingConfig();

  /** Collection of all primitives in topological order */
  std::vector<std::shared_ptr<BNNS::Primitive>> primitives_;

  /** Vector with BNNS tensors. Index of tensor matched with
   *  corresponding EntryID from base JSONRuntimeBase. */
  std::vector<TensorPtr> tensors_eid_;
};

runtime::Module BNNSJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<BNNSJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.BNNSJSONRuntimeCreate").set_body_typed(BNNSJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_bnns_json")
    .set_body_typed(BNNSJSONRuntime::LoadFromBinary<BNNSJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
