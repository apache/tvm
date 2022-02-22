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
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <regex>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      // TODO(@comaniac): Support other data lengths.
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      write_to_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                           offset_in_bytes);
    }

    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      net_.at(i).execute(stream_, net_args_.at(i));
    }
    stream_.wait();

    // Read output buffers.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      read_from_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                            offset_in_bytes);
    }
  }

 private:
  // Build up the engine based on the input graph.
  std::map<std::string, tag> layout_dict{
      {"NCW", tag::ncw},       {"OIW", tag::oiw},     {"GOIW", tag::goiw},   {"NCHW", tag::nchw},
      {"OIHW", tag::oihw},     {"GOIHW", tag::goihw}, {"NCDHW", tag::ncdhw}, {"OIDHW", tag::oidhw},
      {"GOIDHW", tag::goidhw}, {"IOHW", tag::iohw},   {"GIOHW", tag::giohw}, {"IODHW", tag::iodhw},
      {"GIODHW", tag::giodhw},
  };

  std::map<std::string, dnnl::algorithm> elt_name2algo{
      {"abs", dnnl::algorithm::eltwise_abs},
      {"exp", dnnl::algorithm::eltwise_exp},
      {"log", dnnl::algorithm::eltwise_log},
      {"sqrt", dnnl::algorithm::eltwise_sqrt},
      {"round", dnnl::algorithm::eltwise_round},
      {"logsumexp", dnnl::algorithm::eltwise_logsigmoid},
      {"nn.relu", dnnl::algorithm::eltwise_relu},
      {"nn.leaky_relu", dnnl::algorithm::eltwise_relu},
      {"tanh", dnnl::algorithm::eltwise_tanh},
      {"sigmoid", dnnl::algorithm::eltwise_logistic},
      {"clip", dnnl::algorithm::eltwise_clip},
  };

  bool ParsingOpName(const std::string op_name, dnnl::primitive_attr attr) {
    // Define RegExp.
    std::regex bias_add_pat(".*_bias.*");
    std::regex relu_pat(".*_relu.*");
    std::regex tanh_pat(".*_tanh.*");
    std::regex sigmoid_pat(".*_sigmoid.*");

    // Parsing post-ops.
    dnnl::post_ops ops;
    if (std::regex_match(op_name, relu_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    }
    if (std::regex_match(op_name, tanh_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
    }
    if (std::regex_match(op_name, sigmoid_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
    }
    attr.set_post_ops(ops);

    // Parsing bias_add.
    return std::regex_match(op_name, bias_add_pat) ? true : false;
  }

  dnnl::memory::dims TransformStr2Dims(std::vector<std::string> strs, std::string str_name) {
    dnnl::memory::dims out_dims;
    if (str_name == "dilates") {
      std::transform(strs.begin(), strs.end(), std::back_inserter(out_dims),
                     [](const std::string& str) { return std::stoi(str) - 1; });
    } else {
      std::transform(strs.begin(), strs.end(), std::back_inserter(out_dims),
                     [](const std::string& str) { return std::stoi(str); });
    }
    return out_dims;
  }

  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    std::regex conv_pat(".*conv[1-3]d.*");
    std::regex conv_tranpose_pat(".*conv[1-3]d_transpose.*");
    std::regex dense_pat(".*dense.*");
    std::regex max_pool_pat(".*max_pool[1-3]d");
    std::regex avg_pool_pat(".*avg_pool[1-3]d");

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if (std::regex_match(op_name, conv_tranpose_pat)) {
          Deconvolution(nid);
        } else if (std::regex_match(op_name, conv_pat)) {
          Convolution(nid);
        } else if (std::regex_match(op_name, dense_pat)) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if (std::regex_match(op_name, max_pool_pat)) {
          Pooling(nid, dnnl::algorithm::pooling_max);
        } else if (std::regex_match(op_name, avg_pool_pat)) {
          Pooling(nid, dnnl::algorithm::pooling_avg);
        } else if (elt_name2algo.count(op_name)) {
          Eltwise(nid);
        } else if ("nn.softmax" == op_name) {
          Softmax(nid);
        } else if ("add" == op_name) {
          Binary(nid, dnnl::algorithm::binary_add);
        } else if ("multiply" == op_name) {
          Binary(nid, dnnl::algorithm::binary_mul);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Bind a JSON graph node entry to a DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
    }
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    // Since the DNNL memory has been created before calling this function, we assume the entry
    // has not yet been bound to the other DNNL memory; otherwise it may have memory leak.
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK_EQ(dltype.bits, 32);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  void Convolution(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    dnnl::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    dnnl::memory::dim channels =
        node.GetAttr<std::vector<std::string>>("channels")[0] != ""
            ? std::stoi(node.GetAttr<std::vector<std::string>>("channels")[0])
            : out_shape[1];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    std::string data_layout = node.GetAttr<std::vector<std::string>>("data_layout")[0];
    std::string kernel_layout = node.GetAttr<std::vector<std::string>>("kernel_layout")[0];

    // Check layout.
    if (layout_dict.find(data_layout) == layout_dict.end() ||
        layout_dict.find(kernel_layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported layout for conv: " << data_layout << " " << kernel_layout;
    }

    // Memory shapes.
    dnnl::memory::dims src_dims = input_shape;       // {N, IC, ID, IH, IW}
    dnnl::memory::dims weights_dims = weight_shape;  // {OC, IC, KD, KH, KW}
    if (groups > 1) {
      weights_dims = {groups, channels / groups, input_shape[1] / groups};
      weights_dims.insert(weights_dims.end(), weight_shape.begin() + 2, weight_shape.end());
      kernel_layout.insert(0, "G");
    }
    dnnl::memory::dims bias_dims = {channels};
    dnnl::memory::dims dst_dims = out_shape;  // {N, OC, OD, OH, OW}
    dnnl::memory::dims strides_dims = TransformStr2Dims(str_strides, "strides");
    dnnl::memory::dims dilates_dims = TransformStr2Dims(str_dilates, "dilates");
    dnnl::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l, "padding");
    dnnl::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r, "padding");

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::f32, layout_dict[data_layout]);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, layout_dict[kernel_layout]);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, layout_dict[data_layout]);

    // Covn2d description.
    auto conv_desc =
        has_bias ? dnnl::convolution_forward::desc(
                       dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                       conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, strides_dims,
                       dilates_dims, padding_dims_l, padding_dims_r)
                 : dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                                   dnnl::algorithm::convolution_direct, conv_src_md,
                                                   conv_weights_md, conv_dst_md, strides_dims,
                                                   dilates_dims, padding_dims_l, padding_dims_r);

    // Enable elementwise post-ops.
    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    auto conv2d_src_memory = BindDNNLMemory(data_entry, conv_src_md);

    // Weight memory.
    auto conv2d_weights_memory = BindDNNLMemory(weight_entry, conv_weights_md);

    // Output memory.
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, conv2d_prim_desc.dst_desc());

    // Bias memory.
    auto conv2d_bias_memory = dnnl::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, conv2d_bias_memory);

      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                           {DNNL_ARG_BIAS, conv2d_bias_memory},
                           {DNNL_ARG_DST, conv2d_dst_memory}});
    } else {
      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                           {DNNL_ARG_DST, conv2d_dst_memory}});
    }
  }

  void Deconvolution(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    dnnl::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    dnnl::memory::dim channels =
        node.GetAttr<std::vector<std::string>>("channels")[0] != ""
            ? std::stoi(node.GetAttr<std::vector<std::string>>("channels")[0])
            : out_shape[1];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    std::string data_layout = node.GetAttr<std::vector<std::string>>("data_layout")[0];
    std::string kernel_layout = node.GetAttr<std::vector<std::string>>("kernel_layout")[0];

    // Check layout.
    if (layout_dict.find(data_layout) == layout_dict.end() ||
        layout_dict.find(kernel_layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported layout: " << data_layout << " " << kernel_layout;
    }

    // Memory shapes.
    dnnl::memory::dims src_dims = input_shape;       // {N, IC, ID, IH, IW}
    dnnl::memory::dims weights_dims = weight_shape;  // {OC, IC, KD, KH, KW}

    // Check weight shape, transform to `OIHW`
    if (weights_dims[0] == src_dims[1] && weights_dims[1] == channels) {
      std::swap(weights_dims[0], weights_dims[1]);
    }
    if (kernel_layout == "OIDHW") {
      kernel_layout = "IODHW";
    }
    if (groups > 1) {
      weights_dims = {groups, channels / groups, input_shape[1] / groups};
      weights_dims.insert(weights_dims.end(), weight_shape.begin() + 2, weight_shape.end());
      kernel_layout.insert(0, "G");
    }
    dnnl::memory::dims bias_dims = {channels};
    dnnl::memory::dims dst_dims = out_shape;  // {N, OC, OD, OH, OW}
    dnnl::memory::dims strides_dims = TransformStr2Dims(str_strides, "strides");
    dnnl::memory::dims dilates_dims = TransformStr2Dims(str_dilates, "dilates");
    dnnl::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l, "padding");
    dnnl::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r, "padding");

    // Memory descriptions.
    auto deconv_src_md = dnnl::memory::desc(src_dims, dt::f32, layout_dict[data_layout]);
    auto deconv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, layout_dict[kernel_layout]);
    auto deconv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::any);
    auto deconv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, layout_dict[data_layout]);

    // Transposed covn2d description.
    auto deconv_desc =
        has_bias ? dnnl::deconvolution_forward::desc(
                       dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                       deconv_src_md, deconv_weights_md, deconv_bias_md, deconv_dst_md,
                       strides_dims, dilates_dims, padding_dims_l, padding_dims_r)
                 : dnnl::deconvolution_forward::desc(
                       dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                       deconv_src_md, deconv_weights_md, deconv_dst_md, strides_dims, dilates_dims,
                       padding_dims_l, padding_dims_r);

    // Enable elementwise post-ops.
    auto deconv2d_prim_desc =
        dnnl::deconvolution_forward::primitive_desc(deconv_desc, attr, engine_);

    // Push to the network.
    auto deconv = dnnl::deconvolution_forward(deconv2d_prim_desc);
    net_.push_back(deconv);

    // Data memory.
    auto deconv2d_src_memory = BindDNNLMemory(data_entry, deconv_src_md);

    // Weight memory.
    auto deconv2d_weights_memory = BindDNNLMemory(weight_entry, deconv_weights_md);

    // Output memory.
    auto deconv2d_dst_memory = BindDNNLMemory(out_entry, deconv2d_prim_desc.dst_desc());

    // Bias memory.
    auto deconv2d_bias_memory = dnnl::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, deconv2d_bias_memory);

      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, deconv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, deconv2d_weights_memory},
                           {DNNL_ARG_BIAS, deconv2d_bias_memory},
                           {DNNL_ARG_DST, deconv2d_dst_memory}});
    } else {
      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, deconv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, deconv2d_weights_memory},
                           {DNNL_ARG_DST, deconv2d_dst_memory}});
    }
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    dnnl::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    dnnl::memory::dim OC = out_shape[1];

    // Memory shapes.
    dnnl::memory::dims data_dims = input_shape;
    dnnl::memory::dims weight_dims = weight_shape;
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = out_shape;

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);

    // Enable elementwise post-ops.
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, attr, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);

    // Bias memory.
    auto bias_memory = dnnl::memory(bias_md, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, bias_memory);
    } else {
      float bias[OC] = {0};
      write_to_dnnl_memory(bias, bias_memory, OC * sizeof(float));
    }

    // Output memory.
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    dnnl::memory::dims data_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim IC = data_shape[1];
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    // Memory description.
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

    // BN description.
    auto bn_desc = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, data_md, epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(bn_desc, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_prim_desc);
    net_.push_back(bn);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);
    auto mean_memory = BindDNNLMemory(mean_entry, bn_prim_desc.mean_desc());
    auto variance_memory = BindDNNLMemory(variance_entry, bn_prim_desc.variance_desc());

    // In DNNL, weight is composed of gamma+beta, so we point them to the same DNNL memory but
    // assign an offset to beta data for runtime serialization.
    auto weight_memory = BindDNNLMemory(gamma_entry, bn_prim_desc.weights_desc(), 0);
    BindDNNLMemory(beta_entry, weight_memory, IC);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_DST, out_memory},
                         {DNNL_ARG_SCALE_SHIFT, weight_memory},
                         {DNNL_ARG_MEAN, mean_memory},
                         {DNNL_ARG_VARIANCE, variance_memory}});
  }

  void Pooling(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    std::vector<std::string> str_kernel = node.GetAttr<std::vector<std::string>>("pool_size");
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::string layout = node.GetAttr<std::vector<std::string>>("layout")[0];

    // Check layout.
    if (layout_dict.find(layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported layout for pooling: " << layout;
    }

    // Attributes related to AvgPool
    if (algo == dnnl::algorithm::pooling_avg) {
      int int_countpad = std::stoi(node.GetAttr<std::vector<std::string>>("count_include_pad")[0]);
      bool count_include_pad = int_countpad != 0 ? true : false;
      algo = count_include_pad ? dnnl::algorithm::pooling_avg_include_padding
                               : dnnl::algorithm::pooling_avg_exclude_padding;
    }

    dnnl::memory::dims src_dims = input_shape;
    dnnl::memory::dims dst_dims = out_shape;
    dnnl::memory::dims kernel_dims = TransformStr2Dims(str_kernel, "kernel");
    dnnl::memory::dims strides_dims = TransformStr2Dims(str_strides, "strides");
    dnnl::memory::dims dilates_dims = TransformStr2Dims(str_dilates, "dilates");
    dnnl::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l, "padding");
    dnnl::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r, "padding");

    // Memory descriptions.
    auto pool_src_md = dnnl::memory::desc(src_dims, dt::f32, layout_dict[layout]);
    auto pool_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::any);

    // Pooling description.
    auto pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference, algo,
                                                 pool_src_md, pool_dst_md, strides_dims,
                                                 kernel_dims, padding_dims_l, padding_dims_r);

    auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(pool_desc, engine_, true);
    auto pool = dnnl::pooling_forward(pool_prim_desc);
    net_.push_back(pool);

    // Memories.
    auto pool2d_src_memory = BindDNNLMemory(data_entry, pool_src_md);

    auto pool2d_dst_memory = BindDNNLMemory(out_entry, pool_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({{DNNL_ARG_SRC, pool2d_src_memory}, {DNNL_ARG_DST, pool2d_dst_memory}});
  }

  void Eltwise(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    auto algo = elt_name2algo[op_name];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(shape, dt::f32);
    float alpha = 0., beta = 0.;
    if (op_name == "clip") {
      alpha = std::stof(node.GetAttr<std::vector<std::string>>("a_min")[0]);
      beta = std::stof(node.GetAttr<std::vector<std::string>>("a_max")[0]);
    } else if (op_name == "nn.leaky_relu") {
      alpha = std::stof(node.GetAttr<std::vector<std::string>>("alpha")[0]);
    }

    auto elt_desc =
        dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algo, data_md, alpha, beta);
    auto elt_prim_desc = dnnl::eltwise_forward::primitive_desc(elt_desc, engine_);
    ICHECK(data_md == elt_prim_desc.dst_desc());

    auto elt = dnnl::eltwise_forward(elt_prim_desc);
    net_.push_back(elt);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    if (axis < 0) {
      axis = shape.size() + axis;
    }
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(shape, dt::f32);

    auto softmax_desc =
        dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference, data_md, axis);
    auto softmax_prim_desc = dnnl::softmax_forward::primitive_desc(softmax_desc, engine_);
    ICHECK(data_md == softmax_prim_desc.dst_desc());

    auto softmax = dnnl::softmax_forward(softmax_prim_desc);
    net_.push_back(softmax);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Binary(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];

    // Memory and compute description.
    std::vector<dnnl::memory::dims> data_dims;
    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;

    ICHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));
    }
    ICHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    auto binary_desc = dnnl::binary::desc(algo, data_mds[0], data_mds[1], out_md);
    auto binary_prim_desc = dnnl::binary::primitive_desc(binary_desc, engine_);
    auto binary = dnnl::binary(binary_prim_desc);
    net_.push_back(binary);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
  }

  // Generate DNNL memory description and infer the data layout by the given shape.
  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
    dnnl::memory::desc data_md;
    switch (shape.size()) {
      case 2:
        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::pair<dnnl::memory, size_t>> entry_out_mem_;
};

runtime::Module DNNLJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
