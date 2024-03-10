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
#include <string>
#include <vector>

#include "../../../runtime/regex.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

// TODO(@apeskov): Have to mute warning from dnnl headers.
//  -Wzero-as-null-pointer-constant and -Wdocumentation-unknown-command
#include <dnnl.hpp>

#include "dnnl_tensor_requisite.h"
#include "dnnl_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        next_unique_eid_offset_(data_entry_.size()),
        run_arg_eid_(input_var_eid_) {
    for (const auto e : outputs_) run_arg_eid_.push_back(EntryID(e));
  }

  const char* type_key() const override { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
    BuildEngine();
  }

  /* Unused stub implementation */
  void Run() override { LOG(FATAL) << "Unreachable code"; }

  /* Thread safe implementation of Run. Keep runtime instance immutable */
  void Run(const TVMArgs& args) const {
    auto arg_data_provider = makeIODataProvider(args);
    auto mem_solver = tensor_registry_.MakeSolver(arg_data_provider);
    // Execute primitives one by one
    for (const auto& act : net_) {
      auto prim = std::get<0>(act);
      auto arg_reqs = std::get<1>(act);

      // Find proper dnnl::memory buffers
      std::unordered_map<int, dnnl::memory> mem_args;
      for (const auto& kvp : arg_reqs) mem_args[kvp.first] = mem_solver(kvp.second);

      // skip the reorder if src==dst to enable inplace operation
      if (prim.get_kind() == dnnl::primitive::kind::reorder) {
        const auto& mem_src = mem_args.at(DNNL_ARG_SRC);
        const auto& mem_dst = mem_args.at(DNNL_ARG_DST);
        if ((mem_src.get_desc() == mem_dst.get_desc()) &&
            (mem_src.get_data_handle() == mem_dst.get_data_handle())) {
          continue;
        }
      }

      prim.execute(stream_, mem_args);
    }
  }

  /* Override GetFunction to reimplement Run method */
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";

        ICHECK_EQ(args.size(), input_var_eid_.size() + outputs_.size())
            << "Found mismatch in the number of provided data entries and required.";

        Run(args);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name, sptr_to_self);
    }
  }

  /* Same as makeInitDataProvider but in case of InputOutput return real DLTensor */
  TensorRegistry::DLTensorProvider makeIODataProvider(const TVMArgs& args) const {
    auto extract_dl_tensor = [](const TVMArgValue& val) -> const DLTensor* {
      ICHECK(val.type_code() == kTVMNDArrayHandle || val.type_code() == kTVMDLTensorHandle)
          << "Expect NDArray or DLTensor";
      return val.IsObjectRef<NDArray>() ? val.operator NDArray().operator->()
                                        : val.operator DLTensor*();
    };

    std::map<uint32_t, const DLTensor*> io_map;  // eid to dl tensor map
    for (size_t i = 0; i < run_arg_eid_.size(); i++) {
      io_map[run_arg_eid_[i]] = extract_dl_tensor(args[i]);
    }

    // lambda with captured IO data handlers
    return [io_map](uint32_t eid) -> const DLTensor* { return io_map.at(eid); };
  }

 private:
  const std::map<std::string, dnnl::algorithm> elt_name2algo{
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
      {"gelu_erf", dnnl::algorithm::eltwise_gelu_erf},
  };

  dnnl::primitive_attr ParseAttrs(const size_t& nid, TensorRequisite* bias_tr) {
    dnnl::primitive_attr attr;

    // Post op attributes based on named inputs.
    auto dst_zp_tr = GetInputByName(nid, "dst_zp_idx");
    auto o_scl_tr = GetInputByName(nid, "o_scl_idx");
    auto sum_scl_tr = GetInputByName(nid, "sum_scl_idx");

    if (o_scl_tr) {
      ICHECK(o_scl_tr.IsConstant());
      auto data = o_scl_tr.GetConstDataLikeVec<float>();
      attr.set_output_scales(data.size() == 1 ? 0 : (1 << 1), data);
    }

    auto activation = GetNodeAttr<std::vector<std::string>>(nodes_[nid], "activation", {"none"});
    if (activation[0] != "none") {
      auto a_type = elt_name2algo.at(activation[0]);
      auto a_scale = GetInput(nid, std::stoi(activation[1])).GetConstScalarData<float>();
      auto a_alfa = GetInput(nid, std::stoi(activation[2])).GetConstScalarData<float>();
      auto a_beta = GetInput(nid, std::stoi(activation[3])).GetConstScalarData<float>();

      auto ops = attr.get_post_ops();
      ops.append_eltwise(a_scale, a_type, a_alfa, a_beta);
      attr.set_post_ops(ops);
    }

    if (sum_scl_tr) {
      auto scl = sum_scl_tr.GetConstScalarData<float>();
      auto ops = attr.get_post_ops();
      ops.append_sum(scl);
      attr.set_post_ops(ops);
    }

    if (dst_zp_tr) {
      auto zp = dst_zp_tr.GetConstScalarData<float>();
      // Use linear post op instead of set_zero_points(). Because of limitation of int32 type,
      // but we have to use float.
      auto ops = attr.get_post_ops();
      ops.append_eltwise(1.0, dnnl::algorithm::eltwise_linear, 1.0, zp);
      attr.set_post_ops(ops);
    }
    *bias_tr = GetInputByName(nid, "bias_idx");

    if (o_scl_tr || activation[0] != "none" || sum_scl_tr || dst_zp_tr) return attr;

    // Define RegExp.
    std::string bias_add_pat(".*_bias.*");
    std::string relu_pat(".*_relu.*");
    std::string tanh_pat(".*_tanh.*");
    std::string sigmoid_pat(".*_sigmoid.*");
    std::string clip_pat(".*_clip.*");
    std::string gelu_pat(".*_gelu.*");
    std::string swish_pat(".*_swish.*");
    std::string sum_pat(".*_sum.*");
    std::string mish_pat(".*_mish.*");

    // parsing of name to extract attributes
    auto op_name = nodes_[nid].GetOpName();

    // Parsing post-ops.
    dnnl::post_ops ops;
    if (tvm::runtime::regex_match(op_name, sum_pat)) {
      ops.append_sum(1.f);
    }
    if (tvm::runtime::regex_match(op_name, relu_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    }
    if (tvm::runtime::regex_match(op_name, tanh_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
    }
    if (tvm::runtime::regex_match(op_name, clip_pat)) {
      float a_min = GetNodeAttr<float>(nodes_[nid], "a_min");
      float a_max = GetNodeAttr<float>(nodes_[nid], "a_max");
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_clip, a_min, a_max);
    }
    if (tvm::runtime::regex_match(op_name, sigmoid_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
    }
    if (tvm::runtime::regex_match(op_name, swish_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_swish, 1.f, 1.f);
    }
    if (tvm::runtime::regex_match(op_name, gelu_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f);
    }
    if (tvm::runtime::regex_match(op_name, mish_pat)) {
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_mish, 1.f, 0.f);
    }
    if (ops.len() != 0) {
      attr.set_post_ops(ops);
    }

    // Parsing bias_add.
    *bias_tr =
        tvm::runtime::regex_match(op_name, bias_add_pat) ? GetInput(nid, 2) : TensorRequisite{};

    return attr;
  }

  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    std::set<uint32_t> io_eid_set(run_arg_eid_.begin(), run_arg_eid_.end());
    tensor_registry_ = TensorRegistry(engine_, io_eid_set);

    std::string conv_pat(".*conv[1-3]d.*");
    std::string deconv_pat(".*deconv[1-3]d.*");
    std::string conv_transpose_pat(".*conv[1-3]d_transpose.*");
    std::string dense_pat(".*dense.*");
    std::string max_pool_pat(".*max_pool[1-3]d");
    std::string avg_pool_pat(".*avg_pool[1-3]d");

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if (tvm::runtime::regex_match(op_name, deconv_pat) ||
            tvm::runtime::regex_match(op_name, conv_transpose_pat)) {
          Deconvolution(nid);
        } else if (tvm::runtime::regex_match(op_name, conv_pat)) {
          Convolution(nid);
        } else if (tvm::runtime::regex_match(op_name, dense_pat)) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if (tvm::runtime::regex_match(op_name, max_pool_pat)) {
          Pooling(nid, dnnl::algorithm::pooling_max);
        } else if (tvm::runtime::regex_match(op_name, avg_pool_pat)) {
          Pooling(nid, dnnl::algorithm::pooling_avg);
        } else if (elt_name2algo.count(op_name)) {
          Eltwise(nid);
        } else if ("nn.softmax" == op_name) {
          Softmax(nid);
        } else if ("add" == op_name) {
          Binary(nid, dnnl::algorithm::binary_add);
        } else if ("multiply" == op_name) {
          Binary(nid, dnnl::algorithm::binary_mul);
        } else if ("nn.layer_norm" == op_name) {
          LayerNorm(nid);
        } else if ("nn.batch_matmul" == op_name) {
          BatchMatMul(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  void Convolution(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = nodes_[nid].GetOpName();

    // Setup attributes.
    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};

    auto attr = ParseAttrs(nid, &bias_tr);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto strides = GetNodeAttr<std::vector<int64_t>>(node, "strides");
    auto dilates = GetNodeAttr<std::vector<int64_t>>(node, "dilation");
    auto padding = GetNodeAttr<std::vector<int64_t>>(node, "padding");
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());
    auto groups = GetNodeAttr<int>(node, "groups");
    auto src_layout = GetNodeAttr<std::string>(node, "data_layout");
    auto dst_layout = GetNodeAttr<std::string>(node, "out_layout");
    auto wgh_layout = GetNodeAttr<std::string>(node, "kernel_layout");

    // dst_layout == "" means to use data_layout
    if (dst_layout.empty()) dst_layout = src_layout;

    // Minus one for DNNL representation. No dilation for DNNL is 0, for relay is 1.
    for (auto& d : dilates) d--;

    // Take into account provided layout strings
    src_tr = src_tr.TreatAs(src_layout);
    dst_tr = dst_tr.TreatAs(dst_layout);
    wgh_tr = wgh_tr.TreatAs(wgh_layout);

    // Should support G mixed with O. Like { G*O, I, H, W }
    // Use { G, O, I, H, W } weight format even if groups == 1
    if (wgh_layout.find("G") == std::string::npos) {
      auto w_dims = wgh_tr.dims();
      w_dims[0] /= groups;
      w_dims.insert(w_dims.begin(), groups);
      wgh_tr = wgh_tr.Reshape(w_dims);
    }

    // Assumption that bias is correct and can be squeezed to 1D
    bias_tr = bias_tr.Reshape({dst_tr.dims()[1]});

    // TODO(@apeskov): This is WA. In case of padded blocked tensor format we do not know original
    //  shapes. Example tensor {1, 10, 224, 224} with layout "NCNH8c" will lead to tensor
    //  {1, 2, 224, 224, 8}. Identically as for shapes {1, 11, 224, 224} or {1, 15, 224, 224}.
    //
    // Let's try to compensate it for weight tensor. Weight IC should match with source IC.
    // Example src: [1, 3, 224, 224] with layout NCHW
    //         wgh: [16, 3, 3, 3] with layout OIHW2i8o -> [2, 2, 3, 3, 2, 8]
    // Similarly, Weight OC should match with destination OC.
    // Example dst: [1, 1000, 7, 7] with layout NCHW
    //         wgh: [1000, 1024, 1, 1] with layout OIHW48o -> [21, 1024, 1, 1, 48]
    if (wgh_tr.dims()[0] != groups || wgh_tr.dims()[1] != dst_tr.dims()[1] / groups ||
        wgh_tr.dims()[2] != src_tr.dims()[1] / groups) {
      auto wgh_croped_dims = wgh_tr.dims();
      wgh_croped_dims[0] = groups;
      wgh_croped_dims[1] = dst_tr.dims()[1] / groups;  // wgh_OC = dst_OC / groups
      wgh_croped_dims[2] = src_tr.dims()[1] / groups;  // wgh_IC = src_IC / groups
      auto zero_offset = dnnl::memory::dims(wgh_tr.dims().size(), 0);
      wgh_tr = wgh_tr.Crop(wgh_croped_dims, zero_offset);
    }

    // Conv description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        src_tr.LayoutAny().desc(), wgh_tr.LayoutAny().desc(), bias_tr.LayoutAny().desc(),
        dst_tr.LayoutAny().desc(), strides, dilates, padding_l, padding_r);

    // Enable elementwise post-ops.
    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    src_tr = src_tr.RequestLayout(conv_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(conv_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(conv_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(conv_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(conv_prim_desc.scratchpad_desc());

    // TODO(@apeskov): Simulation of inplace primitive. just as PoC.
    auto sum_in_tr = GetInputByName(nid, "sum_idx").TreatAs(dst_layout);
    if (op_name.find("_sum") != std::string::npos) {
      sum_in_tr = GetInput(nid, node.GetInputs().size() - 1);
      sum_in_tr = sum_in_tr.TreatAs(dst_layout);
    }

    Submit(dnnl::convolution_forward(conv_prim_desc),
           {{DNNL_ARG_SRC, src_tr},
            {DNNL_ARG_WEIGHTS, wgh_tr},
            {DNNL_ARG_BIAS, bias_tr},
            {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
            {DNNL_ARG_DST, dst_tr}},
           {sum_in_tr, DNNL_ARG_DST});
  }

  void Deconvolution(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};

    auto attr = ParseAttrs(nid, &bias_tr);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto strides = GetNodeAttr<std::vector<int64_t>>(node, "strides");
    auto dilates = GetNodeAttr<std::vector<int64_t>>(node, "dilation");
    auto padding = GetNodeAttr<std::vector<int64_t>>(node, "padding");
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());
    auto groups = GetNodeAttr<int>(node, "groups");
    auto src_layout = GetNodeAttr<std::string>(node, "data_layout");
    auto dst_layout = GetNodeAttr<std::string>(node, "out_layout");
    auto wgh_layout = GetNodeAttr<std::string>(node, "kernel_layout");

    // dst_layout == "" means to use data_layout
    if (dst_layout.empty()) dst_layout = src_layout;

    // Minus one for DNNL representation. No dilation for DNNL is 0, for relay is 1.
    for (auto& d : dilates) d--;

    // Take into account provided layout strings
    src_tr = src_tr.TreatAs(src_layout);
    dst_tr = dst_tr.TreatAs(dst_layout);
    wgh_tr = wgh_tr.TreatAs(wgh_layout);

    // Should support G mixed with O. Like { G*O, I, H, W }
    if (wgh_layout.find("G") == std::string::npos) {
      auto w_dims = wgh_tr.dims();
      w_dims[0] /= groups;
      w_dims.insert(w_dims.begin(), groups);
      wgh_tr = wgh_tr.Reshape(w_dims);
    }

    // Assumption that bias is correct and can be squeezed to 1D
    bias_tr = bias_tr.Reshape({dst_tr.dims()[1]});

    // Conv description.
    auto deconv_desc = dnnl::deconvolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
        src_tr.LayoutAny().desc(), wgh_tr.LayoutAny().desc(), bias_tr.LayoutAny().desc(),
        dst_tr.LayoutAny().desc(), strides, dilates, padding_l, padding_r);

    // Enable elementwise post-ops.
    auto deconv_prim_desc = dnnl::deconvolution_forward::primitive_desc(deconv_desc, attr, engine_);

    src_tr = src_tr.RequestLayout(deconv_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(deconv_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(deconv_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(deconv_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(deconv_prim_desc.scratchpad_desc());

    Submit(dnnl::deconvolution_forward(deconv_prim_desc), {{DNNL_ARG_SRC, src_tr},
                                                           {DNNL_ARG_WEIGHTS, wgh_tr},
                                                           {DNNL_ARG_BIAS, bias_tr},
                                                           {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
                                                           {DNNL_ARG_DST, dst_tr}});
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();

    // Setup attributes.
    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};

    auto attr = ParseAttrs(nid, &bias_tr);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Assumption that bias is correct and can be squeezed to 1D
    bias_tr = bias_tr.Reshape({dst_tr.dims()[1]});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, src_tr.LayoutAny().desc(), wgh_tr.LayoutAny().desc(),
        bias_tr.LayoutAny().desc(), dst_tr.LayoutAny().desc());

    // Enable elementwise post-ops.
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, attr, engine_);

    src_tr = src_tr.RequestLayout(dense_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(dense_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(dense_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(dense_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(dense_prim_desc.scratchpad_desc());

    // TODO(@apeskov): Simulation of inplace primitive. just as PoC.
    auto sum_in_tr = GetInputByName(nid, "sum_idx");
    if (op_name.find("_sum") != std::string::npos) {
      sum_in_tr = GetInput(nid, node.GetInputs().size() - 1);
    }

    Submit(dnnl::inner_product_forward(dense_prim_desc),
           {{DNNL_ARG_SRC, src_tr},
            {DNNL_ARG_WEIGHTS, wgh_tr},
            {DNNL_ARG_BIAS, bias_tr},
            {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
            {DNNL_ARG_DST, dst_tr}},
           {sum_in_tr, DNNL_ARG_DST});
  }

  void BatchMatMul(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};

    auto attr = ParseAttrs(nid, &bias_tr);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    bool transpose_a = GetNodeAttr<bool>(node, "transpose_a");
    bool transpose_b = GetNodeAttr<bool>(node, "transpose_b");

    if (transpose_a) {
      src_tr = src_tr.Permute({0, 2, 1});
    }
    if (transpose_b) {
      wgh_tr = wgh_tr.Permute({0, 2, 1});
    }

    // Assumption that bias is correct and can be squeezed to 1D
    bias_tr = bias_tr.Reshape({dst_tr.dims()[1]});

    // Matmul description.
    auto bmm_desc = dnnl::matmul::desc(src_tr.LayoutAny().desc(), wgh_tr.LayoutAny().desc(),
                                       bias_tr.LayoutAny().desc(), dst_tr.LayoutAny().desc());

    // Enable elementwise post-ops.
    auto bmm_prim_desc = dnnl::matmul::primitive_desc(bmm_desc, attr, engine_);

    src_tr = src_tr.RequestLayout(bmm_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(bmm_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(bmm_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(bmm_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(bmm_prim_desc.scratchpad_desc());

    Submit(dnnl::matmul(bmm_prim_desc), {{DNNL_ARG_SRC, src_tr},
                                         {DNNL_ARG_WEIGHTS, wgh_tr},
                                         {DNNL_ARG_BIAS, bias_tr},
                                         {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
                                         {DNNL_ARG_DST, dst_tr}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto src_tr = GetInput(nid, 0);
    auto gamma_tr = GetInput(nid, 1);
    auto beta_tr = GetInput(nid, 2);
    auto mean_tr = GetInput(nid, 3);
    auto var_tr = GetInput(nid, 4);
    auto dst_tr = GetOutput(nid, 0);

    auto axis = GetNodeAttr<int>(node, "axis");
    auto epsilon = GetNodeAttr<float>(node, "epsilon");
    auto center = GetNodeAttr<bool>(node, "center");
    auto scale = GetNodeAttr<bool>(node, "scale");

    ICHECK(axis == 1 && center && scale) << "Unimplemented BatchNorm case";

    auto bn_desc = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, src_tr.desc(), epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(bn_desc, engine_);

    // Concatenate scale and shift tensors
    auto scale_shift_tr = TensorRequisite::AsIs(bn_prim_desc.weights_desc(), GenUniqueEid());
    auto sc_sh_dims = scale_shift_tr.dims();
    ICHECK(sc_sh_dims.size() == 2);
    ICHECK(sc_sh_dims[0] == 2);
    sc_sh_dims[0] /= 2;
    auto scale_tr = scale_shift_tr.Crop(sc_sh_dims, {0, 0}).Squeeze();
    auto shift_tr = scale_shift_tr.Crop(sc_sh_dims, {1, 0}).Squeeze();

    auto register_copy = [this](const TensorRequisite& src, const TensorRequisite& dst) {
      dnnl::reorder::primitive_desc copy_pd(engine_, src.desc(), engine_, dst.desc());
      Submit(dnnl::reorder(copy_pd), {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
    };

    register_copy(gamma_tr, scale_tr);
    register_copy(beta_tr, shift_tr);

    Submit(dnnl::batch_normalization_forward(bn_prim_desc), {{DNNL_ARG_SRC, src_tr},
                                                             {DNNL_ARG_DST, dst_tr},
                                                             {DNNL_ARG_SCALE_SHIFT, scale_shift_tr},
                                                             {DNNL_ARG_MEAN, mean_tr},
                                                             {DNNL_ARG_VARIANCE, var_tr}});
  }

  void LayerNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto src_tr = GetInput(nid, 0);
    auto gamma_tr = GetInput(nid, 1);
    auto beta_tr = GetInput(nid, 2);
    auto dst_tr = GetOutput(nid, 0);

    auto axis = GetNodeAttr<int>(node, "axis");
    auto epsilon = GetNodeAttr<float>(node, "epsilon");
    auto center = GetNodeAttr<bool>(node, "center");
    auto scale = GetNodeAttr<bool>(node, "scale");

    ICHECK(axis == -1 && center && scale) << "Unimplemented LayerNorm case";

    // LN description.
    auto lnorm_desc = dnnl::layer_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, src_tr.desc(), epsilon,
        dnnl::normalization_flags::use_scale_shift);

    auto lnorm_prim_desc = dnnl::layer_normalization_forward::primitive_desc(lnorm_desc, engine_);

    // Concatenate scale and shift tensors
    auto scale_shift_tr = TensorRequisite::AsIs(lnorm_prim_desc.weights_desc(), GenUniqueEid());
    auto sc_sh_dims = scale_shift_tr.dims();

    ICHECK(sc_sh_dims.size() == 2);
    ICHECK(sc_sh_dims[0] == 2);
    sc_sh_dims[0] /= 2;
    auto scale_tr = scale_shift_tr.Crop(sc_sh_dims, {0, 0}).Squeeze();
    auto shift_tr = scale_shift_tr.Crop(sc_sh_dims, {1, 0}).Squeeze();

    auto register_copy = [this](const TensorRequisite& src, const TensorRequisite& dst) {
      dnnl::reorder::primitive_desc copy_pd(engine_, src.desc(), engine_, dst.desc());
      Submit(dnnl::reorder(copy_pd), {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
    };

    register_copy(gamma_tr, scale_tr);
    register_copy(beta_tr, shift_tr);

    Submit(
        dnnl::layer_normalization_forward(lnorm_prim_desc),
        {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}, {DNNL_ARG_SCALE_SHIFT, scale_shift_tr}});
  }

  void Pooling(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    bool is_global = op_name.find("global") != std::string::npos;

    auto src_tr = GetInput(nid, 0);
    auto dst_tr = GetOutput(nid, 0);

    // Get layout.
    auto src_layout = GetNodeAttr<std::string>(node, "layout");
    auto dst_layout = GetNodeAttr<std::string>(node, "out_layout");

    // dst_layout == "" means to use data_layout
    if (dst_layout.empty()) dst_layout = src_layout;

    // Take into account provided layout strings
    src_tr = src_tr.TreatAs(src_layout);
    dst_tr = dst_tr.TreatAs(dst_layout);

    ICHECK(src_tr.dims().size() > 2);

    std::vector<int64_t> feature_size;
    for (size_t i = 2; i < src_tr.dims().size(); i++) {
      feature_size.push_back(int64_t(src_tr.dims()[i]));
    }

    // Set attributes.
    auto kernel = is_global ? feature_size : GetNodeAttr<std::vector<int64_t>>(node, "pool_size");
    auto strides = is_global ? std::vector<int64_t>(src_tr.dims().size() - 2, 1)
                             : GetNodeAttr<std::vector<int64_t>>(node, "strides");
    auto dilates = is_global ? std::vector<int64_t>(src_tr.dims().size() - 2, 1)
                             : GetNodeAttr<std::vector<int64_t>>(node, "dilation");
    auto padding = is_global ? std::vector<int64_t>((src_tr.dims().size() - 2) * 2, 0)
                             : GetNodeAttr<std::vector<int64_t>>(node, "padding");
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());

    // Minus one for DNNL representation. No dilation for DNNL is 0, for relay is 1.
    for (auto& d : dilates) d--;

    // Attributes related to AvgPool
    if (!is_global && algo == dnnl::algorithm::pooling_avg) {
      auto include_pad = GetNodeAttr<bool>(node, "count_include_pad");
      algo = include_pad ? dnnl::algorithm::pooling_avg_include_padding
                         : dnnl::algorithm::pooling_avg_exclude_padding;
    }

    // Pooling description.
    auto pool_desc = dnnl::pooling_v2_forward::desc(
        dnnl::prop_kind::forward_inference, algo, src_tr.desc(),  //<= Do not use any for src tensor
        dst_tr.LayoutAny().desc(), strides, kernel, dilates, padding_l, padding_r);
    auto pool_prim_desc = dnnl::pooling_v2_forward::primitive_desc(pool_desc, engine_);

    src_tr = src_tr.RequestLayout(pool_prim_desc.src_desc());
    dst_tr = dst_tr.RequestLayout(pool_prim_desc.dst_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(pool_prim_desc.scratchpad_desc());

    Submit(dnnl::pooling_v2_forward(pool_prim_desc),
           {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}, {DNNL_ARG_SCRATCHPAD, scratchpad_tr}});
  }

  void Eltwise(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    auto algo = elt_name2algo.at(op_name);

    auto src_tr = GetInput(nid, 0);
    auto dst_tr = GetOutput(nid, 0);

    float alpha = 0., beta = 0.;
    if (op_name == "clip") {
      alpha = GetNodeAttr<float>(node, "a_min");
      beta = GetNodeAttr<float>(node, "a_max");
    } else if (op_name == "nn.leaky_relu") {
      alpha = GetNodeAttr<float>(node, "alpha");
    }

    auto elt_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algo,
                                                src_tr.desc(), alpha, beta);
    auto elt_prim_desc = dnnl::eltwise_forward::primitive_desc(elt_desc, engine_);
    ICHECK(src_tr.desc() == elt_prim_desc.dst_desc());

    Submit(dnnl::eltwise_forward(elt_prim_desc), {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];

    auto src_tr = GetInput(nid, 0);
    auto dst_tr = GetOutput(nid, 0);

    auto axis = GetNodeAttr<int>(node, "axis");
    if (axis < 0) {
      axis = src_tr.dims().size() + axis;
    }

    auto softmax_desc =
        dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference, src_tr.desc(), axis);
    auto softmax_prim_desc = dnnl::softmax_forward::primitive_desc(softmax_desc, engine_);
    ICHECK(dst_tr.desc() == softmax_prim_desc.dst_desc());

    Submit(dnnl::softmax_forward(softmax_prim_desc),
           {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  void Binary(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];
    ICHECK_EQ(node.GetInputs().size(), 2U);

    // Memory and compute description.
    auto lhs_tr = GetInput(nid, 0);
    auto rhs_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);

    lhs_tr = lhs_tr.Broadcast(dst_tr.dims());
    rhs_tr = rhs_tr.Broadcast(dst_tr.dims());

    auto binary_desc = dnnl::binary::desc(algo, lhs_tr.desc(), rhs_tr.desc(), dst_tr.desc());
    auto binary_prim_desc = dnnl::binary::primitive_desc(binary_desc, engine_);

    Submit(dnnl::binary(binary_prim_desc),
           {{DNNL_ARG_SRC_0, lhs_tr}, {DNNL_ARG_SRC_1, rhs_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
  T AttrConvert(std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return std::stol(val[0]);
  }

  template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
  T AttrConvert(std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return std::stof(val[0]);
  }

  template <typename T, std::enable_if_t<std::is_same<T, std::string>::value, int> = 0>
  T AttrConvert(std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return val[0];
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, std::vector<typename T::value_type>>::value, int> = 0>
  T AttrConvert(std::vector<std::string> val) {
    T res;
    for (const auto& el : val) res.push_back(AttrConvert<typename T::value_type>({el}));
    return res;
  }

  /*!
   * \brief Helper to extract node attribute with ability to specify default value and result type.
   */
  template <typename T>
  const T GetNodeAttr(const json::JSONGraphNode& node, std::string name,
                      std::vector<std::string> def = {}) {
    auto attr = node.HasAttr(name) ? node.GetAttr<std::vector<std::string>>(name) : def;
    return AttrConvert<T>(attr);
  }

  TensorRequisite GetInput(const size_t& nid, const int idx) {
    if (idx == -1) return {};  // -1 reserved value for empty input.

    const JSONGraphNode& node = nodes_[nid];

    ICHECK_LT(idx, node.GetInputs().size());
    auto data_entry = node.GetInputs()[idx];

    auto shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto dtype = nodes_[data_entry.id_].GetOpDataType()[data_entry.index_];
    auto eid = node_row_ptr_[data_entry.id_] + data_entry.index_;
    auto const_dl_tensor = data_entry_[eid];

    auto desc = MakePlainDesc(shape, dtype);

    TensorRequisite res;
    if (const_dl_tensor) {
      ICHECK(const_dl_tensor->data);
      ICHECK(const_dl_tensor->strides == nullptr);
      auto mem = dnnl::memory(desc, engine_, const_dl_tensor->data);
      res = TensorRequisite::AsIs(mem, eid);
    } else {
      res = TensorRequisite::AsIs(desc, eid);
    }
    return res;
  }

  TensorRequisite GetInputByName(const size_t& nid, const std::string& name) {
    auto idx = GetNodeAttr<int>(nodes_[nid], name, {"-1"});
    return GetInput(nid, idx);
  }

  TensorRequisite GetOutput(const size_t& nid, const int idx) {
    if (idx == -1) return {};  // -1 reserved value for empty input.
    const JSONGraphNode& node = nodes_[nid];

    ICHECK_LT(idx, node.GetNumOutput());
    auto shape = node.GetOpShape()[idx];
    auto dtype = node.GetOpDataType()[idx];
    auto eid = node_row_ptr_[nid] + static_cast<uint32_t>(idx);

    ICHECK(data_entry_[eid] == nullptr);

    auto desc = MakePlainDesc(shape, dtype);

    return TensorRequisite::AsIs(desc, eid).Backward();
  }

  bool IsIntermidate(const TensorRequisite& tr) {
    auto eid = tr.eid();
    bool is_input = std::find(input_nodes_.begin(), input_nodes_.end(), eid) != input_nodes_.end();
    bool is_output = std::any_of(outputs_.begin(), outputs_.end(),
                                 [eid](auto& output) { return output.id_ == eid; });
    if (is_input || is_output) {
      return false;
    } else {
      return true;
    }
  }

  /*! \brief Helper function to register primitive into execution queue */
  void Submit(const dnnl::primitive& prim, const std::unordered_map<int, TensorRequisite>& tr_args,
              const std::pair<TensorRequisite, int>& inplace_conf = {}) {
    // Register all provided TR arguments
    std::unordered_map<int, TensorRegistry::ArgId> prim_arg_id;
    TensorRegistry::ActionQue post_prim_actions;

    // mark inplace tr
    if (auto tr_in = inplace_conf.first) {
      auto tr_out = tr_args.at(inplace_conf.second);
      if (IsIntermidate(tr_in) && IsIntermidate(tr_out)) {
        tensor_registry_.Register(tr_in, &net_);
        tensor_registry_.MarkInplace(tr_out, tr_in);
      }
    }

    for (const auto& kvp : tr_args) {
      const auto& key = kvp.first;
      const auto& tr = kvp.second;

      if (!tr.defined()) continue;  // empty arg is admitted. Just skip it
      auto arg_id = tensor_registry_.Register(tr, tr.IsReversed() ? &post_prim_actions : &net_);
      prim_arg_id[key] = arg_id;
    }

    // Simulate inplace primitive, the reorder with src==dst will be skipped in Run()
    if (auto tr = inplace_conf.first) {
      auto arg_id = tensor_registry_.Register(tr, &net_);
      auto dst_tr = tr_args.at(inplace_conf.second);
      auto dst_arg_id = prim_arg_id.at(inplace_conf.second);

      // Register copy action direct before main primitive
      dnnl::reorder::primitive_desc io_copy_pd(engine_, tr.desc(), engine_, dst_tr.desc());
      net_.push_back(
          {dnnl::reorder(io_copy_pd), {{DNNL_ARG_SRC, arg_id}, {DNNL_ARG_DST, dst_arg_id}}});
    }

    // Register main primitive
    net_.push_back({prim, prim_arg_id});

    // Register post actions
    net_.insert(net_.end(), post_prim_actions.begin(), post_prim_actions.end());
  }

  uint32_t GenUniqueEid() { return next_unique_eid_offset_++; }

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  TensorRegistry::ActionQue net_;
  /* Storage for all memory objects */
  TensorRegistry tensor_registry_;
  /* Generator of new unique eid which doesn't match with existing data entry */
  uint32_t next_unique_eid_offset_;
  /* Map of Run arg idx to corresponding eid */
  std::vector<uint32_t> run_arg_eid_;
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
