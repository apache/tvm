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

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"
#include "dnnl_node_helper.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        g_explorer_(nodes_, data_entry_, node_row_ptr_, engine_) {}

  const char* type_key() const override { return "dnnl_json"; }

  static std::string get_version() {
    auto v = dnnl_version();
    std::stringstream ver_strm;
    ver_strm << v->major << '.' << v->minor << '.' << v->patch;
    return ver_strm.str();
  }

  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    // Setup constants entries for weights.
    SetupConstants(consts);
    // Init internal DNNL specific objects
    BuildEngine();
  }

  /**
   * Override of GetFunction methods to replace main symbol_name_ implementation with
   * thread safe one.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override {
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

  /**
   * @brief Thread safe version of base method Run.
   *
   * The main purpose of this overwrite is to make symbol_name_ function thread safe.
   * The base implementation of that method is using SetInputOutputBuffers() which
   * is not thread safe and lead to changes in DNNLJSONRuntime itself.
   *
   * @param args kernel arguments
   */
  void Run(const TVMArgs& args) const {
    auto io_data_provider = makeIoDataProvider(args);
    // Execute primitives one by one
    for (const auto& act : net_) {
      auto req_args = std::get<TensorRegistry::ArgReqSet>(act);
      auto prim = std::get<dnnl::primitive>(act);

      // Find proper dnnl::memory buffer based on provided ArgRequisite
      auto mem_args = tensor_registry_.solve(req_args, io_data_provider);
      prim.execute(stream_, mem_args);
    }
  }

  /** @brief Stub implementation */
  void Run() override { LOG(ERROR) << "Unimplemented. Should never be called."; }

 private:
  /** Receive tensor memory buffer handler based from provided arg */
  static void* extractDataHandle(const TVMArgValue& val) {
    ICHECK(val.type_code() == kTVMNDArrayHandle || val.type_code() == kTVMDLTensorHandle)
        << "Expect NDArray or DLTensor";
    void* hdl = nullptr;
    if (val.IsObjectRef<NDArray>()) {
      NDArray arr = val;
      hdl = arr.operator->()->data;
    } else {
      hdl = val.operator DLTensor*()->data;
    }
    return hdl;
  }

  TensorRegistry::ExtDataProvider makeIoDataProvider(const TVMArgs& args) const {
    std::map<uint32_t, void*> io_map;  // eid to data handler

    int i = 0;
    for (auto e : input_var_eid_) io_map[e] = extractDataHandle(args[i++]);
    for (auto e : outputs_) io_map[EntryID(e)] = extractDataHandle(args[i++]);

    // lambda with captured IO data handlers
    return [io_map](uint32_t eid) -> void* { return io_map.at(eid); };
  }

  std::set<uint32_t> makeIoEids() const {
    std::set<uint32_t> io_set;  // eid of inputs and outputs
    for (auto e : input_var_eid_) io_set.insert(e);
    for (auto e : outputs_) io_set.insert(EntryID(e));
    return io_set;
  }

  struct SubmitAttr {
    enum AttrType { None, ZeroCopyRequest };

    SubmitAttr() {}
    SubmitAttr(AttrType type, const TensorRequisite& tr, int flag)
        : type_(type), tr_(tr), flag_(flag) {}

    AttrType type_ = AttrType::None;
    const TensorRequisite tr_ = {};
    int flag_ = 0;
  };

  // Helper function to register primitive into execution queue
  void submit(const dnnl::primitive& prim, const std::unordered_map<int, TensorRequisite>& tr_args,
              const SubmitAttr attr = {}) {
    // collection of post action. Dst primitive processing will be stored here
    TensorRegistry::ActionQue post_actions;

    // Helper func to register TensorRequisite and store corresponding Actions in proper place
    auto register_tr = [this, &post_actions](const TensorRequisite& tr) {
      TensorRegistry::ArgReq arg_req;
      TensorRegistry::ActionQue actions;
      std::tie(arg_req, actions) = tensor_registry_.registerTR(tr);

      auto& action_queue = tr.isReversed() ? post_actions : net_;
      action_queue.insert(action_queue.end(), actions.begin(), actions.end());
      return arg_req;
    };

    // Register all provided TR arguments
    std::unordered_map<int, TensorRegistry::ArgReq> arg_reqs;
    for (const auto& kvp : tr_args) {
      const auto& tr = kvp.second;
      const auto& key = kvp.first;

      if (!tr.defined()) continue;  // empty arg is admitted. Just skip it
      arg_reqs[key] = register_tr(tr);
    }

    // ZeroCopyRequest or Inplace memory
    if (attr.type_ == SubmitAttr::ZeroCopyRequest) {
      auto zero_copy_src_tr = attr.tr_;
      auto zero_copy_dst_tr = tr_args.at(attr.flag_);
      auto zero_copy_src_ar = register_tr(zero_copy_src_tr);
      auto zero_copy_dst_ar = arg_reqs.at(attr.flag_);

      // Register copy action direct before main primitive
      dnnl::reorder::primitive_desc io_copy_pd(engine_, zero_copy_src_tr.desc(), engine_,
                                               zero_copy_dst_tr.desc());
      net_.push_back({dnnl::reorder(io_copy_pd),
                      {{DNNL_ARG_SRC, zero_copy_src_ar}, {DNNL_ARG_DST, zero_copy_dst_ar}}});
    }

    // Register main primitive
    net_.push_back({prim, arg_reqs});

    // Register post actions
    net_.insert(net_.end(), post_actions.begin(), post_actions.end());
  }

  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);
    tensor_registry_ = TensorRegistry(engine_, makeIoEids());

    // Build subgraph engine.
    for (uint32_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();

        if ("nn.conv2d" == op_name ||
            "dnnl.conv2d_relu" == op_name ||
            "dnnl.conv2d_tanh" == op_name ||
            "dnnl.conv2d_sigmoid" == op_name ||
            "dnnl.conv2d_bias" == op_name ||
            "dnnl.conv2d_bias_relu" == op_name ||
            "dnnl.conv2d_bias_tanh" == op_name ||
            "dnnl.conv2d_bias_sigmoid" == op_name ||
            "dnnl.qnn.conv2d" == op_name ||
            "dnnl.qnn.conv2d_sum" == op_name) {
          UniConv2d(nid);
        } else if ("nn.dense" == op_name ||
                   "dnnl.dense_relu" == op_name ||
                   "dnnl.dense_tanh" == op_name ||
                   "dnnl.dense_sigmoid" == op_name ||
                   "dnnl.dense_bias" == op_name ||
                   "dnnl.dense_bias_relu" == op_name ||
                   "dnnl.dense_bias_tanh" == op_name ||
                   "dnnl.dense_bias_sigmoid" == op_name ||
                   "dnnl.qnn.dense" == op_name ||
                   "dnnl.qnn.dense_sum" == op_name) {
          UniDense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Eltwise(nid, dnnl::algorithm::eltwise_relu);
        } else if ("tanh" == op_name) {
          Eltwise(nid, dnnl::algorithm::eltwise_tanh);
        } else if ("sigmoid" == op_name) {
          Eltwise(nid, dnnl::algorithm::eltwise_logistic);
        } else if ("add" == op_name) {
          Binary(nid, dnnl::algorithm::binary_add);
        } else if ("multiply" == op_name) {
          Binary(nid, dnnl::algorithm::binary_mul);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }

    tensor_registry_.finalize();
  }

  void UniConv2d(const uint32_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    // Fix position inputs
    auto data_tr = node.getInput(0);
    auto kernel_tr = node.getInput(1);
    auto output_tr = node.getOutput(0);

    // Parse general conv attributes
    auto strides = node.getAttr<std::vector<int>>("strides");
    auto padding = node.getAttr<std::vector<int>>("padding");
    auto dilation = node.getAttr<std::vector<int>>("dilation");
    auto groups = node.getAttr<dnnl::memory::dim>("groups");

    auto data_layout = node.getAttr<std::string>("data_layout");
    auto kernel_layout = node.getAttr<std::string>("kernel_layout");

    auto activation = node.getAttr<std::vector<std::string>>("activation", {"none"});
    auto bias_idx = node.getAttr<int>("bias_idx", {"-1"});
    auto sum_idx = node.getAttr<int>("sum_idx", {"-1"});
    auto sum_scl_idx = node.getAttr<int>("sum_scl_idx", {"-1"});
    auto o_scl_idx = node.getAttr<int>("o_scl_idx", {"-1"});
    auto dst_zp_idx = node.getAttr<int>("dst_zp_idx", {"-1"});

    // may be empty in case if '-1'
    auto bias_tr = node.getInput(bias_idx);
    auto sum_tr = node.getInput(sum_idx);
    auto sum_scl_tr = node.getInput(sum_scl_idx);
    auto o_scl_tr = node.getInput(o_scl_idx);
    auto dst_zp_tr = node.getInput(dst_zp_idx);

    // permute corresponding with provided layouts
    auto data_permutation = utils::permutation(data_layout, "NCHW");
    auto kernel_permutation = utils::permutation(kernel_layout, "OIHW");

    data_tr = data_tr.permute(data_permutation);
    sum_tr = sum_tr.permute(data_permutation);
    output_tr = output_tr.permute(data_permutation);
    kernel_tr = kernel_tr.permute(kernel_permutation);

    // TODO(@apeskov): temp WA. while codegen is not able to guarantee 1D format of bias data
    bias_tr = bias_tr.squeeze();

    // Group weight format
    if (groups > 1) {
      auto k_dims = kernel_tr.dims();  // OIHW -> GOIHW
      k_dims[0] /= groups;
      k_dims.insert(k_dims.begin(), groups);
      kernel_tr = kernel_tr.reshape(k_dims);
    }

    // Attributes setting
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (dst_zp_tr) {
      auto zp = dst_zp_tr.getConstScalarData<int32_t>();
      // Per channel zp is not supported. It was merged into BIAS
      attr.set_zero_points(DNNL_ARG_DST, 0, {zp});
    }

    if (o_scl_tr) {
      ICHECK(o_scl_tr.isConstant());
      auto data = o_scl_tr.getConstDataLikeVec<float>();
      attr.set_output_scales(data.size() == 1 ? 0 : (1 << 1), data);
    }

    if (activation[0] != "none") {
      auto a_type = utils::convert2dnnl_activation(activation[0]);
      auto a_scale = node.getInput(std::stoi(activation[1])).getConstScalarData<float>();
      auto a_alfa = node.getInput(std::stoi(activation[2])).getConstScalarData<float>();
      auto a_beta = node.getInput(std::stoi(activation[3])).getConstScalarData<float>();

      auto ops = attr.get_post_ops();
      ops.append_eltwise(a_scale, a_type, a_alfa, a_beta);
      attr.set_post_ops(ops);
    }

    if (sum_scl_tr) {
      auto scl = sum_scl_tr.getConstScalarData<float>();
      auto ops = attr.get_post_ops();
      ops.append_sum(scl);
      attr.set_post_ops(ops);
    }

    dnnl::memory::dim PW_L = padding[1],  // width padding: left
        PW_R = padding[3],                // width padding: right
        PH_L = padding[0],                // height padding: top
        PH_R = padding[2],                // height padding: bottom
        SH = strides[0],                  // height-wise stride
        SW = strides[1],                  // weight-wise stride
        DH = dilation[0] - 1,  // height-wise dilation, DNNL uses dilation format with - 1
        DW = dilation[1] - 1;  // weight-wise dilation

    // Conv description
    auto conv_d = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        data_tr.layoutAny().desc(), kernel_tr.layoutAny().desc(), bias_tr.layoutAny().desc(),
        output_tr.layoutAny().desc(), {SH, SW} /*strides*/, {DH, DW} /*dilation*/,
        {PH_L, PW_L} /*padding_l*/, {PH_R, PW_R} /*padding_r*/);

    auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_d, attr, engine_);
    auto conv = dnnl::convolution_forward(conv_pd);

    // Specify proper layouts
    data_tr = data_tr.requestLayout(conv_pd.src_desc());
    kernel_tr = kernel_tr.requestLayout(conv_pd.weights_desc());
    output_tr = output_tr.requestLayout(conv_pd.dst_desc());
    bias_tr = bias_tr.requestLayout(conv_pd.bias_desc());

    auto scratchpad_tr = node.makeScratchpad(conv_pd.scratchpad_desc());

    // Inplace request for conv+sum pattern. Match input with dst tensor
    auto submit_attr =
        sum_tr ? SubmitAttr{SubmitAttr::ZeroCopyRequest, sum_tr, DNNL_ARG_DST} : SubmitAttr{};

    // Register prim to execute
    submit(conv,
           {{DNNL_ARG_SRC, data_tr},
            {DNNL_ARG_WEIGHTS, kernel_tr},
            {DNNL_ARG_BIAS, bias_tr},
            {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
            {DNNL_ARG_DST, output_tr}},
           submit_attr);
  }

  void UniDense(const uint32_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto wgh_tr = node.getInput(1);
    auto dst_tr = node.getOutput(0);

    auto activation = node.getAttr<std::vector<std::string>>("activation", {"none"});
    auto bias_idx = node.getAttr<int>("bias_idx", {"-1"});
    auto sum_idx = node.getAttr<int>("sum_idx", {"-1"});
    auto sum_scl_idx = node.getAttr<int>("sum_scl_idx", {"-1"});
    auto o_scl_idx = node.getAttr<int>("o_scl_idx", {"-1"});
    auto dst_zp_idx = node.getAttr<int>("dst_zp_idx", {"-1"});

    // may be empty in case if '-1'
    auto bias_tr = node.getInput(bias_idx);
    auto sum_tr = node.getInput(sum_idx);
    auto sum_scl_tr = node.getInput(sum_scl_idx);
    auto o_scl_tr = node.getInput(o_scl_idx);
    auto dst_zp_tr = node.getInput(dst_zp_idx);

    // TODO(@apeskov): temp WA. while codegen is not able to guarantee 1D format of bias data
    bias_tr = bias_tr.squeeze();

    // Attributes setting
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    ICHECK(!dst_zp_tr) << "DNNL doesn't support input zero point for optimized primitives."
                          "Should be merged into bias";

    if (o_scl_tr) {
      ICHECK(o_scl_tr.isConstant());
      auto data = o_scl_tr.getConstDataLikeVec<float>();
      attr.set_output_scales(data.size() == 1 ? 0 : (1 << 1), data);
    }

    if (activation[0] != "none") {
      auto a_type = utils::convert2dnnl_activation(activation[0]);
      auto a_scale = node.getInput(std::stoi(activation[1])).getConstScalarData<float>();
      auto a_alfa = node.getInput(std::stoi(activation[2])).getConstScalarData<float>();
      auto a_beta = node.getInput(std::stoi(activation[3])).getConstScalarData<float>();

      auto ops = attr.get_post_ops();
      ops.append_eltwise(a_scale, a_type, a_alfa, a_beta);
      attr.set_post_ops(ops);
    }

    if (sum_scl_tr) {
      auto scl = sum_scl_tr.getConstScalarData<float>();
      auto ops = attr.get_post_ops();
      ops.append_sum(scl);
      attr.set_post_ops(ops);
    }

    // Dense description.
    auto dense_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, src_tr.layoutAny().desc(), wgh_tr.layoutAny().desc(),
        bias_tr.layoutAny().desc(), dst_tr.layoutAny().desc());
    auto dense_pd = dnnl::inner_product_forward::primitive_desc(dense_d, attr, engine_);
    auto dense = dnnl::inner_product_forward(dense_pd);

    // Select proper layout
    src_tr = src_tr.requestLayout(dense_pd.src_desc());
    wgh_tr = wgh_tr.requestLayout(dense_pd.weights_desc());
    dst_tr = dst_tr.requestLayout(dense_pd.dst_desc());

    auto scratch_pad_d = node.makeScratchpad(dense_pd.scratchpad_desc());

    // Inplace request for conv+sum pattern. Match input with dst tensor
    auto submit_attr =
        sum_tr ? SubmitAttr{SubmitAttr::ZeroCopyRequest, sum_tr, DNNL_ARG_DST} : SubmitAttr{};

    submit(dense,
           {{DNNL_ARG_SRC, src_tr},
            {DNNL_ARG_WEIGHTS, wgh_tr},
            {DNNL_ARG_BIAS, bias_tr},
            {DNNL_ARG_SCRATCHPAD, scratch_pad_d},
            {DNNL_ARG_DST, dst_tr}},
           submit_attr);
  }

  void BatchNorm(const uint32_t& nid) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto gamma_tr = node.getInput(1);
    auto beta_tr = node.getInput(2);
    auto mean_tr = node.getInput(3);
    auto variance_tr = node.getInput(4);
    auto dst_tr = node.getOutput(0);

    auto axis = node.getAttr<int>("axis");
    auto epsilon = node.getAttr<float>("epsilon");
    auto center = node.getAttr<bool>("center");
    auto scale = node.getAttr<bool>("scale");

    // TODO(@apeskov): Add support of all type of axis, center and scale args
    ICHECK(axis == 1);
    ICHECK(center);
    ICHECK(scale);

    // TODO(@apeskov): Should it use "any" layout to select proper one?
    auto bn_d = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, dst_tr.desc(), epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_pd = dnnl::batch_normalization_forward::primitive_desc(bn_d, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_pd);

    src_tr = src_tr.requestLayout(bn_pd.src_desc());
    dst_tr = dst_tr.requestLayout(bn_pd.dst_desc());
    mean_tr = mean_tr.requestLayout(bn_pd.mean_desc());
    variance_tr = variance_tr.requestLayout(bn_pd.variance_desc());

    // TODO(@apeskov): DNNL v2.5 and late has API for separate scale and shift
    //                 it will eliminate requirements of data copy.
    // Prepare concatenated Scale and Shift tensor
    auto scale_shift_tr = node.makeTemp(bn_pd.weights_desc());
    auto sc_sh_dims = scale_shift_tr.dims();
    ICHECK(sc_sh_dims.size() == 2);
    ICHECK(sc_sh_dims[0] == 2);
    sc_sh_dims[0] /= 2;
    auto scale_tr = scale_shift_tr.crop(sc_sh_dims, {0, 0}).squeeze();
    auto shift_tr = scale_shift_tr.crop(sc_sh_dims, {1, 0}).squeeze();

    auto register_copy = [this](const TensorRequisite& src, const TensorRequisite& dst) {
      dnnl::reorder::primitive_desc copy_pd(engine_, src.desc(), engine_, dst.desc());
      submit(dnnl::reorder(copy_pd), {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
    };

    register_copy(gamma_tr, scale_tr);
    register_copy(beta_tr, shift_tr);

    submit(bn, {{DNNL_ARG_SRC, src_tr},
                {DNNL_ARG_DST, dst_tr},
                {DNNL_ARG_SCALE_SHIFT, scale_shift_tr},
                {DNNL_ARG_MEAN, mean_tr},
                {DNNL_ARG_VARIANCE, variance_tr}});
  }

  void Eltwise(const uint32_t& nid, dnnl::algorithm algo) {
    auto node = NodeHelper{nid, g_explorer_};

    auto src_tr = node.getInput(0);
    auto dst_tr = node.getOutput(0);
    ICHECK(src_tr.dims() == dst_tr.dims());
    // Eltwise op required same layout for src/dst
    src_tr = src_tr.requestLayout(dst_tr.desc());

    auto eltwise_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algo,
                                                 dst_tr.desc());
    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_d, engine_);
    auto eltwise = dnnl::eltwise_forward(eltwise_pd);

    submit(eltwise, {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  void Binary(const uint32_t& nid, dnnl::algorithm algo) {
    auto node = NodeHelper{nid, g_explorer_};

    auto lhs_tr = node.getInput(0);
    auto rhs_tr = node.getInput(1);
    auto out_tr = node.getOutput(0);

    lhs_tr = lhs_tr.broadcast(out_tr.dims());
    rhs_tr = rhs_tr.broadcast(out_tr.dims());

    // Any layouts cannot be used for binary prim
    auto binary_d = dnnl::binary::desc(algo, lhs_tr.desc(), rhs_tr.desc(), out_tr.desc());
    auto binary_pd = dnnl::binary::primitive_desc(binary_d, engine_);
    auto binary = dnnl::binary(binary_pd);

    // Request proper layouts
    lhs_tr = lhs_tr.requestLayout(binary_pd.src0_desc());
    rhs_tr = rhs_tr.requestLayout(binary_pd.src1_desc());
    out_tr = out_tr.requestLayout(binary_pd.dst_desc());

    submit(binary, {{DNNL_ARG_SRC_0, lhs_tr}, {DNNL_ARG_SRC_1, rhs_tr}, {DNNL_ARG_DST, out_tr}});
  }

  /** The dnnl engine. */
  dnnl::engine engine_;
  /** The dnnl stream. */
  dnnl::stream stream_;
  /** Tensor registry which manages all real dnnl memory objects */
  TensorRegistry tensor_registry_;
  /** The network layers that are represented as dnnl primitives plus there args. */
  TensorRegistry::ActionQue net_;
  /** Utility object */
  GraphExplorer g_explorer_;
};

runtime::Module DNNLJSONRuntimeCreate(const String& symbol_name, const String& graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

TVM_REGISTER_GLOBAL("runtime.module.dnnl_version")
    .set_body_typed(DNNLJSONRuntime::get_version);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
