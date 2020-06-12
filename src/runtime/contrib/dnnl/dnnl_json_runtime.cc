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

#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "../../json/json_node.h"
#include "../../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  explicit DNNLJSONRuntime(const std::string& func_name, const std::string& graph_json)
      : JSONRuntimeBase(graph_json), func_name_(func_name) {}
  ~DNNLJSONRuntime() = default;

  const char* type_key() const { return "dnnljsonruntime"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (!this->is_init_) {
      Init();
      BuildEngine();
    }
    this->is_init_ = true;

    if (this->func_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        for (auto i = 0; i < args.size(); ++i) {
          // Setup data entries.
          CHECK(args[i].type_code() == kTVMNDArrayHandle ||
                args[i].type_code() == kTVMDLTensorHandle)
              << "Expect NDArray or DLTensor as inputs\n";
          if (args[i].type_code() == kTVMDLTensorHandle) {
            DLTensor* arg = args[i];
            this->data_entry_[i].CopyFrom(arg);
          } else {
            NDArray arg = args[i];
            this->data_entry_[i].CopyFrom(arg);
          }
        }

        // Execute the subgraph.
        this->Run();

        // Get result.
        auto offset = this->input_nodes_.size();
        for (size_t i = 0; i < this->outputs_.size(); ++i) {
          size_t idx = i + offset;
          if (args[idx].type_code() == kTVMDLTensorHandle) {
            DLTensor* arg = args[idx];
            this->data_entry_[idx].CopyTo(arg);
          } else {
            NDArray arg = args[idx];
            this->data_entry_[idx].CopyTo(arg);
          }
        }

        // FIXME: Multiple outputs.
        //*rv = data_entry_.back();
      });
    } else {
      LOG(WARNING) << "Unknown DNNL symbol " << name;
      return PackedFunc();
    }
  }

  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < this->input_nodes_.size(); ++i) {
      auto nid = this->input_nodes_[i];
      // TODO: Support other data lengths.
      size_t offset_in_bytes = this->node_out_mem_[nid][0].second * 4;
      write_to_dnnl_memory(this->data_entry_[i]->data, this->node_out_mem_[nid][0].first,
                           GetNDArraySize(this->data_entry_[i]), offset_in_bytes);
    }

    // Invoke the engine.
    for (size_t i = 0; i < net_.size(); ++i) {
      net_.at(i).execute(stream_, net_args_.at(i));
    }
    stream_.wait();

    // Read output buffers.
    auto offset = this->input_nodes_.size();
    for (size_t i = 0; i < this->outputs_.size(); ++i) {
      auto out_entry = this->outputs_[i];
      auto nid = out_entry.id_;
      auto idx = out_entry.index_;
      size_t offset_in_bytes = this->node_out_mem_[nid][idx].second * 4;
      read_from_dnnl_memory(this->data_entry_[offset + i]->data,
                            this->node_out_mem_[nid][idx].first,
                            GetNDArraySize(this->data_entry_[offset + i]), offset_in_bytes);
    }
  }

  void Init() override {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);
  }

  void BuildEngine() {
    // Build subgraph engine.
    for (size_t nid = 0; nid < this->nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("add" == op_name) {
          Add(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }

    // Initialize input/output entries.
    DLContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(1);
    ctx.device_id = 0;
    for (size_t i = 0; i < this->input_nodes_.size(); ++i) {
      auto shape = this->nodes_[this->input_nodes_[i]].GetOpShape()[0];
      this->data_entry_.push_back(NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx));
    }
    for (size_t i = 0; i < this->outputs_.size(); ++i) {
      auto entry = this->outputs_[i];
      auto shape = this->nodes_[entry.id_].GetOpShape()[entry.index_];
      this->data_entry_.push_back(NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx));
    }
  }

private:
  // Bind a JSON graph node entry to a DNNL memory.
 dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
                             size_t offset = 0) {
   if (node_out_mem_.count(entry.id_) == 0 || node_out_mem_[entry.id_].count(entry.index_) == 0) {
     return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
   }
   return node_out_mem_[entry.id_][entry.index_].first;
 }

 // Bind a JSON graph node entry to a given DNNL memory.
 dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
                             size_t offset = 0) {
   // Since the DNNL memory has been created before calling this function, we assume the entry
   // has not yet been bind to the other DNNL memory; otherwise it may have memory leak.
   CHECK(node_out_mem_.count(entry.id_) == 0 || node_out_mem_[entry.id_].count(entry.index_) == 0);

   // TODO: Support other data types (i.e., int8).
   auto data_node = nodes_[entry.id_];
   auto dltype = data_node.GetOpDataType()[entry.index_];
   CHECK_EQ(dltype.bits, 32);

   node_out_mem_[entry.id_][entry.index_] = {mem, offset};
   return node_out_mem_[entry.id_][entry.index_].first;
 }

  void Conv2d(const size_t& nid) {
    auto node = this->nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = this->nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape =
        this->nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    int groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    dnnl::memory::dim N = input_shape[0],       // batch size
        IC = input_shape[1],                    // input channels
        IH = input_shape[2],                    // input height
        IW = input_shape[2],                    // input width
        OC = weight_shape[0],                   // output channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width
    // std::cerr << N << ", " << IC << ", " << IH << ", " << IW << "\n";
    // std::cerr << OC << ", " << IC << ", " << KH << ", " << KW << "\n";
    // std::cerr << PH_L << ", " << PH_R << ", " << PW_L << ", " << PW_R << "\n";
    // std::cerr << SH << ", " << SW << "\n";
    // std::cerr << OH << ", " << OW << "\n";

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::nchw);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::a);

    // Covn2d description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);
    dnnl::primitive_attr attr;
    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    CHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
    auto conv2d_src_memory = BindDNNLMemory(data_entry, {src_dims, dt::f32, tag::nchw});

    // Weight memory.
    CHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
    auto conv2d_weights_memory = BindDNNLMemory(
        weight_entry, {weights_dims, dt::f32, (groups > 1) ? tag::goihw : tag::oihw});

    // Bias memory (useless for now as TVM conv2d op has no bias).
    std::vector<float> bias(OC, 0);
    auto conv2d_bias_memory = dnnl::memory({bias_dims, dt::f32, tag::x}, engine_, bias.data());

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, conv2d_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_BIAS, conv2d_bias_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
  }

  void Dense(const size_t& nid) {
    auto node = this->nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = this->nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape =
        this->nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim B = input_shape[0],  // batch size
        IC = input_shape[1],               // input channels
        OC = weight_shape[0];              // output channels

    // Memory shapes.
    dnnl::memory::dims data_dims = {B, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    std::vector<float> bias(OC, 0);
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    auto bias_memory = dnnl::memory(bias_md, engine_, bias.data());
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = this->nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    dnnl::memory::dims data_shape = this->nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
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

  void Relu(const size_t& nid) {
    auto node = this->nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = this->nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto data_md = dnnl::memory::desc{{shape}, dt::f32, tag::abcd};

    auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::eltwise_relu, data_md, 0);
    auto relu_prim_desc = dnnl::eltwise_forward::primitive_desc(relu_desc, engine_);
    CHECK(data_md == relu_prim_desc.dst_desc());

    auto relu = dnnl::eltwise_forward(relu_prim_desc);
    net_.push_back(relu);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto out_md = dnnl::memory::desc(shape, dt::f32, tag::abcd);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Add(const size_t& nid) {
    auto node = this->nodes_[nid];

    // Memory and compute description.
    std::vector<dnnl::memory::dims> data_dims;
    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;

    CHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = this->nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));
    }
    CHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    auto add_desc =
        dnnl::binary::desc(dnnl::algorithm::binary_add, data_mds[0], data_mds[1], out_md);
    auto add_prim_desc = dnnl::binary::primitive_desc(add_desc, engine_);
    auto add = dnnl::binary(add_prim_desc);
    net_.push_back(add);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + size, reinterpret_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy((uint8_t*)handle, (uint8_t*)handle + size, dst + offset);
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

// Calculate the size of a given NDArray in bytes.
inline size_t GetNDArraySize(const NDArray& arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr->ndim; ++i) {
    size *= static_cast<size_t>(arr->shape[i]);
  }
  size *= (arr->dtype.bits * arr->dtype.lanes + 7) / 8;
  return size;
}

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* \brief A simple pool to contain the tensor for each node in the graph. */
  std::vector<NDArray> data_entry_;
  /* The network layers that are represented in dnnl primitives. */
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
  /* The node ID to its corresponding output memory. */
  std::unordered_map < uint32_t,
      std::unordered_map<int, std::pair<dnnl::memory, size_t>>> node_out_mem_;
  /* Indicate if the DNNL engine has been initialized. */
  bool is_init_ = false;
  /* The only subgraph name for this module. */
  std::string func_name_;
};

TVM_REGISTER_GLOBAL("runtime.ext.dnnl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  auto n = tvm::runtime::make_object<DNNLJSONRuntime>(
      args[0].operator std::string(), args[1].operator std::string());
  *rv = Module(n);
});

runtime::Module DNNLJSONRuntimeCreate(std::string func_name, std::string graph_json) {
  auto n = make_object<DNNLJSONRuntime>(func_name, graph_json);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate")
.set_body_typed(DNNLJSONRuntimeCreate);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
