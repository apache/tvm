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
 * \file src/runtime/contrib/cudnn/cudnn_json_runtime.cc
 * \brief A simple JSON runtime for CUDNN.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef TVM_USE_CUDNN_FRONTEND
#include "./cudnn_frontend/attention.h"
#endif
#include "cudnn_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class cuDNNJSONRuntime : public JSONRuntimeBase {
 public:
  cuDNNJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                   const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  void Init(const Array<NDArray>& consts) override {
    op_execs_.resize(nodes_.size());
    // get some config from the graph
    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& node = nodes_[i];
      if (node.GetOpType() == "kernel") {
        std::string op_name = node.GetOpName();
        if (op_name.find("conv2d") != std::string::npos) {
          op_execs_[i] = GetConv2DExec(node);
        } else if (op_name.find("attention") != std::string::npos) {
          op_execs_[i] = GetAttentionExec(node);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  const char* type_key() const override { return "cudnn_json"; }  // May be overridden

  void Run() override {
    for (const auto& f : op_execs_) {
      if (f != nullptr) {
        f();
      }
    }
  }

 private:
  const DLTensor* GetInput(const JSONGraphNode& node, const int idx) {
    ICHECK_LT(idx, node.GetInputs().size());
    auto eid = EntryID(node.GetInputs()[idx]);
    ICHECK(eid < data_entry_.size());
    return data_entry_[eid];
  }

  bool attr_in_name(const std::string& op_name, const std::string& attr_name) {
    return op_name.find(attr_name) != std::string::npos;
  }

  std::vector<int> vstr2vint(const JSONGraphNode& node, const std::string& attrStr) {
    auto string_to_int = [](const std::string& str) { return std::stoi(str); };
    auto string_vec = node.GetAttr<std::vector<std::string>>(attrStr);
    std::vector<int> int_vec(string_vec.size());
    std::transform(string_vec.begin(), string_vec.end(), int_vec.begin(), string_to_int);
    return int_vec;
  }

  std::function<void()> GetConv2DExec(const JSONGraphNode& node) {
    auto* entry_ptr = tvm::contrib::CuDNNThreadEntry::ThreadLocal();
    auto op_name = node.GetOpName();

    std::vector<int> input_dims, kernel_dims, output_dims;
    auto input_node = nodes_[0];
    auto input_shapes = input_node.GetOpShape()[0];
    auto kernel_shapes = nodes_[1].GetOpShape()[0];
    auto output_shapes = node.GetOpShape()[0];
    for (const auto& _i : input_shapes) {
      input_dims.emplace_back(static_cast<int>(_i));
    }
    for (const auto& _i : kernel_shapes) {
      kernel_dims.emplace_back(static_cast<int>(_i));
    }
    for (const auto& _i : output_shapes) {
      output_dims.emplace_back(static_cast<int>(_i));
    }
    bool has_bias = attr_in_name(op_name, "bias");
    int groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    std::vector<int> padding = vstr2vint(node, "padding");
    std::vector<int> strides = vstr2vint(node, "strides");
    std::vector<int> dilation = vstr2vint(node, "dilation");
    auto conv_dtype = node.GetAttr<std::vector<std::string>>("out_dtype")[0];
    std::string layout = node.GetAttr<std::vector<std::string>>("out_layout")[0];
    int dims = layout.size() - 2;  // remove O and I dims

    int format = CUDNN_TENSOR_NHWC;
    if (layout == "NCHW") {
      format = CUDNN_TENSOR_NCHW;
    } else if (layout == "NHWC") {
      format = CUDNN_TENSOR_NHWC;
    } else {
      LOG(FATAL) << "Unsupported layout: " << layout;
    }

    int act = CUDNN_ACTIVATION_IDENTITY;
    double coef = 1.0;
    if (attr_in_name(op_name, "relu")) {
      act = CUDNN_ACTIVATION_RELU;
    } else if (attr_in_name(op_name, "relu6")) {
      act = CUDNN_ACTIVATION_CLIPPED_RELU;
      coef = 6.0;
    } else if (attr_in_name(op_name, "leaky_relu")) {
      act = CUDNN_ACTIVATION_RELU;
      coef = 0.1;
    }

    /*conv mode: CUDNN_CROSS_CORRELATION by default*/
    int mode = CUDNN_CROSS_CORRELATION;

    // find best algo
    TVMRetValue best_algo;

    tvm::contrib::FindAlgo(format, dims, groups, padding.data(), strides.data(), dilation.data(),
                           input_dims.data(), kernel_dims.data(), output_dims.data(), conv_dtype,
                           conv_dtype, false, &best_algo);

    int algo = best_algo.operator int();
    std::function<void()> op_exec = [=]() {
      auto stream = static_cast<cudaStream_t>(GetCUDAStream());
      CUDNN_CALL(cudnnSetStream(entry_ptr->handle, stream));

      auto get_inputs = [this](const JSONGraphNode& node, bool has_bias) {
        const DLTensor* bias = nullptr;
        if (has_bias) {
          bias = GetInput(node, 2);
        }
        return std::make_tuple(GetInput(node, 0), GetInput(node, 1), bias);
      };

      auto [a_ptr, b_ptr, bias_ptr] = get_inputs(node, has_bias);
      uint32_t output_eid = EntryID(outputs_[0]);
      auto out_ptr = data_entry_[output_eid];
      if (has_bias) {
        tvm::contrib::ConvolutionBiasActivationForward(
            mode, format, algo, dims, groups, act, coef, padding.data(), strides.data(),
            dilation.data(), a_ptr, b_ptr, out_ptr, bias_ptr, conv_dtype);
      } else {
        tvm::contrib::ConvolutionForward(mode, format, algo, dims, groups, padding.data(),
                                         strides.data(), dilation.data(), a_ptr, b_ptr, out_ptr,
                                         conv_dtype);
      }
    };
    return op_exec;
  }

  std::function<void()> GetAttentionExec(const JSONGraphNode& node) {
#ifdef TVM_USE_CUDNN_FRONTEND
    auto dtype = node.GetOpDataType()[0];
    int num_heads = vstr2vint(node, "num_heads")[0];
    int num_kv_heads = vstr2vint(node, "num_kv_heads")[0];
    int head_size = vstr2vint(node, "head_size")[0];
    int head_size_v = vstr2vint(node, "head_size_v")[0];
    std::string layout = node.GetAttr<std::vector<std::string>>("layout")[0];
    const auto& input_qkv_node = nodes_[EntryID(node.GetInputs()[0])];
    auto qkv_shapes = input_qkv_node.GetOpShape()[0];

    int64_t batch, seq_len;
    if (layout == "BS3NH") {
      ICHECK_EQ(qkv_shapes.size(), 3);
      batch = qkv_shapes[0];
      seq_len = qkv_shapes[1];
    } else if (layout == "SBN3H") {
      ICHECK_EQ(qkv_shapes.size(), 4);
      batch = qkv_shapes[1];
      seq_len = qkv_shapes[0];
    } else {
      LOG(FATAL) << "Unsupported layout: " << layout;
    }
    double scale = 1 / std::sqrt(head_size);
    std::string scale_attr = node.GetAttr<std::vector<std::string>>("scale")[0];
    if (scale_attr.size()) {
      scale = std::stod(scale_attr);
    }

    auto runner = tvm::contrib::CuDNNSDPARunner::Create();
    runner->Init(batch, seq_len, num_heads, num_kv_heads, head_size, head_size_v, scale, dtype,
                 layout);
    return [=]() {
      auto qkv = GetInput(node, 0);
      auto workspace = const_cast<DLTensor*>(GetInput(node, 1));
      auto out = const_cast<DLTensor*>(data_entry_[EntryID(outputs_[0])]);
      runner->Run(qkv, workspace, out);
    };
#else
    LOG(FATAL) << "Please build with CUDNN frontend to use attention op";
#endif
  }

  std::vector<std::function<void()>> op_execs_;
};

runtime::Module cuDNNJSONRuntimeCreate(String symbol_name, String graph_json,
                                       const Array<String>& const_names) {
  auto n = make_object<cuDNNJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.cuDNNJSONRuntimeCreate").set_body_typed(cuDNNJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cudnn_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<cuDNNJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
